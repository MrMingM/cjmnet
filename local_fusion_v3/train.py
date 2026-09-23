"""Online paired clean/simulated-weather full-scene training; no action cache."""
import argparse
import json
import torch
from gspr_evidence import runtime as er
from gspr_communication.runtime import device, seed_all, new_output, write_json, verify_frozen
from opencood.loss.point_pillar_loss import PointPillarLoss
from . import runtime as rt


def epoch(model, module, loader, criterion, target, settings, optimizer=None):
    from opencood.tools.train_utils import to_device
    module.train(optimizer is not None)
    model.eval()
    sums = dict(loss=0., detection_loss=0., change=0., gate=0.)
    count = updates = 0
    for batch in loader:
        batch = to_device(batch, target)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        for branch in ('clean', 'weather'):
            with torch.no_grad():
                ctx = rt.context(model, batch['ego'], branch, verify=count == 0)
            with torch.set_grad_enabled(optimizer is not None):
                prediction, info = module.predict(model.engine.base, ctx['levels'])
                detection = criterion(prediction, batch['ego']['label_dict'])
                # Both controls receive exactly the same objective and frame budget.
                loss = detection + settings['change_penalty']*info['change']
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite full-scene loss')
                if optimizer is not None and loss.requires_grad:
                    (loss/2).backward()
                for key, value in dict(loss=loss, detection_loss=detection, **info).items():
                    sums[key] += float(value.detach())
            count += 1
        if optimizer is not None:
            if any(p.grad is not None for p in module.parameters()):
                torch.nn.utils.clip_grad_norm_(module.parameters(), 5.)
                optimizer.step()
                updates += 1
        if count == 2 or count % 100 == 0:
            print(f'frames={count//2}/{len(loader)} loss={sums["loss"]/count:.6f}', flush=True)
    if not count:
        raise RuntimeError('Empty split')
    return dict(**{k:v/count for k,v in sums.items()}, frames=count//2, optimizer_updates=updates)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--frontend-config', required=True)
    p.add_argument('--frontend-checkpoint', required=True)
    p.add_argument('--variant', choices=('residual','attention'), required=True)
    p.add_argument('--output-dir', required=True)
    args = p.parse_args()
    from opencood.tools.train_utils import to_device
    verify_frozen()
    options,hypes = er.load_config(args.config,args.frontend_config)
    s = rt.settings(options)
    seed_all(options['seed'])
    target = device()
    model,digest = er.load_model(hypes,options,args.frontend_checkpoint,target)
    model.requires_grad_(False)
    _,train,indices = rt.train_loader(hypes,options)
    _,val,val_indices = er.make_loader(hypes,options,train=False)
    first = to_device(next(iter(train)),target)
    with torch.no_grad():
        ctx = rt.context(model,first['ego'],'clean',verify=True)
    channels = [x.shape[1] for x in ctx['levels']]
    del first,ctx
    # Reset after shape discovery so both variants see the same data stream.
    seed_all(options['seed'])
    _,train,indices = rt.train_loader(hypes,options)
    module = rt.create(channels,options,args.variant,target)
    optimizer = torch.optim.AdamW(module.parameters(),lr=s['learning_rate'],weight_decay=s['weight_decay'])
    criterion = PointPillarLoss(hypes['loss']['args']).to(target)
    output = new_output(args.output_dir)
    specification = rt.contract(options,args.frontend_config,digest)
    write_json(output/'protocol.json',dict(contract=specification,train_indices=indices,validation_indices=val_indices,
        training='full scenes, paired clean/online simulated weather, frozen frontend',variant=args.variant))
    best,history = float('inf'),[]
    for number in range(1,s['epochs']+1):
        seed_all(options['seed']+number)
        train.generator.manual_seed(options['seed']+number)
        training = epoch(model,module,train,criterion,target,s,optimizer)
        seed_all(options['seed'])
        val.generator.manual_seed(options['seed'])
        with torch.no_grad():
            validation = epoch(model,module,val,criterion,target,s)
        row = dict(epoch=number,train=training,validation=validation)
        history.append(row)
        print(json.dumps(row),flush=True)
        state = dict(module=module.state_dict(),optimizer=optimizer.state_dict(),channels=channels,
                     variant=args.variant,contract=specification,epoch=number)
        temporary = output/'last.partial'
        torch.save(state,temporary)
        temporary.replace(output/'last.pth')
        if validation['loss'] < best:
            best = validation['loss']
            torch.save(state,output/'best.pth')
        write_json(output/'history.json',history)
    verify_frozen()


if __name__ == '__main__':
    main()
