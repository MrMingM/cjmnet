"""Run the LR0 epoch engine with frozen modules and branch-only Adam."""
from contextlib import contextmanager
from pathlib import Path
import torch
import engine
from checkpoint import seed, rng, restore, save, state
from geometry_warnings import checked_evaluation
from metrics import valid_nms_candidates
from opencood.hypes_yaml import yaml_utils
from opencood.tools import train_utils
from runtime import SEED, build, create_model, loader
from support import patched
from v2xr_weather_l012.data import tensor_digest
from .audit import WEATHERS, atomic_json, identity, lr, read_json, sha256
from .model import build_model


def trainable(model):
    return {name:p for name,p in model.named_parameters() if p.requires_grad}


def frozen_hash(model):
    return tensor_digest({k:v for k,v in model.state_dict().items() if k not in trainable(model)})


def components(cfg,batches,settings):
    model=build_model(cfg,settings)
    criterion=train_utils.create_loss(cfg).to('cuda:0')
    opts=cfg['optimizer']
    if opts['core_method']!='Adam':
        raise ValueError('Expected LR0 Adam')
    parameters=list(trainable(model).values())
    optimizer=torch.optim.Adam(parameters,lr=opts['lr'],**opts['args'])
    scheduler=train_utils.setup_lr_schedular(cfg,optimizer,n_iter_per_epoch=batches)
    if {id(p) for group in optimizer.param_groups for p in group['params']}!={id(p) for p in parameters}:
        raise ValueError('Optimizer contains parameters outside the interaction branches')
    return model,criterion,optimizer,scheduler


def step(model,criterion,optimizer,batch,epoch,inspect=False):
    model.eval()
    optimizer.zero_grad(set_to_none=True)
    ego=train_utils.to_device(batch['ego'],torch.device('cuda:0'))
    ego['epoch']=epoch
    prediction=model(ego)
    if not bool(torch.stack([torch.isfinite(v).all() for v in prediction.values()]).all()):
        raise FloatingPointError(f'Nonfinite head output at epoch {epoch+1}')
    loss=criterion(prediction,ego['label_dict'])
    if not torch.isfinite(loss):
        raise FloatingPointError(f'Nonfinite LR0 loss at epoch {epoch+1}')
    loss.backward()
    params=trainable(model)
    if any(p.grad is None for p in params.values()) or not bool(
            torch.stack([torch.isfinite(p.grad).all() for p in params.values()]).all()):
        raise FloatingPointError(f'Invalid interaction gradient at epoch {epoch+1}')
    gradient={k:float(p.grad.norm()) for k,p in params.items()} if inspect else None
    optimizer.step()
    if not bool(torch.stack([torch.isfinite(p).all() for p in params.values()]).all()):
        raise FloatingPointError(f'Nonfinite interaction weights at epoch {epoch+1}')
    return float(loss.detach()),gradient


def check_equivalence(cfg,model,batch,native):
    from collections import Counter
    from contextlib import ExitStack
    from geometry_warnings import checked_intersections

    p0=create_model(cfg).eval()
    p0.load_state_dict(torch.load(lr.WEIGHT,map_location='cpu',weights_only=True),strict=True)
    ego=train_utils.to_device(batch['ego'],torch.device('cuda:0'))

    def record(net):
        snapshots={}
        def hook(i):
            def saved(_,args,output):
                snapshots[f'scale{i}_pre']=args[0].detach()
                snapshots[f'scale{i}_fused']=output.detach()
            return saved
        with ExitStack() as stack:
            for i,layer in enumerate(net.backbone.fuse_modules):
                stack.callback(layer.register_forward_hook(hook(i)).remove)
            prediction=net(ego)
        snapshots.update(prediction)
        boxes,scores=native.post_processor.post_process({'ego':ego},{'ego':prediction})
        if boxes is not None:
            snapshots.update(boxes=boxes,scores=scores)
        return snapshots

    mode=model.branches[0].mode
    with torch.no_grad(),checked_intersections(),valid_nms_candidates(Counter()):
        expected=record(p0)
        differences={}
        for test_mode in (mode,'off'):
            for layer in model.branches:
                layer.mode=test_mode
            got=record(model)
            if got.keys()!=expected.keys() or any(got[k].shape!=v.shape for k,v in expected.items()):
                raise ValueError(f'P0 equivalence structure differs in {test_mode}')
            differences[test_mode]={k:float((got[k]-v).abs().max()) if v.numel() else 0.
                                    for k,v in expected.items()}
            if any(v!=0 for v in differences[test_mode].values()):
                raise ValueError(f'P0 equivalence failed: {differences}')
    for layer in model.branches:
        layer.mode=mode
    return differences


def smoke(root,contract,settings):
    from p0_failure import archived_loader
    marker=root/'SMOKE.json'
    if marker.is_file():
        report=read_json(marker)
        if (report['contract']!=contract or any(sha256(root/k)!=v for k,v in report['artifacts'].items())):
            raise ValueError('Smoke evidence changed')
        return report
    seed(SEED)
    dataset,cfg=build(lr.CONFIG,condition='clean',visualize=True)
    validation=next(iter(archived_loader(dataset,'clean')))
    if int(validation['ego']['record_len'][0])<2:
        raise ValueError('Expected a real multi-vehicle equality frame')
    seed(SEED)
    model,criterion,optimizer,scheduler=components(cfg,2,settings)
    equivalence=check_equivalence(cfg,model,validation,dataset.native)
    radar_empty=dict(validation['ego'])
    radar_empty['processed_radar']={k:v[:0] for k,v in radar_empty['processed_radar'].items()}
    with torch.no_grad():
        result=model(train_utils.to_device(radar_empty,torch.device('cuda:0')))
        if not all(torch.isfinite(v).all() for v in result.values()):
            raise ValueError('Real input with empty Radar produced a nonfinite output')
    params=trainable(model)
    original={k:p.detach().clone() for k,p in params.items()}
    initial=tensor_digest(original)
    frozen=frozen_hash(model)
    seed(SEED)
    dataset,_=build(lr.CONFIG,split='train')
    batches=loader(dataset,training=True,limit=16)
    torch.cuda.reset_peak_memory_stats()
    rows=[]
    for i,batch in enumerate(batches):
        data=batch['ego']
        digest=tensor_digest({k:data[k] for k in ('processed_lidar','processed_radar','record_len','label_dict')})
        loss,gradient=step(model,criterion,optimizer,batch,0,inspect=True)
        if any(not torch.isfinite(p).all() for p in params.values()):
            raise FloatingPointError('Invalid smoke weights')
        if i==0:
            for name in params:
                if name.endswith('.W.weight') and (gradient[name]<=0 or torch.equal(original[name],params[name])):
                    raise ValueError(f'First update failed for {name}')
        if i==1:
            for name,p in params.items():
                if gradient[name]<=0 or torch.equal(original[name],p):
                    raise ValueError(f'No trained gradient or update for {name}')
        rows.append(dict(loss=loss,gradients=gradient,input_sha256=digest))
        print('SMOKE UPDATE',i+1,rows[-1],flush=True)
    if len(rows)!=2 or frozen_hash(model)!=frozen:
        raise ValueError('Smoke updates incomplete or frozen state changed')
    peak=torch.cuda.max_memory_allocated()
    scheduler.step(0)
    path=root/'smoke_resume.pth'
    save(path,state(model,optimizer,scheduler,criterion,1,contract))
    expected=tensor_digest([model.state_dict(),optimizer.state_dict(),scheduler.state_dict(),rng()])
    seed(SEED+1)
    restore(path,model,optimizer,scheduler,criterion,contract,1)
    if expected!=tensor_digest([model.state_dict(),optimizer.state_dict(),scheduler.state_dict(),rng()]):
        raise ValueError('Full-state checkpoint round trip differs')
    optimizer.zero_grad(set_to_none=True)
    del optimizer,scheduler
    torch.cuda.empty_cache()
    seed(20261016)
    evaluation=checked_evaluation(engine.evaluate)(model,None,lr.CONFIG,root/'smoke_evaluation',
                                                  WEATHERS,12,contract,sha256(path))
    report=dict(status='complete',contract=contract,initial_interaction_sha256=initial,
        frozen_sha256=frozen,equivalence=equivalence,updates=rows,peak_train_allocated_bytes=peak,
        evaluation=evaluation,parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in params.values()),resume_roundtrip=True,real_empty_radar=True,
        artifacts={str(p.relative_to(root)):sha256(p) for p in (path,root/'smoke_evaluation/summary.json')})
    atomic_json(marker,report)
    return report


def formal(root,contract,settings,expected_frozen):
    def validate(model,*args,**kwargs):
        if frozen_hash(model)!=expected_frozen:
            raise RuntimeError('Frozen P0 changed before validation')
        return original_validate(model,*args,**kwargs)

    original_validate=engine.clean_validation
    with patched(engine,'components',lambda cfg,batches:components(cfg,batches,settings)), \
            patched(engine,'train_step',step),patched(engine,'clean_validation',validate), \
            patched(engine,'evaluate',checked_evaluation(engine.evaluate)):
        rows=engine.train(root,lr.CONFIG,contract)
    if len(rows)!=20 or sum(row['train_updates'] for row in rows)!=20200:
        raise ValueError('Incomplete formal training')
    best=min(rows,key=lambda row:(-row['clean_AP']['O']['ap70'],row['epoch']))
    selected=dict(epoch=best['epoch'],checkpoint=best['checkpoint'],sha256=best['checkpoint_sha256'],
                  clean_selection_ap70=best['clean_AP']['O']['ap70'])
    if sha256(selected['checkpoint'])!=selected['sha256']:
        raise ValueError('Selected weight differs from committed weight')
    atomic_json(root/'selected.json',selected)
    cfg=yaml_utils.load_yaml(str(lr.CONFIG))
    model=build_model(cfg,settings)
    model.load_state_dict(torch.load(selected['checkpoint'],map_location='cpu',weights_only=True),strict=True)
    if frozen_hash(model)!=expected_frozen:
        raise ValueError('Selected checkpoint changed frozen parameters or buffers')
    return model,selected,max(row['peak_train_gpu_allocated_bytes'] for row in rows)
