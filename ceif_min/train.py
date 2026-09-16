"""Train query decoder then CEIF/auxiliary-only control on simulated OPV2V train."""
import argparse
import copy
import json
import time
from pathlib import Path
import torch
import yaml
from .network import QueryDecoder,MiniCEIF,interval_loss
from .observations import decoder_loss
from .runtime import rt,contract,loader,prepare,predict,progress,validate


def save(path,value):
    temporary=path.with_suffix('.partial')
    torch.save(value,temporary)
    temporary.replace(path)


def warm_epoch(model,decoder,hypes,options,settings,device,training,epoch,smoke,optimizer=None):
    from opencood.tools.train_utils import to_device
    ds,batches,indices = loader(hypes,options,training,'clean',epoch,smoke)
    decoder.train(training)
    total=frames=positive=negative=correct=label_count=0; start=time.perf_counter()
    for count,batch in enumerate(batches,1):
        batch=to_device(batch,device); ego=batch['ego']; sample=int(ego['communication_sample_index'][0])
        feature,packet=prepare(model,ego,'clean',options['seed']+sample+(epoch*100000 if training else 0),settings['queries'],verify=count==1)
        with torch.set_grad_enabled(training):
            logits=decoder(feature,packet['xyz'],model.lidar_range)
            loss=decoder_loss(logits,packet)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite decoder loss')
            if training:
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(decoder.parameters(),5.); optimizer.step()
        valid=(packet['label_weights']>=.6)&(packet['lower']<=packet['upper'])
        positive+=int((valid&(packet['labels']==1)).sum()); negative+=int((valid&(packet['labels']==0)).sum())
        correct+=int(((logits.detach()>=0)==packet['labels'].bool())[valid].sum()); label_count+=int(valid.sum())
        total+=float(loss.detach()); frames+=1
        progress('query/'+('train' if training else 'validation'),count,len(indices),start)
    if frames!=len(indices) or not min(positive,negative):
        raise RuntimeError('Query calibration requires complete split and both hit/free samples')
    return dict(loss=total/frames,frames=frames,positive=positive,negative=negative,
                sensor_pseudolabel_accuracy=correct/max(label_count,1))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='gspr_evidence/experiment.yaml')
    p.add_argument('--frontend-config',required=True); p.add_argument('--frontend-checkpoint',required=True)
    p.add_argument('--output-dir',required=True); p.add_argument('--resume',action='store_true')
    p.add_argument('--epochs',type=int,default=3); p.add_argument('--decoder-epochs',type=int,default=1)
    p.add_argument('--queries',type=int,default=1024); p.add_argument('--hidden',type=int,default=32)
    p.add_argument('--lr',type=float,default=2e-4); p.add_argument('--observation-weight',type=float,default=.1)
    p.add_argument('--smoke',type=int,default=0)
    a=p.parse_args()
    if min(a.epochs,a.decoder_epochs,a.queries,a.hidden)<1 or a.smoke<0 or a.lr<=0 or a.observation_weight<0:
        p.error('Invalid prototype settings')
    from opencood.tools.train_utils import to_device
    from opencood.loss.point_pillar_loss import PointPillarLoss
    rt.verify_frozen()
    options,hypes=rt.load_config(a.config,a.frontend_config)
    if Path(hypes['root_dir']).name!='train' or Path(hypes['validate_dir']).name!='validate':
        raise ValueError('Only train for updates and validate for model selection; test is forbidden')
    options['seed']=20260916
    rt.seed_all(options['seed']); device=rt.device()
    model,digest=rt.load_model(hypes,options,a.frontend_checkpoint,device)
    model.eval().requires_grad_(False)
    settings=dict(epochs=a.epochs,decoder_epochs=a.decoder_epochs,queries=a.queries,hidden=a.hidden,
        lr=a.lr,observation_weight=a.observation_weight,smoke=a.smoke,communication='full',
        selection='maximum unweighted mean validation AP70 over clean/fog/rain/snow',
        supervision='clean sensor hit/free query calibration; paired clean+online mixed physics detection training')
    identity=contract(options,a.frontend_config,digest,settings)
    out=Path(a.output_dir).resolve() if a.resume else rt.new_output(a.output_dir)
    if a.resume:
        previous=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
        if previous['contract']!=identity: raise ValueError('Resume contract differs')
    else:
        rt.write_json(out/'protocol.json',dict(contract=identity,frontend_config=str(Path(a.frontend_config).resolve()),
            frontend_checkpoint=str(Path(a.frontend_checkpoint).resolve()),train_root=hypes['root_dir'],
            validation_root=hypes['validate_dir'],test_used_for_selection=False))
        (out/'experiment.yaml').write_text(yaml.safe_dump(options,sort_keys=False),encoding='utf-8')
    channels=model.engine.base.cls_head.in_channels
    decoder=QueryDecoder(channels).to(device)
    warm_path=out/'query_last.pth'
    warm=torch.load(warm_path,map_location=device,weights_only=True) if a.resume and warm_path.exists() else None
    optimizer=torch.optim.AdamW(decoder.parameters(),lr=a.lr)
    begin=0; best=float('inf')
    if warm:
        decoder.load_state_dict(warm['decoder']); optimizer.load_state_dict(warm['optimizer'])
        begin=warm['epoch']+1; best=warm['best']
    for epoch in range(begin,a.decoder_epochs):
        rt.seed_all(options['seed']+epoch)
        train_metrics=warm_epoch(model,decoder,hypes,options,settings,device,True,epoch,a.smoke,optimizer)
        val_metrics=warm_epoch(model,decoder,hypes,options,settings,device,False,epoch,a.smoke)
        if val_metrics['loss']<best:
            best=val_metrics['loss']; save(out/'query_best.pth',dict(decoder=decoder.state_dict(),contract=identity,epoch=epoch))
        save(warm_path,dict(decoder=decoder.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,best=best))
        rt.write_json(out/f'query_epoch_{epoch+1}.json',dict(train=train_metrics,validation=val_metrics))
    decoder.load_state_dict(torch.load(out/'query_best.pth',map_location=device,weights_only=True)['decoder'])
    decoder.eval().requires_grad_(False)
    variants={'ceif':MiniCEIF(channels,a.hidden).to(device)}
    variants['aux_only']=copy.deepcopy(variants['ceif'])
    optimizers={name:torch.optim.AdamW(module.parameters(),lr=a.lr,weight_decay=1e-4) for name,module in variants.items()}
    best={name:-float('inf') for name in variants}; begin=0
    if a.resume and (out/'last.pth').exists():
        checkpoint=torch.load(out/'last.pth',map_location=device,weights_only=True)
        if checkpoint['contract']!=identity: raise ValueError('Checkpoint contract mismatch')
        for name,module in variants.items():
            module.load_state_dict(checkpoint['models'][name]); optimizers[name].load_state_dict(checkpoint['optimizers'][name])
        best=checkpoint['best']; begin=checkpoint['epoch']+1
    criterion=PointPillarLoss(hypes['loss']['args'])
    for epoch in range(begin,a.epochs):
        rt.seed_all(options['seed']+1000+epoch)
        ds,batches,indices=loader(hypes,options,True,'mixed',epoch,a.smoke)
        totals={name:dict(det=0.,obs=0.,loss=0.) for name in variants}; views=0; start=time.perf_counter()
        for module in variants.values(): module.train()
        for count,batch in enumerate(batches,1):
            batch=to_device(batch,device); ego=batch['ego']; sample=int(ego['communication_sample_index'][0])
            for optimizer in optimizers.values(): optimizer.zero_grad(set_to_none=True)
            for branch in ('clean','weather'):
                feature,packet=prepare(model,ego,branch,options['seed']+sample+epoch*100000,a.queries,verify=count==1)
                for name,module in variants.items():
                    corrected=module(feature,decoder,packet,model.lidar_range,project=name=='ceif')
                    det=criterion(predict(model.engine.base,corrected),ego['label_dict'])
                    obs=interval_loss(decoder(corrected,packet['xyz'],model.lidar_range),packet['lower'],packet['upper'])
                    loss=det+a.observation_weight*obs+1e-4*(corrected-feature).square().mean()
                    if not torch.isfinite(loss): raise FloatingPointError(f'Nonfinite {name} loss at {sample}')
                    (loss/2).backward()
                    for key,value in (('det',det),('obs',obs),('loss',loss)):
                        totals[name][key]+=float(value.detach())
                    del corrected,det,obs,loss
                views+=1
            for name,module in variants.items():
                torch.nn.utils.clip_grad_norm_(module.parameters(),5.); optimizers[name].step()
            progress(f'fusion/epoch{epoch+1}',count,len(indices),start)
        if count!=len(indices): raise RuntimeError('Incomplete training epoch')
        validation={}
        for weather in ('clean','fog','rain','snow'):
            rt.seed_all(options['seed'])
            validation[weather]=validate(model,decoder,variants,hypes,options,
                dict(settings,_validation_weather=weather),device,a.smoke)
        for name,module in variants.items():
            score=sum(row['results'][name]['ap70'] for row in validation.values())/4
            if score>best[name]:
                best[name]=score
                save(out/f'{name}_best.pth',dict(model=module.state_dict(),decoder=decoder.state_dict(),
                    contract=identity,epoch=epoch,variant=name,validation_mean_ap70=score))
        rt.write_json(out/f'epoch_{epoch+1}.json',dict(training={n:{k:v/views for k,v in t.items()} for n,t in totals.items()},
            validation=validation,best=best))
        save(out/'last.pth',dict(models={n:m.state_dict() for n,m in variants.items()},
            optimizers={n:o.state_dict() for n,o in optimizers.items()},epoch=epoch,best=best,contract=identity))
        print(f'Epoch {epoch+1}: best validation mean AP70 {best}',flush=True)
        rt.verify_frozen()
    if rt.sha256(a.frontend_checkpoint)!=digest: raise RuntimeError('Frontend checkpoint changed')
    rt.write_json(out/'training_complete.json',dict(epochs=a.epochs,best=best,smoke=bool(a.smoke),contract=identity))
    print(f'Training complete: {out}; test has NOT been used.',flush=True)


if __name__=='__main__': main()
