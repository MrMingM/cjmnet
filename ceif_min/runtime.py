import copy
import time
from pathlib import Path
import torch
from gspr_evidence import runtime as rt
from ceif_audit.replay import received_levels, fused_levels
from .observations import make_packet


def contract(options, frontend, digest, settings):
    result = rt.contract(options,frontend,digest)
    paths = list(Path(__file__).parent.glob('*.py'))+[rt.ROOT/'ceif_audit'/f for f in ('core.py','run.py','replay.py')]
    result['ceif_sources'] = {str(p.relative_to(rt.ROOT)):rt.sha256(p) for p in paths}
    result['prototype'] = settings
    return result


def loader(hypes,options,train,weather,epoch=0,smoke=0):
    ds, original, indices = rt.make_loader(hypes,options,train=train,weather=weather,smoke=smoke)
    if hasattr(ds,'set_weather_augmentation_epoch'):
        ds.set_weather_augmentation_epoch(epoch if train else 0)
    if train:
        from torch.utils.data import DataLoader
        generator = torch.Generator().manual_seed(options['seed']+epoch)
        original = DataLoader(original.dataset,batch_size=1,shuffle=True,num_workers=options['workers'],
            collate_fn=ds.collate_batch_test,worker_init_fn=rt.seed_worker,generator=generator)
    return ds,original,indices


@torch.no_grad()
def prepare(model,ego,branch,seed,queries,verify=False):
    inp = rt.input_branch(ego,branch)
    encoded = model.encode(inp)
    masks = torch.ones_like(model.empty_masks(encoded))
    received = received_levels(encoded,masks,model.engine.value_bytes)
    fused = fused_levels(received,masks)
    base = model.engine.base
    feature = torch.cat([d(x) for d,x in zip(base.backbone.deblocks,fused)],1).detach()
    if verify:
        original = base(inp)
        for key,value in predict(base,feature).items():
            torch.testing.assert_close(value,original[key],atol=2e-4,rtol=2e-4)
    packet = make_packet(model,inp,encoded,masks,seed,queries)
    return feature,packet


def predict(base,feature):
    # Frozen head weights still transmit gradients to corrected features.
    return dict(psm=base.cls_head(feature),rm=base.reg_head(feature))


def progress(name,index,total,start):
    if index == 1 or index % 20 == 0:
        seconds = (time.perf_counter()-start)/index
        print(f'{name}: {index}/{total}, {seconds:.2f}s/frame, ETA {(total-index)*seconds/3600:.2f}h',flush=True)


def validate(model,decoder,variants,hypes,options,settings,device,smoke=0,formal_weather=None,output=None):
    from opencood.tools.train_utils import to_device
    from opencood.utils import eval_utils
    from ceif_audit.scoring import empty_stats,ap_values
    from .network import interval_loss
    weather = formal_weather or settings.pop('_validation_weather', 'clean')
    branch = 'clean' if formal_weather is not None else weather
    ds,batches,indices = loader(hypes,options,False,branch,smoke=smoke)
    stats = {name:empty_stats() for name in ['baseline',*variants]}
    diagnostics = {name:dict(violation=0.,correction_rms=0.,projection_rms=0.) for name in variants}
    queries = conflicts = extra = 0
    start = time.perf_counter()
    with torch.no_grad():
        for count,batch in enumerate(batches,1):
            batch = to_device(batch,device); ego = batch['ego']
            sample = int(ego['communication_sample_index'][0])
            feature,packet = prepare(model,ego,branch,options['seed']+sample,settings['queries'],verify=count==1)
            features = {'baseline':feature}
            for name,module in variants.items():
                module.eval()
                features[name],prior = module(feature,decoder,packet,model.lidar_range,project=name=='ceif',return_prior=True)
                logits = decoder(features[name],packet['xyz'],model.lidar_range)
                diagnostics[name]['violation'] += float(interval_loss(logits,packet['lower'],packet['upper']))
                diagnostics[name]['correction_rms'] += float((features[name]-feature).square().mean().sqrt())
                diagnostics[name]['projection_rms'] += float((features[name]-prior).square().mean().sqrt())
            for name,value in features.items():
                boxes,scores,gt = ds.post_process(batch,{'ego':predict(model.engine.base,value)})
                for threshold in stats[name]:
                    eval_utils.caluclate_tp_fp(boxes,scores,gt,stats[name],threshold)
            queries += len(packet['xyz']); conflicts += int((packet['lower']>packet['upper']).sum())
            extra += packet['extra_bytes']
            progress('validation/'+weather,count,len(indices),start)
    if not indices or count != len(indices) or not stats['baseline'][.7]['gt']:
        raise RuntimeError('Incomplete/empty validation')
    results = {name:ap_values(rows,eval_utils) for name,rows in stats.items()}
    result = dict(weather=weather,frames=count,data_root=hypes['validate_dir'],online_weather=formal_weather is None and weather!='clean',
        global_sort=False,smoke=bool(smoke),results=results,mean_queries=queries/count,
        conflict_fraction=conflicts/max(queries,1),mean_geometry_extra_bytes=extra/count,
        diagnostics={name:{k:v/count for k,v in values.items()} for name,values in diagnostics.items()})
    for name,module in variants.items():
        result['diagnostics'][name]['projection_step'] = float(module.step_logit.sigmoid()) if name=='ceif' else 0.
    if output:
        output.mkdir(parents=True,exist_ok=False)
        for name,rows in stats.items():
            destination=output/name; destination.mkdir()
            eval_utils.eval_final_results(copy.deepcopy(rows),str(destination),False)
        rt.write_json(output/'summary.json',result)
    return result
