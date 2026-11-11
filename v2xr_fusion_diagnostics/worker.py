"""Run one frozen diagnostic on its assigned physical card."""
import importlib.util
import json
from pathlib import Path
import sys
import traceback

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'common'))
from diag_bootstrap import OUT,WEATHERS,audit,idle,map_device,atomic_json,identity,pot,read_json,sha256
from common import lock

MODULES={'D1':'d1_perturbation','D2':'d2_multiscale','D3':'d3_crossmodal'}


def experiment(group):
    spec=importlib.util.spec_from_file_location(group,HERE/MODULES[group]/'experiment.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run


def run_frame(group,model,weather,original,clean,cfg,native,fixed,live,known,artifact,repeat):
    from diag_data import regions,splice
    from diag_model import ego_only,radar_off,error
    from diag_eval import predictions
    from diag_stats import measure
    from predict import live_gt_observer
    from support import compare_gt
    from opencood.tools import train_utils
    import torch

    targets,selected=regions(fixed,known)
    alt={}
    splices={}
    base_input=train_utils.to_device(original,torch.device('cuda:0'))
    with torch.inference_mode():
        base=model.capture(base_input)
        repetition=error(base.prediction,model.capture(base_input).prediction) if repeat else None
        if repetition not in (None,0):
            raise ValueError(f'Repeated frozen input differs: {repetition}')
        reconstruction=error(base.prediction,model.predict(base.post))
        if reconstruction!=0:
            raise ValueError(f'Original head replay differs: {reconstruction}')
        scopes={'D1':('neighbor',),'D2':('both',),'D3':('ego','neighbor','both')}[group]
        if group=='D2':
            alt['single']=model.capture(ego_only(base_input))
        if group=='D3':
            alt['weather_r0']=model.capture(radar_off(base_input))
        for scope in scopes:
            mixed,splices[scope]=splice(original,clean,selected,cfg,scope,weather)
            unchanged=not (splices[scope]['removed'] or splices[scope]['added'])
            moved=base_input if unchanged else train_utils.to_device(mixed,torch.device('cuda:0'))
            if group=='D3':
                alt[f'{scope}_r1']=base if unchanged else model.capture(moved)
                alt[f'{scope}_r0']=alt['weather_r0'] if unchanged else model.capture(radar_off(moved))
            else:
                alt[scope]=base if unchanged else model.capture(moved)
        modes,measures,raw,checks=experiment(group)(model,base,alt,known)
        spatial,objects=measure(cfg,measures,raw,targets,known,artifact)
        records,aliases,post=predictions(native,base_input,modes)
        _,gt=live_gt_observer(native.post_processor,{'ego':base_input},native.post_processor.generate_gt_bbx)
        compare_gt(gt,live['official_gt'],f'{weather}/{fixed["frame_key"]}')
    return dict(frame=fixed['frame_key'],vehicle_count=int(base.record_len[0]),known_ids=known,
                selected_context_ids=[str(r['object_id']) for r in selected],
                splices=splices,checks=dict(repeat_head_error=repetition,head_replay_error=reconstruction,**checks),
                official_gt=live['official_gt'],fixed_gt=fixed['official_gt'],prediction=records,
                aliases=aliases,spatial=spatial,targets=objects,postprocess=post,
                sample_sha256=sha256(artifact) if artifact.is_file() else None)


def test_units(group,model):
    import torch
    from diag_model import attention,fuse
    device='cuda:0'
    native=model.backbone.fuse_modules[0]
    for cars in (1,3):
        x=torch.arange(cars*128*2*2,device=device,dtype=torch.float32).reshape(cars,128,2,2)/1000
        record=torch.tensor([cars],device=device)
        a=attention(x,native)
        torch.testing.assert_close(a.sum(-1),torch.ones_like(a[...,0]))
        torch.testing.assert_close(fuse(a,x),native(x,record),atol=1e-6,rtol=1e-6)
        if group=='D1':
            y=x.clone()
            y[1:]+=.01
            b=attention(y,native)
            torch.testing.assert_close(fuse(b,y),native(y,record),atol=1e-6,rtol=1e-6)
        elif group=='D2':
            torch.testing.assert_close(native(x[:1],record.new_ones(1)),x[:1],atol=0,rtol=0)
        else:
            a,b,c,d=(x+1,x,x+3,x+2)
            torch.testing.assert_close((c-a)-(d-b),torch.zeros_like(x),atol=1e-6,rtol=0)
    return dict(cars=[1,3],native_attention_equivalence=True,group_algebra=True)


def worker(group,stage):
    from diag_data import samples
    from diag_model import FrozenP0
    from diag_eval import score
    from checkpoint import seed
    from v2xr_weather_l012.data import tensor_digest
    import torch

    dest=OUT/group
    contract=read_json(OUT/'CONTRACT.json')
    for file,expected in contract['sources'].items():
        if sha256(HERE/file)!=expected:
            raise ValueError(f'Frozen source changed: {file}')
    run_id=identity(contract)
    if stage=='full':
        smoke=read_json(dest/'SMOKE.json')
        if smoke['status']!='complete' or smoke['contract']!=run_id:
            raise ValueError('Smoke prerequisite differs')
    idle([group])
    device=map_device(group)
    seed(20261016)
    model=FrozenP0()
    print('DEVICE',group,device,flush=True)
    weight_before=tensor_digest(model.model.state_dict())
    pot.bind(dest/'PREFLIGHT.json',dict(status='complete',contract=run_id,device=device,
             weight_sha256=sha256(pot.WEIGHT),frozen_state_sha256=weight_before))
    cohort=read_json(OUT/'COHORT.json')
    if stage=='smoke':
        units=test_units(group,model)
        rows={}
        for weather in WEATHERS:
            for i,ego,clean,cfg,native,fixed,live in samples(weather,12):
                if int(ego['record_len'][0])<=1 or not fixed['official_gt']:
                    continue
                known=cohort.get(weather,{}).get(str(i),[])
                row=run_frame(group,model,weather,ego,clean,cfg,native,fixed,live,known,
                              dest/'smoke_samples'/f'{weather}_{i}.npz',True)
                if weather!='clean' and not any(v['added'] or v['removed'] for v in row['splices'].values()):
                    continue
                stats,hits=score(row['prediction']['base'],row['official_gt'],row['fixed_gt'],weather=='clean')
                rows[weather]=dict(index=i,cars=row['vehicle_count'],modes=sorted(set(row['prediction'])|set(row['aliases'])),
                                   checks=row['checks'],splices=row['splices'],fixed_hits70=len(hits),
                                   protocol_names=sorted(stats),known_ids=known)
                break
            if weather not in rows:
                raise ValueError(f'{weather} smoke did not exercise real data')
        if tensor_digest(model.model.state_dict())!=weight_before:
            raise ValueError('Smoke changed frozen model state')
        atomic_json(dest/'SMOKE.json',dict(status='complete',contract=run_id,units=units,frames=rows))
        print('SMOKE PASSED',group,rows,flush=True)
        return
    for weather in WEATHERS:
        for i,ego,clean,cfg,native,fixed,live in samples(weather,829):
            path=dest/'frames'/weather/f'{i:06d}.json'
            artifact=dest/'features'/weather/f'{i:06d}.npz'
            if path.exists():
                stored=read_json(path)
                if (stored['contract']!=run_id or stored['frame']!=fixed['frame_key']
                        or identity(stored['data'])!=stored['sha256']
                        or (stored['data']['sample_sha256'] and sha256(artifact)!=stored['data']['sample_sha256'])):
                    raise ValueError(f'Invalid frame cache {path}')
                continue
            known=cohort.get(weather,{}).get(str(i),[])
            row=run_frame(group,model,weather,ego,clean,cfg,native,fixed,live,known,artifact,i==0)
            atomic_json(path,dict(contract=run_id,frame=row['frame'],sha256=identity(row),data=row))
            if (i+1)%25==0 or i==828:
                print(group,weather,i+1,'/ 829',flush=True)
    if tensor_digest(model.model.state_dict())!=weight_before:
        raise ValueError('Frozen model state changed during inference')
    del model
    from diag_summary import summarize
    summarize(group,run_id)


if __name__=='__main__':
    if sys.argv[1:] == ['audit']:
        audit()
    else:
        group,stage=sys.argv[1:]
        with lock(OUT/group/'.lock'):
            try:
                atomic_json(OUT/group/'STATUS.json',dict(status='running',stage=stage))
                worker(group,stage)
                atomic_json(OUT/group/'STATUS.json',dict(status='complete',stage=stage))
            except BaseException:
                atomic_json(OUT/group/'FAILURE.json',dict(status='failed',stage=stage,traceback=traceback.format_exc()))
                atomic_json(OUT/group/'STATUS.json',dict(status='failed',stage=stage))
                raise
