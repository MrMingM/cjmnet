"""Frozen paired single/multi V2X-R evaluation on the official fixed targets."""
from collections import Counter, defaultdict
from pathlib import Path
import json
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v2xr_fusion_hypothesis_audit.shared.input_audit import (
    CLEAN, REPO, WEATHERS, VAL_MANIFEST, atomic_json, devices, lr,
    read_json, sha256, source_contract, verify_gt,
)

OUT = Path(os.environ['NEG_OUT'])
LOGS = lr.LR0.parent
ROOTS = {'E0': lr.LR0, 'E1': Path(os.getenv('LRG_RUN_DIR', LOGS/'v2xr-lr1-v1/LRG')),
         'E2': LOGS/'v2xr-radar-ra-rb-v1/F0'}
devices.DEVICES = {'E0': (4,'0000:1d:00.0','7137d9b2a09a2061'),
                   'E1': (3,'0000:16:00.0','7137d9b2a09238c1'),
                   'E2': (5,'0000:20:00.0','7137d9aff40e38c1')}


def evidence(group):
    root=ROOTS[group]
    final=read_json(root/'FINAL_RESULTS.json')
    chosen=read_json(root/'selected.json')
    if final['status']!='complete' or len(chosen['candidates'])!=20:
        raise ValueError('Training selection is incomplete')
    if group=='E1' and read_json(root/'CONTRACT.json').get('variant')!='LRG':
        raise ValueError('E1 must be the original trained LRG variant')
    weight=root/'epochs'/f"{chosen['epoch']:02d}"/'weights.pth'
    for i in range(1,21):
        if not (root/'epochs'/f'{i:02d}'/'complete.json').is_file():
            raise ValueError(f'Epoch {i} is not committed')
    if sha256(weight)!=chosen['sha256'] or sha256(root/'config.yaml')!=sha256(lr.CONFIG):
        raise ValueError('Checkpoint SHA256 or inference configuration mismatch')
    verify_gt(root/'final_evaluation')
    return dict(weight=str(weight),sha256=chosen['sha256'],epoch=chosen['epoch'])


def preflight(allow_missing_lrg=False):
    frozen=read_json(lr.LR0/'CONTRACT.json')
    data=read_json(lr.LR0/'DATA.json')['splits']['validate']
    if source_contract()!=frozen['sources'] or sha256(lr.CONFIG)!=frozen['config_sha256']:
        raise ValueError('Official runtime or LR0 config changed')
    if data['frames']!=829 or sha256(VAL_MANIFEST)!=data['input_manifest_sha256']:
        raise ValueError('Validation manifest changed')
    models,failures={},{}
    for group in ROOTS:
        try:
            models[group]=evidence(group)
        except (OSError,ValueError,KeyError) as exc:
            failures[group]=str(exc)
    atomic_json(OUT/'PREFLIGHT.json',dict(models=models,failures=failures,data=data,config=frozen['config_sha256']))
    print('PREFLIGHT', json.dumps(dict(models=models, failures=failures)), flush=True)
    if failures and not (allow_missing_lrg and set(failures)=={'E1'}):
        raise SystemExit(f'Preflight blocked: {failures}')
    devices.idle(ROOTS if allow_missing_lrg else models)


def ego_only(ego):
    from v2xr_fusion_hypothesis_audit.shared.extract import one_car
    if len(ego['record_len'])!=1 or not bool(ego['ego_flag'][0][0]):
        raise ValueError('Expected one scene with ego in vehicle slot zero')
    result=one_car(ego,0)
    for key in ('lidar_pose','ego_flag'):
        result[key]=[ego[key][0][:1]]
    return result


def model_for(group,cfg):
    from runtime import create_model
    from empty_input import preserve_empty_vehicle_slots
    if group=='E0':
        return create_model(cfg)
    if group=='E1':
        from v2xr_lr1.lr1_model import LR1Model
        model=LR1Model(cfg['model']['args'],'LRG')
        preserve_empty_vehicle_slots(model)
        return model.cuda()
    sys.path.insert(0,str(REPO/'v2xr_radar_ra_rb'))
    from v2xr_radar_ra_rb.model import Model
    settings=read_json(REPO/'v2xr_radar_ra_rb/settings.json')['F0']
    if settings['ra'] or settings['rb']:
        raise ValueError('F0 unexpectedly has radar enhancement enabled')
    return Model(cfg['model']['args'],settings).cuda()


class BoxCapture:
    """Save exact official final prediction order while tracing candidate stages."""
    def __init__(self, processor):
        self.processor=processor

    def __getattr__(self, name):
        return getattr(self.processor,name)

    def post_process(self,*args):
        self.final=self.processor.post_process(*args)
        return self.final


def failure_stages(arrays,target):
    import numpy as np
    from p0_failure import overlap
    corners,_,_,scores,stage=arrays
    ids,iou=overlap(corners,np.asarray(target['corners']))
    readings=[]
    for s in range(5):
        keep=stage[ids]>=s
        valid=keep&(iou>=.7)
        readings.append(dict(max_iou=float(iou[keep].max()) if keep.any() else 0.,
                             best_score=float(scores[ids[valid]].max()) if valid.any() else None))
    labels=('localization','score_threshold','geometry_filter','nms','range_filter')
    return readings,next((x for x,r in zip(labels,readings) if r['best_score'] is None),'one_to_one_competition')


def run(group,smoke):
    import torch
    from opencood.hypes_yaml import yaml_utils
    from opencood.tools import train_utils
    from opencood.utils.eval_utils_opv2v import calculate_ap
    from runtime import build,loader
    from checkpoint import seed
    from diagnostics import object_support
    from evaluate import statistics,official_match_trace
    from metrics import valid_nms_candidates
    from geometry_warnings import checked_intersections
    from support import corner_array
    from v2xr_fusion_hypothesis_audit.shared.native_trace import trace

    devices.idle([group])
    device=devices.device(group)
    print(group,device,flush=True)
    chosen=evidence(group)
    preflight_result=read_json(OUT/'PREFLIGHT.json')
    if not smoke and preflight_result['failures'] and not (
            group in ('E0','E2') and set(preflight_result['failures'])=={'E1'}
            and '--with-lrg-training' in sys.argv):
        raise ValueError('Full experiment requires all three models')
    if chosen!=preflight_result['models'][group]:
        raise ValueError('Model evidence changed after preflight')
    seed(20261016)
    cfg=yaml_utils.load_yaml(str(lr.CONFIG))
    model=model_for(group,cfg).eval().requires_grad_(False)
    model.load_state_dict(torch.load(chosen['weight'],map_location='cpu',weights_only=True),strict=True)
    generator=torch.Generator()
    generator.set_state(torch.get_rng_state())
    root=OUT/group
    history=read_json(lr.OUT/'p0-frozen-failure-v1/COHORT.json')
    results={}
    for weather in WEATHERS:
        ds,_=build(lr.CONFIG,condition=weather,visualize=True)
        if len(ds)!=829: raise ValueError('Validation length differs')
        batches=loader(ds,evaluation=True,limit=3 if smoke else 829)
        batches.generator=generator
        processor=BoxCapture(ds.native.post_processor)
        params=processor.params['target_args']
        official=float(params['score_threshold'])
        thresholds=(official,.75*official,min(.99,1.25*official))
        aggregate={str(t):{m:statistics() for m in ('single','multi')} for t in thresholds}
        counts=defaultdict(Counter)
        n=0
        temp=root/(('smoke_' if smoke else '')+weather+'.tmp')
        with temp.open('w',encoding='utf-8') as rows, torch.inference_mode(), checked_intersections():
            for i,batch in enumerate(batches):
                ref=read_json(CLEAN/f'{i:06d}.json')
                if ds.keys[i]!=ref['frame_key']:
                    raise ValueError('Fixed target frame differs')
                targets=ref['official_gt']
                gt=torch.from_numpy(corner_array(targets))
                ids=[t['object_id'] for t in targets]
                n+=len(ids)
                cpu=batch['ego']
                low=cfg['model']['args']['lidar_range']
                support=object_support(cpu,targets,None,(low[0],low[1],1.6,1.6,50,176),
                                       dict(distance_edges_m=[30,60]))
                ego=train_utils.to_device(cpu,torch.device('cuda:0'))
                inputs={'single':ego_only(ego),'multi':ego}
                outputs={mode:model(data) for mode,data in inputs.items()}
                if int(ego['record_len'][0])==1:
                    for k in outputs['single']:
                        torch.testing.assert_close(outputs['single'][k],outputs['multi'][k])
                for t in thresholds:
                    pairs={}
                    diagnostics={}
                    for mode,data in inputs.items():
                        params['score_threshold']=t
                        try:
                            with valid_nms_candidates(Counter()):
                                arrays,_=trace(lambda _:outputs[mode],data,processor,targets)
                        finally:
                            params['score_threshold']=official
                        diagnostics[mode]=arrays
                        boxes,scores=processor.final
                        for x in (.3,.5,.7):
                            matched,_=official_match_trace(boxes,scores,gt,ids,aggregate[str(t)][mode],x)
                            if x==.7:
                                pairs[mode]={str(p['object_id']):dict(p,score=float(scores[p['prediction_index']]),
                                    corners=boxes[p['prediction_index']].tolist()) for p in matched}
                    for target in targets:
                        oid=str(target['object_id'])
                        a,b=pairs['single'].get(oid),pairs['multi'].get(oid)
                        state=('both' if b else 'negative') if a else ('positive' if b else 'neither')
                        past=oid in history.get(weather,{}).get(str(i),[])
                        row=dict(index=i,frame=ref['frame_key'],object=oid,threshold=t,state=state,
                                 historical=past,support=support[oid],single=a,multi=b)
                        if state=='negative':
                            reading,label=failure_stages(diagnostics['multi'],target)
                            score=reading[0]['best_score']
                            row.update(stages=reading,reason=label,
                                       score_delta=None if score is None else score-a['score'])
                        rows.write(json.dumps(row)+'\n')
                        p=support[oid]['ego_lidar']
                        point_bin='0' if p==0 else '1-5' if p<=5 else '6-20' if p<=20 else '21+'
                        for dim in ('all','scene/'+ref['frame_key'][0],'distance/'+support[oid]['distance'],
                                    'ego_lidar/'+point_bin,'historical/'+str(past)):
                            c=counts[f'{t}/{dim}']
                            c[state]+=1
                            if state=='negative': c['reason/'+label]+=1
                if i==0 or (i+1)%50==0: print(group,weather,i+1,flush=True)
        if not smoke and n!=6235:
            raise ValueError(f'{weather} expected 6235 fixed targets, got {n}')
        temp.replace(temp.with_suffix('.jsonl'))
        ap={t:{m:{str(x):float(calculate_ap(stats,x,False)[0]) for x in (.3,.5,.7)}
               for m,stats in modes.items()} for t,modes in aggregate.items()}
        results[weather]=dict(official=official,targets=n,ap=ap,counts=counts)
    atomic_json(root/('SMOKE.json' if smoke else 'RESULTS.json'),
                dict(status='complete',device=device,model=chosen,weather=results))


if __name__=='__main__':
    if sys.argv[1]=='preflight':
        preflight('--with-lrg-training' in sys.argv)
    elif sys.argv[1]=='register-lrg':
        receipt=read_json(OUT/'PREFLIGHT.json')
        receipt['models']['E1']=evidence('E1')
        receipt['failures'].pop('E1',None)
        atomic_json(OUT/'PREFLIGHT.json', receipt)
        print('Registered E1',receipt['models']['E1'],flush=True)
    elif sys.argv[1]=='report':
        from report import report
        report(OUT)
    else:
        run(sys.argv[1],sys.argv[2]=='smoke')
