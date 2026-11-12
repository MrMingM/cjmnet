"""Official LR0 evaluation, paired targets and observation-conditioned readouts."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import numpy as np
import shapely
import torch
from diagnostics import fixed_metrics,object_support,verify_gt
from geometry_warnings import checked_evaluation
from metrics import evaluate
from opencood.hypes_yaml import yaml_utils
from opencood.tools import train_utils
from p0_failure import archived_loader
from runtime import build,create_model
from checkpoint import seed
from support import atomic_jsonl
from .audit import OUT,WEATHERS,atomic_json,identity,lr,read_json,sha256

LOCATIONS=('empty','ego_unique','neighbor_unique','multiple')
FIELDS=('base','active','F2','F3','F4','F2_F3','F3_F4')


def baseline(contract,device):
    root=OUT/'BASELINE'
    seed(20261016)
    model=create_model(yaml_utils.load_yaml(str(lr.CONFIG))).eval()
    model.load_state_dict(torch.load(lr.WEIGHT,map_location='cpu',weights_only=True),strict=True)
    result=checked_evaluation(evaluate)(model,None,lr.CONFIG,root/'final_evaluation',WEATHERS,829,
                                        identity([contract,'P0']),sha256(lr.WEIGHT))
    verify_gt(root/'final_evaluation')
    atomic_json(root/'FINAL_RESULTS.json',dict(status='complete',contract=contract,group='P0',device=device,
        parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=0,evaluation=result,
        weight_sha256=sha256(lr.WEIGHT)))
    del model
    torch.cuda.empty_cache()


def readouts(report):
    row=Counter(cars=report['cars'])
    if report['cars']==1:
        row['single_vehicle_zero_delta']=1
        return row
    for i,label in enumerate(LOCATIONS):
        mask=report['location']==i
        row[label+'/cells']=int(mask.sum())
        row[label+'/nonzero_delta']=int((report['active'][mask]>0).sum())
        for field in FIELDS:
            values=report[field][mask]
            row[label+'/'+field+'_sum']=float(values.sum())
            row[label+'/'+field+'_square_sum']=float(values.square().sum())
            row[label+'/'+field+'_nonzero']=int((values>0).sum())
    for i in range(3):
        mask=report['pair_category']==i
        label=f'pair_{i}'
        row[label+'/cells']=int(mask.sum())
        row[label+'/raw_square_sum']=float(report['pair_raw'][mask].square().sum())
        row[label+'/gated_square_sum']=float(report['pair_gated'][mask].square().sum())
    row['lidar_pillars']=int(report['sensor_pillars'][:,0].sum())
    row['radar_pillars']=int(report['sensor_pillars'][:,1].sum())
    return row


def target_readouts(report,targets,cfg,scale):
    if report['cars']==1:
        return {str(t['object_id']):{'single_vehicle_zero_delta':True} for t in targets}
    field={k:report[k].cpu().numpy() for k in (*FIELDS,'location')}
    height,width=field['location'].shape
    low=np.asarray(cfg['preprocess']['cav_lidar_range'][:2])
    voxel=np.asarray(cfg['preprocess']['args']['voxel_size'][:2])
    spacing=voxel*2**(scale+1)
    center=low+voxel/2
    radius=np.linalg.norm(spacing)/2+1e-5
    rows={}
    for target in targets:
        poly=np.asarray(target['corners'])[:4,:2]
        begin=np.maximum(np.floor((poly.min(0)-radius-center)/spacing).astype(int),0)
        stop=np.minimum(np.ceil((poly.max(0)+radius-center)/spacing).astype(int)+1,[width,height])
        x,y=np.meshgrid(np.arange(begin[0],stop[0]),np.arange(begin[1],stop[1]))
        xy=center+np.stack((x,y),axis=-1)*spacing
        mask=shapely.distance(shapely.points(xy),shapely.Polygon(poly))<=radius
        take=(y[mask],x[mask])
        rows[str(target['object_id'])]=dict(cells=int(mask.sum()),
            location_counts={k:int(np.count_nonzero(field['location'][take]==j)) for j,k in enumerate(LOCATIONS)},
            mean_cell_rms={k:float(field[k][take].mean()) if mask.any() else None for k in FIELDS})
    return rows


def observe(root,model,contract,weight_hash):
    marker=root/'MECHANISM.json'
    if marker.is_file():
        result=read_json(marker)
        if (result['identity']!=identity([contract,weight_hash])
                or any(sha256(root/k)!=v for k,v in result['artifacts'].items())):
            raise ValueError('Mechanism evidence changed')
        return result
    history=read_json(OUT/'COHORT.json')
    totals,artifacts={},{}
    for branch in model.branches:
        branch.observe=True
    try:
        for weather in WEATHERS:
            dataset,cfg=build(lr.CONFIG,condition=weather,visualize=True)
            weather_totals=defaultdict(Counter)
            rows=[]
            with torch.no_grad():
                for i,batch in enumerate(archived_loader(dataset,weather)):
                    reference=read_json(lr.LR0/'final_evaluation/predictions/clean'/f'{i:06d}.json')
                    if reference['frame_key']!=list(dataset.keys[i]):
                        raise ValueError('Mechanism frame order differs')
                    ego=train_utils.to_device(batch['ego'],torch.device('cuda:0'))
                    model(ego)
                    targets=reference['official_gt']
                    args=cfg['model']['args']
                    vx,vy,_=args['voxel_size']
                    low=args['lidar_range']
                    support=object_support(batch['ego'],targets,None,
                         (low[0]+vx/2,low[1]+vy/2,vx*4,vy*4,50,176),dict(distance_edges_m=[30,60]))
                    objects=support['objects']
                    for oid in objects:
                        objects[oid].pop('center_unmatched_rms')
                        objects[oid]['cohort']=oid in history.get(weather,{}).get(str(i),[])
                    scales={}
                    for s,branch in enumerate(model.branches):
                        if len(branch.report)!=1:
                            raise ValueError('Missing mechanism frame report')
                        capture=branch.report[0]
                        stats=readouts(capture)
                        scales[str(s)]=stats
                        weather_totals[str(s)+'/all'].update(stats)
                        weather_totals[f'{s}/cars{capture["cars"]}'].update(stats)
                        local=target_readouts(capture,targets,cfg,s)
                        for oid,entry in local.items():
                            objects[oid].setdefault('scales',{})[str(s)]=entry
                    rows.append(dict(index=i,frame=reference['frame_key'],scales=scales,targets=objects))
                    if (i+1)%100==0:
                        print('MECHANISM',weather,i+1,flush=True)
            if len(rows)!=829 or sum(len(r['targets']) for r in rows)!=6235:
                raise ValueError('Mechanism run is incomplete')
            file=root/f'mechanism_{weather}.jsonl'
            atomic_jsonl(file,rows)
            artifacts[file.name]=sha256(file)
            totals[weather]=weather_totals
    finally:
        for branch in model.branches:
            branch.observe=False
            branch.report=[]
    result=dict(identity=identity([contract,weight_hash]),weather=totals,artifacts=artifacts)
    atomic_json(marker,result)
    return result


def final_evaluation(root,model,group,contract,device,selected,peak):
    comparisons={'F2':(), 'F3':('F2','uniform2'), 'F4':('F3','uniform3')}[group]
    evaluations={}
    for mode in (group,*comparisons):
        for branch in model.branches:
            branch.mode=mode
        dest=root/'final_evaluation' if mode==group else root/'controls'/mode
        seed(20261016)
        evaluations[mode]=checked_evaluation(evaluate)(model,None,lr.CONFIG,dest,WEATHERS,829,
                         identity([contract,group,mode]),selected['sha256'])
        verify_gt(dest)
    for branch in model.branches:
        branch.mode=group
    mechanism=observe(root,model,contract,selected['sha256'])
    params=sum(p.numel() for p in model.parameters())
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    atomic_json(root/'EVALUATED.json',dict(status='complete',group=group,contract=contract,device=device,
        selected=selected,evaluation=evaluations[group],controls={k:evaluations[k] for k in comparisons},
        mechanism_sha256=sha256(root/'MECHANISM.json'),parameters=params,trainable_parameters=trainable,
        peak_train_allocated_bytes=peak,mechanism_artifacts=mechanism['artifacts'],
        added_linear_macs_per_scene_per_vehicle_plus_one=sum(h*w*c*16 for h,w,c in
            ((100,352,128),(50,176,256)))))
