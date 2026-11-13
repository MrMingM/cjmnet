"""Collect F0 candidate evidence through the unchanged native detector."""
from collections import Counter
from contextlib import contextmanager
import gzip
import json
from pathlib import Path
import sys

from context import F0, OUT, WEATHER, bind, frozen_clean, load, setup, sha, write

run=setup()
import numpy as np
import shapely
import torch

from checkpoint import seed
from diagnostics import input_context
from metrics import valid_nms_candidates
from opencood.tools import train_utils
from opencood.utils import box_utils, common_utils
from opencood.pcdet_utils.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu
from support import patched
from torch.utils.data import DataLoader, Subset


def as_boxes(boxes):
    return np.asarray(boxes,dtype=np.float32).reshape(-1,8,3)


def iou_matrix(pred,gt):
    # Same four projected corners and area ratio as OpenCOOD, vectorized.
    pred,gt=as_boxes(pred),as_boxes(gt)
    if not len(pred) or not len(gt):
        return np.zeros((len(pred),len(gt)))
    a,b=(shapely.polygons(x[:,:4,:2]) for x in (pred,gt))
    result=np.zeros((len(a),len(b)))
    good=shapely.is_valid(a)&(shapely.area(a)>0)
    a=a[good]
    lo,hi=pred[good,:,:2].min(1),pred[good,:,:2].max(1)
    for j,g in enumerate(b):
        x0,y0,x1,y1=g.bounds
        k=(lo[:,0]<=x1)&(hi[:,0]>=x0)&(lo[:,1]<=y1)&(hi[:,1]>=y0)
        if k.any():
            z=shapely.intersection(a[k],g)
            result[np.flatnonzero(good)[k],j]=shapely.area(z)/(shapely.area(a[k])+g.area-shapely.area(z))
    return result


def stats(x):
    return [float(x.mean()),*np.quantile(x,[.1,.5,.9]).tolist()] if len(x) else [None]*4


def inside(points,box):
    # Official box representation and native points-in-box test.
    if not len(points):
        return np.zeros(0,bool)
    return np.asarray(points_in_boxes_cpu(np.ascontiguousarray(points[:,:3],dtype=np.float32),
        np.ascontiguousarray(box[None,[0,1,2,5,4,3,6]],dtype=np.float32)))[0].astype(bool)


def feature_context(ego,model):
    raw=[list(ego['projected_'+s][0]) for s in ('lidar','radar')]
    vox=ego['processed_lidar']
    cloud=vox['voxel_features'].numpy()
    coords=vox['voxel_coords'].numpy()
    out=model._point_outputs
    if out is None:
        return raw,np.empty((0,3)),np.empty(0),np.empty(0),np.empty(0)
    valid=out['point_valid_mask'].cpu().numpy()
    return (raw,cloud[:,:,:3][valid],out['point_reliability'].cpu().numpy()[valid],
            out['point_uncertainty'].cpu().numpy()[valid],np.broadcast_to(coords[:,0,None],valid.shape)[valid])


def evidence(box,score,context):
    (ls,rs),xyz,reliability,uncertainty,agents=context
    lidar=[p[inside(p,box)] for p in ls]
    radar=[p[inside(p,box)] for p in rs]
    l,r=np.concatenate(lidar),np.concatenate(radar)
    mask=inside(xyz,box)
    # Coarse 0.4m XY bin agreement inside candidate, pooled across agents.
    cells=lambda p:set(map(tuple,np.floor(p[:,:2]/.4).astype(int)))
    lc,rc=cells(l),cells(r)
    offset=r[:,:3]-box[:3]
    center=offset.mean(0).tolist() if len(r) else [None]*3
    spread=offset.std(0).tolist() if len(r) else [None]*3
    return dict(score=float(score),distance=float(np.linalg.norm(box[:2])),
        lidar=len(l),radar=len(r),lidar_per_car=[len(x) for x in lidar],
        radar_per_car=[len(x) for x in radar],radar_offset=center,radar_std=spread,
        lidar_radar_overlap=len(lc&rc),radar_unique=len(rc-lc),lidar_unique=len(lc-rc),
        gspr_points=int(mask.sum()),reliability=stats(reliability[mask]),
        uncertainty=stats(uncertainty[mask]),
        gspr_per_car=[stats(reliability[mask&(agents==i)]) for i in range(len(ls))])


def inspect_head(ego, output, pp):
    p=output['psm'].permute(0,2,3,1).sigmoid().flatten()
    decoded=pp.delta_to_boxes3d(output['rm'],ego['anchor_box'])[0]
    if 'dm' in output or len(p)!=len(decoded):
        raise ValueError('Unsupported official head layout')
    corners=box_utils.project_box3d(box_utils.boxes_to_corners_3d(decoded,order=pp.params['order']),
                                     ego['transformation_matrix'])
    return p,decoded,corners


def native_post(batch,out,pp,threshold):
    import copy
    from opencood.utils import box_utils as bu
    keep_stage={}
    native=bu.nms_rotated
    def trace(boxes,scores,threshold):
        keep=native(boxes,scores,threshold)
        keep_stage.update(nms_in=boxes.detach().cpu().numpy(),nms_out=boxes[keep].detach().cpu().numpy())
        return keep
    params=copy.deepcopy(pp.params)
    params['target_args']['score_threshold']=threshold
    with patched(pp,'params',params), patched(bu,'nms_rotated',trace), valid_nms_candidates(Counter()):
        pred,score=pp.post_process(batch,{'ego':out})
    return pred,score,keep_stage



def pack(pred,score,all_corners,decoded,raw_score,ctx):
    if pred is None:
        return []
    corners=pred.cpu().numpy()
    scores=score.cpu().numpy()
    # Lookup exact decoded box identity, including overlapping anchors.
    lookup={}
    for i in np.flatnonzero(raw_score>.01):
        lookup.setdefault(all_corners[i].tobytes(),[]).append(i)
    eligible=np.flatnonzero(raw_score>.01)
    rows=[]
    for box,s in zip(corners,scores):
        ids=lookup.get(box.tobytes(),[])
        ids=[i for i in ids if abs(float(raw_score[i]-s))<1e-6]
        if not ids:
            same=eligible[np.isclose(raw_score[eligible],s,atol=1e-6,rtol=1e-6)]
            errors=np.max(np.abs(all_corners[same]-box),axis=(1,2))
            ids=same[errors<=1e-4].tolist()
            if len(ids)!=1:
                raise ValueError(f'Native final box association ambiguous: {len(ids)} at score {s}')
        i=ids[0]
        rows.append(dict(anchor=int(i),corners=box.tolist(),**evidence(decoded[i],s,ctx)))
    return rows


def dev_data(config,weather):
    dataset,_=run.build(config,split='train',visualize=True)
    dataset.native.train_sim=False
    if weather=='fog':
        for scene in dataset.native.scenario_database.values():
            for car in scene.values():
                for stamp,rec in car.items():
                    if stamp!='ego':
                        rec['lidar']=rec['lidar'].replace('.pcd','_fog_0.060.pcd')
    dataset.native.train=False
    dataset.native.training=False
    scenes={}
    for i,(s,_) in enumerate(dataset.keys):
        scenes.setdefault(s,[]).append(i)
    # Fixed training-only development: four equally spaced timestamps per scene.
    chosen=sorted({v[int(k)] for v in scenes.values() for k in np.linspace(0,len(v)-1,4)})
    return dataset,chosen


def dev_gt(pp,batch):
    from predict import live_gt_observer
    return live_gt_observer(pp,batch,pp.generate_gt_bbx)[1]


def run_collect(split):
    model,config,digest=load(run,'F0')
    root=OUT/split
    identity=dict(weight=digest,code=sha(__file__),split=split,low_floor=.05,
                  config=sha(config),postprocessor=sha(run.RUNTIME/'opencood/data_utils/post_processor/voxel_postprocessor.py'))
    record=root/'IDENTITY.json'
    if record.exists():
        previous=__import__('context').read(record)
        if any(previous.get(k)!=v for k,v in identity.items() if k!='code'):
            raise ValueError('Frozen cache inputs differ')
        bind(root/f'COLLECTOR_{identity["code"]}.json',identity)
    else:
        bind(record,identity)
    conditions=WEATHER if split=='validate' else ('clean','fog')
    for weather in conditions:
        seed(20261016)
        if split=='train_dev':
            data,chosen=dev_data(config,weather)
        else:
            data,_=run.build(config,condition=weather,visualize=True)
            chosen=list(range(len(data)))
        loader=DataLoader(Subset(data,chosen),batch_size=1,num_workers=6,collate_fn=data.native.collate_batch_test)
        pp=data.native.post_processor
        official_threshold=float(pp.params['target_args']['score_threshold'])
        if official_threshold<=.05:
            raise ValueError('Lowered threshold must be below official threshold')
        destination=root/weather
        destination.mkdir(parents=True,exist_ok=True)
        for i,batch in zip(chosen,loader):
            path=destination/f'{i:06d}.json.gz'
            if path.exists():
                continue
            key=data.keys[i]
            device=train_utils.to_device(batch,torch.device('cuda:0'))
            with torch.inference_mode():
                # Some historical train collates omit anchors when train=True.
                out=model(device['ego'])
                prob,dec,corners=inspect_head(device['ego'],out,pp)
                p,d,c=(x.detach().cpu().numpy() for x in (prob,dec,corners))
                ctx=feature_context(batch['ego'],model)
                native,s,trace=native_post(device,out,pp,official_threshold)
                low,low_s,low_trace=native_post(device,out,pp,.05)
                if split=='validate':
                    gt=frozen_clean(i,key)
                    clean_obs=F0/'final_evaluation/observations/clean'/f'{i:06d}.json'
                    clean_counts=__import__('context').read(clean_obs)['objects']
                elif weather=='clean':
                    gt=dev_gt(pp,device)
                    clean_counts=input_context(batch,gt)['objects']
                else:
                    with gzip.open(root/'clean'/f'{i:06d}.json.gz','rt') as f:
                        clean=json.load(f)
                    gt,clean_counts=clean['gt'],clean['clean_counts']
                pred=pack(native,s,c,d,p,ctx)
                lower=pack(low,low_s,c,d,p,ctx)
                raw_good=np.isfinite(c).all((1,2))&np.isfinite(p)
                g=[r['corners'] for r in gt]
                raw_ious=iou_matrix(c[raw_good],g)
                yes=raw_ious>=.7
                valid_scores=p[raw_good]
                nms_in=trace.get('nms_in',[])
                nms_out=trace.get('nms_out',[])
                coverage=dict(raw=yes.any(0).tolist(), above=yes[valid_scores>official_threshold].any(0).tolist(),
                    nms_in=(iou_matrix(nms_in,g)>=.7).any(0).tolist(),
                    nms_out=(iou_matrix(nms_out,g)>=.7).any(0).tolist(),
                    final=(iou_matrix([v['corners'] for v in pred],g)>=.7).any(0).tolist(),
                    counts=[int(len(p)),int((p>official_threshold).sum()),len(nms_in),len(nms_out),len(pred),len(lower)])
                if i==chosen[0]:
                    # Check vectorized overlap against the frozen geometry utility.
                    a=common_utils.convert_format(as_boxes([v['corners'] for v in pred]))
                    b=common_utils.convert_format(as_boxes(g))
                    expected=np.asarray([common_utils.compute_iou(x,b) for x in a]).reshape(len(a),len(b))
                    if not np.allclose(expected,iou_matrix([v['corners'] for v in pred],g),atol=1e-8):
                        raise ValueError('Diagnostic IoU differs from official geometry')
                    print('SMOKE',split,weather,coverage['counts'],flush=True)
                observed=input_context(batch,gt)['objects']
            record=dict(index=i,key=key,weather=weather,gt=gt,clean_counts=clean_counts,observed=observed,
                        final=pred,low=lower,coverage=coverage,score_threshold=official_threshold,
                        nms_threshold=float(pp.params['nms_thresh']))
            tmp=path.with_suffix('.tmp')
            with gzip.open(tmp,'wt',encoding='utf-8') as stream:
                json.dump(record,stream,allow_nan=False)
            tmp.replace(path)
            if (i+1)%100==0:
                print(split,weather,i+1,flush=True)
        print('COLLECTED',split,weather,len(chosen),flush=True)
    write(root/'COMPLETE.json',dict(weight=digest,conditions=list(conditions)))


if __name__=='__main__':
    run_collect(sys.argv[1])

