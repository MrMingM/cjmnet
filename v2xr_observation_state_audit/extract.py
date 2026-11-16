"""One frozen F0 pass per weather/frame with read-only feature hooks."""
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from v2xr_fusion_hypothesis_audit.shared.input_audit import (
    CLEAN, REPO, WEATHERS, VAL_MANIFEST, atomic_json, devices, lr,
    identity, read_json, sha256, source_contract, verify_gt,
)

import numpy as np
import torch
from collections import Counter
from contextlib import contextmanager
from inputs import TRAIN_MANIFEST
from runtime import build, loader
from checkpoint import seed
from v2xr_weather_l012.data import tensor_digest
from metrics import valid_nms_candidates
from geometry_warnings import checked_intersections
from evaluate import official_match_trace, statistics
from diagnostics import fixed_metrics
from support import corner_array
from opencood.hypes_yaml import yaml_utils
from opencood.tools import train_utils
from states import rays, labels

OUT=Path(os.environ['OBS_OUT'])
F0=lr.LR0.parent/'v2xr-radar-ra-rb-v1/F0'
EXPECTED='a5e12d10cde649c5a198b814277134ae06f8069a5b8e4d32505438c19b1fe8b1'
CARDS={4:('0000:1d:00.0','7137d9b2a09a2061'),5:('0000:20:00.0','7137d9aff40e38c1'),
       6:('0000:25:00.0','71010022e8ce1081'),7:('0000:2c:00.0','715219ae3dec3841')}


def preflight(mode):
    frozen=read_json(lr.LR0/'CONTRACT.json')
    selected=read_json(F0/'selected.json')
    weight=F0/'epochs/20/weights.pth'
    final=read_json(F0/'FINAL_RESULTS.json')
    if (selected['epoch']!=20 or selected['sha256']!=EXPECTED or sha256(weight)!=EXPECTED
            or final['status']!='complete' or sha256(F0/'config.yaml')!=frozen['config_sha256']
            or source_contract()!=frozen['sources']):
        raise ValueError('F0 frozen model or runtime identity differs')
    data=read_json(lr.LR0/'DATA.json')
    if sha256(lr.LR0/'DATA.json')!=frozen['data_sha256']:
        raise ValueError('Dataset verification receipt differs')
    split=[]
    for name,path,count in [('train',TRAIN_MANIFEST,8083),('validate',VAL_MANIFEST,829)]:
        row=data['splits'][name]
        if row['frames']!=count or sha256(path)!=row['input_manifest_sha256']:
            raise ValueError(f'Input manifest mismatch: {name}')
        split.append({s['scene'] for s in read_json(path)['scenes']})
    if split[0]&split[1]:
        raise ValueError('Training and validation share scene identifiers')
    verify_gt(F0/'final_evaluation')
    files=[*Path(__file__).parent.glob('*.py'),Path(__file__).parent/'run_all.sh',
           REPO/'v2xr_radar_ra_rb/model.py',REPO/'v2xr_lr1/lr1_model.py',
           REPO/'attfuse_gspr/reliability.py',REPO/'gspr_ggca/modules.py',
           REPO/'v2xr_radar_ra_rb/settings.json',F0/'final_evaluation/gt_manifest.json',
           TRAIN_MANIFEST,VAL_MANIFEST]
    record=dict(weight=EXPECTED,config=frozen['config_sha256'],train_scenes=len(split[0]),
                validation_scenes=len(split[1]),overlap=0,mode=mode,
                files={str(p):sha256(p) for p in files})
    file=OUT/mode/'CONTRACT.json'
    if file.exists() and read_json(file)!=record:
        raise ValueError('Prior cache contract differs')
    atomic_json(file,record)
    return weight,identity(record)


def make_model(cfg,weight):
    sys.path.insert(0,str(REPO/'v2xr_radar_ra_rb'))
    from v2xr_radar_ra_rb.model import Model
    settings=read_json(REPO/'v2xr_radar_ra_rb/settings.json')['F0']
    if settings['ra'] or settings['rb']:
        raise ValueError('F0 unexpectedly has Radar enrichment')
    model=Model(cfg['model']['args'],settings).cuda().eval().requires_grad_(False)
    model.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True),strict=True)
    return model


@contextmanager
def capture(model):
    maps={}
    scatter=[]
    def save_scatter(_, args, result):
        scatter.append(result['spatial_features'][0])
    def fusion_before(_, args):
        maps['trunk']=args[0][0]
    def fusion_after(_, args, value):
        maps['fused']=value[0]
    def backbone_before(_, args):
        maps['concat']=args[0]['spatial_features'][0]
    # Sample the first shared/fused scale so both stages have identical receptive-grid centers.
    hooks=[model.scatter.register_forward_hook(save_scatter),
           model.backbone.register_forward_pre_hook(backbone_before),
           model.backbone.fuse_modules[0].register_forward_pre_hook(fusion_before),
           model.backbone.fuse_modules[0].register_forward_hook(fusion_after)]
    try:
        yield maps,scatter
    finally:
        for hook in hooks: hook.remove()


def raw(ego,sensor):
    points=ego['projected_'+sensor][0][0]
    return np.asarray(points,dtype=np.float32)


def nearest(xy,origin,spacing,shape):
    h,w=shape
    ix=np.rint((xy-origin)/spacing).astype(int)
    return ix[:,1].clip(0,h-1),ix[:,0].clip(0,w-1)


def gspr_neighborhood(xy,origin,model):
    from scipy.ndimage import maximum_filter, uniform_filter
    values=np.zeros((3,200,704),dtype=np.float32)
    if model._point_outputs is not None:
        coords=model._raw['voxel_coords']
        ego=coords[:,0]==0
        _,_,iy,ix=coords[ego].long().cpu().numpy().T
        values[0,iy,ix]=model._point_outputs['pillar_reliability'].reshape(-1)[ego].cpu().numpy()
        values[1,iy,ix]=model._point_outputs['pillar_uncertainty'].reshape(-1)[ego].cpu().numpy()
        values[2,iy,ix]=1
    y,x=nearest(xy,origin,.4,values.shape[-2:])
    avg=uniform_filter(values,size=(1,5,5))[:,y,x]
    mx=maximum_filter(values[0],size=5)[y,x]
    return np.column_stack((avg[:2].T/np.maximum(avg[2:, :].T,1/25),mx,avg[2]*25))


def sample(maps,xy,origin):
    feat={}
    for name,array in maps.items():
        expected=(200,704) if name in ('lidar','radar','concat') else (100,352)
        if tuple(array.shape[-2:])!=expected:
            raise ValueError(f'Unexpected {name} grid shape {array.shape}')
        stride=.4 if expected[0]==200 else .8
        y,x=nearest(xy,origin,stride,expected)
        feat[name]=array[:,y,x].T.float().cpu().numpy().astype(np.float16)
    return feat


def point_counts(points,xy):
    from scipy.spatial import cKDTree
    tree=cKDTree(points[:,:2])
    return np.asarray(tree.query_ball_point(xy,.57,p=np.inf,return_length=True))


def rows_for(xy, lidar, radar, clean, model, maps, origin, detector_score):
    l,r,c=(point_counts(p,xy) for p in (lidar,radar,clean))
    ray=rays(lidar,xy)
    state=labels(l,r,c,ray)
    if ((state[:,0]==1)&(l!=0)).any() or ((state[:,1]==1)&((c<3)|(l!=0))).any():
        raise ValueError('State annotation violates measurement rules')
    return dict(states=state,points=np.column_stack((l,r,c)),ray=ray,
                base=np.column_stack((np.log1p(l),np.linalg.norm(xy,axis=1)/80,xy/100)),
                radar_info=np.column_stack((np.log1p(r),point_counts(radar,xy+np.array([.8,0]))>0,
                                           point_counts(radar,xy+np.array([0,.8]))>0)),
                gspr=gspr_neighborhood(xy,origin,model), score=detector_score, xy=xy,
                **sample(maps,xy,origin))


def extract(mode):
    limit=24 if mode=='smoke' else 829
    root=OUT/mode
    root.mkdir(parents=True,exist_ok=True)
    weight,contract=preflight(mode)
    gpu=int(os.environ['GPU'])
    devices.DEVICES={'audit':(gpu,*CARDS[gpu])}
    devices.idle(['audit'])
    device=devices.device('audit')
    print('F0 device',device,flush=True)
    seed(20261016)
    cfg=yaml_utils.load_yaml(str(F0/'config.yaml'))
    model=make_model(cfg,weight)
    frozen=tensor_digest(model.state_dict())
    generator=torch.Generator()
    generator.set_state(torch.get_rng_state())
    datasets=[build(F0/'config.yaml',condition=w,visualize=True)[0] for w in WEATHERS]
    if any(len(ds)!=829 or ds.keys!=datasets[0].keys for ds in datasets):
        raise ValueError('Weather splits not paired')
    loaders=[loader(ds,evaluation=True,limit=limit) for ds in datasets]
    for batches in loaders: batches.generator=generator
    grid_origin=np.array(cfg['model']['args']['lidar_range'][:2],dtype=float)+.2
    y,x=np.mgrid[0:100:8,0:352:8]
    grid=np.column_stack((x.ravel(),y.ravel()))*.8+grid_origin
    files={}
    expected_counts=Counter()
    original=fixed_metrics(F0/'final_evaluation')
    drift=Counter()
    with torch.inference_mode(),checked_intersections(),valid_nms_candidates(Counter()):
        for i,batches in enumerate(zip(*loaders,strict=True)):
            path=root/'cache'/f'{i:06d}.npz'
            ref=read_json(CLEAN/f'{i:06d}.json')
            frame=ref['frame_key']
            if frame!=datasets[0].keys[i]:
                raise ValueError('Fixed target identity differs')
            targets=ref['official_gt']
            expected_counts['targets']+=len(targets)
            if path.exists():
                with np.load(path,allow_pickle=False) as old:
                    if old['contract'].item()!=contract or old['frame'].tolist()!=frame:
                        raise ValueError('Incompatible frame cache')
                files[path.name]=sha256(path)
                continue
            payload=dict(contract=np.array(contract),frame=np.array(frame),
                         targets=np.array([str(t['object_id']) for t in targets]))
            clean=raw(batches[0]['ego'],'lidar')
            reference_radar=raw(batches[0]['ego'],'radar')
            for w,ds,batch in zip(WEATHERS,datasets,batches):
                cpu=batch['ego']
                lidar,radar=raw(cpu,'lidar'),raw(cpu,'radar')
                # Native independently shuffles input points. Compare summary and exact XYZ multiset.
                if radar.shape!=reference_radar.shape:
                    raise ValueError('Paired Radar point counts differ')
                if len(radar) and w!='clean':
                    sort=lambda a:a[np.lexsort(a[:,:3].T),:3]
                    np.testing.assert_allclose(sort(radar),sort(reference_radar),atol=1e-6,rtol=0)
                ego=train_utils.to_device(cpu,torch.device('cuda:0'))
                with capture(model) as (maps,scatter):
                    outputs=model(ego)
                if len(scatter)!=2 or set(maps)!= {'trunk','fused','concat'}:
                    raise ValueError('F0 feature hooks did not capture expected tensors')
                maps.update(lidar=scatter[0],radar=scatter[1])
                if not torch.equal(maps['concat'],torch.cat(scatter)):
                    raise ValueError('LiDAR/Radar scatter order differs from F0 concatenation')
                boxes,scores=ds.native.post_processor.post_process({'ego':ego},{'ego':outputs})
                gt=torch.from_numpy(corner_array(targets))
                matches,_=official_match_trace(boxes,scores,gt,[t['object_id'] for t in targets],statistics(),.7)
                matched_ids={str(m['object_id']) for m in matches}
                matched_boxes={m['prediction_index'] for m in matches}
                historic=original[f'{w}/{i}']
                previous={str(m['object_id']) for m in historic['matched_pairs']['0.7']}
                payload[f'{w}_match_drift']=np.array([len(matched_ids^previous),
                    abs((0 if boxes is None else len(boxes))-len(matches)-historic['thresholds']['0.7']['fp'])])
                target_xy=np.asarray([t['box'][:2] for t in targets],dtype=float).reshape(-1,2)
                pred_xy=boxes[:,:4,:2].mean(1).cpu().numpy() if boxes is not None else np.zeros((0,2))
                score_map=outputs['psm'][0].sigmoid().max(0).values
                for kind,xy,label in (
                    ('grid',grid,np.full(len(grid),-1)),
                    ('target',target_xy,np.array([str(t['object_id']) not in matched_ids for t in targets])),
                    ('prediction',pred_xy,np.array([j not in matched_boxes for j in range(len(pred_xy))])),
                ):
                    gy,gx=nearest(xy,grid_origin,.8,score_map.shape)
                    score=(scores.cpu().numpy() if kind=='prediction' and scores is not None
                           else score_map[gy,gx].cpu().numpy())
                    data=rows_for(xy,lidar,radar,clean,model,maps,grid_origin,score)
                    sy,sx=nearest(xy,grid_origin,.4,(200,704))
                    encoded=[]
                    for sensor in ('lidar','radar'):
                        coords=cpu['processed_'+sensor]['voxel_coords'].numpy()
                        coords=coords[coords[:,0]==0]
                        occupancy=np.zeros((200,704),dtype=bool)
                        occupancy[coords[:,2],coords[:,3]]=True
                        encoded.append(occupancy[sy,sx])
                    data['encoded']=np.column_stack(encoded)
                    if kind=='prediction':
                        data['geometry']=boxes.cpu().numpy().reshape(-1,24) if boxes is not None else np.zeros((0,24))
                    data['error']=label.astype(np.int8)
                    for k,v in data.items():
                        if not np.isfinite(v).all():
                            raise ValueError(f'Nonfinite cache array {w}/{kind}/{k}')
                        payload[f'{w}_{kind}_{k}']=v
            path.parent.mkdir(exist_ok=True)
            temp=path.with_suffix('.tmp')
            with temp.open('wb') as f: np.savez_compressed(f,**payload)
            temp.replace(path)
            files[path.name]=sha256(path)
            if i==0 or (i+1)%25==0: print(mode,'paired frames',i+1,limit,flush=True)
    if frozen!=tensor_digest(model.state_dict()):
        raise ValueError('Frozen F0 state changed during inference')
    if len(files)!=limit or (mode=='full' and expected_counts['targets']!=6235):
        raise ValueError('Incomplete frame or target count')
    for filename in files:
        with np.load(root/'cache'/filename,allow_pickle=False) as row:
            for w in WEATHERS:
                a,b=map(int,row[f'{w}_match_drift'])
                drift[w+'/target_symmetric_difference']+=a
                drift[w+'/absolute_fp_count_difference']+=b
    atomic_json(root/'CACHE_MANIFEST.json',dict(contract=contract,frames=len(files),files=files,
                targets=expected_counts['targets'],device=device,frozen_state=frozen,archive_drift=drift))


if __name__=='__main__':
    extract(sys.argv[1])
