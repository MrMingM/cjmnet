"""Frozen P0 candidate audit and paired local weather intervention."""
from collections import Counter, defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run
from common import atomic_json, read_json, sha256
from device import pre_device, map_device
from diagnostics import fixed_metrics, verify_gt
from geometry_warnings import checked_intersections
from metrics import valid_nms_candidates
from support import corner_array, patched
from evaluate import official_match_trace, statistics
from runtime import build, loader
from checkpoint import seed
import numpy as np
import shapely
import torch
from opencood.tools import train_utils
from opencood.utils import box_utils

OUT = run.OUT / 'p0-frozen-failure-v1'
THRESHOLD = .7
RADIUS = 12.


def select_targets():
    paths = {'single': run.OUT/'ablations/P0/single',
             'multi': run.OUT/'ablations/P0/same_weights_gpu4'}
    for p in paths.values():
        verify_gt(p)
    matches = {mode: fixed_metrics(path) for mode, path in paths.items()}
    cohort = {}
    for weather, count in (('fog',103),('snow',224)):
        frames = {}
        for i in range(829):
            key=f'{weather}/{i}'
            a,b=({str(p['object_id']) for p in matches[m][key]['matched_pairs']['0.7']}
                 for m in ('single','multi'))
            if a-b:
                frames[i]=sorted(a-b)
        if sum(map(len,frames.values())) != count:
            raise ValueError(f'{weather} archived cohort differs')
        cohort[weather]=frames
    return cohort


def feature_hooks(model):
    rows = {}
    def save(i):
        def capture(_, args):
            rows[i] = args[0].detach().clone()
        return capture
    handles=[layer.register_forward_pre_hook(save(i)) for i,layer in enumerate(model.backbone.fuse_modules)]
    return rows, handles


def polygons(corners):
    return shapely.polygons(corners[:, :4, :2])


def overlap(boxes, target):
    points = boxes[:, :4, :2]
    finite = np.isfinite(points).all((1,2))
    lo,hi=points.min(1),points.max(1)
    t=target[:4,:2]
    intersects=finite & ((lo<=t.max(0)) & (hi>=t.min(0))).all(1)
    nearby=finite & (np.linalg.norm(points.mean(1)-t.mean(0),axis=1)<=6.)
    ids=np.flatnonzero(intersects | nearby)
    ious=np.zeros(len(ids),dtype=np.float32)
    if len(ids):
        poly=polygons(boxes[ids])
        valid=shapely.is_valid(poly) & (shapely.area(poly)>0)
        valid_ids=np.flatnonzero(valid & intersects[ids])
        gt=shapely.Polygon(t)
        selected=poly[valid_ids]
        with np.errstate(invalid='ignore'):
            values=shapely.area(shapely.intersection(selected,gt))/shapely.area(shapely.union(selected,gt))
        if not np.isfinite(values).all():
            raise ValueError('Invalid target overlap')
        ious[valid_ids]=values
    return ids,ious


def trace(model, ego, processor, reference, features, single):
    model.single_vehicle=single
    features.clear()
    native=box_utils.nms_rotated
    nms={}
    def capture(boxes,scores,threshold):
        keep=native(boxes,scores,threshold)
        nms.update(boxes=boxes, scores=scores, keep=keep)
        return keep
    outputs=model(ego)
    logits=outputs['psm'].permute(0,2,3,1).reshape(-1)
    scores=logits.sigmoid()
    decoded=processor.delta_to_boxes3d(outputs['rm'],ego['anchor_box'])[0]
    corners=box_utils.project_box3d(box_utils.boxes_to_corners_3d(decoded, order=processor.params['order']),
                                   ego['transformation_matrix'])
    selected=torch.nonzero(scores>processor.params['target_args']['score_threshold']).flatten()
    flags=np.zeros(len(decoded),dtype=np.uint8)
    flags[selected.cpu().numpy()]=1
    with patched(box_utils,'nms_rotated',capture):
        final, final_scores=processor.post_process({'ego':ego},{'ego':outputs})
    if len(selected):
        above=corners[selected]
        valid=box_utils.remove_large_pred_bbx(above,processor.dataset)&box_utils.remove_bbx_abnormal_z(above)
        indices=selected[valid]
        if not torch.equal(corners[indices],nms['boxes']) or not torch.equal(scores[indices],nms['scores']):
            raise ValueError('Official pre-NMS candidates differ from head index trace')
        flags[indices.cpu().numpy()]=2
        indices=indices[nms['keep']]
        flags[indices.cpu().numpy()]=3
        in_range=box_utils.get_mask_for_boxes_within_range_torch(corners[indices],processor.params['anchor_args']['cav_lidar_range'])
        indices=indices[in_range]
        if not torch.equal(corners[indices],final) or not torch.equal(scores[indices],final_scores):
            raise ValueError('Official final candidates differ from trace')
        flags[indices.cpu().numpy()]=4
    gt=torch.from_numpy(corner_array(reference))
    pairs,_=official_match_trace(final,final_scores,gt,[r['object_id'] for r in reference],statistics(),THRESHOLD)
    hits={str(p['object_id']) for p in pairs}
    arrays=(corners.cpu().numpy(),decoded.cpu().numpy(),logits.cpu().numpy(),scores.cpu().numpy(),flags)
    return arrays,dict(features),hits


def candidate_rows(arrays, target):
    corners,boxes,logits,scores,flags=arrays
    ids,iou=overlap(corners,np.asarray(target['corners'],dtype=np.float32))
    qualifies=iou>=THRESHOLD
    first=next((name for stage,name in enumerate(('localization','classification_threshold','box_size_or_height',
                                                  'nms','range_filter')) if not np.any(qualifies & (flags[ids]>=stage))),
               'evaluation_assignment')
    maxima=[float(iou[flags[ids]>=stage].max(initial=0)) for stage in range(5)]
    return dict(first_missing_stage=first,max_iou_by_stage=maxima,
                eligible_by_stage=[int(np.count_nonzero(qualifies & (flags[ids]>=stage))) for stage in range(5)],
                candidates=dict(anchor_index=ids,decoded_hwl_yaw=boxes[ids],logit=logits[ids],
                                score=scores[ids],iou=iou,last_stage=flags[ids]))


def voxel_masks(coords, low, spacing, targets):
    xy=low+(coords[:,[3,2]]+.5)*spacing
    point=shapely.points(xy)
    protected=np.zeros(len(coords),dtype=bool)
    nearby=np.zeros(len(coords),dtype=bool)
    for target in targets:
        b=np.asarray(target['corners'],dtype=np.float32)[:4,:2]
        # Circumscribed radius of a voxel prevents replacing any voxel intersecting a target.
        protected |= shapely.distance(point,shapely.Polygon(b)) <= np.linalg.norm(spacing)/2+1e-5
        nearby |= np.linalg.norm(xy-b.mean(0),axis=1)<=RADIUS
    return protected,nearby


def replace_context(ego, clean, targets, config):
    if (not torch.equal(ego['record_len'],clean['record_len'])
            or ego['ego_flag']!=clean['ego_flag'] or not np.array_equal(ego['lidar_pose'],clean['lidar_pose'])):
        raise ValueError('Paired weather CAV layout differs')
    low=np.asarray(config['preprocess']['cav_lidar_range'][:2])
    spacing=np.asarray(config['preprocess']['args']['voxel_size'][:2])
    original,other=(data['processed_lidar'] for data in (ego,clean))
    p0,n0=voxel_masks(original['voxel_coords'].numpy(),low,spacing,targets)
    p1,n1=voxel_masks(other['voxel_coords'].numpy(),low,spacing,targets)
    keep=p0|~n0
    insert=~p1&n1
    mixed={k:torch.cat((original[k][keep],other[k][insert])) for k in original}
    new_p,new_near=voxel_masks(mixed['voxel_coords'].numpy(),low,spacing,targets)
    if not all(torch.equal(mixed[k][new_p],original[k][p0]) and
               torch.equal(mixed[k][~new_near],original[k][~n0]) for k in original):
        raise ValueError('Target or distant voxel features changed')
    if len(torch.unique(mixed['voxel_coords'],dim=0))!=len(mixed['voxel_coords']):
        raise ValueError('Duplicated hybrid voxels')
    return dict(ego,processed_lidar=mixed),dict(protected=int(p0.sum()),removed=int((~keep).sum()),added=int(insert.sum()))


def feature_delta(before, after, target, config):
    low=np.asarray(config['preprocess']['cav_lidar_range'][:2])
    spacing=np.asarray(config['preprocess']['args']['voxel_size'][:2])
    corners=np.asarray(target['corners'])[:4,:2]
    results={}
    for i,a in before.items():
        b=after[i]
        stride=2**(i+1)
        x=low[0]+(np.arange(a.shape[-1])*stride+.5)*spacing[0]
        y=low[1]+(np.arange(a.shape[-2])*stride+.5)*spacing[1]
        xx,yy=np.meshgrid(x,y)
        p=shapely.points(xx.ravel(),yy.ravel())
        near=shapely.distance(p,shapely.Polygon(corners))<=np.linalg.norm(spacing)*stride/2
        mask=torch.from_numpy(near.reshape(a.shape[-2:])).to(a.device)
        av,bv=a[...,mask].float(),b[...,mask].float()
        norms=av.square().sum((1,2)).sqrt()
        differences=(av-bv).square().sum((1,2)).sqrt()
        results[str(stride)]=dict(cells=int(near.sum()),base_norm=norms.cpu().tolist(),
                                  delta_norm=differences.cpu().tolist(),
                                  relative=(differences/norms.clamp_min(1e-8)).cpu().tolist())
    return results


def archived_loader(dataset, weather):
    # Match the worker base seeds of the archived Clean -> Fog -> Snow evaluation.
    generator=torch.Generator().manual_seed(20261016)
    for _ in range(('clean','fog','snow').index(weather)):
        torch.empty((),dtype=torch.int64).random_(generator=generator)
    batches=loader(dataset,evaluation=True,limit=829)
    batches.generator=generator
    return batches


def main():
    pre_device()
    device=map_device()
    seed(20261016)
    cohort=select_targets()
    weight=run.WEIGHT
    digest=sha256(weight)
    if digest!='0eb0ec8bd6a6e0bc368c7cc8aacd49a33cfe014413463f8b314fdbf76c6defd2':
        raise ValueError('P0 weight identity changed')
    from opencood.hypes_yaml import yaml_utils
    config=yaml_utils.load_yaml(str(run.CONFIG))
    model=run.make_model(config,'P2').eval()
    model.fusion.mode='module_off'
    missing,extra=model.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True),strict=False)
    if extra or not all(key.startswith('backbone.fuse_modules.2.') for key in missing):
        raise ValueError('P0 weight mismatch')
    features,hooks=feature_hooks(model)
    clean_ds,_=build(run.CONFIG,condition='clean',visualize=True)
    summary={}
    atomic_json(OUT/'COHORT.json',cohort)
    for weather,frames in cohort.items():
        ds,_=build(run.CONFIG,condition=weather,visualize=True)
        assert ds.keys==clean_ds.keys
        counts=Counter()
        for i,(weather_batch,clean_batch) in enumerate(zip(archived_loader(ds,weather),
                                                           archived_loader(clean_ds,'clean'),strict=True)):
            if i not in frames:
                continue
            oids=frames[i]
            path=OUT/weather/f'{i:06d}.json'
            if path.exists():
                old=read_json(path)
                if old['weight_sha256']!=digest or set(old['targets'])!=set(oids):
                    raise ValueError('Resume frame identity changed')
                counts.update(old['counts'])
                continue
            weather_ego=weather_batch['ego']
            clean_ego=clean_batch['ego']
            clean_reference=read_json(run.LR0/'final_evaluation/predictions/clean'/f'{i:06d}.json')
            if list(ds.keys[i])!=clean_reference['frame_key']:
                raise ValueError('Fixed target frame key differs')
            reference=clean_reference['official_gt']
            targets={str(row['object_id']):row for row in reference if str(row['object_id']) in oids}
            hybrid,splice=replace_context(weather_ego,clean_ego,list(targets.values()),config)
            device_data=lambda x:train_utils.to_device(x,torch.device('cuda:0'))
            with torch.no_grad(), checked_intersections(), valid_nms_candidates(Counter()):
                single,_,single_hits=trace(model,device_data(weather_ego),ds.native.post_processor,reference,features,True)
                multi,before,multi_hits=trace(model,device_data(weather_ego),ds.native.post_processor,reference,features,False)
                context,after,context_hits=trace(model,device_data(hybrid),ds.native.post_processor,reference,features,False)
            if not set(oids)<=single_hits or set(oids)&multi_hits:
                raise ValueError(f'Archived lost targets did not replay: {weather}/{i} single missing={set(oids)-single_hits} multi={set(oids)&multi_hits}')
            rows={}
            current=Counter()
            for oid,target in targets.items():
                row={label:candidate_rows(array,target) for label,array in
                     (('single',single),('multi',multi),('clean_surroundings',context))}
                row['clean_surroundings_detected']=oid in context_hits
                row['feature_change']=feature_delta(before,after,target,config)
                row['target']=target
                current[row['multi']['first_missing_stage']]+=1
                current['context_recovered']+=oid in context_hits
                current['target_count']+=1
                rows[oid]=row
            counts.update(current)
            atomic_json(path,dict(weight_sha256=digest,frame=ds.keys[i],targets=rows,counts=current,splice=splice))
            if len(list((OUT/weather).glob('*.json')))%20==0:
                print(weather,i,dict(counts),flush=True)
        summary[weather]=dict(counts)
        print('COMPLETE',weather,dict(counts),flush=True)
    atomic_json(OUT/'RESULTS.json',dict(device=device,weight_sha256=digest,summary=summary,
       radius_m=RADIUS,confidence_threshold=.20,iou_threshold=THRESHOLD,
       stages=['decoded_head','confidence','geometry_pre_nms','post_nms','final_range'],
       candidate_columns=['anchor_index','decoded_hwl_yaw','logit','score','iou','last_stage']))
    for hook in hooks:
        hook.remove()


if __name__=='__main__':
    from common import lock
    with lock(OUT/'.lock'):
        main()
