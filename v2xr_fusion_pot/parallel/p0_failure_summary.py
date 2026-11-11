"""Summarize candidate-stage and local-context effects from committed P0 frames."""
from collections import Counter, defaultdict
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run
from common import atomic_json, read_json
import numpy as np
import shapely
import torch
from opencood.utils import box_utils

ROOT=run.OUT/'p0-frozen-failure-v1'


def nms_witness(row):
    """Find a surviving higher-score candidate that overlaps a removed good box."""
    data=row['candidates']
    boxes=np.asarray(data['decoded_hwl_yaw'],dtype=np.float32).reshape(-1,7)
    score=np.asarray(data['score'])
    iou=np.asarray(data['iou'])
    stage=np.asarray(data['last_stage'])
    dropped=np.flatnonzero((stage==2)&(iou>=.7))
    kept=np.flatnonzero(stage>=3)
    if not len(dropped) or not len(kept):
        return None
    corners=box_utils.boxes_to_corners_3d(torch.from_numpy(boxes),order='hwl').numpy()
    poly=shapely.polygons(corners[:,:4,:2])
    for i in dropped[np.argsort(-score[dropped])]:
        larger=kept[score[kept]>score[i]]
        if not len(larger):
            continue
        with np.errstate(invalid='ignore'):
            overlaps=shapely.area(shapely.intersection(poly[i],poly[larger]))/shapely.area(shapely.union(poly[i],poly[larger]))
        over=larger[overlaps>.15]
        if len(over):
            j=over[np.argmax(score[over])]
            return dict(removed_anchor=data['anchor_index'][i],surviving_anchor=data['anchor_index'][j],
                removed_score=float(score[i]),surviving_score=float(score[j]),
                removed_target_iou=float(iou[i]),surviving_target_iou=float(iou[j]),
                mutual_iou=float(overlaps[larger==j][0]))
    return None


def quantiles(values):
    return dict(n=len(values),median=float(np.median(values)),q25=float(np.quantile(values,.25)),
                q75=float(np.quantile(values,.75))) if values else dict(n=0)


def main():
    results=read_json(ROOT/'RESULTS.json')
    cohort=read_json(ROOT/'COHORT.json')
    summary={}
    examples={}
    for weather,frames in cohort.items():
        affected=Counter()
        transitions=defaultdict(Counter)
        metrics=defaultdict(list)
        counts=Counter()
        splice=Counter()
        for index,ids in frames.items():
            file=ROOT/weather/f'{int(index):06d}.json'
            payload=read_json(file)
            if set(payload['targets'])!=set(ids) or payload['weight_sha256']!=results['weight_sha256']:
                raise ValueError('Candidate artifact identity differs')
            counts.update(payload['counts'])
            splice.update(payload['splice'])
            for oid,row in payload['targets'].items():
                base,changed=(row[mode] for mode in ('multi','clean_surroundings'))
                mode=base['first_missing_stage']
                recovered=row['clean_surroundings_detected']
                transitions[mode]['targets']+=1
                transitions[mode]['context_recovered']+=recovered
                label='recovered' if recovered else 'still_missed'
                if recovered or abs(changed['max_iou_by_stage'][0]-base['max_iou_by_stage'][0])>1e-4:
                    affected['head_iou_changed']+=1
                if mode=='nms':
                    witness=nms_witness(base)
                    if witness:
                        affected['nms_higher_score_witness']+=1
                        examples.setdefault(weather,dict(index=int(index),frame=payload['frame'],object_id=oid,**witness))
                        for key in ('removed_score','surviving_score','removed_target_iou',
                                    'surviving_target_iou','mutual_iou'):
                            metrics[f'nms_conflict/{key}'].append(witness[key])
                if mode=='classification_threshold':
                    for branch in ('single','multi','clean_surroundings'):
                        candidates=row[branch]['candidates']
                        good=np.asarray(candidates['iou'])>=.7
                        if good.any():
                            metrics[f'classification/{branch}_best_good_box_score'].append(
                                float(np.asarray(candidates['score'])[good].max()))
                for scale,features in row['feature_change'].items():
                    for name,norms,delta in (
                            ('ego',features['base_norm'][:1],features['delta_norm'][:1]),
                            ('neighbor',features['base_norm'][1:],features['delta_norm'][1:])):
                        base_norm=float(np.linalg.norm(norms))
                        if base_norm>1e-8:
                            value=float(np.linalg.norm(delta)/base_norm)
                            for key in ('all',label):
                                metrics[f'{key}/{name}/stride{scale}'].append(value)
                for stage,label_stage in enumerate(('head','confidence','pre_nms','post_nms','final')):
                    metrics[f'{label}/{label_stage}_iou_delta'].append(
                        changed['max_iou_by_stage'][stage]-base['max_iou_by_stage'][stage])
        if dict(counts)!=results['summary'][weather]:
            raise ValueError('Mismatch in aggregated stage counts')
        summary[weather]=dict(frame_count=len(frames),counts=counts,spliced_voxels=splice,by_original_stage=transitions,
                              affected=affected,feature_and_iou_quantiles={k:quantiles(v) for k,v in metrics.items()})
    atomic_json(ROOT/'DETAIL_SUMMARY.json',dict(summary=summary,nms_examples=examples,
        feature_metric='L2 of feature change divided by original L2; zero originals excluded; target-covering cells'))
    for w,row in summary.items():
        print(w,row['counts'],row['affected'],row['by_original_stage'],flush=True)


if __name__=='__main__':
    main()
