"""Oracle candidate coverage and actual official union recall."""
from collections import Counter, defaultdict
import json

import numpy as np
import torch
from shared import cache
from shared.input_audit import OUT, WEATHERS, atomic_json
from shared.metrics import overlap
from shared.statistics import ratio_interval
from evaluate import official_match_trace, statistics
from support import corner_array
from geometry_warnings import checked_intersections

THRESHOLDS=(.3,.5,.7)
SETS=('final','pre_nms','top20','top50','top100','top200')


def car_candidates(arrays, weather, car):
    prefix=f'{weather}_{car}'
    corners,scores,stage,top=(arrays[f'{prefix}_{key}'] for key in ('corners','score','stage','top'))
    indices={'final':np.flatnonzero(stage==4),'pre_nms':np.flatnonzero(stage>=2)}
    indices.update({f'top{k}':top[:k] for k in (20,50,100,200)})
    return corners,scores,indices


def analyze(smoke=False):
    root=OUT/('V1_smoke' if smoke else 'V1')
    root.mkdir(exist_ok=True)
    blocks=defaultdict(lambda:defaultdict(lambda:[0,0]))
    counts=defaultdict(Counter)
    with (root/'TARGET_DETAILS.tmp').open('w',encoding='utf-8') as stream, checked_intersections():
        for meta,arr in cache.frames(smoke):
            scene=meta['frame'][0]
            gt=meta['targets']
            gt_corners=corner_array(gt)
            ids=[target['object_id'] for target in gt]
            for weather in WEATHERS:
                frame=meta['weather'][weather]
                cars=[car_candidates(arr,weather,k) for k in range(len(frame['cars']))]
                counts[weather][f'cars_{len(cars)}']+=1
                for _,_,indices in cars:
                    for mode,ix in indices.items():
                        counts[weather]['candidates_'+mode]+=len(ix)
                union=np.concatenate([boxes[ix['final']] for boxes,_,ix in cars])
                score=np.concatenate([scores[ix['final']] for _,scores,ix in cars])
                for t in THRESHOLDS:
                    pairs,_=official_match_trace(torch.from_numpy(union),torch.from_numpy(score),
                                               torch.from_numpy(gt_corners),ids,statistics(),t)
                    b=blocks[weather,'official_final_union',t][scene]
                    b[0]+=len(pairs)
                    b[1]+=len(gt)
                    counts[weather][f'union_unmatched_{t}']+=len(union)-len(pairs)
                p0=set(frame['multi_reference_hits'])
                for target,corner in zip(gt,gt_corners,strict=True):
                    oid=str(target['object_id'])
                    support=frame['support'][oid]
                    points=sum(support[k] for k in ('ego_lidar','ego_radar','neighbor_lidar','neighbor_radar'))
                    groups=['all',support['source'],support['distance'],
                            'points_'+('0' if points==0 else '1-5' if points<=5 else '6-20' if points<=20 else '21+')]
                    if oid in p0: groups.append('P0_success')
                    values={mode:[] for mode in SETS}
                    details=[]
                    for box,score,ix in cars:
                        selected,ious=overlap(box,corner)
                        all_iou=np.zeros(len(box),dtype=np.float32)
                        all_iou[selected]=ious
                        row={}
                        for mode,picks in ix.items():
                            hits=int((all_iou[picks]>=.7).sum())
                            if len(picks):
                                best=picks[np.argmax(all_iou[picks])]
                                value=float(all_iou[best])
                                row[mode]=[value,float(score[best]),hits]
                            else:
                                value=0.
                                row[mode]=[0.,None,0]
                            values[mode].append(value)
                        details.append(row)
                    for mode,ious in values.items():
                        for t in THRESHOLDS:
                            for group in groups:
                                b=blocks[weather,group+'/'+mode,t][scene]
                                b[0]+=max(ious)>=t
                                b[1]+=1
                        ego=ious[0]>=.7
                        neighbor=any(x>=.7 for x in ious[1:])
                        label='both' if ego and neighbor else 'ego_only' if ego else 'neighbor_only' if neighbor else 'neither'
                        counts[weather][mode+'_'+label]+=1
                        counts[weather][mode+'_target_duplicate_edges']+=max(0,sum(x[mode][2] for x in details)-1)
                    stream.write(json.dumps(dict(frame=meta['frame'],weather=weather,object_id=oid,p0=oid in p0,
                                                  groups=groups,car_details=details))+'\n')
            if (meta['index']+1)%100==0: print('V1',meta['index']+1,flush=True)
    (root/'TARGET_DETAILS.tmp').replace(root/'TARGET_DETAILS.jsonl')
    metrics={'/'.join(map(str,k)):ratio_interval(v) for k,v in blocks.items()}
    result=dict(metrics=metrics, counts=counts)
    atomic_json(root/'CANDIDATE_COVERAGE.json',result)
    lines=['# V1 单车候选覆盖','', '覆盖允许同一候选命中多个目标。最终框并集的一对一召回复用官方匹配。','',
           '| 天气 | 对象 | 候选 | IoU | 覆盖/召回 | 95% 区间 | 目标数 | 场景数 |',
           '|---|---|---|---:|---:|---|---:|---:|']
    for key,r in sorted(metrics.items()):
        if r['value'] is not None:
            lines.append('| '+' | '.join([*key.split('/'),f"{r['value']:.2%}",str(r['interval95']),str(r['n']),str(r['scenes'])])+' |')
    (root/'CANDIDATE_COVERAGE.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result
