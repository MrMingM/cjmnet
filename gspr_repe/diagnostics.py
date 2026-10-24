"""Stratify existing official matching traces; never implement a new matcher/AP."""
import json
import math
from collections import defaultdict
from auditlib import atomic_json, read_json


def summarize(root, variants):
    run=read_json(root/'run_identity.json')
    evidence_path=run['roots']['gt_audit_root']+'/per_frame.jsonl'
    evidence=[json.loads(line) for line in open(evidence_path,encoding='utf-8')]
    bins=defaultdict(lambda:dict(gt=0,tp=0))
    matches={}; frame_counts={}
    for variant in variants:
        folder=root/variant/'final_evaluation'
        for line in (folder/'per_frame_metrics.jsonl').read_text(encoding='utf-8').splitlines():
            row=json.loads(line); weather=row['condition']; index=row['index']
            frame=evidence[index]
            if frame['frame_key']!=row['frame_key']:
                raise ValueError('Stratification frame mismatch')
            matched={x['object_id'] for x in row['protocols']['F'].get('matched_pairs',{}).get('0.7',[])}
            matches[(variant,weather,index)]=matched
            frame_counts[(variant,weather,index)]=row['protocols']['F']['thresholds']['0.7']
            current=frame['conditions'][weather]
            for gt in frame['conditions']['clean']['final_gt']:
                oid=gt['object_id']; corners=gt['corners']
                center=[sum(c[k] for c in corners)/len(corners) for k in (0,1)]
                distance=math.hypot(*center)
                counts=[c['official_points_in_box'][c['before_point_filter_ids'].index(oid)]
                        for c in current['cavs'] if oid in c['before_point_filter_ids']]
                count=sum(counts)
                distance_bin='0-30' if distance<30 else '30-60' if distance<60 else '60+'
                point_bin='0' if count==0 else '1-5' if count<=5 else '6-20' if count<=20 else '21+'
                for group,label in (('distance',distance_bin),('point_count_sum_across_cavs',point_bin)):
                    b=bins[(variant,weather,group,label)]; b['gt']+=1; b['tp']+=oid in matched
    strata=[dict(model=k[0],weather=k[1],group=k[2],bin=k[3],**v,fn=v['gt']-v['tp'],recall=v['tp']/v['gt']) for k,v in bins.items()]
    changes=[]
    for (variant,weather,index),matched in matches.items():
        if variant=='B0':
            continue
        baseline=matches[('B0',weather,index)]
        a,b=frame_counts[(variant,weather,index)],frame_counts[('B0',weather,index)]
        changes.append(dict(model=variant,weather=weather,index=index,recovered=sorted(matched-baseline),
            lost=sorted(baseline-matched),tp_delta=a['tp']-b['tp'],fp_delta=a['fp']-b['fp'],fn_delta=a['fn']-b['fn']))
    data=dict(status='complete',strata=strata,target_changes=changes,
        definitions='Fixed Clean GT, official F matching at IoU .7. Point counts sum native per-CAV counts, may observe same surface multiple times; FP stays at frame level.')
    atomic_json(root/'reports/STRATIFIED_DETECTION.json',data)
    return data
