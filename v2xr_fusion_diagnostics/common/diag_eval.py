"""Use the unchanged LR0 postprocessor and official AP matching."""
from collections import Counter
import copy
import numpy as np
import torch
from evaluate import official_match_trace,statistics
from metrics import valid_nms_candidates
from opencood.utils import eval_utils_opv2v as official
from support import THRESHOLDS,corner_array
from geometry_warnings import checked_intersections


def predictions(native,ego,modes):
    results, aliases, seen = {},{},{}
    rejected=Counter()
    with checked_intersections() as warning_counts,valid_nms_candidates(rejected):
        for mode,head in modes.items():
            previous=next((name for name,value in seen.items() if
                           torch.equal(value['psm'],head['psm']) and torch.equal(value['rm'],head['rm'])),None)
            if previous is not None:
                aliases[mode]=previous
                continue
            seen[mode]=head
            boxes,scores=native.post_processor.post_process({'ego':ego},{'ego':head})
            if boxes is not None and not bool(torch.isfinite(boxes).all() & torch.isfinite(scores).all()):
                raise FloatingPointError('Nonfinite official prediction')
            results[mode]=None if boxes is None else [boxes.cpu().tolist(),scores.cpu().tolist()]
    return results,aliases,dict(geometry_rejected=dict(rejected),warnings=dict(warning_counts))


def score(prediction, weather_gt, fixed_gt, clean):
    stat_o, stat_f=statistics(),statistics()
    boxes,scores=(None,None) if prediction is None else (
        torch.tensor(prediction[0],dtype=torch.float32).reshape(-1,8,3),torch.tensor(prediction[1],dtype=torch.float32))
    weather=torch.from_numpy(corner_array(weather_gt))
    fixed=torch.from_numpy(corner_array(fixed_gt))
    ids=[r['object_id'] for r in fixed_gt]
    with checked_intersections():
        for t in THRESHOLDS:
            if t==.7:
                pairs,_=official_match_trace(boxes,scores,fixed,ids,stat_f,t)
            else:
                official.caluclate_tp_fp(boxes,scores,fixed,stat_f,t)
            if boxes is not None and not clean:
                official.caluclate_tp_fp(boxes,scores,weather,stat_o,t)
    if clean and boxes is not None:
        stat_o=copy.deepcopy(stat_f)
    stat_ostar={t:{**v,'gt':len(weather_gt)} for t,v in stat_o.items()}
    return {'O':stat_o,'O*':stat_ostar,'F':stat_f},[str(r['object_id']) for r in pairs]


def merge(dst,src):
    for protocol,stats in src.items():
        for t,row in stats.items():
            dest=dst[protocol][t]
            for k in ('tp','fp','score'):
                dest[k].extend(row[k])
            dest['gt']+=row['gt']


def finish(total):
    return {protocol:dict(AP={f'AP{int(t*100)}':float(official.calculate_ap(copy.deepcopy(stat),t,False)[0])
                    for t in THRESHOLDS},
                    gt=stat[.7]['gt'], tp=sum(stat[.7]['tp']), fp=sum(stat[.7]['fp']))
            for protocol,stat in total.items()}
