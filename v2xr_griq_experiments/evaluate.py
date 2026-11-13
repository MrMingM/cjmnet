"""Replay scores through the native decoder's postprocessing and official evaluator."""
import numpy as np
import torch

from v2xr_griq_experiments.native import replay


def frame_result(d, pp, logits, threshold, all_protocols=True, trace=True):
    from evaluate import official_match_trace, statistics
    from opencood.utils.eval_utils_opv2v import caluclate_tp_fp

    pred, scores = replay(pp, d['decoded'], logits, d['transform'], threshold)
    row = d['meta']
    fixed = row['fgt']
    official = row['ogt']
    result = dict(scene=row['key'][0], key=row['key'], observations=row['observations'],
                  none=pred is None, empty=pred is not None and not len(pred), hits={})
    from support import corner_array
    # Exact source-anchor identity for final selected predictions, including ties.
    identity = {}
    if pred is not None and trace:
        for i, b in enumerate(d['corners']):
            identity.setdefault(b.tobytes(), []).append(i)
    for protocol, g in [('F',fixed), ('O',official), ('O*',official)][:3 if all_protocols else 1]:
        stat = statistics()
        gt = torch.from_numpy(corner_array(g))
        if protocol != 'O' or pred is not None:
            for t in stat:
                if not all_protocols and t != .7:
                    continue
                if protocol == 'F' and t == .7 and trace:
                    pairs,_ = official_match_trace(pred, scores, gt, [str(o['object_id']) for o in g], stat, t)
                    for pair in pairs:
                        i = pair['prediction_index']
                        geometry = pred[i].numpy()
                        ids = identity.get(geometry.tobytes(), [])
                        if not ids:
                            ids = np.flatnonzero(np.max(np.abs(d['corners']-geometry), axis=(1,2)) <= 1e-4)
                        if not len(ids):
                            raise ValueError('Official output is not in the common candidate pool')
                        j = min(ids, key=lambda j: abs(float(logits[j])-float(torch.logit(scores[i].clamp(1e-7,1-1e-7)))))
                        result['hits'][pair['object_id']] = dict(iou=pair['iou'],
                            low=bool(d['prob'][j] <= d['native_threshold']), anchor=int(d['ids'][j]))
                else:
                    caluclate_tp_fp(pred, scores, gt, stat, t)
        result[protocol] = stat
    return result
