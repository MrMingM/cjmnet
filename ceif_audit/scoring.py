"""Evaluation-only GT access. Never imported by proposal/evidence generation."""
import copy
import numpy as np


def empty_stats():
    return {t: dict(tp=[], fp=[], gt=0, score=[]) for t in (.3, .5, .7)}


def merge(total, frame):
    for threshold in total:
        total[threshold]['gt'] += frame[threshold]['gt']
        for key in ('tp', 'fp', 'score'):
            total[threshold][key].extend(frame[threshold][key])


def greedy(iou_matrix, scores, threshold=.7):
    """Same np.argsort order and greedy maximum-IoU removal as eval_utils."""
    remaining = list(range(iou_matrix.shape[1]))
    assigned = np.full(len(scores), -1, dtype=int)
    for i in np.argsort(-scores):
        if remaining:
            row = iou_matrix[i, remaining]
            best = int(np.argmax(row))
            if row[best] >= threshold:
                assigned[i] = remaining.pop(best)
    return assigned


def assignments(boxes, scores, gt):
    from opencood.utils import common_utils as cu
    polygons = list(cu.convert_format(boxes))
    targets = list(cu.convert_format(gt))
    matrix = np.asarray([cu.compute_iou(p, targets) for p in polygons]).reshape(len(boxes), len(gt))
    return greedy(matrix, scores)


def choose_hindsight(base_match, base_fp, candidates, eligible_only):
    """Finite pool hindsight, NOT an AP oracle or a global CEIF upper bound.

    Require retaining every baseline GT match and no extra FP at IoU .7.
    Rank by recovered GT, then fewer FP. Score/AP effects remain measured.
    """
    best, key = None, (0, 0)
    for index, row in enumerate(candidates):
        if eligible_only and not row['eligible']:
            continue
        matched = set(row['matched'])
        fp = row['fp']
        if not base_match.issubset(matched) or fp > base_fp:
            continue
        gain = (len(matched-base_match), base_fp-fp)
        if gain > key:
            key, best = gain, index
    return best


def ap_values(stats, evaluation):
    if not stats[.7]['gt']:
        return dict(ap30=None, ap50=None, ap70=None)
    return {f'ap{int(t*100)}': float(evaluation.calculate_ap(copy.deepcopy(stats), t, False)[0])
            for t in (.3, .5, .7)}
