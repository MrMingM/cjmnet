"""Inference-only feature extractor: accepts predicted tensors and ROI masks.

No batch, dataset, GT association or outcome object crosses this boundary.
Competition interactions are explicit products: a candidate-only constant
cancels from a linear pairwise score, so it alone cannot help source ranking.
"""
import re
import numpy as np
from .common import KEEP

SHARED = (
    'shared_score', 'shared_rank', 'shared_x', 'shared_y', 'shared_z',
    'shared_length', 'shared_width', 'shared_height', 'shared_yaw_sin',
    'shared_yaw_cos', 'source_count', 'shared_local_max', 'shared_local_mean',
    'shared_local_std', 'shared_top1_top2_margin',
)
IDENTITY = ('is_keep', 'is_single', 'is_query', 'is_ego')
CLASSIFICATION = (
    'anchor_probability', 'roi_max', 'roi_mean', 'roi_std', 'roi_topk_mean',
    'anchor_delta_shared', 'roi_delta_shared', 'source_relative_rank',
    'source_delta_mean', 'source_disagreement', 'single_query_delta',
    'absolute_source_delta_mean',
)
REGRESSION = (
    'delta_x', 'delta_y', 'delta_z', 'delta_length', 'delta_width', 'delta_height',
    'delta_yaw_sin', 'delta_yaw_cos', 'source_shared_pred_iou', 'center_distance',
    'center_to_mean', 'center_to_median', 'size_to_mean', 'size_to_median',
    'yaw_to_consensus', 'center_variance', 'size_variance', 'yaw_variance',
)
COMPETITION = (
    'competition_nearby_count', 'competition_max_pred_iou',
    'competition_iou_over_0p1_count', 'competition_iou_over_0p3_count',
    'competition_iou_over_0p5_count', 'competition_highest_score',
    'competition_score_margin', 'competition_higher_score_overlap_count',
    'competition_local_score_density',
    'competition_source_max_pred_iou', 'competition_source_higher_overlap_count',
    'competition_roi_delta_x_higher_count', 'competition_anchor_delta_x_max_iou',
    'competition_geometry_distance_x_overlap_count',
)


def feature_leakage_check(names):
    allowed = set(SHARED + IDENTITY + CLASSIFICATION + REGRESSION + COMPETITION)
    forbidden = ('gt', 'target', 'focal', 'matched', 'unmatched', 'oracle',
                 'recovered', 'lost', 'new_fp', 'was_shared_missed', 'post_nms')
    for name in names:
        tokens = re.split('[^a-z0-9]+', name.lower())
        if name not in allowed or any(x in tokens for x in forbidden):
            raise ValueError('Feature leakage/unknown feature field: ' + name)
    if len(set(names)) != len(names):
        raise ValueError('Duplicate feature names')
    return True


def schema(task):
    names = SHARED + IDENTITY + (CLASSIFICATION if task == 'cls' else REGRESSION) + COMPETITION
    if task not in ('cls', 'reg'):
        raise ValueError('Invalid task')
    feature_leakage_check(names)
    return list(names)


def columns(task, variant):
    names = schema(task)
    return [i for i, name in enumerate(names)
            if variant == 'with_competition_features' or not name.startswith('competition_')]


def source_names(pool):
    def key(name):
        if name == KEEP:
            return (0, 0)
        family, index = name.split(':')
        if family not in ('single', 'query') or int(index) < 0:
            raise ValueError('Unknown source: ' + name)
        return (1 if family == 'single' else 2, int(index))
    return sorted(pool, key=key)


def geometry(corners):
    value = np.asarray(corners, dtype=np.float64)
    center = value.mean(axis=-2)
    length_vector = value[..., 0, :] - value[..., 3, :]
    width_vector = value[..., 1, :] - value[..., 0, :]
    height_vector = value[..., 4, :] - value[..., 0, :]
    sizes = np.stack([np.linalg.norm(v, axis=-1) for v in
                      (length_vector, width_vector, height_vector)], -1)
    yaw = np.arctan2(length_vector[..., 1], length_vector[..., 0])
    return center, sizes, yaw


def pred_iou(left, right):
    from opencood.utils import common_utils as cu
    if not len(left) or not len(right):
        return np.zeros((len(left), len(right)), dtype=np.float64)
    polygons = list(cu.convert_format(np.asarray(right, dtype=np.float32)))
    result = np.asarray([cu.compute_iou(p, polygons) for p in
                         cu.convert_format(np.asarray(left, dtype=np.float32))], dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite prediction-prediction IoU')
    return result


def extract_features(proxies, ids, scores, corners, masks, spec):
    names = source_names(proxies)
    shared = proxies[KEEP]
    n, s = len(ids), len(names)
    result = {task: np.empty((n, s, len(schema(task))), np.float32) for task in ('cls', 'reg')}
    if not n:
        return names, result
    centers, sizes, yaws = geometry(corners)
    pair_iou = pred_iou(corners, corners)
    src_corners = np.stack([proxies[name]['corners'][ids] for name in names], axis=1)
    if not np.isfinite(src_corners).all():
        raise ValueError('Nonfinite source geometry')
    src_centers, src_sizes, src_yaws = geometry(src_corners)
    # Mean/median summaries are symmetric under source enumeration permutations.
    evidence_indices = [i for i, name in enumerate(names) if name != KEEP]
    if not evidence_indices:
        raise ValueError('Source pool must include single/query predictions')
    ev_center = src_centers[:, evidence_indices]
    ev_size = src_sizes[:, evidence_indices]
    ev_yaw = src_yaws[:, evidence_indices]
    center_mean, center_median = ev_center.mean(1), np.median(ev_center, axis=1)
    size_mean, size_median = ev_size.mean(1), np.median(ev_size, axis=1)
    sin_mean, cos_mean = np.sin(ev_yaw).mean(1), np.cos(ev_yaw).mean(1)
    yaw_consensus = np.arctan2(sin_mean, cos_mean)
    center_variance = ev_center.var(1).mean(1)
    size_variance = ev_size.var(1).mean(1)
    yaw_variance = 1 - np.hypot(sin_mean, cos_mean)
    anchors = shared['anchor_num']
    for p, anchor_id in enumerate(ids):
        roi_ids = np.flatnonzero(np.repeat(np.asarray(masks[p], bool)[..., None], anchors, axis=2).reshape(-1))
        if not len(roi_ids):
            raise ValueError('Empty Shared proposal ROI')
        local = [np.asarray(proxies[name]['scores'][roi_ids], dtype=np.float64) for name in names]
        maxima = np.array([v.max() for v in local])
        anchor_probs = np.array([proxies[name]['scores'][anchor_id] for name in names])
        if not np.isfinite(maxima).all() or not np.isfinite(anchor_probs).all():
            raise ValueError('Nonfinite classification output')
        top = np.sort(local[0])[-2:]
        margin = float(top[-1] - top[-2]) if len(top) > 1 else 0.
        base = [float(scores[p]), (p + 1) / spec['shared_top_k'],
                *centers[p], *sizes[p], np.sin(yaws[p]), np.cos(yaws[p]),
                s, maxima[0], local[0].mean(), local[0].std(), margin]
        others = np.arange(n) != p
        distances = np.linalg.norm(centers[:, :2] - centers[p, :2], axis=1)
        near = others & (distances <= spec['nearby_distance_m'])
        ious = pair_iou[p]
        overlap = others & (ious > .1)
        high = overlap & (scores > scores[p])
        highest = float(np.max(scores[overlap])) if overlap.any() else 0.
        competition = [near.sum(), float(np.max(ious[others])) if others.any() else 0.,
                       *[int((others & (ious > t)).sum()) for t in (.1, .3, .5)],
                       highest, float(scores[p]) - highest, high.sum(), float(scores[near].sum())]
        source_iou = pred_iou(src_corners[p], corners)
        source_shared_iou = pred_iou(src_corners[p], corners[p:p+1])[:, 0]
        for i, name in enumerate(names):
            identity = [name == KEEP, name.startswith('single:'), name.startswith('query:'),
                        name != KEEP and name.split(':')[1] == '0']
            counterpart = ('query:' if name.startswith('single:') else 'single:') + name.split(':')[-1]
            sq_delta = 0.
            if name != KEEP and counterpart in names:
                single = i if name.startswith('single:') else names.index(counterpart)
                query = names.index(counterpart) if name.startswith('single:') else i
                sq_delta = maxima[single] - maxima[query]
            cls = [anchor_probs[i], maxima[i], local[i].mean(), local[i].std(),
                   np.sort(local[i])[-spec['local_top_k']:].mean(),
                   anchor_probs[i] - anchor_probs[0], maxima[i] - maxima[0],
                   float((maxima[evidence_indices] > maxima[i]).sum()) / len(evidence_indices),
                   maxima[i] - maxima[evidence_indices].mean(),
                   maxima[evidence_indices].std(), sq_delta,
                   abs(maxima[i] - maxima[evidence_indices].mean())]
            delta_yaw = src_yaws[p, i] - yaws[p]
            distance = np.linalg.norm(src_centers[p, i] - centers[p])
            reg = [*(src_centers[p, i] - centers[p]), *(src_sizes[p, i] - sizes[p]),
                   np.sin(delta_yaw), np.cos(delta_yaw), source_shared_iou[i], distance,
                   np.linalg.norm(src_centers[p, i] - center_mean[p]),
                   np.linalg.norm(src_centers[p, i] - center_median[p]),
                   np.linalg.norm(src_sizes[p, i] - size_mean[p]),
                   np.linalg.norm(src_sizes[p, i] - size_median[p]),
                   1 - np.cos(src_yaws[p, i] - yaw_consensus[p]),
                   center_variance[p], size_variance[p], yaw_variance[p]]
            extra = [float(np.max(source_iou[i, others])) if others.any() else 0.,
                     int((others & (source_iou[i] > .1) & (scores > anchor_probs[i])).sum()),
                     (maxima[i] - maxima[0]) * high.sum(),
                     (anchor_probs[i] - anchor_probs[0]) * competition[1],
                     distance * overlap.sum()]
            for task, evidence in (('cls', cls), ('reg', reg)):
                result[task][p, i] = base + identity + evidence + competition + extra
    for value in result.values():
        if not np.isfinite(value).all():
            raise ValueError('Feature matrix contains nonfinite values')
    return names, result
