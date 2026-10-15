"""Matched decoded-output perturbations and valid-CAV random feature mixing."""
import hashlib
import math
import numpy as np
from .common import KEEP, settings

MAGNITUDES = ('abs_score', 'center_shift', 'size_change', 'yaw_change', 'candidate_iou_change')


def rng_for(seed, weather, index, family, extra=''):
    raw = f'{seed}/{weather}/{index}/{family}/{extra}'.encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(raw).digest()[:8], 'little'))


def wrap_yaw(value):
    return (value + np.pi) % (2*np.pi) - np.pi


def delta_vector(scores, boxes, base_scores, base_boxes):
    boxes, base_boxes = np.asarray(boxes), np.asarray(base_boxes)
    delta = boxes-base_boxes
    delta[:, 6] = wrap_yaw(delta[:, 6])
    return np.column_stack((np.asarray(scores)-np.asarray(base_scores), delta))


def choose_pool(pools, weather, family):
    minimum = settings()['calibration']['minimum_pool_rows']
    family_values = [v for k, v in sorted(pools.items()) if k.endswith('/'+family)]
    if not pools:
        raise ValueError('Empty train delta pool')
    choices = [(f'{weather}/{family}', pools.get(f'{weather}/{family}')),
               (family, np.concatenate(family_values, axis=0) if family_values else None)]
    choices.append(('global', np.concatenate([v for k,v in sorted(pools.items())], axis=0)))
    for label, values in choices:
        if values is not None and len(values) >= minimum:
            return values, label
    raise ValueError('Insufficient train output delta pool even after frozen fallback hierarchy')


def sample_vectors(pools, weather, family, count, rng):
    values, fallback = choose_pool(pools, weather, family)
    return values[rng.integers(len(values), size=count)], fallback


def dirichlet_weights(source_count, candidate_count, alpha, rng):
    if source_count < 1 or candidate_count != 2 * source_count:
        raise ValueError('N2 source/action count mismatch')
    return rng.dirichlet(np.full(source_count, alpha), size=candidate_count)


def mix_predictions(b, base, levels, shared, alpha, rng):
    """Levels are the SAME aligned received tensors used by build_candidate_pool."""
    count = len(levels[0])
    weights = dirichlet_weights(count, 2*count, alpha, rng)
    pool = {KEEP: shared}
    for index, row in enumerate(weights):
        fused = []
        for level in levels:
            if len(level) != count:
                raise ValueError('Valid CAV count differs across aligned levels')
            weight = b.torch.as_tensor(row, dtype=level.dtype, device=level.device)[:, None, None, None]
            fused.append((level*weight).sum(0, keepdim=True))
        pool[f'mix:{index:03d}'] = b.predict_from_levels(base, fused)
    return pool


def decode(dataset, batch, prediction):
    import torch
    score = torch.sigmoid(prediction['psm']).permute(0, 2, 3, 1).reshape(-1)
    boxes = dataset.post_processor.delta_to_boxes3d(prediction['rm'], batch['ego']['anchor_box'])[0]
    return score, boxes


def encode(dataset, batch, shared, scores, boxes):
    """Inverse of the ACTUAL voxel postprocessor xyzhwl+yaw decoder."""
    import torch
    anchors = batch['ego']['anchor_box'].reshape(-1, 7).to(boxes).float()
    diagonal = torch.sqrt(anchors[:, 4]**2 + anchors[:, 5]**2)
    values = torch.zeros_like(boxes)
    values[:, :2] = (boxes[:, :2]-anchors[:, :2])/diagonal[:, None]
    values[:, 2] = (boxes[:, 2]-anchors[:, 2])/anchors[:, 3]
    values[:, 3:6] = torch.log(boxes[:, 3:6]/anchors[:, 3:6])
    values[:, 6] = boxes[:, 6]-anchors[:, 6]
    h, w = shared['psm'].shape[-2:]
    count = shared['psm'].shape[1]
    rm = values.reshape(1, h, w, count*7).permute(0, 3, 1, 2).contiguous()
    eps = settings()['calibration']['probability_epsilon']
    psm = torch.logit(scores.clamp(eps, 1-eps)).reshape(1, h, w, count).permute(0, 3, 1, 2).contiguous()
    result = dict(psm=psm, rm=rm)
    # Before generating any null, ensure inverse representation has not drifted.
    decoded = dataset.post_processor.delta_to_boxes3d(rm, batch['ego']['anchor_box'])[0]
    if not torch.allclose(decoded, boxes, rtol=1e-5, atol=1e-4):
        raise ValueError('Decoded-output inverse representation mismatch')
    return result


def empirical_predictions(dataset, batch, shared, source_names, pools, weather, rng, score_only=False):
    import torch
    score, boxes = decode(dataset, batch, shared)
    pool, fallback, clipping = {KEEP: shared}, {}, {}
    for index, name in enumerate(source_names):
        family = name.split(':')[0]
        vector, label = sample_vectors(pools, weather, family, len(score), rng)
        delta = torch.as_tensor(vector, dtype=boxes.dtype, device=boxes.device)
        requested = score + delta[:, 0]
        eps = settings()['calibration']['probability_epsilon']
        new_score = requested.clamp(eps, 1-eps)
        changed = boxes.clone() if score_only else boxes + delta[:, 1:]
        minimum = settings()['calibration']['minimum_size_m']
        clipped = (changed[:, 3:6] < minimum).any(1)
        changed[:, 3:6].clamp_(min=minimum)
        prediction = encode(dataset, batch, shared, new_score, changed)
        if score_only:
            prediction['rm'] = shared['rm']  # exact geometry preservation
        null_name = f'empirical:{index:03d}'
        pool[null_name] = prediction
        fallback[null_name] = label
        clipping[null_name] = dict(score=int((new_score != requested).sum()), size=int(clipped.sum()), anchors=len(score))
    return pool, dict(fallback=fallback, clipping=clipping)


def assert_fair(source, null):
    if KEEP not in source or KEEP not in null or len(source) != len(null):
        raise ValueError('Null action count unfair')
    if null[KEEP] is not source[KEEP]:
        raise ValueError('KEEP must be the identical Shared tensor mapping')


def candidate_ids(b, dataset, batch, shared, limit=256):
    from local_fusion_task_split_pilot.proposal_constrained_oracle import top_shared_proposals
    ids, _, _ = top_shared_proposals(b._proxy_cache(dataset, batch, shared), limit)
    return np.asarray(ids, dtype=int)


def pool_deltas(dataset, batch, shared, pool, ids):
    s, box = decode(dataset, batch, shared)
    base_score = s.detach().cpu().numpy()[ids]
    base_boxes = box.detach().cpu().numpy()[ids]
    result = {}
    for name, prediction in pool.items():
        if name == KEEP:
            continue
        score, boxes = decode(dataset, batch, prediction)
        result[name] = delta_vector(score.detach().cpu().numpy()[ids], boxes.detach().cpu().numpy()[ids],
                                    base_score, base_boxes)
    return result


def pool_magnitudes(b, dataset, batch, shared, pool, ids):
    """IoU change = 1-IoU(candidate box, Shared box), without GT access."""
    base_proxy = b._proxy_cache(dataset, batch, shared)
    deltas = pool_deltas(dataset, batch, shared, pool, ids)
    result = {k: [] for k in MAGNITUDES}
    for name, vector in deltas.items():
        proxy = b._proxy_cache(dataset, batch, pool[name])
        original = list(b.cu.convert_format(base_proxy['corners'][ids]))
        altered = list(b.cu.convert_format(proxy['corners'][ids]))
        ious = [float(b.cu.compute_iou(x, [y])[0]) for x, y in zip(original, altered)]
        result['abs_score'].extend(abs(vector[:, 0]).tolist())
        result['center_shift'].extend(np.linalg.norm(vector[:, 1:4], axis=1).tolist())
        result['size_change'].extend(np.linalg.norm(vector[:, 4:7], axis=1).tolist())
        result['yaw_change'].extend(abs(vector[:, 7]).tolist())
        result['candidate_iou_change'].extend((1-np.asarray(ious)).clip(0, 1).tolist())
    return result


def magnitude_summary(values):
    return {k: dict(n=len(v), median=float(np.median(v)) if v else None,
                    p90=float(np.quantile(v, .9)) if v else None) for k, v in values.items()}


def alpha_distance(source, random, epsilon=1e-8):
    terms = [abs(math.log(max(random[k][q], epsilon)/max(source[k][q], epsilon)))
             for k in MAGNITUDES for q in ('median', 'p90')]
    return float(np.mean(terms))
