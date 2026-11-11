"""Fixed-Clean object attribution and inference-only interventions."""
from collections import defaultdict
from contextlib import contextmanager
import json

import numpy as np
import torch

from common import atomic_json, read_json
from inputs import DATA
from support import patched

LR0 = DATA / 'logs/v2xr-lr0-v3'
WEATHERS = ('clean', 'fog', 'snow')


def verify_gt(out):
    native = read_json(LR0 / 'final_evaluation/gt_manifest.json')['frames']
    current = read_json(out / 'gt_manifest.json')['frames']
    fields = ('condition', 'index', 'frame_key', 'fixed_ids', 'fixed_gt_identity')
    if (len(native) != 3*829 or len(current) != len(native)
            or any(any(a[k] != b[k] for k in fields) for a, b in zip(native, current))
            or any(sum(r['fixed_count'] for r in current if r['condition'] == w) != 6235 for w in WEATHERS)):
        raise ValueError('Formal F ground truth differs from the frozen LR0 6235-target reference')


def input_context(batch, reference):
    from opencood.pcdet_utils.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu

    ego = batch['ego']
    boxes = np.asarray([r['box'] for r in reference], dtype=np.float32).reshape(-1, 7)
    boxes = boxes[:, [0, 1, 2, 5, 4, 3, 6]]
    centers = np.linalg.norm(boxes[:, :2], axis=-1)
    sums = {}
    cars = {}
    for sensor in ('lidar', 'radar'):
        streams = ego[f'projected_{sensor}'][0]
        populated = set(ego[f'processed_{sensor}']['voxel_coords'][:, 0].detach().cpu().tolist())
        sums[sensor] = np.zeros(len(boxes), dtype=np.int64)
        cars[sensor] = []
        for i, points in enumerate(streams):
            xyz = points[:, :3].detach().cpu().numpy() if torch.is_tensor(points) else points[:, :3]
            if len(boxes) and len(xyz):
                inside = points_in_boxes_cpu(xyz.astype(np.float32, copy=False), boxes)
                sums[sensor] += np.asarray(inside).sum(1)
            cars[sensor].append('occupied' if i in populated else 'projected_empty' if not len(xyz) else 'voxel_empty')
    rows = {str(r['object_id']): dict(lidar=int(l), radar=int(a), distance=float(d))
            for r, l, a, d in zip(reference, sums['lidar'], sums['radar'], centers)}
    return dict(objects=rows, car_observations=cars)


@contextmanager
def observe_eval(model, destination):
    """Sidecar-only instrumentation. Native dataset, postprocess and replay are unchanged."""
    import metrics
    from opencood.tools import inference_utils

    build, infer = metrics.build, inference_utils.inference_intermediate_fusion
    cursor = dict(index=0, weather=None)

    def track_build(*args, **kwargs):
        cursor.update(index=0, weather=kwargs['condition'])
        return build(*args, **kwargs)

    def observed(batch, net, dataset):
        w, i = cursor['weather'], cursor['index']
        cursor['index'] += 1
        clean = read_json(LR0 / 'final_evaluation/predictions/clean' / f'{i:06d}.json')
        context = input_context(batch, clean['official_gt'])
        result = infer(batch, net, dataset)
        path = destination / 'observations' / w / f'{i:06d}.json'
        atomic_json(path, dict(frame_key=clean['frame_key'], **context, features=model.observation))
        return result

    model.observe = True
    try:
        with patched(metrics, 'build', track_build), patched(inference_utils, 'inference_intermediate_fusion', observed):
            yield
    finally:
        model.observe = False


def bin_points(n):
    return '0' if n == 0 else '1–5' if n <= 5 else '6–20' if n <= 20 else '21+'


def aggregate(out):
    """Use original F matched target identities and IoU, with explicit unassigned false positives."""
    grouped = defaultdict(lambda: defaultdict(lambda: dict(tp=0, fn=0, iou_sum=0.)))
    context_counts = defaultdict(lambda: defaultdict(int))
    features = defaultdict(lambda: defaultdict(float))
    target_rows = {}
    with (out / 'per_frame_metrics.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            w, i = row['condition'], row['index']
            obs = read_json(out / 'observations' / w / f'{i:06d}.json')
            clean = read_json(out / 'observations/clean' / f'{i:06d}.json')
            if obs['frame_key'] != row['frame_key']:
                raise ValueError('Object diagnostic frame mismatch')
            pairs = row['protocols']['F']['matched_pairs'].get('0.7', [])
            matched = {str(pair['object_id']): pair['iou'] for pair in pairs}
            context_counts[w]['unassigned_fp70'] += row['protocols']['F']['thresholds']['0.7']['fp']
            for sensor, observations in obs['car_observations'].items():
                for source in observations:
                    context_counts[w][f'{sensor}_{source}'] += 1
            for key, value in obs['features'].items():
                features[w][key] += value
            features[w]['frames'] += 1
            for oid, obj in obs['objects'].items():
                lidar, radar, distance = obj['lidar'], obj['radar'], obj['distance']
                clean_lidar = clean['objects'][oid]['lidar']
                damage = 'unknown' if clean_lidar == 0 else 'damaged' if lidar <= .7*clean_lidar else 'other'
                bins = (f'points:{bin_points(lidar)}', f'radar:{bool(radar)}',
                        f'distance:{"0–30" if distance < 30 else "30–60" if distance < 60 else "60+"}',
                        f'{damage}|radar:{bool(radar)}')
                target_rows[f'{w}/{i}/{oid}'] = dict(iou=matched.get(oid), bins=bins)
                for group in bins:
                    stats = grouped[w][group]
                    stats['tp'] += oid in matched
                    stats['fn'] += oid not in matched
                    stats['iou_sum'] += matched.get(oid, 0.)
    for w, s in features.items():
        n = s.get('radar_cells', 0)
        if n:
            mean = s['gate_sum']/n
            s['gate_mean'] = mean
            s['gate_std'] = max(0., s['gate_sq_sum']/n - mean**2)**.5
            s['residual_mean'] = s['residual_sum']/n
            for corrected, count in (('corrected_missing','missing_lidar'), ('corrected_observed','observed'),
                                     ('corrected_low_reliability','low_reliability')):
                s[corrected+'_fraction'] = s[corrected]/s[count] if s.get(count) else None
    return dict(grouped=grouped, fixed_target_rows=target_rows, observations=context_counts, features=features)


def target_changes(old, new):
    a, b = old['fixed_target_rows'], new['fixed_target_rows']
    if a.keys() != b.keys():
        raise ValueError('Unmatched diagnostic target identities')
    report = defaultdict(lambda: defaultdict(lambda: dict(recovered=0, lost=0, common=0, iou_delta_sum=0.)))
    for key, row in a.items():
        other = b[key]
        if row['bins'] != other['bins']:
            raise ValueError('Target diagnostic strata differ across predictions')
        x, y = row['iou'], other['iou']
        for group in row['bins']:
            cell = report[key.split('/')[0]][group]
            cell['recovered'] += x is None and y is not None
            cell['lost'] += x is not None and y is None
            if x is not None and y is not None:
                cell['common'] += 1
                cell['iou_delta_sum'] += y-x
    return report

