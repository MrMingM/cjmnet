"""Inference hooks for the existing sparse Radar compensation path."""
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import torch

from common import atomic_json, read_json
from diagnostics import LR0
from enhancement import point_statistics
from support import patched
from opencood.pcdet_utils.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu


def transform(support, mode):
    if mode == 'D0':
        return support
    if mode == 'D1':
        return support.sqrt()
    if mode == 'D2':
        return torch.ones_like(support)
    raise ValueError(mode)


@contextmanager
def hook(model, mode):
    captured = []

    def before(_, inputs):
        support = inputs[3]
        return (*inputs[:3], transform(support, mode))

    def after(_, inputs, output):
        coords, stats = model.radar_stats
        delta, gate = output
        if not torch.isfinite(delta).all() or not torch.isfinite(gate).all():
            raise ValueError('Nonfinite Radar correction')
        captured.append(dict(coords=coords.detach(), support=stats['support'].detach(),
            effective=inputs[3].detach(), neighbors=stats['neighbors'].detach(),
            ra_rms=stats['residual'].detach()/8, gate=gate.detach(),
            rb_rms=delta.detach().square().mean(-1).sqrt()))

    a = model.rb.register_forward_pre_hook(before)
    b = model.rb.register_forward_hook(after)
    try:
        yield captured
    finally:
        b.remove()
        a.remove()


def boxes_mask(xyz, boxes, padding=0.):
    """Horizontal association in the ego grid, with an explicit margin per side."""
    if not len(boxes) or not len(xyz):
        return np.zeros((len(boxes),len(xyz)), dtype=bool)
    xy = np.asarray(xyz, dtype=np.float32).copy()
    xy[:,2] = 0
    roi = np.asarray(boxes, dtype=np.float32).reshape(-1,7)[:,[0,1,2,5,4,3,6]].copy()
    roi[:,2] = 0
    roi[:,3:5] += 2*padding
    roi[:,5] = 2
    return np.asarray(points_in_boxes_cpu(xy, roi), dtype=bool)


def snapshot(captured, radar, objects):
    if not captured:
        if len(radar['voxel_coords']):
            raise ValueError('Occupied Radar did not reach the compensation hook')
        empty = dict(pillars=0, corrected=0, support=0., effective=0., gate=0., ra_rms=0., rb_rms=0., max_neighbors=None)
        return dict(neighbors={}, near_targets={str(x["object_id"]):dict(empty) for x in objects})
    if len(captured) != 1 or not torch.equal(captured[0]['coords'], radar['voxel_coords']):
        raise ValueError('Compensation coordinates differ from true occupied Radar coordinates')
    values = {k:v.cpu().numpy() for k,v in captured[0].items() if k != 'coords'}
    neighbors = values.pop('neighbors')
    fields = ('support', 'effective', 'gate', 'ra_rms', 'rb_rms')

    def reduce(mask):
        return dict(pillars=int(mask.sum()), corrected=int((values['rb_rms'][mask]>1e-6).sum()),
            **{k:float(values[k][mask].sum(dtype=np.float64)) for k in fields})

    groups = {str(i):reduce(neighbors == i) for i in range(9)}
    xyz = point_statistics(radar)[1].cpu().numpy()
    inside = boxes_mask(xyz, [x['box'] for x in objects], .4)
    per_object = {}
    near = inside.any(0)
    for obj, mask in zip(objects, inside):
        row = reduce(mask)
        row['max_neighbors'] = int(neighbors[mask].max()) if mask.any() else None
        per_object[str(obj['object_id'])] = row
    return dict(neighbors=groups, near_neighbors={str(i):reduce(near & (neighbors==i)) for i in range(9)},
                near_targets=per_object)


def max_difference(reference, tested):
    maximum = max(float((v-tested[k]).abs().max()) for k,v in reference.items())
    if any(not torch.allclose(v,tested[k],atol=1e-5,rtol=1e-5) for k,v in reference.items()):
        raise ValueError(f"Model output identity exceeded tolerance. Max absolute difference {maximum:.8g}")
    return maximum


def preflight(model, ego):
    radar = ego['processed_radar']
    if not len(radar['voxel_coords']):
        raise ValueError('Identity fixture requires real Radar occupancy')
    empty = dict(ego, processed_radar={k:v[:0] for k,v in radar.items()})
    result = {}
    with torch.no_grad():
        native = model(ego)
        empty_ref = model(empty)
        native_repeat = max_difference(native,model(ego))
        empty_repeat = max_difference(empty_ref,model(empty))
        runs = {}
        for mode in ('D0','D1','D2'):
            with hook(model, mode) as trace:
                pred = model(ego)
                if len(trace) != 1 or not torch.equal(trace[0]['coords'], radar['voxel_coords']):
                    raise ValueError('Intervention added or reordered Radar positions')
                runs[mode] = trace.pop()
                empty_pred = model(empty)
                if trace:
                    raise ValueError('Empty Radar reached the compensation hook')
                empty_difference = max_difference(empty_ref,empty_pred)
                if any(not torch.isfinite(v).all() for v in pred.values()):
                    raise ValueError('Nonfinite intervention fixture')
                d0_difference = max_difference(native,pred) if mode == 'D0' else None
                result[mode] = dict(occupied=len(runs[mode]['coords']),
                    max_rms=float(runs[mode]['rb_rms'].max()), empty_max_abs=empty_difference,
                    d0_max_abs=d0_difference)
        original = runs['D0']
        for mode, run in runs.items():
            if (not torch.allclose(run['gate'], original['gate'], atol=1e-5, rtol=1e-5)
                    or not torch.allclose(run['ra_rms'], original['ra_rms'], atol=1e-6, rtol=1e-5)):
                raise ValueError('Intervention changed native gate or R-A residual')
            support = original['support']
            expected = original['rb_rms'] * transform(support, mode) / support
            if not torch.allclose(run['rb_rms'], expected, atol=1e-6, rtol=1e-5):
                raise ValueError('Intervention did not isolate the R-B residual multiplier')
    return dict(status='passed', atol=1e-5, rtol=1e-5, native_repeat_max_abs=native_repeat,
                empty_repeat_max_abs=empty_repeat, modes=result)


@contextmanager
def instrument(model, out, mode, baseline):
    import metrics
    from opencood.tools import inference_utils

    native_build, infer = metrics.build, inference_utils.inference_intermediate_fusion
    cursor = {}

    def build(*args, **kwargs):
        cursor.update(weather=kwargs['condition'], index=0)
        return native_build(*args, **kwargs)

    def observed(batch, network, dataset):
        w, i = cursor['weather'], cursor['index']
        cursor['index'] += 1
        clean = read_json(baseline/'observations/clean'/f'{i:06d}.json')
        weather = read_json(baseline/'observations'/w/f'{i:06d}.json')
        fixed = read_json(LR0/'final_evaluation/predictions/clean'/f'{i:06d}.json')
        if clean['frame_key'] != weather['frame_key'] or clean['frame_key'] != fixed['frame_key']:
            raise ValueError('Archived target frame identity differs')
        damaged = []
        for target in fixed['official_gt']:
            key = str(target['object_id'])
            ref, obs = clean['objects'][key], weather['objects'][key]
            if ref['lidar'] > 0 and obs['lidar'] <= .7*ref['lidar'] and obs['radar'] > 0:
                damaged.append(target)
        with hook(model, mode) as captured:
            result = infer(batch, network, dataset)
        evidence = snapshot(captured, batch['ego']['processed_radar'], damaged)
        atomic_json(out/'feature_rows'/w/f'{i:06d}.json',
                    dict(frame_key=fixed['frame_key'], **evidence))
        return result

    with patched(metrics, 'build', build), patched(inference_utils, 'inference_intermediate_fusion', observed):
        yield


def merge_features(path, weathers, limit):
    import json
    frames = {}
    with (path/'per_frame_metrics.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            frames[(row['condition'],row['index'])] = row
    output = {}
    for weather in weathers:
        groups = defaultdict(lambda: defaultdict(float))
        targets = {}
        for i in range(limit):
            obs = read_json(path/'feature_rows'/weather/f'{i:06d}.json')
            frame = frames[(weather,i)]
            if obs['frame_key'] != frame['frame_key']:
                raise ValueError('Feature evidence frame differs from evaluator')
            pairs = frame['protocols']['F'].get('matched_pairs', {}).get('0.7', [])
            matched = {str(p['object_id']):p['iou'] for p in pairs}
            for name in ('neighbors','near_neighbors'):
                for k,v in obs.get(name,{}).items():
                    for field,value in v.items():
                        groups[(name,k)][field] += value
            for key,row in obs['near_targets'].items():
                targets[f'{i}/{key}'] = dict(row, iou=matched.get(key))
        output[weather] = dict(
            neighbors={k:dict(v) for (name,k),v in groups.items() if name=='neighbors'},
            near_neighbors={k:dict(v) for (name,k),v in groups.items() if name=='near_neighbors'},
            targets=targets)
    return output

