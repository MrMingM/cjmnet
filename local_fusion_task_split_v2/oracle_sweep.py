"""No-training structural upper-bound audit for B1.

Part A: fixed constant gate sweep from Collapse (g=0) to Split (g=1).
Part B: hindsight local Oracle gates.  A GT region is opened (g=1) only when
B0-Split detects that GT at IoU 0.7 while B0-Collapse misses it.  Everywhere
else stays Collapse (g=0).

GT and paired endpoint outcomes are used only for the Oracle diagnostic.  This
script is not deployable, does not train or tune a model, and never touches
OPV2V-W.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from ceif_audit.scoring import ap_values, empty_stats
from gspr_communication.runtime import (
    ROOT,
    device,
    new_output,
    seed_all,
    sha256,
    verify_frozen,
    write_json,
)
from gspr_evidence import runtime as er
from local_fusion_task_split_pilot.pipeline import selected_loader
from local_fusion_utility_v2.outcomes import compare_predictions, detection_outcome
from local_fusion_v3 import runtime as v3rt
from opencood.tools.train_utils import to_device
from opencood.utils import eval_utils
from qa_local_intervention.common import grid_xy, roi_mask

from .model import (
    _interpolate,
    prepare_frozen_context,
)
from local_fusion_task_split_pilot.model import (
    _decode_task,
    _fuse_from_weights,
)
from .pipeline import (
    _load_json,
    _load_split,
    _verify_b0_source_contract,
)


SWEEP = (0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0)
ORACLE_EXPANSIONS = {
    'Oracle-Box': 1.0,
    'Oracle-Context': 1.5,
}
ENDPOINTS = ('B0-Collapse', 'B0-Split')


def _gate_name(value):
    return f'Gate-{value:.2f}'


def _predict_with_maps(base, frozen, gate_maps):
    levels = frozen['levels']
    cls_weights, reg_weights = [], []
    for scale, (common, cls_raw, reg_raw) in enumerate(zip(
            frozen['common_weights'],
            frozen['weights_cls'],
            frozen['weights_reg'])):
        gate = gate_maps.get(scale)
        if gate is None:
            cls_weights.append(common)
            reg_weights.append(common)
        else:
            cls_weights.append(_interpolate(common, cls_raw, gate))
            reg_weights.append(_interpolate(common, reg_raw, gate))

    cls_levels = _fuse_from_weights(levels, cls_weights)
    reg_levels = _fuse_from_weights(levels, reg_weights)
    return {
        'psm': _decode_task(base, cls_levels, base.cls_head),
        'rm': _decode_task(base, reg_levels, base.reg_head),
    }


def _constant_maps(frozen, value):
    maps = {}
    for scale, signal in frozen['signals'].items():
        tv = signal['total_variation']
        maps[scale] = tv.new_full(tv.shape, float(value))
    return maps


def _oracle_maps(frozen, gt, split_only, lidar_range, expansion):
    maps = {}
    fallback = {}
    active_cells = {}
    total_cells = {}
    for scale, signal in frozen['signals'].items():
        tv = signal['total_variation']
        h, w = tv.shape[-2:]
        union = np.zeros((h, w), dtype=bool)
        fallback_count = 0
        xy = grid_xy((h, w), lidar_range)
        for target_index in sorted(split_only):
            mask, used_fallback = roi_mask(
                xy,
                np.asarray(gt[target_index], dtype=np.float64),
                float(expansion),
            )
            union |= mask
            fallback_count += int(used_fallback)
        tensor = torch.from_numpy(union.astype(np.float32)).to(
            device=tv.device, dtype=tv.dtype)
        maps[scale] = tensor.unsqueeze(0).unsqueeze(0)
        fallback[str(scale)] = fallback_count
        active_cells[str(scale)] = int(union.sum())
        total_cells[str(scale)] = int(union.size)
    return maps, fallback, active_cells, total_cells


def _zero_harm():
    return {'recovered': 0, 'lost': 0, 'new_fp': 0}


def _add_harm(store, value):
    for key in store:
        store[key] += int(value[key])


def _post(dataset, batch, prediction):
    return dataset.post_process(batch, {'ego': prediction})


def _assert_same_gt(left, right):
    a = left[2].detach().cpu().numpy()
    b = right[2].detach().cpu().numpy()
    if a.shape != b.shape or not np.allclose(a, b):
        raise RuntimeError('GT changed across paired predictions')


def _evaluate_condition(model, split_arm, dataset, loader, indices,
                        branch, target, lidar_range, saved_b0, folder):
    method_names = [_gate_name(x) for x in SWEEP]
    method_names += list(ORACLE_EXPANSIONS)
    methods = tuple(ENDPOINTS) + tuple(method_names)
    stats = {name: empty_stats() for name in methods}
    harms_vs_collapse = {
        name: _zero_harm() for name in method_names
    }
    harms_vs_split = {
        name: _zero_harm() for name in method_names
    }
    oracle_counts = {
        name: {
            'split_only_targets': 0,
            'frames_with_split_only': 0,
            'fallbacks': {'0': 0, '1': 0},
            'active_cells': {'0': 0, '1': 0},
            'total_cells': {'0': 0, '1': 0},
        }
        for name in ORACLE_EXPANSIONS
    }
    frame_rows = []
    count = 0

    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, target)
            context = v3rt.context(
                model, batch['ego'], branch, verify=count == 0)
            frozen = prepare_frozen_context(
                model.engine.base, context['levels'], split_arm, scales=(0, 1))

            collapse_post = _post(
                dataset, batch, frozen['collapse_prediction'])
            split_post = _post(
                dataset, batch, frozen['split_prediction'])
            _assert_same_gt(collapse_post, split_post)
            gt_tensor = split_post[2]
            gt = gt_tensor.detach().cpu().numpy()

            matched_collapse, _ = detection_outcome(*collapse_post, .7)
            matched_split, _ = detection_outcome(*split_post, .7)
            split_only = set(matched_split) - set(matched_collapse)

            predictions = {
                'B0-Collapse': frozen['collapse_prediction'],
                'B0-Split': frozen['split_prediction'],
            }
            posts = {
                'B0-Collapse': collapse_post,
                'B0-Split': split_post,
            }

            for value in SWEEP:
                name = _gate_name(value)
                prediction = _predict_with_maps(
                    model.engine.base,
                    frozen,
                    _constant_maps(frozen, value),
                )
                predictions[name] = prediction
                posts[name] = _post(dataset, batch, prediction)
                _assert_same_gt(collapse_post, posts[name])

            oracle_frame = {}
            for name, expansion in ORACLE_EXPANSIONS.items():
                maps, fallback, active, total = _oracle_maps(
                    frozen, gt, split_only, lidar_range, expansion)
                prediction = _predict_with_maps(
                    model.engine.base, frozen, maps)
                predictions[name] = prediction
                posts[name] = _post(dataset, batch, prediction)
                _assert_same_gt(collapse_post, posts[name])

                row = oracle_counts[name]
                row['split_only_targets'] += len(split_only)
                row['frames_with_split_only'] += int(bool(split_only))
                for scale in ('0', '1'):
                    row['fallbacks'][scale] += int(fallback.get(scale, 0))
                    row['active_cells'][scale] += int(active.get(scale, 0))
                    row['total_cells'][scale] += int(total.get(scale, 0))
                oracle_frame[name] = {
                    'active_cells': active,
                    'total_cells': total,
                    'fallbacks': fallback,
                }

            for name in methods:
                boxes, scores, gt_eval = posts[name]
                for threshold in stats[name]:
                    eval_utils.caluclate_tp_fp(
                        boxes, scores, gt_eval, stats[name], threshold)

            for name in method_names:
                _add_harm(
                    harms_vs_collapse[name],
                    compare_predictions(
                        posts['B0-Collapse'], posts[name], .7, .7),
                )
                _add_harm(
                    harms_vs_split[name],
                    compare_predictions(
                        posts['B0-Split'], posts[name], .7, .7),
                )

            frame_rows.append({
                'sample_index': int(
                    batch['ego']['communication_sample_index'][0]),
                'source_count': int(len(context['levels'][0])),
                'split_only_targets': sorted(int(x) for x in split_only),
                'split_only_count': int(len(split_only)),
                'oracle': oracle_frame,
            })

            count += 1
            if count == 1 or count % 20 == 0:
                print(
                    f'oracle/sweep {folder.name}: {count}/{len(indices)}',
                    flush=True,
                )

    if count != len(indices) or not count:
        raise RuntimeError('Incomplete oracle/sweep validation')

    results = {
        name: ap_values(value, eval_utils)
        for name, value in stats.items()
    }

    # Endpoint reproduction is mandatory.
    mapping = {
        'B0-Collapse': 'Split-Collapse',
        'B0-Split': 'Split',
    }
    for current_name, saved_name in mapping.items():
        for metric in ('ap30', 'ap50', 'ap70'):
            left = float(results[current_name][metric])
            right = float(saved_b0['results'][saved_name][metric])
            if abs(left - right) > 1e-6:
                raise RuntimeError(
                    f'B0 endpoint reproduction mismatch '
                    f'{current_name}.{metric}: {left} vs {right}')

    # Constant endpoints must also reproduce Collapse/Split exactly.
    for metric in ('ap30', 'ap50', 'ap70'):
        if abs(
            float(results['Gate-0.00'][metric])
            - float(results['B0-Collapse'][metric])
        ) > 1e-6:
            raise RuntimeError('g=0 failed to reproduce Collapse')
        if abs(
            float(results['Gate-1.00'][metric])
            - float(results['B0-Split'][metric])
        ) > 1e-6:
            raise RuntimeError('g=1 failed to reproduce Split')

    with (folder / 'frames.jsonl').open('w', encoding='utf-8') as stream:
        for row in frame_rows:
            stream.write(json.dumps(
                row, ensure_ascii=False, allow_nan=False) + '\n')

    for name, row in oracle_counts.items():
        row['active_fraction'] = {}
        for scale in ('0', '1'):
            denominator = max(1, row['total_cells'][scale])
            row['active_fraction'][scale] = (
                row['active_cells'][scale] / denominator
            )

    return {
        'frames': count,
        'results': results,
        'diagnostics_vs_collapse': harms_vs_collapse,
        'diagnostics_vs_split': harms_vs_split,
        'oracle_statistics': oracle_counts,
        'global_sort': False,
    }


def _posthoc_curve_summary(conditions):
    weathers = ('fog', 'rain', 'snow')
    curves = {}
    for condition, row in conditions.items():
        curves[condition] = {
            str(value): row['results'][_gate_name(value)]['ap70']
            for value in SWEEP
        }

    weather_mean = {}
    for value in SWEEP:
        name = _gate_name(value)
        weather_mean[str(value)] = float(np.mean([
            conditions[weather]['results'][name]['ap70']
            for weather in weathers
        ]))

    best_value = max(
        SWEEP, key=lambda x: weather_mean[str(x)])
    endpoint_split = weather_mean['1.0']
    endpoint_collapse = weather_mean['0.0']

    oracle_weather = {}
    for name in ORACLE_EXPANSIONS:
        oracle_weather[name] = {
            weather: (
                conditions[weather]['results'][name]['ap70']
                - conditions[weather]['results']['B0-Split']['ap70']
            )
            for weather in weathers
        }
        oracle_weather[name]['mean_vs_split'] = float(np.mean([
            oracle_weather[name][weather] for weather in weathers
        ]))
        oracle_weather[name]['mean_vs_collapse'] = float(np.mean([
            conditions[weather]['results'][name]['ap70']
            - conditions[weather]['results']['B0-Collapse']['ap70']
            for weather in weathers
        ]))

    return {
        'fixed_gate_ap70_curves': curves,
        'fixed_gate_weather_mean_ap70': weather_mean,
        'posthoc_best_fixed_gate': float(best_value),
        'posthoc_best_fixed_gate_weather_mean_ap70': (
            weather_mean[str(best_value)]
        ),
        'collapse_weather_mean_ap70': endpoint_collapse,
        'split_weather_mean_ap70': endpoint_split,
        'oracle_weather_ap70_deltas': oracle_weather,
        'selection_boundary': (
            'The best fixed gate is reported only to describe the curve. '
            'It must not be reused as a tuned deployment value because the '
            'same validation set was inspected.'
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--b1-run', required=True,
        help='Completed B1 run; used only to recover exact lineage/indices')
    parser.add_argument(
        '--v3-config', default='local_fusion_v3/experiment.yaml')
    parser.add_argument('--frontend-config', required=True)
    parser.add_argument('--frontend-checkpoint', required=True)
    parser.add_argument('--v3-checkpoint', required=True)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    verify_frozen()
    b1_run = Path(args.b1_run).resolve()
    b1_protocol = _load_json(b1_run / 'protocol.json')
    b1_results = _load_json(b1_run / 'decision_results.json')
    b0_run = Path(b1_protocol['b0_run']).resolve()
    b0_protocol = _load_json(b0_run / 'protocol.json')
    b0_results = _load_json(b0_run / 'decision_results.json')
    _verify_b0_source_contract(b0_protocol)

    output = new_output(
        Path(args.output).resolve()
        if args.output else b1_run / 'oracle_sweep'
    )

    # Ensure the audit is attached to the exact completed B1/B0 lineage.
    if sha256(b0_run / 'protocol.json') != b1_protocol['b0_protocol_sha256']:
        raise ValueError('B0 protocol changed since B1 training')
    if sha256(b0_run / 'Split.pth') != b1_protocol['b0_split_checkpoint_sha256']:
        raise ValueError('B0 Split checkpoint changed since B1 training')
    if b1_results.get('scope') is None:
        raise ValueError('Invalid completed B1 result file')

    options, hypes = er.load_config(
        args.v3_config, args.frontend_config)
    target = device()
    model, frontend_digest = er.load_model(
        hypes, options, args.frontend_checkpoint, target)
    model.requires_grad_(False).eval()
    if frontend_digest != b1_protocol['frontend_sha256']:
        raise ValueError('Frontend checkpoint differs from B1 run')

    v3_contract = v3rt.contract(
        options, args.frontend_config, frontend_digest)
    if v3_contract != b1_protocol['v3_contract']:
        raise ValueError('v3 contract differs from B1 run')
    if sha256(args.v3_checkpoint) != b1_protocol['v3_checkpoint_sha256']:
        raise ValueError('v3 checkpoint differs from B1 run')

    source, checkpoint = v3rt.load(
        args.v3_checkpoint, v3_contract, target)
    if checkpoint['variant'] != 'residual':
        raise ValueError('Expected residual v3 checkpoint')
    split_arm = _load_split(
        b0_run, source, target, sha256(args.v3_checkpoint))
    del source

    validation_indices = [
        int(x) for x in b1_protocol['validation_indices']
    ]
    if validation_indices != [
        int(x) for x in b0_protocol['validation_indices']
    ]:
        raise ValueError('B1/B0 validation indices differ')

    lidar_range = hypes['postprocess']['anchor_args']['cav_lidar_range']
    if len(lidar_range) != 6:
        raise ValueError('Invalid lidar range')

    conditions = {}
    for condition in ('clean', 'fog', 'rain', 'snow'):
        seed_all(int(b1_protocol['config']['seed']) + 1)
        dataset, loader = selected_loader(
            hypes, options, 'validation', condition, validation_indices)
        folder = output / condition
        folder.mkdir(parents=True)
        conditions[condition] = _evaluate_condition(
            model,
            split_arm,
            dataset,
            loader,
            validation_indices,
            'clean' if condition == 'clean' else 'weather',
            target,
            lidar_range,
            b0_results['conditions'][condition],
            folder,
        )
        write_json(folder / 'summary.json', conditions[condition])
        del dataset, loader

    report = {
        'scope': (
            'no-training structural upper-bound audit on exact completed B1 '
            'validation indices; official validation + online weather only; '
            'no OPV2V-W'
        ),
        'fixed_gate_values': list(SWEEP),
        'oracle_definition': (
            'For each frame, identify GT detected by frozen B0-Split at IoU '
            '0.7 but missed by frozen B0-Collapse. Set g=1 only inside the '
            'union of those GT ROIs, g=0 elsewhere. Oracle-Box uses the exact '
            'oriented box footprint; Oracle-Context uses 1.5x footprint.'
        ),
        'oracle_boundary': (
            'Oracle uses GT and hindsight endpoint outcomes, so it is an '
            'upper-bound diagnostic only and cannot be deployed or used as '
            'a learned target without an independent protocol.'
        ),
        'b1_run': str(b1_run),
        'b0_run': str(b0_run),
        'conditions': conditions,
        'summary': _posthoc_curve_summary(conditions),
    }
    write_json(output / 'oracle_sweep_results.json', report)
    verify_frozen()
    print(
        'B1 ORACLE/SWEEP COMPLETE:',
        output / 'oracle_sweep_results.json',
        flush=True,
    )


if __name__ == '__main__':
    main()
