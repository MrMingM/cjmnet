"""Evaluation and feasibility gate for B1 task-conflict gating."""
import copy
import json

import numpy as np
import torch

from ceif_audit.scoring import ap_values, empty_stats
from local_fusion_utility_v2.outcomes import compare_predictions, detection_outcome
from local_fusion_v3 import runtime as v3rt
from opencood.tools.train_utils import to_device
from opencood.utils import eval_utils

from .model import prepare_frozen_context, predict_with_gate


METHODS = ('B0-Collapse', 'B0-Split', 'Global-Gate', 'Local-Gate')


def _zero_harm():
    return dict(recovered=0, lost=0, new_fp=0)


def _append_gate_rows(store, info):
    for scale, value in info['gate_maps'].items():
        signal = info['signals'][scale]
        row = store.setdefault(
            str(scale),
            {'gate': [], 'tv': [], 'disagree': [], 'activity': []},
        )
        row['gate'].append(value.detach().cpu().reshape(-1).numpy())
        row['tv'].append(
            signal['total_variation'].detach().cpu().reshape(-1).numpy())
        row['disagree'].append(
            signal['top_disagreement'].detach().cpu().reshape(-1).numpy())
        row['activity'].append(
            signal['activity'].detach().cpu().reshape(-1).numpy())


def _safe_corr(left, right):
    if len(left) < 2 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _gate_summary(store):
    result = {}
    all_gate, all_tv, all_disagree, all_activity = [], [], [], []
    for scale, row in sorted(store.items()):
        gate = np.concatenate(row['gate']) if row['gate'] else np.empty(0)
        tv = np.concatenate(row['tv']) if row['tv'] else np.empty(0)
        disagree = (
            np.concatenate(row['disagree']) if row['disagree'] else np.empty(0))
        activity = (
            np.concatenate(row['activity']) if row['activity'] else np.empty(0))
        if not len(gate):
            result[scale] = {'cells': 0}
            continue
        q = np.quantile(gate, [.5, .75, .9])
        tv75 = float(np.quantile(tv, .75))
        high_tv = tv >= tv75
        disagree_mask = disagree > .5
        result[scale] = {
            'cells': int(len(gate)),
            'mean': float(gate.mean()),
            'p50': float(q[0]),
            'p75': float(q[1]),
            'p90': float(q[2]),
            'corr_gate_total_variation': _safe_corr(gate, tv),
            'corr_gate_activity': _safe_corr(gate, activity),
            'top_disagreement_rate': float(disagree_mask.mean()),
            'mean_gate_top_disagree': (
                float(gate[disagree_mask].mean())
                if disagree_mask.any() else None
            ),
            'mean_gate_top_agree': (
                float(gate[~disagree_mask].mean())
                if (~disagree_mask).any() else None
            ),
            'total_variation_p75': tv75,
            'mean_gate_high_tv_quartile': (
                float(gate[high_tv].mean()) if high_tv.any() else None
            ),
            'mean_gate_lower_tv_75pct': (
                float(gate[~high_tv].mean()) if (~high_tv).any() else None
            ),
        }
        all_gate.append(gate)
        all_tv.append(tv)
        all_disagree.append(disagree)
        all_activity.append(activity)

    if all_gate:
        gate = np.concatenate(all_gate)
        tv = np.concatenate(all_tv)
        disagree = np.concatenate(all_disagree)
        activity = np.concatenate(all_activity)
        q = np.quantile(gate, [.5, .75, .9])
        disagree_mask = disagree > .5
        result['all_scales'] = {
            'cells': int(len(gate)),
            'mean': float(gate.mean()),
            'p50': float(q[0]),
            'p75': float(q[1]),
            'p90': float(q[2]),
            'corr_gate_total_variation': _safe_corr(gate, tv),
            'corr_gate_activity': _safe_corr(gate, activity),
            'top_disagreement_rate': float(disagree_mask.mean()),
            'mean_gate_top_disagree': (
                float(gate[disagree_mask].mean())
                if disagree_mask.any() else None
            ),
            'mean_gate_top_agree': (
                float(gate[~disagree_mask].mean())
                if (~disagree_mask).any() else None
            ),
        }
    return result


def evaluate_condition(model, split_arm, gates, dataset, loader, indices,
                       branch, target, output):
    stats = {name: empty_stats() for name in METHODS}
    vs_global = {name: _zero_harm() for name in
                 ('B0-Collapse', 'B0-Split', 'Local-Gate')}
    vs_split = {name: _zero_harm() for name in ('Global-Gate', 'Local-Gate')}
    global_tp = split_tp = 0
    gate_rows = {'Global-Gate': {}, 'Local-Gate': {}}
    count = 0

    with torch.no_grad(), (output / 'frames.jsonl').open(
            'w', encoding='utf-8') as stream:
        for batch in loader:
            batch = to_device(batch, target)
            context = v3rt.context(
                model, batch['ego'], branch, verify=count == 0)
            frozen = prepare_frozen_context(
                model.engine.base,
                context['levels'],
                split_arm,
                gates['Local-Gate'].scales,
            )
            global_prediction, global_info = predict_with_gate(
                model.engine.base, frozen, gates['Global-Gate'])
            local_prediction, local_info = predict_with_gate(
                model.engine.base, frozen, gates['Local-Gate'])

            predictions = {
                'B0-Collapse': frozen['collapse_prediction'],
                'B0-Split': frozen['split_prediction'],
                'Global-Gate': global_prediction,
                'Local-Gate': local_prediction,
            }
            post = {
                name: dataset.post_process(batch, {'ego': prediction})
                for name, prediction in predictions.items()
            }
            for name in METHODS:
                boxes, scores, gt = post[name]
                for threshold in stats[name]:
                    eval_utils.caluclate_tp_fp(
                        boxes, scores, gt, stats[name], threshold)

            matched_global, _ = detection_outcome(*post['Global-Gate'], .7)
            matched_split, _ = detection_outcome(*post['B0-Split'], .7)
            global_tp += len(matched_global)
            split_tp += len(matched_split)

            for name in vs_global:
                difference = compare_predictions(
                    post['Global-Gate'], post[name], .7, .7)
                for key in vs_global[name]:
                    vs_global[name][key] += int(difference[key])
            for name in vs_split:
                difference = compare_predictions(
                    post['B0-Split'], post[name], .7, .7)
                for key in vs_split[name]:
                    vs_split[name][key] += int(difference[key])

            if len(context['levels'][0]) > 1:
                _append_gate_rows(gate_rows['Global-Gate'], global_info)
                _append_gate_rows(gate_rows['Local-Gate'], local_info)

            row = {
                'sample_index': int(
                    batch['ego']['communication_sample_index'][0]),
                'source_count': int(len(context['levels'][0])),
                'global_gate_mean': float(
                    global_info['gate_mean'].detach().cpu()),
                'local_gate_mean': float(
                    local_info['gate_mean'].detach().cpu()),
            }
            for scale in gates['Local-Gate'].scales:
                row[f'global_gate_s{scale}'] = float(
                    global_info[f'gate_s{scale}'].detach().cpu())
                row[f'local_gate_s{scale}'] = float(
                    local_info[f'gate_s{scale}'].detach().cpu())
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')

            count += 1
            if count == 1 or count % 20 == 0:
                print(f'B1 validation {count}/{len(indices)}', flush=True)

    if count != len(indices) or not count or not stats['B0-Split'][.7]['gt']:
        raise RuntimeError('Incomplete or empty B1 validation')

    results = {name: ap_values(value, eval_utils)
               for name, value in stats.items()}
    for name, value in stats.items():
        folder = output / name
        folder.mkdir()
        eval_utils.eval_final_results(copy.deepcopy(value), str(folder), False)

    return {
        'frames': count,
        'results': results,
        'diagnostics_vs_global': vs_global,
        'diagnostics_vs_split': vs_split,
        'global_tp': int(global_tp),
        'split_tp': int(split_tp),
        'gate_statistics': {
            name: _gate_summary(rows)
            for name, rows in gate_rows.items()
        },
        'global_sort': False,
    }


def assert_b0_reproduction(condition, current, saved, tolerance=1e-6):
    mapping = {
        'B0-Collapse': 'Split-Collapse',
        'B0-Split': 'Split',
    }
    for current_name, b0_name in mapping.items():
        for metric in ('ap30', 'ap50', 'ap70'):
            left = float(current['results'][current_name][metric])
            right = float(saved['conditions'][condition]['results'][b0_name][metric])
            if abs(left - right) > tolerance:
                raise RuntimeError(
                    f'B0 reproduction mismatch {condition} '
                    f'{current_name}.{metric}: {left} versus {right}')


def decide(conditions, settings):
    weathers = ('fog', 'rain', 'snow')
    local_vs_global = {
        weather: (
            conditions[weather]['results']['Local-Gate']['ap70']
            - conditions[weather]['results']['Global-Gate']['ap70']
        )
        for weather in weathers
    }
    local_vs_split = {
        weather: (
            conditions[weather]['results']['Local-Gate']['ap70']
            - conditions[weather]['results']['B0-Split']['ap70']
        )
        for weather in weathers
    }
    global_vs_split = {
        weather: (
            conditions[weather]['results']['Global-Gate']['ap70']
            - conditions[weather]['results']['B0-Split']['ap70']
        )
        for weather in weathers
    }

    mean_local_vs_global = sum(local_vs_global.values()) / len(weathers)
    mean_local_vs_split = sum(local_vs_split.values()) / len(weathers)
    positive = sum(value > 0 for value in local_vs_global.values())

    clean = conditions['clean']['results']
    checks = {
        'weather_gain_vs_global': (
            mean_local_vs_global
            >= settings['minimum_weather_mean_ap70_gain_vs_global']
        ),
        'positive_weathers_vs_global': (
            positive >= settings['minimum_positive_weathers']
        ),
        'weather_mean_above_b0_split': (
            mean_local_vs_split
            > settings['minimum_weather_mean_ap70_gain_vs_split']
        ),
        'clean_ap70_vs_global': (
            clean['Local-Gate']['ap70']
            >= clean['Global-Gate']['ap70']
            - settings['clean_ap70_tolerance']
        ),
    }

    for condition, row in conditions.items():
        checks[f'{condition}_ap50_vs_global'] = (
            row['results']['Local-Gate']['ap50']
            >= row['results']['Global-Gate']['ap50']
            - settings['ap50_tolerance']
        )
        difference = row['diagnostics_vs_global']['Local-Gate']
        denominator = max(1, row['global_tp'])
        checks[f'{condition}_lost_tp_vs_global'] = (
            difference['lost']
            <= settings['max_lost_tp_fraction'] * denominator
        )
        checks[f'{condition}_new_fp_vs_global'] = (
            difference['new_fp']
            <= settings['max_new_fp_per_reference_tp'] * denominator
        )

    return {
        'expand': all(checks.values()),
        'checks': checks,
        'local_vs_global_weather_ap70_gains': local_vs_global,
        'mean_local_vs_global_weather_ap70_gain': mean_local_vs_global,
        'positive_weathers_vs_global': positive,
        'local_vs_b0_split_weather_ap70_gains': local_vs_split,
        'mean_local_vs_b0_split_weather_ap70_gain': mean_local_vs_split,
        'global_vs_b0_split_weather_ap70_gains': global_vs_split,
        'mean_global_vs_b0_split_weather_ap70_gain': (
            sum(global_vs_split.values()) / len(weathers)
        ),
        'interpretation': (
            'Primary test is Local-Gate versus equal-parameter Global-Gate. '
            'Local must also beat the frozen B0 Split on mean weather AP70. '
            'Gate statistics are mechanism diagnostics, not extra feasibility '
            'thresholds.'
        ),
    }
