"""Evaluation for the shared-versus-task-split fusion pilot."""
import copy
import json

import torch

from ceif_audit.scoring import ap_values, empty_stats
from local_fusion_utility_v2.outcomes import compare_predictions, detection_outcome
from local_fusion_v3 import runtime as v3rt
from opencood.tools.train_utils import to_device
from opencood.utils import eval_utils


METHODS = (
    'baseline',
    'v3_start',
    'Shared',
    'Split',
    'Split-Collapse',
    'Split-Swap',
)


def _zero_harm():
    return dict(recovered=0, lost=0, new_fp=0)


def evaluate_condition(model, start_fusion, arms, dataset, loader, indices,
                       branch, target, output):
    stats = {name: empty_stats() for name in METHODS}
    harms_shared = {name: _zero_harm() for name in
                    ('Split', 'Split-Collapse', 'Split-Swap')}
    harms_start = {name: _zero_harm() for name in ('Shared', 'Split')}
    shared_tp = start_tp = 0
    separation_sums = {
        'task_gap': 0.0,
        'task_disagreement': 0.0,
        'gap_s0': 0.0,
        'gap_s1': 0.0,
        'disagree_s0': 0.0,
        'disagree_s1': 0.0,
    }
    multi_source = 0
    count = 0

    with torch.no_grad(), (output / 'frames.jsonl').open('w', encoding='utf-8') as stream:
        for batch in loader:
            batch = to_device(batch, target)
            context = v3rt.context(model, batch['ego'], branch, verify=count == 0)
            start_prediction, _ = start_fusion.predict(
                model.engine.base, context['levels'])
            shared_prediction, _ = arms['Shared'].predict(
                model.engine.base, context['levels'])
            split_prediction, split_info = arms['Split'].predict(
                model.engine.base, context['levels'])
            collapse_prediction, _ = arms['Split'].predict(
                model.engine.base, context['levels'], collapse=True)
            swap_prediction, _ = arms['Split'].predict(
                model.engine.base, context['levels'], swap=True)

            predictions = {
                'baseline': context['baseline_prediction'],
                'v3_start': start_prediction,
                'Shared': shared_prediction,
                'Split': split_prediction,
                'Split-Collapse': collapse_prediction,
                'Split-Swap': swap_prediction,
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

            matched_shared, _ = detection_outcome(*post['Shared'], .7)
            matched_start, _ = detection_outcome(*post['v3_start'], .7)
            shared_tp += len(matched_shared)
            start_tp += len(matched_start)

            for name in harms_shared:
                difference = compare_predictions(post['Shared'], post[name], .7, .7)
                for key in harms_shared[name]:
                    harms_shared[name][key] += int(difference[key])
            for name in harms_start:
                difference = compare_predictions(post['v3_start'], post[name], .7, .7)
                for key in harms_start[name]:
                    harms_start[name][key] += int(difference[key])

            row = {
                'sample_index': int(batch['ego']['communication_sample_index'][0]),
                'source_count': int(len(context['levels'][0])),
            }
            if len(context['levels'][0]) > 1:
                multi_source += 1
                for key in separation_sums:
                    value = float(split_info[key].detach())
                    separation_sums[key] += value
                    row[key] = value
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')

            count += 1
            if count == 1 or count % 20 == 0:
                print(f'validation {count}/{len(indices)}', flush=True)

    if count != len(indices) or not count or not stats['baseline'][.7]['gt']:
        raise RuntimeError('Incomplete or empty validation')

    results = {name: ap_values(value, eval_utils) for name, value in stats.items()}
    for name, value in stats.items():
        folder = output / name
        folder.mkdir()
        eval_utils.eval_final_results(copy.deepcopy(value), str(folder), False)

    denominator = max(1, multi_source)
    separation = {
        key: value / denominator for key, value in separation_sums.items()
    }
    separation['multi_source_frames'] = int(multi_source)
    separation['frames'] = int(count)

    return {
        'frames': count,
        'results': results,
        'diagnostics_vs_shared': harms_shared,
        'diagnostics_vs_v3_start': harms_start,
        'shared_tp': int(shared_tp),
        'v3_start_tp': int(start_tp),
        'task_separation': separation,
        'global_sort': False,
    }


def decide(conditions, settings):
    weather = ('fog', 'rain', 'snow')
    gains = {
        name: (conditions[name]['results']['Split']['ap70']
               - conditions[name]['results']['Shared']['ap70'])
        for name in weather
    }
    mean_gain = sum(gains.values()) / len(gains)
    positive = sum(value > 0 for value in gains.values())

    collapse_vs_shared = {
        name: (conditions[name]['results']['Split-Collapse']['ap70']
               - conditions[name]['results']['Shared']['ap70'])
        for name in weather
    }
    split_vs_collapse = {
        name: (conditions[name]['results']['Split']['ap70']
               - conditions[name]['results']['Split-Collapse']['ap70'])
        for name in weather
    }
    swap_vs_split = {
        name: (conditions[name]['results']['Split-Swap']['ap70']
               - conditions[name]['results']['Split']['ap70'])
        for name in weather
    }

    clean = conditions['clean']['results']
    checks = {
        'weather_gain': mean_gain >= settings['minimum_weather_mean_ap70_gain'],
        'positive_weathers': positive >= settings['minimum_positive_weathers'],
        'clean_ap70': (
            clean['Split']['ap70']
            >= clean['Shared']['ap70'] - settings['clean_ap70_tolerance']
        ),
        'clean_ap70_vs_v3_start': (
            clean['Split']['ap70']
            >= clean['v3_start']['ap70'] - settings['clean_ap70_tolerance']
        ),
    }

    for name, row in conditions.items():
        checks[f'{name}_ap50'] = (
            row['results']['Split']['ap50']
            >= row['results']['Shared']['ap50'] - settings['ap50_tolerance']
        )
        difference = row['diagnostics_vs_shared']['Split']
        denominator = max(1, row['shared_tp'])
        checks[f'{name}_lost_tp'] = (
            difference['lost']
            <= settings['max_lost_tp_fraction'] * denominator
        )
        checks[f'{name}_new_fp'] = (
            difference['new_fp']
            <= settings['max_new_fp_per_reference_tp'] * denominator
        )

    mean_collapse_vs_shared = (
        sum(collapse_vs_shared.values()) / len(collapse_vs_shared)
    )
    mean_split_vs_collapse = (
        sum(split_vs_collapse.values()) / len(split_vs_collapse)
    )
    mean_swap_vs_split = sum(swap_vs_split.values()) / len(swap_vs_split)
    retention = None
    if mean_gain > 0:
        retention = mean_collapse_vs_shared / mean_gain

    return {
        'expand': all(checks.values()),
        'checks': checks,
        'weather_ap70_gains_split_vs_shared': gains,
        'mean_weather_ap70_gain_split_vs_shared': mean_gain,
        'positive_weathers': positive,
        'mechanism_diagnostics': {
            'collapse_vs_shared_weather_ap70_gains': collapse_vs_shared,
            'mean_collapse_vs_shared_ap70_gain': mean_collapse_vs_shared,
            'split_vs_collapse_weather_ap70_gains': split_vs_collapse,
            'mean_split_vs_collapse_ap70_gain': mean_split_vs_collapse,
            'collapse_retained_fraction_of_split_gain': retention,
            'swap_vs_split_weather_ap70_gains': swap_vs_split,
            'mean_swap_vs_split_ap70_gain': mean_swap_vs_split,
            'interpretation': (
                'Primary feasibility is Split versus Shared. Stronger task-specific '
                'evidence requires the trained Split advantage to shrink when its '
                'two task weights are collapsed; Swap is diagnostic only.'
            ),
        },
    }
