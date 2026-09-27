"""Paired validation in the repository's frame-order, BEV-IoU AP protocol."""
import copy
import json

import torch

from ceif_audit.scoring import ap_values, empty_stats
from gspr_evidence import runtime as er
from local_fusion_utility_v2.outcomes import compare_predictions, detection_outcome
from local_fusion_v3 import runtime as v3rt
from opencood.tools.train_utils import to_device
from opencood.utils import eval_utils


def evaluate_condition(model, arms, dataset, loader, indices, branch, target, output):
    names = ('baseline', 'F', 'F+D')
    stats = {name: empty_stats() for name in names}
    harms = {name: dict(recovered=0, lost=0, new_fp=0) for name in ('F', 'F+D')}
    harms_baseline = {name: dict(recovered=0, lost=0, new_fp=0) for name in ('F', 'F+D')}
    baseline_tp = F_tp = 0
    count = 0
    with torch.no_grad(), (output / 'frames.jsonl').open('w', encoding='utf-8') as stream:
        for batch in loader:
            batch = to_device(batch, target)
            ctx = v3rt.context(model, batch['ego'], branch, verify=count == 0)
            predictions = {'baseline': ctx['baseline_prediction']}
            for name, arm in arms.items():
                predictions[name], _ = arm.predict(model.engine.base, ctx['levels'])
            post = {name: dataset.post_process(batch, {'ego': prediction})
                    for name, prediction in predictions.items()}
            for name in names:
                boxes, scores, gt = post[name]
                for threshold in stats[name]:
                    eval_utils.caluclate_tp_fp(boxes, scores, gt, stats[name], threshold)
            matched, _ = detection_outcome(*post['baseline'], .7)
            baseline_tp += len(matched)
            matched_F, _ = detection_outcome(*post['F'], .7)
            F_tp += len(matched_F)
            for name in harms:
                difference = compare_predictions(post['F'], post[name], .7, .7)
                for key in harms[name]:
                    harms[name][key] += difference[key]
                difference = compare_predictions(post['baseline'], post[name], .7, .7)
                for key in harms_baseline[name]:
                    harms_baseline[name][key] += difference[key]
            stream.write(json.dumps({'sample_index': int(batch['ego']['communication_sample_index'][0])}) + '\n')
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
    return dict(frames=count, results=results, diagnostics_vs_F=harms,
                diagnostics_vs_baseline=harms_baseline,
                baseline_tp=baseline_tp, F_tp=F_tp, global_sort=False)


def decide(conditions, settings):
    """Predeclared expansion gate, comparing F+D directly with F."""
    gains = {weather: conditions[weather]['results']['F+D']['ap70']
             - conditions[weather]['results']['F']['ap70']
             for weather in ('fog', 'rain', 'snow')}
    mean_gain = sum(gains.values()) / len(gains)
    positive = sum(gain > 0 for gain in gains.values())
    clean = conditions['clean']['results']
    checks = {
        'weather_gain': mean_gain >= settings['minimum_weather_mean_ap70_gain'],
        'positive_weathers': positive >= settings['minimum_positive_weathers'],
        'clean_ap70': clean['F+D']['ap70'] >= clean['F']['ap70'] - settings['clean_ap70_tolerance'],
        'clean_ap70_vs_baseline': clean['F+D']['ap70'] >= clean['baseline']['ap70'] - settings['clean_ap70_tolerance'],
    }
    for weather, row in conditions.items():
        checks[f'{weather}_ap50'] = (row['results']['F+D']['ap50'] >=
                                    row['results']['F']['ap50'] - settings['ap50_tolerance'])
        checks[f'{weather}_ap50_vs_baseline'] = (row['results']['F+D']['ap50'] >=
                                                row['results']['baseline']['ap50'] - settings['ap50_tolerance'])
        difference = row['diagnostics_vs_F']['F+D']
        # The reference TP count comes from F's actual post-NMS predictions.
        denominator = max(1, row['F_tp'])
        checks[f'{weather}_lost_tp'] = difference['lost'] <= settings['max_lost_tp_fraction'] * denominator
        checks[f'{weather}_new_fp'] = difference['new_fp'] <= settings['max_new_fp_per_reference_tp'] * denominator
        baseline_harm = row['diagnostics_vs_baseline']['F+D']
        baseline_denominator = max(1, row['baseline_tp'])
        checks[f'{weather}_lost_tp_vs_baseline'] = baseline_harm['lost'] <= settings['max_lost_tp_fraction'] * baseline_denominator
        checks[f'{weather}_new_fp_vs_baseline'] = baseline_harm['new_fp'] <= settings['max_new_fp_per_reference_tp'] * baseline_denominator
    return dict(expand=all(checks.values()), checks=checks,
                weather_ap70_gains=gains, mean_weather_ap70_gain=mean_gain,
                positive_weathers=positive)
