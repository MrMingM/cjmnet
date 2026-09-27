"""Post-hoc B0 task-gap benefit audit.

No retraining.  Replays the completed task-split pilot on the exact saved
validation indices and asks whether GT targets helped by Split relative to
Split-Collapse (primary) or Shared (secondary) lie in regions where the two
task routers actually prefer different source mixtures.

GT is used only to define evaluation cohorts and target ROIs.  The resulting
statistics are diagnostic evidence, not a deployable trigger or threshold.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ceif_audit.scoring import ap_values, empty_stats
from gspr_communication.runtime import device, seed_all, sha256, verify_frozen, write_json
from gspr_evidence import runtime as er
from local_fusion_utility_v2.outcomes import detection_outcome
from local_fusion_v3 import runtime as v3rt
from opencood.tools.train_utils import to_device
from opencood.utils import eval_utils
from qa_local_intervention.common import grid_xy, roi_mask

from .model import TaskFusionArm, _route
from .pipeline import selected_loader


METHODS = ('Shared', 'Split', 'Split-Collapse')
COMPARISONS = {
    'split_vs_collapse': ('Split', 'Split-Collapse'),
    'split_vs_shared': ('Split', 'Shared'),
}
EXPANSIONS = (1.0, 1.5)
SCALES = (0, 1)


def _load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _load_arm(run, name, source, target):
    state = torch.load(run / (name + '.pth'), map_location='cpu', weights_only=True)
    expected_mode = name.lower()
    if state.get('mode') != expected_mode:
        raise ValueError(f'{name} checkpoint mode mismatch')
    arm = TaskFusionArm(source, expected_mode).to(target)
    arm.load_state_dict(state['arm'], strict=True)
    if int(state.get('parameter_count', -1)) != arm.parameter_count():
        raise ValueError(f'{name} checkpoint parameter count mismatch')
    return arm.eval()


def _category(primary, reference, target):
    a = target in primary
    b = target in reference
    if a and not b:
        return 'split_only'
    if b and not a:
        return 'reference_only'
    if a and b:
        return 'both'
    return 'neither'


def _weight_maps(weights_a, weights_b):
    """Per-cell source-distribution differences.

    mean_abs matches the spirit of B0 task_gap but is local.
    total_variation = 0.5 * sum_source |p-q| lies in [0,1].
    """
    output = {}
    for scale in SCALES:
        left, right = weights_a[scale], weights_b[scale]
        delta = (left - right).abs()
        output[scale] = {
            'mean_abs': delta.mean(0).squeeze(0),
            'total_variation': (0.5 * delta.sum(0)).squeeze(0),
            'top_disagreement': (
                left.argmax(0) != right.argmax(0)
            ).float().squeeze(0),
        }
    return output


def _roi_stats(metric, corners, lidar_range, expansion):
    values = metric.detach().cpu().numpy()
    mask, fallback = roi_mask(
        grid_xy(values.shape, lidar_range),
        np.asarray(corners, dtype=np.float64),
        float(expansion),
    )
    selected = values[mask]
    if not selected.size or not np.isfinite(selected).all():
        raise RuntimeError('Invalid task-gap ROI values')
    return {
        'mean': float(selected.mean()),
        'max': float(selected.max()),
        'p90': float(np.quantile(selected, .9)),
        'cells': int(selected.size),
        'fallback': bool(fallback),
    }


def target_metrics(maps, corners, lidar_range):
    result = {}
    for expansion in EXPANSIONS:
        label = 'box' if expansion == 1.0 else 'context'
        per_scale = {}
        for scale in SCALES:
            per_scale[str(scale)] = {
                name: _roi_stats(metric, corners, lidar_range, expansion)
                for name, metric in maps[scale].items()
            }
        result[label] = per_scale

        # Primary compact target scores: average scale-0/1 ROI means.
        result[label]['combined'] = {}
        for name in ('mean_abs', 'total_variation', 'top_disagreement'):
            means = [per_scale[str(scale)][name]['mean'] for scale in SCALES]
            maxima = [per_scale[str(scale)][name]['max'] for scale in SCALES]
            result[label]['combined'][name + '_mean'] = float(np.mean(means))
            result[label]['combined'][name + '_max'] = float(np.max(maxima))
        result[label]['combined']['any_top_disagreement'] = bool(
            result[label]['combined']['top_disagreement_max'] > 0
        )
    return result


def _metric(row, path):
    value = row['task_metrics']
    for key in path:
        value = value[key]
    return float(value)


def _quantiles(values):
    array = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    if not len(array):
        return {'n': 0, 'mean': None, 'p10': None, 'p25': None,
                'median': None, 'p75': None, 'p90': None}
    q = np.quantile(array, [.1, .25, .5, .75, .9])
    return {
        'n': int(len(array)),
        'mean': float(array.mean()),
        'p10': float(q[0]),
        'p25': float(q[1]),
        'median': float(q[2]),
        'p75': float(q[3]),
        'p90': float(q[4]),
    }


def _auc(positive, negative):
    """Pairwise AUROC = P(pos > neg) + .5 P(tie), diagnostic only."""
    p = np.asarray(positive, dtype=np.float64)
    n = np.asarray(negative, dtype=np.float64)
    p = p[np.isfinite(p)]
    n = n[np.isfinite(n)]
    if not len(p) or not len(n):
        return None
    # Target counts are small enough for an exact pairwise diagnostic.
    relation = p[:, None] - n[None, :]
    return float(((relation > 0).sum() + 0.5 * (relation == 0).sum()) /
                 relation.size)


def _enrichment(rows, comparison, path):
    relevant = [row for row in rows if row['comparison'][comparison] is not None]
    if not relevant:
        return None
    values = np.asarray([_metric(row, path) for row in relevant], dtype=np.float64)
    threshold = float(np.quantile(values, .75))
    high = values >= threshold
    overall_rate = float(high.mean())
    positive = np.asarray([
        row['comparison'][comparison] == 'split_only' for row in relevant
    ], dtype=bool)
    if not positive.any():
        return {
            'threshold_p75': threshold,
            'overall_high_rate': overall_rate,
            'split_only_n': 0,
            'split_only_high_rate': None,
            'enrichment': None,
        }
    positive_rate = float(high[positive].mean())
    return {
        'threshold_p75': threshold,
        'overall_high_rate': overall_rate,
        'split_only_n': int(positive.sum()),
        'split_only_high_rate': positive_rate,
        'enrichment': (
            positive_rate / overall_rate if overall_rate > 0 else None
        ),
    }


def summarize(rows):
    paths = {
        'box_total_variation_mean':
            ('box', 'combined', 'total_variation_mean'),
        'box_mean_abs_mean':
            ('box', 'combined', 'mean_abs_mean'),
        'box_top_disagreement_mean':
            ('box', 'combined', 'top_disagreement_mean'),
        'context_total_variation_mean':
            ('context', 'combined', 'total_variation_mean'),
        'context_top_disagreement_mean':
            ('context', 'combined', 'top_disagreement_mean'),
    }
    report = {}
    for comparison in COMPARISONS:
        categories = {}
        for category in ('split_only', 'reference_only', 'both', 'neither'):
            subset = [
                row for row in rows
                if row['comparison'][comparison] == category
            ]
            categories[category] = {
                'targets': len(subset),
                'frames': len({row['sample_index'] for row in subset}),
                'any_box_top_disagreement_rate': (
                    float(np.mean([
                        row['task_metrics']['box']['combined']
                           ['any_top_disagreement']
                        for row in subset
                    ])) if subset else None
                ),
                'metrics': {
                    name: _quantiles([_metric(row, path) for row in subset])
                    for name, path in paths.items()
                },
            }

        positives = [
            row for row in rows
            if row['comparison'][comparison] == 'split_only'
        ]
        stable = [
            row for row in rows
            if row['comparison'][comparison] == 'both'
        ]
        all_other = [
            row for row in rows
            if row['comparison'][comparison] != 'split_only'
        ]
        aucs = {}
        for name, path in paths.items():
            pos = [_metric(row, path) for row in positives]
            aucs[name] = {
                'split_only_vs_both': _auc(
                    pos, [_metric(row, path) for row in stable]),
                'split_only_vs_all_other': _auc(
                    pos, [_metric(row, path) for row in all_other]),
            }

        report[comparison] = {
            'categories': categories,
            'auc_diagnostics': aucs,
            'top_quartile_enrichment': {
                name: _enrichment(rows, comparison, path)
                for name, path in paths.items()
            },
            'interpretation_boundary': (
                'These are GT-localized post-hoc associations. AUROC/enrichment '
                'do not define a deployable gate and are not significance tests.'
            ),
        }
    return report


def _assert_reproduction(weather, observed, saved, tolerance):
    for name in METHODS:
        for metric in ('ap30', 'ap50', 'ap70'):
            left = float(observed[name][metric])
            right = float(saved[name][metric])
            if abs(left - right) > tolerance:
                raise RuntimeError(
                    f'{weather} reproduction mismatch {name}.{metric}: '
                    f'{left} versus saved {right}')


def evaluate_weather(model, arms, dataset, loader, indices, branch, target,
                     lidar_range, saved, folder):
    stats = {name: empty_stats() for name in METHODS}
    rows = []
    count = 0

    with torch.no_grad(), (folder / 'targets.jsonl').open(
            'w', encoding='utf-8') as stream:
        for batch in loader:
            batch = to_device(batch, target)
            context = v3rt.context(
                model, batch['ego'], branch, verify=count == 0)

            shared, _ = arms['Shared'].predict(
                model.engine.base, context['levels'])
            split, _ = arms['Split'].predict(
                model.engine.base, context['levels'])
            collapse, _ = arms['Split'].predict(
                model.engine.base, context['levels'], collapse=True)

            predictions = {
                'Shared': shared,
                'Split': split,
                'Split-Collapse': collapse,
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

            matched = {
                name: detection_outcome(*post[name], .7)[0]
                for name in METHODS
            }
            gt = post['Split'][2].detach().cpu().numpy()
            _, weights_a, _ = _route(
                arms['Split'].router_a, context['levels'])
            _, weights_b, _ = _route(
                arms['Split'].router_b, context['levels'])
            maps = _weight_maps(weights_a, weights_b)

            sample_index = int(
                batch['ego']['communication_sample_index'][0])
            for target_index, corners in enumerate(gt):
                metrics = target_metrics(maps, corners, lidar_range)
                center = np.asarray(corners)[:, :2].mean(0)
                row = {
                    'sample_index': sample_index,
                    'target_index': int(target_index),
                    'distance_xy': float(np.linalg.norm(center)),
                    'comparison': {
                        label: _category(
                            matched[primary], matched[reference],
                            target_index)
                        for label, (primary, reference)
                        in COMPARISONS.items()
                    },
                    'matched': {
                        name: bool(target_index in matched[name])
                        for name in METHODS
                    },
                    'task_metrics': metrics,
                }
                rows.append(row)
                stream.write(json.dumps(
                    row, ensure_ascii=False, allow_nan=False) + '\n')

            count += 1
            if count == 1 or count % 20 == 0:
                print(f'task-gap audit {count}/{len(indices)}', flush=True)

    if count != len(indices) or not count:
        raise RuntimeError('Incomplete task-gap audit')
    results = {name: ap_values(value, eval_utils)
               for name, value in stats.items()}
    _assert_reproduction(
        branch if branch == 'clean' else folder.name,
        results, saved['results'], 1e-6)
    return {
        'frames': count,
        'targets': len(rows),
        'results': results,
        'association': summarize(rows),
        'fallback_counts': {
            label: {
                str(scale): sum(
                    row['task_metrics'][label][str(scale)]
                       ['total_variation']['fallback']
                    for row in rows)
                for scale in SCALES
            }
            for label in ('box', 'context')
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--run', required=True,
        help='Completed local_fusion_task_split_pilot run')
    parser.add_argument(
        '--v3-config', default='local_fusion_v3/experiment.yaml')
    parser.add_argument('--frontend-config', required=True)
    parser.add_argument('--frontend-checkpoint', required=True)
    parser.add_argument('--v3-checkpoint', required=True)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    verify_frozen()
    run = Path(args.run).resolve()
    protocol = _load_json(run / 'protocol.json')
    saved = _load_json(run / 'decision_results.json')
    output = (
        Path(args.output).resolve()
        if args.output else run / 'task_gap_audit'
    )
    output.mkdir(parents=True, exist_ok=False)

    # Refuse to reinterpret a run with drifted trained core code.
    for name in (
        'local_fusion_task_split_pilot/model.py',
        'local_fusion_task_split_pilot/pipeline.py',
        'local_fusion_task_split_pilot/evaluate.py',
        'local_fusion_task_split_pilot/experiment.yaml',
    ):
        expected = protocol['source_hashes'].get(name)
        if expected is None or sha256(Path(name)) != expected:
            raise ValueError('Task-split source drift since training: ' + name)

    options, hypes = er.load_config(
        args.v3_config, args.frontend_config)
    target = device()
    model, frontend_digest = er.load_model(
        hypes, options, args.frontend_checkpoint, target)
    model.requires_grad_(False).eval()
    if frontend_digest != protocol['frontend_sha256']:
        raise ValueError('Frontend checkpoint differs from completed B0 run')

    v3_contract = v3rt.contract(
        options, args.frontend_config, frontend_digest)
    if v3_contract != protocol['v3_contract']:
        raise ValueError('v3 config/source contract differs from completed B0 run')
    if sha256(args.v3_checkpoint) != protocol['v3_checkpoint_sha256']:
        raise ValueError('v3 checkpoint differs from completed B0 run')

    source, checkpoint = v3rt.load(
        args.v3_checkpoint, v3_contract, target)
    if checkpoint['variant'] != 'residual':
        raise ValueError('Expected residual v3 checkpoint')
    arms = {
        name: _load_arm(run, name, source, target)
        for name in ('Shared', 'Split')
    }
    del source

    indices = [int(x) for x in protocol['validation_indices']]
    lidar_range = hypes['postprocess']['anchor_args']['cav_lidar_range']
    if len(lidar_range) != 6:
        raise ValueError('Invalid lidar range')

    conditions = {}
    for weather in ('clean', 'fog', 'rain', 'snow'):
        seed_all(int(protocol['pilot']['seed']) + 1)
        dataset, loader = selected_loader(
            hypes, options, 'validation', weather, indices)
        folder = output / weather
        folder.mkdir(parents=True)
        conditions[weather] = evaluate_weather(
            model,
            arms,
            dataset,
            loader,
            indices,
            'clean' if weather == 'clean' else 'weather',
            target,
            lidar_range,
            saved['conditions'][weather],
            folder,
        )
        write_json(folder / 'summary.json', conditions[weather])
        del dataset, loader

    # Aggregate weather rows without changing their per-condition identities.
    aggregate_rows = []
    for weather in ('fog', 'rain', 'snow'):
        with (output / weather / 'targets.jsonl').open(
                encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                row['weather'] = weather
                aggregate_rows.append(row)

    report = {
        'scope': (
            'post-hoc GT-localized task-gap benefit audit on exact completed '
            'B0 validation indices; no retraining; no OPV2V-W'
        ),
        'question': (
            'Are GT targets helped by task-specific routing concentrated in '
            'regions where classification/regression source weights differ?'
        ),
        'metrics': {
            'mean_abs': (
                'mean absolute per-source weight difference; local analogue '
                'of the B0 task-gap statistic'
            ),
            'total_variation': (
                '0.5 * sum_source |w_cls-w_reg|; 0 means identical source '
                'mixtures, 1 means maximally different'
            ),
            'top_disagreement': (
                'fraction of target ROI cells where classification and '
                'regression choose different top-weight sources'
            ),
            'box': 'exact oriented GT footprint on each BEV scale',
            'context': '1.5x expanded oriented GT footprint',
        },
        'primary_comparison': (
            'Split versus Split-Collapse isolates whether retaining trained '
            'task-specific weights helps relative to collapsing them.'
        ),
        'secondary_comparison': (
            'Split versus Shared checks whether the same local conflict signal '
            'also aligns with the equal-budget trained control.'
        ),
        'conditions': conditions,
        'weather_aggregate': summarize(aggregate_rows),
        'boundary': [
            'GT is used to define target cohorts and ROIs, so this cannot be used as a deployable trigger.',
            'Association does not prove that task-gap caused the recovery.',
            'This audit should decide whether a local residual task-split B1 is motivated, not tune a threshold on validation.',
        ],
    }
    write_json(output / 'task_gap_audit.json', report)
    verify_frozen()
    print('TASK-GAP AUDIT COMPLETE:', output / 'task_gap_audit.json', flush=True)


if __name__ == '__main__':
    main()
