"""Target-wise task/source Oracle death test for direction B.

This module deliberately gives direction B much more information than a
deployable method can have.  It uses GT to choose a local source action per
target and reports whether classification/regression source separation has
enough AP70 headroom to justify further method development.

Two Oracles start from the trained B0 Shared prediction:

Oracle-Same
    In a GT-local ROI, classification and regression must use the SAME
    candidate source representation.

Oracle-Task
    In the same ROI, classification and regression may independently choose
    DIFFERENT candidate source representations.

Candidate pool (computed once per frame):
    KEEP_SHARED
    single:i  - frozen detector output from aligned CAV i alone
    query:i   - frozen detector output after attention fusion using CAV i as
                query while retaining all available source features

For each target the Oracle may also choose ROI expansion 1.0 or 1.5.  The
selection is greedy and GT-aware: targets missed by Shared are processed first;
an action is preferred only through true post-NMS frame outcomes, with matched
GT count first, then preservation of current GTs, FP harm, focal detection/
score/IoU, and finally simpler actions.  KEEP_SHARED is always legal.

This is intentionally a strong diagnostic, not a deployable method and not a
mathematical proof of the absolute optimum over arbitrary continuous source
weights.  If even this GT-aware source-choice Oracle fails the pre-registered
1-2 AP-point bar, the current direction B should be stopped.
"""
import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml

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
from local_fusion_task_split_pilot.model import TaskFusionArm
from local_fusion_task_split_pilot.pipeline import selected_loader
from local_fusion_utility_v2.fusion import (
    attention_fusion,
    predict_each_source,
    predict_from_levels,
)
from local_fusion_utility_v2.outcomes import (
    count_new_false_positives,
    detection_outcome,
)
from local_fusion_v3 import runtime as v3rt
from opencood.tools.train_utils import to_device
from opencood.utils import common_utils as cu
from opencood.utils import eval_utils
from qa_local_intervention.operators import masks_for_frame


METHODS = ('Shared', 'Oracle-Same', 'Oracle-Task')
KEEP = 'KEEP_SHARED'


def settings(path):
    value = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if type(value.get('seed')) is not int:
        raise ValueError('seed must be an integer')
    if not 0 < float(value.get('target_iou', 0)) <= 1:
        raise ValueError('target_iou must be in (0,1]')
    expansions = tuple(float(x) for x in value.get('roi_expansions', ()))
    if not expansions or any(x < 1 for x in expansions):
        raise ValueError('roi_expansions must contain values >= 1')
    if len(set(expansions)) != len(expansions):
        raise ValueError('roi_expansions must be unique')
    value['roi_expansions'] = expansions
    families = tuple(value.get('candidate_families', ()))
    if not families or any(x not in ('single', 'query') for x in families):
        raise ValueError('candidate_families must use single/query')
    if len(set(families)) != len(families):
        raise ValueError('candidate_families must be unique')
    value['candidate_families'] = families
    for key in (
        'minimum_mean_task_vs_shared',
        'per_weather_task_vs_shared',
        'minimum_mean_task_vs_same',
    ):
        if not 0 <= float(value.get(key, -1)) <= 1:
            raise ValueError(key + ' must be in [0,1]')
    n = value.get('minimum_weathers_ge_1pp')
    if type(n) is not int or not 0 <= n <= 3:
        raise ValueError('minimum_weathers_ge_1pp must be 0..3')
    return value


def _load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _load_b0_arm(run, filename, source, target, expected_mode, start_digest):
    state = torch.load(run / filename, map_location='cpu', weights_only=True)
    if state.get('mode') != expected_mode:
        raise ValueError(filename + ' mode mismatch')
    if state.get('start_checkpoint_sha256') != start_digest:
        raise ValueError(filename + ' starts from a different v3 checkpoint')
    arm = TaskFusionArm(source, expected_mode).to(target)
    arm.load_state_dict(state['arm'], strict=True)
    if int(state.get('parameter_count', -1)) != arm.parameter_count():
        raise ValueError(filename + ' parameter count mismatch')
    return arm.requires_grad_(False).eval()


def _verify_b0_source_contract(protocol):
    recorded = protocol.get('source_hashes') or {}
    for name in (
        'local_fusion_task_split_pilot/model.py',
        'local_fusion_task_split_pilot/pipeline.py',
        'local_fusion_task_split_pilot/evaluate.py',
        'local_fusion_task_split_pilot/experiment.yaml',
    ):
        expected = recorded.get(name)
        if expected is None:
            raise ValueError('B0 protocol lacks source hash: ' + name)
        if not (ROOT / name).is_file() or sha256(ROOT / name) != expected:
            raise ValueError('B0 source drift since training: ' + name)


def _slice_prediction(prediction, index):
    return {
        'psm': prediction['psm'][index:index+1],
        'rm': prediction['rm'][index:index+1],
    }


def build_candidate_pool(base, levels, shared_prediction, families):
    """Inference tensors available before GT selection."""
    if not levels or any(len(x) != len(levels[0]) for x in levels):
        raise ValueError('Invalid per-source levels')
    count = len(levels[0])
    pool = {KEEP: shared_prediction}

    if 'single' in families:
        local = predict_each_source(base, levels)
        if len(local['psm']) != count or len(local['rm']) != count:
            raise RuntimeError('Per-source detector batch differs from source count')
        for index in range(count):
            pool[f'single:{index}'] = _slice_prediction(local, index)

    if 'query' in families:
        for index in range(count):
            fused = [
                attention_fusion(level, query_index=index)[0]
                for level in levels
            ]
            pool[f'query:{index}'] = predict_from_levels(base, fused)

    return pool


def compose_prediction(current, pool, mask, cls_name, reg_name):
    """Replace all anchor channels inside one output-space GT ROI."""
    m = torch.as_tensor(
        mask, device=current['psm'].device, dtype=torch.bool)[None, None]
    if tuple(m.shape[-2:]) != tuple(current['psm'].shape[-2:]):
        raise ValueError('ROI/output shape mismatch')
    output = {
        'psm': current['psm'],
        'rm': current['rm'],
    }
    if cls_name != KEEP:
        output['psm'] = torch.where(m, pool[cls_name]['psm'], current['psm'])
    if reg_name != KEEP:
        output['rm'] = torch.where(m, pool[reg_name]['rm'], current['rm'])
    return output


def _focal_quality(post, target_index, threshold):
    boxes, scores, gt = post
    if boxes is None or scores is None or not len(boxes):
        return {
            'detected': False,
            'score': -1.0,
            'iou': 0.0,
        }
    box_array = boxes.detach().cpu().numpy()
    gt_array = gt.detach().cpu().numpy()
    if not 0 <= target_index < len(gt_array):
        raise ValueError('Invalid target index')
    target = list(cu.convert_format(gt_array[target_index:target_index+1]))
    polygons = list(cu.convert_format(box_array))
    ious = np.asarray([
        float(cu.compute_iou(poly, target)[0]) if target else 0.0
        for poly in polygons
    ], dtype=np.float64)
    score_array = scores.detach().cpu().numpy().reshape(-1)
    good = ious >= float(threshold)
    if not good.any():
        return {
            'detected': False,
            'score': float(score_array[ious.argmax()]) if len(ious) else -1.0,
            'iou': float(ious.max()) if len(ious) else 0.0,
        }
    return {
        'detected': True,
        'score': float(score_array[good].max()),
        'iou': float(ious[good].max()),
    }


def _state(post, target_index, threshold):
    matched, fp = detection_outcome(*post, threshold)
    focal = _focal_quality(post, target_index, threshold)
    return {
        'matched': matched,
        'fp': fp,
        'focal': focal,
    }


def _action_key(current_state, candidate_state, cls_name, reg_name):
    current_match = current_state['matched']
    match = candidate_state['matched']
    lost = len(current_match - match)
    recovered = len(match - current_match)
    new_fp = count_new_false_positives(
        candidate_state['fp'], current_state['fp'], .7)
    same = cls_name == reg_name and cls_name != KEEP
    keep = cls_name == KEEP and reg_name == KEEP
    simplicity = 2 if keep else (1 if same else 0)
    return (
        len(match),
        -lost,
        -new_fp,
        int(candidate_state['focal']['detected']),
        recovered,
        -len(candidate_state['fp']),
        candidate_state['focal']['score'],
        candidate_state['focal']['iou'],
        simplicity,
    )


def _target_order(shared_post, threshold):
    matched, _ = detection_outcome(*shared_post, threshold)
    gt_count = int(len(shared_post[2]))
    return (
        [index for index in range(gt_count) if index not in matched]
        + [index for index in range(gt_count) if index in matched]
    )


def _actions(pool_names, expansions, mode):
    yield None, KEEP, KEEP
    if mode == 'same':
        for expansion in expansions:
            for name in pool_names:
                if name != KEEP:
                    yield expansion, name, name
        return
    if mode != 'task':
        raise ValueError('mode must be same or task')
    for expansion in expansions:
        for cls_name in pool_names:
            for reg_name in pool_names:
                if cls_name == KEEP and reg_name == KEEP:
                    continue
                yield expansion, cls_name, reg_name


def run_oracle(dataset, batch, base, levels, shared_prediction, pool,
               lidar_range, expansions, mode, threshold):
    """Greedy GT-aware target oracle.  KEEP is always legal."""
    current = {
        'psm': shared_prediction['psm'].clone(),
        'rm': shared_prediction['rm'].clone(),
    }
    current_post = dataset.post_process(batch, {'ego': current})
    shared_matched, _ = detection_outcome(*current_post, threshold)
    order = _target_order(current_post, threshold)
    source_names = tuple(sorted(pool.keys()))
    records = []

    gt_np = current_post[2].detach().cpu().numpy()
    masks = {}
    for target_index in order:
        masks[target_index] = {}
        for expansion in expansions:
            head_mask, _, stats = masks_for_frame(
                batch['ego']['anchor_box'],
                gt_np[target_index],
                levels,
                lidar_range,
                expansion,
            )
            masks[target_index][expansion] = (head_mask, stats)

    for target_index in order:
        current_state = _state(current_post, target_index, threshold)
        best = {
            'prediction': current,
            'post': current_post,
            'state': current_state,
            'key': _action_key(current_state, current_state, KEEP, KEEP),
            'expansion': None,
            'cls_source': KEEP,
            'reg_source': KEEP,
            'mask_stats': None,
        }

        for expansion, cls_name, reg_name in _actions(
                source_names, expansions, mode):
            if expansion is None:
                continue
            mask, mask_stats = masks[target_index][expansion]
            prediction = compose_prediction(
                current, pool, mask, cls_name, reg_name)
            post = dataset.post_process(batch, {'ego': prediction})
            candidate_state = _state(post, target_index, threshold)
            key = _action_key(
                current_state, candidate_state, cls_name, reg_name)
            if key > best['key']:
                best = {
                    'prediction': prediction,
                    'post': post,
                    'state': candidate_state,
                    'key': key,
                    'expansion': float(expansion),
                    'cls_source': cls_name,
                    'reg_source': reg_name,
                    'mask_stats': mask_stats,
                }

        before = current_state
        after = best['state']
        current = best['prediction']
        current_post = best['post']
        records.append({
            'target_index': int(target_index),
            'was_shared_missed': bool(target_index not in shared_matched),
            'expansion': best['expansion'],
            'cls_source': best['cls_source'],
            'reg_source': best['reg_source'],
            'changed': not (
                best['cls_source'] == KEEP and best['reg_source'] == KEEP),
            'task_separated': bool(
                best['cls_source'] != best['reg_source']),
            'matched_before': len(before['matched']),
            'matched_after': len(after['matched']),
            'focal_before': before['focal'],
            'focal_after': after['focal'],
            'fp_before': len(before['fp']),
            'fp_after': len(after['fp']),
            'mask_stats': best['mask_stats'],
        })

    return current, current_post, records


def _zero_delta():
    return {
        'recovered': 0,
        'lost': 0,
        'new_fp': 0,
    }


def _compare(base_post, action_post, threshold):
    base_match, base_fp = detection_outcome(*base_post, threshold)
    act_match, act_fp = detection_outcome(*action_post, threshold)
    return {
        'recovered': int(len(act_match - base_match)),
        'lost': int(len(base_match - act_match)),
        'new_fp': int(count_new_false_positives(act_fp, base_fp, .7)),
    }


def _add_delta(total, row):
    for key in total:
        total[key] += int(row[key])


def evaluate_condition(model, shared_arm, dataset, loader, indices, branch,
                       target, spec, lidar_range, folder, saved_b0):
    stats = {name: empty_stats() for name in METHODS}
    deltas = {
        'Oracle-Same': _zero_delta(),
        'Oracle-Task': _zero_delta(),
    }
    selected = {
        'Oracle-Same': Counter(),
        'Oracle-Task': Counter(),
    }
    target_totals = {
        'Oracle-Same': {
            'targets': 0, 'changed': 0, 'task_separated': 0,
        },
        'Oracle-Task': {
            'targets': 0, 'changed': 0, 'task_separated': 0,
        },
    }
    frames = 0

    with torch.no_grad(), (folder / 'frames.jsonl').open(
            'w', encoding='utf-8') as stream:
        for batch in loader:
            batch = to_device(batch, target)
            context = v3rt.context(
                model, batch['ego'], branch, verify=frames == 0)
            shared_prediction, _ = shared_arm.predict(
                model.engine.base, context['levels'])
            pool = build_candidate_pool(
                model.engine.base,
                context['levels'],
                shared_prediction,
                spec['candidate_families'],
            )
            shared_post = dataset.post_process(
                batch, {'ego': shared_prediction})

            same_prediction, same_post, same_records = run_oracle(
                dataset,
                batch,
                model.engine.base,
                context['levels'],
                shared_prediction,
                pool,
                lidar_range,
                spec['roi_expansions'],
                'same',
                spec['target_iou'],
            )
            task_prediction, task_post, task_records = run_oracle(
                dataset,
                batch,
                model.engine.base,
                context['levels'],
                shared_prediction,
                pool,
                lidar_range,
                spec['roi_expansions'],
                'task',
                spec['target_iou'],
            )

            posts = {
                'Shared': shared_post,
                'Oracle-Same': same_post,
                'Oracle-Task': task_post,
            }
            for name, post in posts.items():
                for threshold in stats[name]:
                    eval_utils.caluclate_tp_fp(
                        *post, stats[name], threshold)

            for name, post, records in (
                ('Oracle-Same', same_post, same_records),
                ('Oracle-Task', task_post, task_records),
            ):
                _add_delta(
                    deltas[name],
                    _compare(shared_post, post, spec['target_iou']),
                )
                totals = target_totals[name]
                totals['targets'] += len(records)
                totals['changed'] += sum(x['changed'] for x in records)
                totals['task_separated'] += sum(
                    x['task_separated'] for x in records)
                for row in records:
                    if row['changed']:
                        key = (
                            f"{row['cls_source']}|{row['reg_source']}"
                            f"|r={row['expansion']}"
                        )
                        selected[name][key] += 1

            stream.write(json.dumps({
                'sample_index': int(
                    batch['ego']['communication_sample_index'][0]),
                'source_count': int(len(context['levels'][0])),
                'candidate_pool': sorted(pool),
                'same': same_records,
                'task': task_records,
            }, ensure_ascii=False, allow_nan=False) + '\n')

            frames += 1
            if frames == 1 or frames % 10 == 0:
                print(
                    f'task-source oracle {folder.name}: '
                    f'{frames}/{len(indices)}',
                    flush=True,
                )

    if frames != len(indices) or not frames:
        raise RuntimeError('Incomplete task/source oracle')

    results = {
        name: ap_values(value, eval_utils)
        for name, value in stats.items()
    }

    # The B0 Shared branch must replay exactly before interpreting the Oracle.
    saved = saved_b0['results']['Shared']
    for metric in ('ap30', 'ap50', 'ap70'):
        if abs(
            float(results['Shared'][metric]) - float(saved[metric])
        ) > 1e-6:
            raise RuntimeError(
                f'Shared reproduction mismatch {folder.name}.{metric}: '
                f'{results["Shared"][metric]} vs {saved[metric]}')

    return {
        'frames': frames,
        'results': results,
        'diagnostics_vs_shared': deltas,
        'target_action_counts': target_totals,
        'selected_actions': {
            name: dict(counter.most_common())
            for name, counter in selected.items()
        },
        'candidate_families': list(spec['candidate_families']),
        'roi_expansions': list(spec['roi_expansions']),
        'global_sort': False,
    }


def decide(conditions, spec):
    weathers = ('fog', 'rain', 'snow')
    task_vs_shared = {
        weather: (
            conditions[weather]['results']['Oracle-Task']['ap70']
            - conditions[weather]['results']['Shared']['ap70']
        )
        for weather in weathers
    }
    same_vs_shared = {
        weather: (
            conditions[weather]['results']['Oracle-Same']['ap70']
            - conditions[weather]['results']['Shared']['ap70']
        )
        for weather in weathers
    }
    task_vs_same = {
        weather: (
            conditions[weather]['results']['Oracle-Task']['ap70']
            - conditions[weather]['results']['Oracle-Same']['ap70']
        )
        for weather in weathers
    }
    mean_task_shared = float(np.mean(list(task_vs_shared.values())))
    mean_same_shared = float(np.mean(list(same_vs_shared.values())))
    mean_task_same = float(np.mean(list(task_vs_same.values())))
    weathers_ge_1pp = sum(
        value >= float(spec['per_weather_task_vs_shared'])
        for value in task_vs_shared.values()
    )
    checks = {
        'mean_task_vs_shared_at_least_1p5pp': (
            mean_task_shared
            >= float(spec['minimum_mean_task_vs_shared'])
        ),
        'at_least_two_weathers_ge_1pp': (
            weathers_ge_1pp
            >= int(spec['minimum_weathers_ge_1pp'])
        ),
        'mean_task_vs_same_at_least_0p5pp': (
            mean_task_same
            >= float(spec['minimum_mean_task_vs_same'])
        ),
    }
    survive = all(checks.values())
    return {
        'direction_b_survives': bool(survive),
        'direction_b_kill': bool(not survive),
        'checks': checks,
        'task_vs_shared_ap70_gains': task_vs_shared,
        'same_vs_shared_ap70_gains': same_vs_shared,
        'task_vs_same_ap70_gains': task_vs_same,
        'mean_task_vs_shared_ap70_gain': mean_task_shared,
        'mean_same_vs_shared_ap70_gain': mean_same_shared,
        'mean_task_vs_same_ap70_gain': mean_task_same,
        'weathers_task_vs_shared_ge_1pp': int(weathers_ge_1pp),
        'thresholds': {
            'minimum_mean_task_vs_shared': float(
                spec['minimum_mean_task_vs_shared']),
            'minimum_weathers_ge_1pp': int(
                spec['minimum_weathers_ge_1pp']),
            'per_weather_task_vs_shared': float(
                spec['per_weather_task_vs_shared']),
            'minimum_mean_task_vs_same': float(
                spec['minimum_mean_task_vs_same']),
        },
        'interpretation': (
            'Survival requires a journal-scale Oracle ceiling: >=1.5 pp mean '
            'weather AP70 over Shared, >=1 pp in at least two weather '
            'conditions, and >=0.5 pp mean extra AP70 from allowing '
            'classification/regression to choose different candidates versus '
            'the Same-Source Oracle. Failure means stop the current direction '
            'B under the pre-registered publication bar.'
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='local_fusion_task_source_oracle/experiment.yaml')
    parser.add_argument(
        '--v3-config', default='local_fusion_v3/experiment.yaml')
    parser.add_argument('--frontend-config', required=True)
    parser.add_argument('--frontend-checkpoint', required=True)
    parser.add_argument('--v3-checkpoint', required=True)
    parser.add_argument('--b0-run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    verify_frozen()
    spec = settings(args.config)
    b0_run = Path(args.b0_run).resolve()
    protocol = _load_json(b0_run / 'protocol.json')
    saved_b0 = _load_json(b0_run / 'decision_results.json')
    _verify_b0_source_contract(protocol)

    options, hypes = er.load_config(
        args.v3_config, args.frontend_config)
    target = device()
    model, frontend_digest = er.load_model(
        hypes, options, args.frontend_checkpoint, target)
    model.requires_grad_(False).eval()
    if protocol.get('frontend_sha256') != frontend_digest:
        raise ValueError('B0 frontend differs')

    contract = v3rt.contract(
        options, args.frontend_config, frontend_digest)
    if protocol.get('v3_contract') != contract:
        raise ValueError('B0 v3 contract differs')
    start_digest = sha256(args.v3_checkpoint)
    if protocol.get('v3_checkpoint_sha256') != start_digest:
        raise ValueError('B0 v3 checkpoint differs')

    source, checkpoint = v3rt.load(
        args.v3_checkpoint, contract, target)
    if checkpoint['variant'] != 'residual':
        raise ValueError('Expected residual v3 checkpoint')
    shared_arm = _load_b0_arm(
        b0_run, 'Shared.pth', source, target, 'shared', start_digest)
    del source

    indices = [int(x) for x in protocol['validation_indices']]
    if not indices or len(indices) != len(set(indices)):
        raise ValueError('Invalid B0 validation indices')
    lidar_range = hypes['postprocess']['anchor_args']['cav_lidar_range']

    output = new_output(Path(args.output).resolve())
    own_sources = [
        'local_fusion_task_source_oracle/__init__.py',
        'local_fusion_task_source_oracle/experiment.yaml',
        'local_fusion_task_source_oracle/oracle.py',
        'local_fusion_task_source_oracle/test_core.py',
        'local_fusion_task_source_oracle/run_all.sh',
        'local_fusion_task_source_oracle/README.md',
    ]
    source_hashes = {
        name: sha256(ROOT / name)
        for name in own_sources
        if (ROOT / name).is_file()
    }
    write_json(output / 'protocol.json', {
        'purpose': (
            'Direction-B death test: GT-aware target-local Same-Source versus '
            'Task-Source Oracle from a strong single/query candidate pool'
        ),
        'test_data_used': False,
        'b0_run': str(b0_run),
        'b0_protocol_sha256': sha256(b0_run / 'protocol.json'),
        'b0_shared_checkpoint_sha256': sha256(b0_run / 'Shared.pth'),
        'frontend_sha256': frontend_digest,
        'v3_checkpoint_sha256': start_digest,
        'v3_contract': contract,
        'validation_indices': indices,
        'config': spec,
        'source_hashes': source_hashes,
        'selection_boundary': (
            'GT and post-NMS hindsight are intentionally used. This is a '
            'research ceiling/death test, not a deployable method. Candidate '
            'source tensors remain frozen and no training occurs.'
        ),
    })

    conditions = {}
    for condition in ('clean', 'fog', 'rain', 'snow'):
        seed_all(int(spec['seed']) + 1)
        dataset, loader = selected_loader(
            hypes, options, 'validation', condition, indices)
        folder = output / condition
        folder.mkdir(parents=True)
        conditions[condition] = evaluate_condition(
            model,
            shared_arm,
            dataset,
            loader,
            indices,
            'clean' if condition == 'clean' else 'weather',
            target,
            spec,
            lidar_range,
            folder,
            saved_b0['conditions'][condition],
        )
        write_json(folder / 'summary.json', conditions[condition])
        del dataset, loader

    decision = decide(conditions, spec)
    report = {
        'scope': (
            'GT-aware no-training direction-B death test on exact B0 '
            'development validation indices; no OPV2V-W'
        ),
        'methods': {
            'Shared': 'trained B0 Shared control',
            'Oracle-Same': (
                'per-target local GT-aware selection; classification and '
                'regression must use the same candidate'),
            'Oracle-Task': (
                'per-target local GT-aware selection; classification and '
                'regression independently select candidate sources'),
        },
        'candidate_pool': (
            'KEEP_SHARED plus each aligned source alone and each '
            'source-query attention fusion, as enabled by config'
        ),
        'conditions': conditions,
        'decision': decision,
        'boundary': (
            'This is intentionally stronger than deployable inference but is '
            'still not a mathematical optimum over arbitrary continuous '
            'source weights or all possible target-action orderings.'
        ),
    }
    write_json(output / 'death_test_results.json', report)
    verify_frozen()
    print(
        'DIRECTION-B DEATH TEST COMPLETE:',
        output / 'death_test_results.json',
        flush=True,
    )


if __name__ == '__main__':
    main()
