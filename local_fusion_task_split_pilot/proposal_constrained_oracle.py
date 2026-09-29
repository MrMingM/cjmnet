"""Direction-B Oracle with regions generated from Shared top-K predictions.

The proposal boxes and their rank use inference-visible Shared outputs only.
GT still chooses the best overlapping proposal for each target and chooses the
source action through post-NMS hindsight. This isolates the *region coverage*
gap; it is not a deployable source selector or a new trained detector.
Run only on the remote research server.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from ceif_audit.scoring import empty_stats
from gspr_communication.runtime import (
    ROOT, device, new_output, seed_all, sha256, verify_frozen, write_json,
)
from gspr_evidence import runtime as er
from local_fusion_task_source_oracle import oracle as b
from local_fusion_task_split_pilot.pipeline import selected_loader
from local_fusion_v3 import runtime as v3rt
from opencood.tools.train_utils import to_device
from opencood.utils import box_utils
from opencood.utils import common_utils as cu
from opencood.utils import eval_utils
from qa_local_intervention.operators import masks_for_frame


WEATHERS = ('clean', 'fog', 'rain', 'snow')
METHODS = ('Shared', 'Proposal-Same', 'Proposal-Task')


def proposal_settings(path):
    value = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('Proposal configuration must be a mapping')
    top_k = value.get('shared_top_k')
    minimum = value.get('minimum_gt_proposal_iou')
    if type(top_k) is not int or top_k < 1:
        raise ValueError('shared_top_k must be a positive integer')
    if not isinstance(minimum, (int, float)) or not 0 < minimum <= 1:
        raise ValueError('minimum_gt_proposal_iou must be in (0,1]')
    return {'shared_top_k': top_k,
            'minimum_gt_proposal_iou': float(minimum)}


def top_shared_proposals(cache, top_k):
    """Match Stage-0 top-K ranking, then retain its in-range proposals."""
    scores = np.asarray(cache['scores'], dtype=np.float64)
    corners = np.asarray(cache['corners'], dtype=np.float32)
    if corners.shape != (len(scores), 8, 3):
        raise ValueError('Shared score/box shape mismatch')
    finite = np.isfinite(scores) & np.isfinite(corners).all(axis=(1, 2))
    safe = np.where(finite)[0]
    if len(safe):
        boxes = torch.as_tensor(corners[safe], dtype=torch.float32)
        valid = (
            box_utils.remove_large_pred_bbx(boxes)
            & box_utils.remove_bbx_abnormal_z(boxes)
        ).detach().cpu().numpy().astype(bool)
        safe = safe[valid]
    safe = safe[scores[safe] > 0]
    order = np.lexsort((safe, -scores[safe]))
    chosen = safe[order[:top_k]].astype(np.int64)
    if len(chosen):
        in_range = box_utils.get_mask_for_boxes_within_range_torch(
            torch.as_tensor(corners[chosen], dtype=torch.float32)
        ).detach().cpu().numpy().astype(bool)
        chosen = chosen[in_range]
    return chosen, scores[chosen], corners[chosen]


def assign_proposals(gt_corners, candidate_ids, candidate_scores,
                     candidate_corners, min_iou):
    """GT-aware association to a precomputed, inference-visible region pool."""
    targets = np.asarray(gt_corners, dtype=np.float32)
    if targets.ndim != 3 or targets.shape[1:] != (8, 3):
        raise ValueError('Expected GT corners as N/8/3')
    if not len(targets):
        return [], []
    if not len(candidate_ids):
        return [None] * len(targets), [0.0] * len(targets)
    proposal_polygons = list(cu.convert_format(candidate_corners))
    output = []
    best_ious = []
    for polygon in cu.convert_format(targets):
        ious = np.asarray(cu.compute_iou(polygon, proposal_polygons),
                          dtype=np.float64).reshape(-1)
        if len(ious) != len(candidate_ids) or not np.isfinite(ious).all():
            raise ValueError('Invalid GT/proposal IoU vector')
        best = int(np.argmax(ious))
        best_ious.append(float(ious[best]))
        if float(ious[best]) < min_iou:
            output.append(None)
        else:
            output.append({
                'anchor_id': int(candidate_ids[best]),
                'score': float(candidate_scores[best]),
                'gt_iou': float(ious[best]),
                'corners': candidate_corners[best],
            })
    return output, best_ious


def _summary_state(state):
    return {
        'matched': len(state['matched']),
        'fp': len(state['fp']),
        'focal_detected': bool(state['focal']['detected']),
        'focal_score': float(state['focal']['score']),
        'focal_iou': float(state['focal']['iou']),
    }


def _choice(current, post, state, cls_name=b.KEEP, reg_name=b.KEEP,
            expansion=None, mask_stats=None, proxy=None):
    return {
        'prediction': current, 'post': post, 'state': state,
        'key': b._action_key(state, state, cls_name, reg_name),
        'cls_source': cls_name, 'reg_source': reg_name,
        'expansion': expansion, 'mask_stats': mask_stats, 'proxy': proxy,
    }


def run_constrained_oracle(dataset, batch, levels, shared_prediction, pool,
                           fixed_proxy_pool, lidar_range, spec, proposal_map,
                           mode):
    """Replay Same/Task; Task also compares actions from an identical state."""
    current = {key: shared_prediction[key].clone() for key in ('psm', 'rm')}
    current_post = dataset.post_process(batch, {'ego': current})
    shared_matched, _ = b.detection_outcome(*current_post, spec['target_iou'])
    order = b._target_order(current_post, spec['target_iou'])
    pool_names = tuple(sorted(pool))
    gt = current_post[2].detach().cpu().numpy()
    if len(proposal_map) != len(gt):
        raise ValueError('Proposal/GT count mismatch')
    records = []
    for ordinal, target_index in enumerate(order, 1):
        before = b._state(current_post, target_index, spec['target_iou'])
        proposal = proposal_map[target_index]
        keep = _choice(current, current_post, before)
        best = keep
        best_same = keep
        shortlist_info = None
        if proposal is not None:
            masks = {}
            for expansion in spec['roi_expansions']:
                mask, _, stats = masks_for_frame(
                    batch['ego']['anchor_box'], proposal['corners'],
                    levels, lidar_range, expansion,
                )
                masks[expansion] = (mask, stats)
            current_cache = b._proxy_cache(dataset, batch, current)
            shortlist, shortlist_info = b._shortlist_actions(
                current_cache, fixed_proxy_pool, gt[target_index],
                masks, pool_names, spec['roi_expansions'], mode,
                spec['target_iou'], spec['shortlist_k'],
            )
            for row in shortlist:
                prediction = b.compose_prediction(
                    current, pool, row['mask'], row['cls_source'],
                    row['reg_source'],
                )
                post = dataset.post_process(batch, {'ego': prediction})
                state = b._state(post, target_index, spec['target_iou'])
                choice = {
                    'prediction': prediction, 'post': post, 'state': state,
                    'key': b._action_key(
                        before, state, row['cls_source'], row['reg_source']),
                    'cls_source': row['cls_source'],
                    'reg_source': row['reg_source'],
                    'expansion': row['expansion'],
                    'mask_stats': row['mask_stats'],
                    'proxy': list(row['proxy']),
                }
                if choice['key'] > best['key']:
                    best = choice
                if (row['cls_source'] == row['reg_source']
                        and choice['key'] > best_same['key']):
                    best_same = choice

        task_specific_win = (mode == 'task' and best['key'] > best_same['key'])
        records.append({
            'target_index': int(target_index),
            'was_shared_missed': target_index not in shared_matched,
            'proposal': None if proposal is None else {
                key: proposal[key] for key in ('anchor_id', 'score', 'gt_iou')},
            'selected': {
                'cls_source': best['cls_source'],
                'reg_source': best['reg_source'],
                'expansion': best['expansion'],
                'mask_stats': best['mask_stats'],
                'proxy': best['proxy'],
            },
            'before': _summary_state(before),
            'after': _summary_state(best['state']),
            'same_state_same_best': _summary_state(best_same['state'])
                if mode == 'task' else None,
            'same_state_same_best_action': {
                'source': best_same['cls_source'],
                'expansion': best_same['expansion'],
            } if mode == 'task' else None,
            'task_specific_win': task_specific_win,
            'task_specific_match_gain': bool(
                task_specific_win and
                len(best['state']['matched']) > len(best_same['state']['matched'])),
            'task_specific_focal_gain': bool(
                task_specific_win and best['state']['focal']['detected']
                and not best_same['state']['focal']['detected']),
            'shortlist': shortlist_info,
        })
        current, current_post = best['prediction'], best['post']
        if ordinal == 1 or ordinal % 5 == 0 or ordinal == len(order):
            print(f'  proposal {mode} target {ordinal}/{len(order)}', flush=True)
    final_matched, final_fp = b.detection_outcome(
        *current_post, spec['target_iou'])
    return current_post, records, [int(x) for x in sorted(final_matched)], len(final_fp)


def evaluate_condition(model, shared_arm, dataset, loader, indices, branch,
                       target, spec, proposal_spec, lidar_range, folder,
                       saved_b0):
    stats = {name: empty_stats() for name in METHODS}
    totals = Counter()
    frames = 0
    with (folder / 'frames.jsonl').open('w', encoding='utf-8') as stream:
        for batch in loader:
            batch = to_device(batch, target)
            sample_index = int(batch['ego']['communication_sample_index'][0])
            if sample_index != int(indices[frames]):
                raise RuntimeError('Validation loader order differs from B0')
            context = v3rt.context(
                model, batch['ego'], branch, verify=frames == 0)
            shared, _ = shared_arm.predict(model.engine.base, context['levels'])
            pool = b.build_candidate_pool(
                model.engine.base, context['levels'], shared,
                spec['candidate_families'],
            )
            fixed_proxy = b.build_proxy_pool(dataset, batch, pool)
            shared_post = dataset.post_process(batch, {'ego': shared})
            shared_cache = b._proxy_cache(dataset, batch, shared)
            ids, scores, corners = top_shared_proposals(
                shared_cache, proposal_spec['shared_top_k'])
            gt = shared_post[2].detach().cpu().numpy()
            proposal_map, best_ious = assign_proposals(
                gt, ids, scores, corners,
                proposal_spec['minimum_gt_proposal_iou'],
            )
            same_post, same_records, same_matched, same_fp = (
                run_constrained_oracle(
                    dataset, batch, context['levels'], shared, pool,
                    fixed_proxy, lidar_range, spec, proposal_map, 'same'))
            task_post, task_records, task_matched, task_fp = (
                run_constrained_oracle(
                    dataset, batch, context['levels'], shared, pool,
                    fixed_proxy, lidar_range, spec, proposal_map, 'task'))
            for name, post in (
                ('Shared', shared_post), ('Proposal-Same', same_post),
                ('Proposal-Task', task_post),
            ):
                for threshold in stats[name]:
                    eval_utils.caluclate_tp_fp(
                        *post, stats[name], threshold)
            shared_matched, shared_fp = b.detection_outcome(
                *shared_post, spec['target_iou'])
            totals['targets'] += len(gt)
            totals['proposal_covered_targets'] += sum(
                row is not None for row in proposal_map)
            totals['shared_missed_targets'] += len(gt) - len(shared_matched)
            totals['proposal_covered_shared_missed'] += sum(
                row is not None and index not in shared_matched
                for index, row in enumerate(proposal_map))
            totals['proposal_same_changed_actions'] += sum(
                row['selected']['cls_source'] != b.KEEP
                or row['selected']['reg_source'] != b.KEEP
                for row in same_records)
            totals['proposal_task_changed_actions'] += sum(
                row['selected']['cls_source'] != b.KEEP
                or row['selected']['reg_source'] != b.KEEP
                for row in task_records)
            totals['proposal_task_separated_actions'] += sum(
                row['selected']['cls_source']
                != row['selected']['reg_source']
                for row in task_records)
            for threshold in (.1, .3, .5, .7):
                label = str(threshold).replace('.', '')
                totals[f'gt_with_proposal_iou_ge_{label}'] += sum(
                    value >= threshold for value in best_ious)
                totals[f'shared_missed_with_proposal_iou_ge_{label}'] += sum(
                    value >= threshold and index not in shared_matched
                    for index, value in enumerate(best_ious))
            totals['task_specific_same_state_wins'] += sum(
                row['task_specific_win'] for row in task_records)
            totals['task_specific_same_state_match_gains'] += sum(
                row['task_specific_match_gain'] for row in task_records)
            totals['task_specific_same_state_focal_gains'] += sum(
                row['task_specific_focal_gain'] for row in task_records)
            totals['final_task_only_gt'] += len(
                set(task_matched) - set(same_matched))
            totals['final_same_only_gt'] += len(
                set(same_matched) - set(task_matched))
            totals['final_task_recovered_vs_shared'] += len(
                set(task_matched) - shared_matched)
            totals['final_task_lost_vs_shared'] += len(
                shared_matched - set(task_matched))
            totals['final_same_recovered_vs_shared'] += len(
                set(same_matched) - shared_matched)
            totals['final_same_lost_vs_shared'] += len(
                shared_matched - set(same_matched))
            stream.write(json.dumps({
                'sample_index': sample_index,
                'proposal_pool_size': int(len(ids)),
                'best_shared_proposal_iou_by_gt': best_ious,
                'shared_matched_gt_ids': [int(x) for x in sorted(shared_matched)],
                'same_matched_gt_ids': same_matched,
                'task_matched_gt_ids': task_matched,
                'shared_fp_count': len(shared_fp),
                'same_fp_count': same_fp,
                'task_fp_count': task_fp,
                'same': same_records, 'task': task_records,
            }, ensure_ascii=False, allow_nan=False) + '\n')
            frames += 1
            if frames == 1 or frames % 10 == 0:
                print(f'proposal Oracle {folder.name}: '
                      f'{frames}/{len(indices)}', flush=True)
    if frames != len(indices):
        raise RuntimeError('Incomplete proposal Oracle')
    results = {name: b.ap_values(value, eval_utils)
               for name, value in stats.items()}
    for metric in ('ap30', 'ap50', 'ap70'):
        if abs(results['Shared'][metric]
               - float(saved_b0['results']['Shared'][metric])) > 1e-6:
            raise RuntimeError(f'Shared reproduction mismatch: {folder.name}.{metric}')
    return {
        'frames': frames,
        'results': results,
        'target_counts': dict(totals),
        'global_sort': False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--oracle-config', default=
                        'local_fusion_task_source_oracle/experiment.yaml')
    parser.add_argument('--proposal-config', default=
                        'local_fusion_task_split_pilot/proposal_oracle.yaml')
    parser.add_argument('--v3-config', default='local_fusion_v3/experiment.yaml')
    parser.add_argument('--frontend-config', required=True)
    parser.add_argument('--frontend-checkpoint', required=True)
    parser.add_argument('--v3-checkpoint', required=True)
    parser.add_argument('--b0-run', required=True)
    parser.add_argument('--original-oracle-run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    verify_frozen()
    spec = b.settings(args.oracle_config)
    proposal_spec = proposal_settings(args.proposal_config)
    b0_run = Path(args.b0_run).resolve()
    original_run = Path(args.original_oracle_run).resolve()
    b0_protocol = b._load_json(b0_run / 'protocol.json')
    saved_b0 = b._load_json(b0_run / 'decision_results.json')
    old_protocol = b._load_json(original_run / 'protocol.json')
    old_results = b._load_json(original_run / 'death_test_results.json')
    b._verify_b0_source_contract(b0_protocol)
    if old_protocol.get('b0_protocol_sha256') != sha256(b0_run / 'protocol.json'):
        raise ValueError('Original Oracle used a different B0 protocol')
    if old_protocol.get('b0_shared_checkpoint_sha256') != sha256(b0_run / 'Shared.pth'):
        raise ValueError('Original Oracle used a different Shared checkpoint')
    # JSON-loaded tuples become lists in the saved protocol.
    normalized = {**spec, 'roi_expansions': list(spec['roi_expansions']),
                  'candidate_families': list(spec['candidate_families'])}
    if old_protocol.get('config') != normalized:
        raise ValueError('Oracle settings differ from original run')
    original_hash = old_protocol.get('source_hashes', {}).get(
        'local_fusion_task_source_oracle/oracle.py')
    if original_hash != sha256(ROOT / 'local_fusion_task_source_oracle/oracle.py'):
        raise ValueError('Current Oracle implementation differs from original run')

    options, hypes = er.load_config(args.v3_config, args.frontend_config)
    target = device()
    model, frontend_digest = er.load_model(
        hypes, options, args.frontend_checkpoint, target)
    model.requires_grad_(False).eval()
    if b0_protocol.get('frontend_sha256') != frontend_digest:
        raise ValueError('B0 frontend differs')
    contract = v3rt.contract(options, args.frontend_config, frontend_digest)
    if b0_protocol.get('v3_contract') != contract:
        raise ValueError('B0 v3 contract differs')
    start_digest = sha256(args.v3_checkpoint)
    if b0_protocol.get('v3_checkpoint_sha256') != start_digest:
        raise ValueError('B0 v3 checkpoint differs')
    source, checkpoint = v3rt.load(args.v3_checkpoint, contract, target)
    if checkpoint['variant'] != 'residual':
        raise ValueError('Expected residual v3 checkpoint')
    shared_arm = b._load_b0_arm(
        b0_run, 'Shared.pth', source, target, 'shared', start_digest)
    del source
    indices = [int(x) for x in b0_protocol['validation_indices']]
    if indices != [int(x) for x in old_protocol['validation_indices']]:
        raise ValueError('Original Oracle used different validation frames')
    lidar_range = hypes['postprocess']['anchor_args']['cav_lidar_range']
    output = new_output(Path(args.output).resolve())
    own = (
        'local_fusion_task_split_pilot/audit_oracle_actions.py',
        'local_fusion_task_split_pilot/proposal_constrained_oracle.py',
        'local_fusion_task_split_pilot/proposal_oracle.yaml',
        'local_fusion_task_split_pilot/run_proposal_constrained_oracle.sh',
    )
    write_json(output / 'protocol.json', {
        'purpose': 'Inference-visible Shared top-K region restriction of B Oracle',
        'test_data_used': False,
        'original_oracle_run': str(original_run),
        'original_oracle_protocol_sha256': sha256(original_run / 'protocol.json'),
        'original_oracle_results_sha256': sha256(
            original_run / 'death_test_results.json'),
        'b0_protocol_sha256': sha256(b0_run / 'protocol.json'),
        'b0_shared_checkpoint_sha256': sha256(b0_run / 'Shared.pth'),
        'validation_indices': indices,
        'oracle_config': spec,
        'proposal_config': proposal_spec,
        'source_hashes': {name: sha256(ROOT / name) for name in own},
        'boundary': ('Proposal generation uses Shared outputs only; GT picks '
                     'the best overlapping proposal and the best action. '
                     'Results are a research ceiling, not deployable AP.'),
    })
    conditions = {}
    with torch.no_grad():
        for weather in WEATHERS:
            seed_all(int(spec['seed']) + 1)
            dataset, loader = selected_loader(
                hypes, options, 'validation', weather, indices)
            folder = output / weather
            folder.mkdir(parents=True)
            conditions[weather] = evaluate_condition(
                model, shared_arm, dataset, loader, indices,
                'clean' if weather == 'clean' else 'weather', target,
                spec, proposal_spec, lidar_range, folder,
                saved_b0['conditions'][weather],
            )
            write_json(folder / 'summary.json', conditions[weather])
            del dataset, loader
    comparisons = {}
    for weather in WEATHERS:
        row = conditions[weather]['results']
        old = old_results['conditions'][weather]['results']
        for metric in ('ap30', 'ap50', 'ap70'):
            if abs(row['Shared'][metric] - old['Shared'][metric]) > 1e-6:
                raise RuntimeError(f'Original Shared differs: {weather}.{metric}')
        same_gain = row['Proposal-Same']['ap70'] - row['Shared']['ap70']
        task_gain = row['Proposal-Task']['ap70'] - row['Shared']['ap70']
        task_extra = row['Proposal-Task']['ap70'] - row['Proposal-Same']['ap70']
        old_extra = old['Oracle-Task']['ap70'] - old['Oracle-Same']['ap70']
        comparisons[weather] = {
            'proposal_same_minus_shared_ap70': same_gain,
            'proposal_task_minus_shared_ap70': task_gain,
            'proposal_task_minus_same_ap70': task_extra,
            'original_task_minus_same_ap70': old_extra,
            'task_extra_retention_ratio': task_extra / old_extra
                if old_extra > 0 else None,
        }
    write_json(output / 'proposal_oracle_results.json', {
        'scope': 'GT-aware restricted-region B Oracle on development validation',
        'conditions': conditions,
        'comparisons': comparisons,
        'interpretation_boundary': (
            'GT still associates proposals with targets, ranks source actions, '
            'and checks final outcomes. Task/Same AP differences are comparisons '
            'of independent greedy paths; same-state action wins are logged '
            'separately in each frame.'),
    })
    verify_frozen()
    print('PROPOSAL ORACLE COMPLETE:',
          output / 'proposal_oracle_results.json', flush=True)


if __name__ == '__main__':
    main()
