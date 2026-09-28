"""Read-only Stage 0 pool sweep and candidate/source falsification audit.

All GT labels are post-hoc diagnostics. Source-only predictions and leave-one-out
effects are proxies/conditional interventions, not additive causal attribution.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ceif_audit.scoring import ap_values, empty_stats
from gspr_communication.runtime import device, seed_all, sha256, verify_frozen, write_json
from gspr_evidence import runtime as er
from gspr_evidence.stage3_trace import trace_branch
from local_fusion_utility_v2.fusion import predict_each_source
from local_fusion_v3 import runtime as v3rt
from opencood.tools.train_utils import to_device
from opencood.utils import box_utils, common_utils, eval_utils

from .model import AdaptationArm
from .pipeline import selected_loader


WEATHERS = ('clean', 'fog', 'rain', 'snow')
ARMS = ('F', 'F+D')
SWAPS = ('F_fusion_F_detector', 'FD_fusion_FD_detector',
         'F_fusion_FD_detector', 'FD_fusion_F_detector')
POOL_SPECS = (
    ('original', .2, None),
    ('threshold_010', .1, None),
    ('threshold_005', .05, None),
    ('top_128', 0., 128),
    ('top_256', 0., 256),
    ('top_512', 0., 512),
    ('threshold_005_top_256', .05, 256),
    ('all_geometry', 0., None),
)


def geometry_ids(trace):
    corners = torch.as_tensor(trace['corners'], dtype=torch.float32)
    valid = torch.logical_and(box_utils.remove_large_pred_bbx(corners),
                              box_utils.remove_bbx_abnormal_z(corners))
    return torch.nonzero(valid, as_tuple=False).reshape(-1).cpu().numpy()


def candidate_ids(trace, threshold, top_k=None, valid_ids=None):
    """Score and geometry filtering before NMS, retaining stable anchor IDs."""
    if threshold == .2 and top_k is None:
        return np.asarray(trace['ids']['geometry'], dtype=np.int64)
    scores = np.asarray(trace['scores'])
    valid_ids = geometry_ids(trace) if valid_ids is None else valid_ids
    ids = valid_ids[scores[valid_ids] > threshold]
    if not len(ids):
        return ids.astype(np.int64)
    if top_k is not None and len(ids) > top_k:
        ids = ids[np.argsort(scores[ids])[::-1][:top_k]]
    return ids.astype(np.int64)


def coverage(trace, ids):
    ious = np.asarray(trace['ious'])
    gt = ious.shape[1]
    chosen = ious[ids]
    valid_range = box_utils.get_mask_for_boxes_within_range_torch(
        torch.as_tensor(trace['corners'][ids], dtype=torch.float32)).cpu().numpy()
    in_range = chosen[valid_range]
    return {
        'candidates': int(len(ids)),
        'range_candidates': int(valid_range.sum()),
        'gt': int(gt),
        'covered_70': int((chosen.max(0) >= .7).sum()) if len(chosen) and gt else 0,
        'range_covered_70': int((in_range.max(0) >= .7).sum()) if len(in_range) and gt else 0,
    }


def summarize_pools(frame_rows):
    out = {}
    for name, _, _ in POOL_SPECS:
        rows = [row[name] for row in frame_rows]
        counts = np.asarray([row['candidates'] for row in rows], dtype=np.int64)
        out[name] = {
            'frames': len(rows), 'total_candidates': int(counts.sum()),
            'candidates_per_frame_p50': float(np.percentile(counts, 50)),
            'candidates_per_frame_p90': float(np.percentile(counts, 90)),
            'candidates_per_frame_max': int(counts.max()),
            'gt': sum(row['gt'] for row in rows),
            'covered_70': sum(row['covered_70'] for row in rows),
            'range_covered_70': sum(row['range_covered_70'] for row in rows),
        }
        out[name]['coverage_70'] = out[name]['covered_70'] / max(1, out[name]['gt'])
        out[name]['range_coverage_70'] = out[name]['range_covered_70'] / max(1, out[name]['gt'])
    return out


def _score_band(score):
    if score < .3:
        return 'below_030'
    if score < .5:
        return '030_050'
    if score < .7:
        return '050_070'
    return '070_100'


def _summarize_candidate_groups(groups):
    return [dict(weather=key[0], arm=key[1], quality_band=key[2],
                 score_band=key[3], source_count=key[4], within_range=key[5], **value,
                 disagreement_rate=value['disagreements']/value['candidates'])
            for key, value in sorted(groups.items())]


def pool_gate(conditions):
    """Prespecified bounded-pool screen; validation-only, not a test result."""
    checks = {}
    for name, _, _ in POOL_SPECS:
        if name == 'all_geometry':
            continue
        per_condition = {}
        for weather in WEATHERS:
            for arm in ARMS:
                pools = conditions[weather]['pools'][arm]
                row, ceiling = pools[name], pools['all_geometry']
                budget = row['candidates_per_frame_p90'] <= 256
                retention = (row['range_covered_70'] >=
                             .9 * ceiling['range_covered_70'])
                per_condition[f'{weather}_{arm}'] = {
                    'p90_at_most_256': budget,
                    'at_least_90pct_of_all_geometry_range_coverage': retention,
                }
        checks[name] = {
            'pass': all(all(value.values()) for value in per_condition.values()),
            'conditions': per_condition,
        }
    return {'rule': ('Every weather and both arms: p90 <= 256 candidates/frame '
                     'and >= 90% of all-geometry IoU70 GT coverage after range filtering.'),
            'checks': checks,
            'passing_pools': [name for name, value in checks.items() if value['pass']]}


def _same_box_iou(first, second):
    """Pairwise planar BEV IoU for matching anchor IDs; no best-box matching."""
    a = common_utils.convert_format(first)
    b = common_utils.convert_format(second)
    return [float(common_utils.compute_iou(x, [y])[0]) for x, y in zip(a, b)]


def _project_selected(pp, prediction, ego, ids):
    decoded = pp.delta_to_boxes3d(prediction['rm'], ego['anchor_box'])[:, ids]
    if not torch.isfinite(decoded).all():
        raise ValueError('Non-finite source or ablation boxes')
    sources, count = decoded.shape[:2]
    corners = box_utils.boxes_to_corners_3d(
        decoded.reshape(-1, 7), order=pp.params['order'])
    projected = box_utils.project_box3d(corners, ego['transformation_matrix'])
    if not torch.isfinite(projected).all():
        raise ValueError('Non-finite projected source or ablation boxes')
    return projected.reshape(sources, count, 8, 3).detach().cpu().numpy()


def _candidate_rows(trace, ids, local, pp, ego, condition, arm_name, frame_index,
                    leaveout=None, feature_cache=None):
    if not len(ids):
        return []
    fused = np.asarray(trace['corners'])[ids]
    within_range = box_utils.get_mask_for_boxes_within_range_torch(
        torch.as_tensor(fused, dtype=torch.float32)).cpu().numpy()
    if not torch.isfinite(local['psm']).all():
        raise ValueError('Non-finite source-only classification logits')
    source_scores = torch.sigmoid(local['psm'].permute(0, 2, 3, 1).reshape(len(local['psm']), -1))
    source_boxes = _project_selected(pp, local, ego, ids)
    score_array = source_scores[:, ids].detach().cpu().numpy()
    source_ious = np.asarray([_same_box_iou(boxes, fused) for boxes in source_boxes])
    source_centers = source_boxes[:, :, :4, :2].mean(axis=2)
    fused_centers = fused[:, :4, :2].mean(axis=1)
    center_shifts = np.linalg.norm(source_centers - fused_centers[None, :, :], axis=2)
    source_pairs = [_same_box_iou(source_boxes[first], source_boxes[second])
                    for first in range(len(source_boxes))
                    for second in range(first + 1, len(source_boxes))]
    pairwise_ious = (np.mean(source_pairs, axis=0) if source_pairs
                     else np.ones(len(ids), dtype=np.float32))
    if not np.isfinite(source_ious).all():
        raise ValueError('Non-finite source-box agreement')
    quality = trace['ious'][ids].max(1) if len(trace['gt']) else np.zeros(len(ids))
    final_assignment = dict(zip(trace['ids']['range'], trace['assignment']))
    rows = []
    for j, cid in enumerate(ids):
        score_source = int(np.argmax(score_array[:, j]))
        geometry_source = int(np.argmax(source_ious[:, j]))
        q = float(quality[j])
        row = {
            'weather': condition, 'arm': arm_name, 'sample_index': frame_index,
            'candidate_id': int(cid), 'source_count': int(score_array.shape[0]),
            'score': float(trace['scores'][cid]), 'max_gt_iou': q,
            'fused_logit': float(trace['logits'][cid]),
            'fused_regression_deltas': trace['regression_deltas'][cid].tolist(),
            'fused_decoded_box': trace['decoded'][cid].tolist(),
            'within_range': bool(within_range[j]),
            'quality_band': 'good' if q >= .7 else 'bad' if q < .5 else 'middle',
            'high_score_bad': bool(q < .5 and trace['scores'][cid] >= .5),
            'background': bool(q == 0),
            'final_duplicate_fp': bool(cid in final_assignment and
                                       final_assignment[cid] == -1 and q >= .7),
            'source_scores_same_anchor': score_array[:, j].tolist(),
            'source_box_iou_same_anchor': source_ious[:, j].tolist(),
            'source_box_center_shift_to_fused': center_shifts[:, j].tolist(),
            'source_box_pairwise_iou_mean': float(pairwise_ious[j]),
            # Four BEV vertices are sufficient for an offline NMS-competition audit.
            'fused_bev_corners': fused[j, :4, :2].tolist(),
            'score_source': score_source, 'geometry_source': geometry_source,
            'source_proxy_disagreement': bool(score_source != geometry_source),
        }
        if leaveout is not None:
            effects = {}
            for source, value in leaveout.items():
                full_box = trace['decoded'][cid]
                reduced_box = value['decoded'][j]
                yaw_change = np.arctan2(np.sin(full_box[6] - reduced_box[6]),
                                        np.cos(full_box[6] - reduced_box[6]))
                effects[str(source)] = {
                    'logit_drop': float(trace['logits'][cid] - value['logits'][cid]),
                    'score_drop': float(trace['scores'][cid] - value['scores'][cid]),
                    'box_iou_to_full': float(_same_box_iou(
                        value['corners'][j:j+1], fused[j:j+1])[0]),
                    'center_shift': float(np.linalg.norm(full_box[:3] - reduced_box[:3])),
                    'size_shift_l1': float(np.abs(full_box[3:6] - reduced_box[3:6]).sum()),
                    'yaw_shift_abs': float(abs(yaw_change)),
                    'gt_quality_change': float(q - value['quality'][j]),
                }
            row['leave_one_source_out'] = effects
        if feature_cache is not None:
            row['fused_feature_cache'] = feature_cache
        rows.append(row)
    return rows


def _leave_one_out(arm, levels, pp, ego, ids, gt):
    if len(levels[0]) <= 1 or not len(ids):
        return {}
    from gspr_evidence.stage3_trace import polygon_ious
    result = {}
    for source in range(len(levels[0])):
        if source == 0:
            # Ego is the fusion query; deleting it changes the problem definition.
            continue
        reduced = [torch.cat((level[:source], level[source+1:]), 0) for level in levels]
        prediction, _ = arm.predict(None, reduced)
        logits = prediction['psm'].permute(0, 2, 3, 1).reshape(-1).detach().cpu().numpy()
        scores = torch.sigmoid(prediction['psm']).permute(0, 2, 3, 1).reshape(-1)
        decoded = pp.delta_to_boxes3d(prediction['rm'], ego['anchor_box'])[0, ids]
        corners = _project_selected(pp, prediction, ego, ids)[0]
        ious = polygon_ious(corners, gt)
        result[source] = {
            'logits': logits, 'scores': scores.detach().cpu().numpy(),
            'decoded': decoded.detach().cpu().numpy(), 'corners': corners,
            'quality': ious.max(1) if len(gt) else np.zeros(len(ids)),
        }
    return result


def _fixed_candidate_features(arm, levels, prediction, ids):
    """Full detector input at each fixed anchor cell; never uses GT or source IDs."""
    fused, _ = arm.fusion(levels)
    joined = torch.cat([deblock(value) for deblock, value in
                        zip(arm.detector.backbone.deblocks, fused)], dim=1)
    torch.testing.assert_close(arm.detector.cls_head(joined), prediction['psm'],
                               atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(arm.detector.reg_head(joined), prediction['rm'],
                               atol=1e-6, rtol=1e-5)
    h, w = joined.shape[-2:]
    anchors_per_cell = prediction['psm'].shape[1]
    if joined.shape[0] != 1 or len(ids) and int(np.max(ids)) >= h*w*anchors_per_cell:
        raise ValueError('Candidate IDs do not match detector feature grid')
    flattened = joined.permute(0, 2, 3, 1).reshape(h*w, joined.shape[1])
    return flattened[torch.as_tensor(ids // anchors_per_cell,
                                     device=joined.device)].detach().cpu().numpy()


def _swap_predictions(arms, levels):
    fused_f, _ = arms['F'].fusion(levels)
    fused_fd, _ = arms['F+D'].fusion(levels)
    return {
        'F_fusion_F_detector': arms['F'].detector(fused_f),
        'FD_fusion_FD_detector': arms['F+D'].detector(fused_fd),
        'F_fusion_FD_detector': arms['F+D'].detector(fused_f),
        'FD_fusion_F_detector': arms['F'].detector(fused_fd),
    }


def evaluate_weather(model, arms, dataset, loader, indices, branch, weather,
                     folder, audit_pool, ablation_positions, stage0_only,
                     ablation_arm='both', candidate_features=False):
    pool_rows = {name: [] for name in ARMS}
    stats = {name: empty_stats() for name in (SWAPS[:2] if stage0_only else SWAPS)}
    groups = {}
    pp = dataset.post_processor
    count = 0
    with torch.no_grad(), (folder / 'candidate_rows.jsonl').open(
            'w', encoding='utf-8') as stream, (folder / 'frame_targets.jsonl').open(
            'w', encoding='utf-8') as target_stream:
        for batch in loader:
            batch = to_device(batch, next(model.parameters()).device)
            ctx = v3rt.context(model, batch['ego'], branch, verify=count == 0)
            frame_index = int(batch['ego']['communication_sample_index'][0])
            if frame_index != indices[count]:
                raise RuntimeError('Validation loader order differs from saved pilot indices')
            predictions = _swap_predictions(arms, ctx['levels'])
            local = predict_each_source(model.engine.base, ctx['levels']) if not stage0_only else None
            frame_pools = {}
            for arm_name, key in (('F', SWAPS[0]), ('F+D', SWAPS[1])):
                trace = trace_branch(dataset, batch, predictions[key])
                if arm_name == 'F':
                    target_stream.write(json.dumps({
                        'weather': weather, 'sample_index': frame_index,
                        'gt_bev_corners': np.asarray(trace['gt'])[:, :4, :2].tolist(),
                    }, ensure_ascii=False) + '\n')
                valid_ids = geometry_ids(trace)
                frame_pools[arm_name] = {}
                for name, threshold, top_k in POOL_SPECS:
                    ids = candidate_ids(trace, threshold, top_k, valid_ids)
                    frame_pools[arm_name][name] = coverage(trace, ids)
                pool_rows[arm_name].append(frame_pools[arm_name])
                if not stage0_only:
                    ids = candidate_ids(trace, *audit_pool, valid_ids=valid_ids)
                    selected = count in ablation_positions and ablation_arm in ('both', arm_name)
                    leaveout = (_leave_one_out(arms[arm_name], ctx['levels'], pp,
                                               batch['ego'], ids, trace['gt'])
                                if selected else None)
                    cache_name = None
                    if candidate_features and selected and len(ids):
                        cache_name = f'{arm_name}_features_{frame_index}.npz'
                        features = _fixed_candidate_features(
                            arms[arm_name], ctx['levels'], predictions[key], ids)
                        np.savez_compressed(folder / cache_name,
                                            candidate_id=ids, feature=features.astype(np.float32))
                    for row in _candidate_rows(trace, ids, local, pp, batch['ego'],
                                               weather, arm_name, frame_index,
                                               leaveout, cache_name):
                        stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                        key = (weather, arm_name, row['quality_band'],
                               _score_band(row['score']), row['source_count'],
                               row['within_range'])
                        item = groups.setdefault(key, {'candidates': 0, 'disagreements': 0,
                                                       'background': 0, 'high_score_bad': 0,
                                                       'final_duplicate_fp': 0})
                        item['candidates'] += 1
                        item['disagreements'] += int(row['source_proxy_disagreement'])
                        for field in ('background', 'high_score_bad', 'final_duplicate_fp'):
                            item[field] += int(row[field])
            for name in stats:
                output = dataset.post_process(batch, {'ego': predictions[name]})
                for threshold in stats[name]:
                    eval_utils.caluclate_tp_fp(*output, stats[name], threshold)
            count += 1
            if count == 1 or count % 20 == 0:
                print(f'{weather} candidate audit {count}/{len(indices)}', flush=True)
    if count != len(indices) or not count:
        raise RuntimeError('Candidate audit incomplete')
    return {
        'frames': count,
        'frame_targets_file': 'frame_targets.jsonl',
        'original_score_threshold': float(pp.params['target_args']['score_threshold']),
        'nms_iou_threshold': float(pp.params['nms_thresh']),
        'pools': {name: summarize_pools(rows) for name, rows in pool_rows.items()},
        'swap_ap': {name: ap_values(value, eval_utils) for name, value in stats.items()},
        'candidate_groups': _summarize_candidate_groups(groups),
        'ablation_frames': [indices[i] for i in sorted(ablation_positions)]
                           if not stage0_only else [],
        'ablation_arm': ablation_arm,
        'candidate_feature_cache': bool(candidate_features),
    }


def _load_arm(path, source, base, target, expected_adapt):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if bool(state.get('adapt_detector')) != expected_adapt:
        raise ValueError('Saved arm type mismatch: ' + str(path))
    arm = AdaptationArm(source, base, expected_adapt).to(target)
    arm.load_state_dict(state['arm'], strict=True)
    return arm.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--v3-config', default='local_fusion_v3/experiment.yaml')
    parser.add_argument('--frontend-config', required=True)
    parser.add_argument('--frontend-checkpoint', required=True)
    parser.add_argument('--v3-checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--stage0-only', action='store_true')
    parser.add_argument('--audit-pool', default='original',
                        choices=[name for name, _, _ in POOL_SPECS
                                 if name != 'all_geometry'])
    parser.add_argument('--ablation-frames', type=int, default=12)
    parser.add_argument('--ablation-arm', choices=('F', 'F+D', 'both'), default='both')
    parser.add_argument('--candidate-features', action='store_true',
                        help='Save fixed-candidate detector-input vectors for ablation frames')
    parser.add_argument('--reproduction-tolerance', type=float, default=1e-6)
    args = parser.parse_args()
    if args.ablation_frames < 0:
        parser.error('--ablation-frames must be nonnegative')
    if args.candidate_features and (args.stage0_only or args.audit_pool != 'top_256'
                                    or args.ablation_frames == 0):
        parser.error('--candidate-features requires top_256 source audit and ablation frames')
    verify_frozen()
    run = Path(args.run).resolve()
    protocol = json.loads((run / 'protocol.json').read_text(encoding='utf-8'))
    saved = json.loads((run / 'decision_results.json').read_text(encoding='utf-8'))
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    options, hypes = er.load_config(args.v3_config, args.frontend_config)
    target = device()
    model, digest = er.load_model(hypes, options, args.frontend_checkpoint, target)
    model.requires_grad_(False).eval()
    if digest != protocol['frontend_sha256']:
        raise ValueError('Frontend checkpoint differs from pilot')
    contract = v3rt.contract(options, args.frontend_config, digest)
    if contract != protocol['v3_contract'] or sha256(args.v3_checkpoint) != protocol['v3_checkpoint_sha256']:
        raise ValueError('v3 config/checkpoint differs from pilot')
    source, checkpoint = v3rt.load(args.v3_checkpoint, contract, target)
    if checkpoint['variant'] != 'residual':
        raise ValueError('Expected residual v3 checkpoint')
    arms = {
        'F': _load_arm(run / 'F.pth', source, model.engine.base, target, False),
        'F+D': _load_arm(run / 'F+D.pth', source, model.engine.base, target, True),
    }
    del source
    indices = [int(x) for x in protocol['validation_indices']]
    # Even spacing avoids choosing only early scenes from sorted validation indices.
    selected = {int(i) for i in np.linspace(
        0, len(indices)-1, min(args.ablation_frames, len(indices)), dtype=int)} if args.ablation_frames else set()
    audit_pool = next((threshold, top_k) for name, threshold, top_k in POOL_SPECS
                      if name == args.audit_pool)
    implementation_hash = sha256(Path(__file__).resolve())
    report = {
        'scope': 'same saved validation frames; online synthetic weather; no training or OPV2V-W test',
        'validation_indices': indices,
        'frontend_sha256': digest,
        'v3_checkpoint_sha256': sha256(args.v3_checkpoint),
        'arm_sha256': {name: sha256(run / (name + '.pth')) for name in ARMS},
        'candidate_pool_specs': {name: {'score_greater_than': threshold, 'top_k': top_k}
                                 for name, threshold, top_k in POOL_SPECS},
        'audit_pool': args.audit_pool, 'stage0_only': args.stage0_only,
        'ablation_arm': args.ablation_arm,
        'candidate_feature_cache': bool(args.candidate_features),
        'feature_precision': 'float32' if args.candidate_features else None,
        'source_interpretation': ('Source-only heads are standalone proxies. Leave-one-peer-out is a '
                                  'conditional full-forward effect; it renormalizes remaining sources '
                                  'and is not additive causal attribution. GT quality changes are '
                                  'post-hoc only.'),
        'implementation_sha256': implementation_hash, 'conditions': {},
    }
    write_json(output / 'protocol.json', {key: value for key, value in report.items()
                                          if key != 'conditions'})
    for weather in WEATHERS:
        seed_all(int(protocol['pilot']['seed']) + 1)
        dataset, loader = selected_loader(hypes, options, 'validation', weather, indices)
        scene_ends = np.asarray(dataset.len_record, dtype=np.int64)
        if not len(scene_ends) or any(index < 0 or index >= scene_ends[-1]
                                      for index in indices):
            raise ValueError('Validation frame index is outside scene boundaries')
        scene_map = {str(index): int(np.searchsorted(scene_ends, index, side='right'))
                     for index in indices}
        if 'validation_scene_map' in report and report['validation_scene_map'] != scene_map:
            raise RuntimeError('Weather datasets disagree on validation scene identity')
        report['validation_scene_map'] = scene_map
        folder = output / weather
        folder.mkdir()
        result = evaluate_weather(model, arms, dataset, loader, indices,
                                  'clean' if weather == 'clean' else 'weather',
                                  weather, folder, audit_pool, selected, args.stage0_only,
                                  args.ablation_arm, args.candidate_features)
        for arm, swap in (('F', SWAPS[0]), ('F+D', SWAPS[1])):
            for metric in ('ap30', 'ap50', 'ap70'):
                observed = result['swap_ap'][swap][metric]
                expected = saved['conditions'][weather]['results'][arm][metric]
                if abs(observed - expected) > args.reproduction_tolerance:
                    raise RuntimeError(f'{weather} {arm}.{metric} AP reproduction failed: '
                                       f'{observed} versus {expected}')
        report['conditions'][weather] = result
        write_json(folder / 'summary.json', result)
        del dataset, loader
    report['stage0_gate'] = pool_gate(report['conditions'])
    if sha256(Path(__file__).resolve()) != implementation_hash:
        raise RuntimeError('Audit source changed while running')
    write_json(output / 'candidate_audit.json', report)
    verify_frozen()
    print('CANDIDATE AUDIT COMPLETE:', output / 'candidate_audit.json', flush=True)


if __name__ == '__main__':
    main()
