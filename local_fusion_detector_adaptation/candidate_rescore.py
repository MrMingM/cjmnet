"""Frozen, no-training candidate score controls on the completed F/F+D pilot.

Compare original fused confidence, ego-only confidence, maximum standalone
source confidence, and maximum (source confidence * same-anchor source/fused
BEV box IoU). The scorepass and Stage-0 top-256 pools are both fixed before
evaluation. No GT enters a score or pool choice; GT is used only by AP.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from ceif_audit.scoring import ap_values, empty_stats
from gspr_communication.runtime import ROOT, device, seed_all, sha256, verify_frozen, write_json
from gspr_evidence import runtime as er
from local_fusion_utility_v2.fusion import predict_each_source
from local_fusion_v3 import runtime as v3rt
from opencood.tools.train_utils import to_device
from opencood.utils import box_utils, eval_utils

from .candidate_audit import (_load_arm, _project_selected, _same_box_iou,
                              candidate_ids, geometry_ids)
from .pipeline import selected_loader


WEATHERS = ('clean', 'fog', 'rain', 'snow')
ARMS = ('F', 'F+D')
POOLS = ('scorepass', 'top256')
SCORES = ('fused', 'ego', 'max_source', 'source_agreement')
METHODS = ('original',) + tuple(f'{pool}_{score}' for pool in POOLS for score in SCORES)


def extract_candidates(pp, ego, prediction):
    """Keep the classic anchor order and scorepass projection precision."""
    if abs(float(pp.params['target_args']['score_threshold']) - .2) > 1e-7:
        raise ValueError('This diagnostic fixes the original score threshold at 0.2')
    logits = prediction['psm'].permute(0, 2, 3, 1).reshape(-1)
    scores = torch.sigmoid(logits)
    decoded = pp.delta_to_boxes3d(prediction['rm'], ego['anchor_box'])[0]
    if len(scores) != len(decoded) or not torch.isfinite(decoded).all() or not torch.isfinite(scores).all():
        raise ValueError('Invalid frozen detector output')
    corners = box_utils.project_box3d(
        box_utils.boxes_to_corners_3d(decoded, order=pp.params['order']),
        ego['transformation_matrix'])
    score_ids = torch.nonzero(
        scores > pp.params['target_args']['score_threshold'], as_tuple=False).reshape(-1)
    if len(score_ids):
        # The original postprocessor projects only the score-selected boxes.
        # Recompute that subset just as stage3_trace does for exact replay.
        selected = box_utils.project_box3d(
            box_utils.boxes_to_corners_3d(decoded[score_ids], order=pp.params['order']),
            ego['transformation_matrix'])
        corners[score_ids] = selected
        valid = torch.logical_and(box_utils.remove_large_pred_bbx(selected),
                                  box_utils.remove_bbx_abnormal_z(selected))
        scorepass = score_ids[valid].detach().cpu().numpy().astype(np.int64)
    else:
        scorepass = np.empty(0, dtype=np.int64)
    trace_like = {
        'scores': scores.detach().cpu().numpy(),
        'corners': corners.detach().cpu().numpy(),
        'score_selected_count': int(len(score_ids)),
        'ids': {'geometry': scorepass},
    }
    all_valid = geometry_ids(trace_like)
    return trace_like, {
        'scorepass': candidate_ids(trace_like, .2, None, all_valid),
        'top256': candidate_ids(trace_like, 0., 256, all_valid),
    }


def source_score_variants(trace, ids, local, pp, ego):
    """Inference-visible scores only; no GT or weather-specific calibration."""
    if not len(ids):
        return {name: np.empty(0, dtype=np.float32) for name in SCORES}
    if not torch.isfinite(local['psm']).all():
        raise ValueError('Non-finite standalone source logits')
    source_scores = torch.sigmoid(
        local['psm'].permute(0, 2, 3, 1).reshape(len(local['psm']), -1))[:, ids]
    source_scores = source_scores.detach().cpu().numpy()
    source_boxes = _project_selected(pp, local, ego, ids)
    fused_boxes = trace['corners'][ids]
    agreement = np.asarray([_same_box_iou(boxes, fused_boxes)
                            for boxes in source_boxes], dtype=np.float32)
    if agreement.shape != source_scores.shape or not np.isfinite(agreement).all():
        raise ValueError('Invalid source/fused box agreement')
    values = {
        'fused': np.asarray(trace['scores'][ids], dtype=np.float32),
        'ego': source_scores[0],
        'max_source': source_scores.max(axis=0),
        'source_agreement': (source_scores * agreement).max(axis=0),
    }
    if any(not np.isfinite(value).all() or (value < 0).any() or (value > 1 + 1e-5).any()
           for value in values.values()):
        raise ValueError('Invalid candidate score')
    return values


def rescore_postprocess(trace, ids, values, pp, gt):
    """Original rotated NMS and post-NMS range check, with fixed box geometry."""
    if not len(ids):
        if trace['score_selected_count']:
            return torch.empty((0, 8, 3)), torch.empty((0,)), gt
        return None, None, gt
    boxes = torch.as_tensor(trace['corners'][ids], dtype=torch.float32)
    scores = torch.as_tensor(values, dtype=torch.float32)
    if scores.shape != (len(ids),):
        raise ValueError('Candidate score and box count differ')
    keep = torch.as_tensor(box_utils.nms_rotated(
        boxes, scores, pp.params['nms_thresh']), dtype=torch.long)
    boxes, scores = boxes[keep], scores[keep]
    within = box_utils.get_mask_for_boxes_within_range_torch(boxes)
    return boxes[within], scores[within], gt


def assert_scorepass_replay(reference, replay):
    for actual, expected in zip(replay[:2], reference[:2]):
        if expected is None:
            if actual is not None:
                raise AssertionError('Original scorepass None semantics changed')
        elif actual is None:
            raise AssertionError('Original scorepass unexpectedly empty')
        else:
            torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=1e-6, rtol=1e-6)


def global_ap_values(stats):
    return {f'ap{int(threshold*100)}': float(eval_utils.calculate_ap(
        copy.deepcopy(stats), threshold, True)[0]) for threshold in (.3, .5, .7)}


def evaluate_weather(model, arms, dataset, loader, indices, branch, target, folder):
    stats = {arm: {method: empty_stats() for method in METHODS} for arm in ARMS}
    counts = {arm: {pool: 0 for pool in POOLS} for arm in ARMS}
    pp = dataset.post_processor
    count = 0
    with torch.no_grad(), (folder / 'frames.jsonl').open('w', encoding='utf-8') as stream:
        for batch in loader:
            batch = to_device(batch, target)
            ctx = v3rt.context(model, batch['ego'], branch, verify=count == 0)
            sample = int(batch['ego']['communication_sample_index'][0])
            if sample != indices[count]:
                raise RuntimeError('Validation order differs from saved pilot')
            local = predict_each_source(model.engine.base, ctx['levels'])
            frame = {'sample_index': sample, 'arms': {}}
            for name in ARMS:
                prediction, _ = arms[name].predict(model.engine.base, ctx['levels'])
                original = dataset.post_process(batch, {'ego': prediction})
                trace, pools = extract_candidates(pp, batch['ego'], prediction)
                top_ids = pools['top256']
                positions = {int(cid): i for i, cid in enumerate(top_ids)}
                if any(int(cid) not in positions for cid in pools['scorepass']):
                    raise RuntimeError('Original scorepass pool is not contained in Stage-0 top256')
                top_variants = source_score_variants(
                    trace, top_ids, local, pp, batch['ego'])
                outputs = {'original': original}
                frame['arms'][name] = {}
                for pool, ids in pools.items():
                    counts[name][pool] += len(ids)
                    frame['arms'][name][pool] = {'candidates': int(len(ids))}
                    if pool == 'top256':
                        variants = top_variants
                    else:
                        select = [positions[int(cid)] for cid in ids]
                        variants = {score_name: values[select]
                                    for score_name, values in top_variants.items()}
                    for score_name, values in variants.items():
                        method = f'{pool}_{score_name}'
                        outputs[method] = rescore_postprocess(
                            trace, ids, values, pp, original[2])
                    if pool == 'scorepass':
                        assert_scorepass_replay(original, outputs['scorepass_fused'])
                for method, output in outputs.items():
                    for threshold in (.3, .5, .7):
                        eval_utils.caluclate_tp_fp(*output, stats[name][method], threshold)
                frame['arms'][name]['final_boxes'] = {
                    method: 0 if output[0] is None else int(len(output[0]))
                    for method, output in outputs.items()
                }
            stream.write(json.dumps(frame, ensure_ascii=False) + '\n')
            count += 1
            if count == 1 or count % 20 == 0:
                print(f'rescore {folder.name} {count}/{len(indices)}', flush=True)
    if count != len(indices) or not count:
        raise RuntimeError('Incomplete candidate rescore evaluation')
    results = {arm: {
        'ap': {method: ap_values(value, eval_utils) for method, value in methods.items()},
        'global_sort_ap': {method: global_ap_values(value) for method, value in methods.items()},
        'candidate_counts': counts[arm],
    } for arm, methods in stats.items()}
    return {'frames': count, 'results': results, 'global_sort_primary': False}


def _assert_reproduction(weather, observed, saved, tolerance):
    for arm in ARMS:
        for method in ('original', 'scorepass_fused'):
            for metric in ('ap30', 'ap50', 'ap70'):
                actual = observed['results'][arm]['ap'][method][metric]
                expected = saved['conditions'][weather]['results'][arm][metric]
                if abs(actual - expected) > tolerance:
                    raise RuntimeError(f'{weather} {arm} {method}.{metric} reproduction failed: '
                                       f'{actual} versus {expected}')


def continuation_screen(conditions):
    """Fixed development screen for whether this simple control warrants follow-up."""
    gain = {}
    for weather in WEATHERS:
        ap = conditions[weather]['results']['F']['ap']
        reference = max(ap['original']['ap70'], ap['top256_max_source']['ap70'])
        gain[weather] = ap['top256_source_agreement']['ap70'] - reference
    clean = conditions['clean']['results']['F']['ap']
    checks = {
        'weather_mean_gain_at_least_0.5pp_vs_best_simple_control':
            np.mean([gain[weather] for weather in WEATHERS[1:]]) >= .005,
        'positive_in_at_least_two_weathers_vs_best_simple_control':
            sum(gain[weather] > 0 for weather in WEATHERS[1:]) >= 2,
        'clean_ap70_no_more_than_0.1pp_below_original':
            clean['top256_source_agreement']['ap70'] >= clean['original']['ap70'] - .001,
    }
    for weather in WEATHERS:
        ap = conditions[weather]['results']['F']['ap']
        checks[f'{weather}_ap50_no_more_than_0.1pp_below_original'] = (
            ap['top256_source_agreement']['ap50'] >= ap['original']['ap50'] - .001)
    checks = {key: bool(value) for key, value in checks.items()}
    return {
        'meaning': ('Development-only screen for further investigation of a simple control. '
                    'Passing does not establish novelty or generalization.'),
        'arm': 'F', 'method': 'top256_source_agreement',
        'reference': 'better of original and top256_max_source AP70 in each weather',
        'weather_ap70_gain_pp': {weather: 100 * value for weather, value in gain.items()},
        'checks': checks, 'continue': bool(all(checks.values())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--v3-config', default='local_fusion_v3/experiment.yaml')
    parser.add_argument('--frontend-config', required=True)
    parser.add_argument('--frontend-checkpoint', required=True)
    parser.add_argument('--v3-checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--reproduction-tolerance', type=float, default=1e-6)
    args = parser.parse_args()
    verify_frozen()
    run = Path(args.run).resolve()
    protocol = json.loads((run / 'protocol.json').read_text(encoding='utf-8'))
    saved = json.loads((run / 'decision_results.json').read_text(encoding='utf-8'))
    options, hypes = er.load_config(args.v3_config, args.frontend_config)
    target = device()
    model, digest = er.load_model(hypes, options, args.frontend_checkpoint, target)
    model.requires_grad_(False).eval()
    if digest != protocol['frontend_sha256']:
        raise ValueError('Frontend checkpoint differs from completed pilot')
    contract = v3rt.contract(options, args.frontend_config, digest)
    if contract != protocol['v3_contract'] or sha256(args.v3_checkpoint) != protocol['v3_checkpoint_sha256']:
        raise ValueError('v3 config/checkpoint differs from completed pilot')
    source, checkpoint = v3rt.load(args.v3_checkpoint, contract, target)
    if checkpoint['variant'] != 'residual':
        raise ValueError('Expected residual v3 checkpoint')
    arms = {
        'F': _load_arm(run / 'F.pth', source, model.engine.base, target, False),
        'F+D': _load_arm(run / 'F+D.pth', source, model.engine.base, target, True),
    }
    del source
    indices = [int(x) for x in protocol['validation_indices']]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    implementation_paths = (
        'local_fusion_detector_adaptation/candidate_rescore.py',
        'local_fusion_detector_adaptation/candidate_audit.py',
        'opencood/data_utils/post_processor/voxel_postprocessor.py',
        'opencood/utils/box_utils.py',
        'opencood/utils/eval_utils.py',
    )
    implementation_hashes = {name: sha256(ROOT / name) for name in implementation_paths}
    report = {
        'scope': 'same saved validation frames; online synthetic weather; exploratory; no training or test data',
        'candidate_pool': 'Stage-0 top256 and original score>0.2 geometry-valid pools; fixed before this run',
        'scoring': {
            'fused': 'original F or F+D fused classification sigmoid',
            'ego': 'frozen standalone ego classification sigmoid at same anchor',
            'max_source': 'maximum frozen standalone source classification sigmoid at same anchor',
            'source_agreement': ('maximum over sources of standalone source score times BEV IoU '
                                 'of its same-anchor box with the fused candidate box'),
        },
        'interpretation': ('Source predictions are standalone proxies, not causal attribution. '
                           'Scores use no GT, weather label, or validation-fitted parameter. '
                           'Primary AP follows original frame-order evaluation; global-sorted AP '
                           'is a diagnostic only. This is a control, not a novel method claim.'),
        'continuation_rule': ('For F/top256_source_agreement: mean weather AP70 gain >=0.5pp '
                              'against the better of original and top256_max_source per weather; '
                              'positive in at least two weathers; Clean AP70 and every-condition '
                              'AP50 no more than 0.1pp below original F. Development screen only.'),
        'validation_indices': indices,
        'frontend_sha256': digest,
        'v3_checkpoint_sha256': sha256(args.v3_checkpoint),
        'arm_sha256': {name: sha256(run / (name + '.pth')) for name in ARMS},
        'implementation_sha256': implementation_hashes,
        'conditions': {},
    }
    write_json(output / 'protocol.json', {key: value for key, value in report.items()
                                          if key != 'conditions'})
    for weather in WEATHERS:
        seed_all(int(protocol['pilot']['seed']) + 1)
        dataset, loader = selected_loader(hypes, options, 'validation', weather, indices)
        folder = output / weather
        folder.mkdir()
        result = evaluate_weather(model, arms, dataset, loader, indices,
                                  'clean' if weather == 'clean' else 'weather', target, folder)
        _assert_reproduction(weather, result, saved, args.reproduction_tolerance)
        result['online_weather'] = weather != 'clean'
        report['conditions'][weather] = result
        write_json(folder / 'summary.json', result)
        del dataset, loader
    report['continuation_screen'] = continuation_screen(report['conditions'])
    if any(sha256(ROOT / name) != expected
           for name, expected in implementation_hashes.items()):
        raise RuntimeError('Rescore implementation changed while running')
    write_json(output / 'candidate_rescore.json', report)
    verify_frozen()
    print('CANDIDATE RESCORE COMPLETE:', output / 'candidate_rescore.json', flush=True)


if __name__ == '__main__':
    main()
