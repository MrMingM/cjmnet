"""GT-IoU oracle for the completed F versus F+D validation pilot.

No model is trained or modified.  For baseline, F, and F+D predictions on the
exact saved validation indices, this module compares:

original
    The repository's unchanged VoxelPostprocessor.

quality_scorepass
    Keep only proposals that already passed the original classification score
    threshold and geometry checks, replace their score by max IoU to any GT,
    then run the original NMS/range filtering.

quality_all
    Start before the original classification score threshold: use every
    decoded proposal that passes the original geometry checks, replace its
    score by max IoU to any GT, discard only exact-zero-quality proposals, then
    run the original NMS/range filtering.

ceiling{30,50,70}_{scorepass,all}
    Stronger per-metric ceilings.  They use the same GT-IoU quality score but
    remove candidates whose true max GT IoU is below the named evaluation IoU
    threshold before NMS.  Only the AP at the matching threshold should be
    interpreted as that metric's ceiling.

These are diagnostic upper bounds with GT access.  They are not deployable
methods and they do not estimate the upper bound of methods that can change
box geometry or generate new proposals.
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
from local_fusion_v3 import runtime as v3rt
from opencood.tools.train_utils import to_device
from opencood.utils import box_utils
from opencood.utils import eval_utils

from .model import AdaptationArm
from .pipeline import selected_loader


SOURCES = ('baseline', 'F', 'F+D')
METHODS = (
    'original',
    'quality_scorepass',
    'quality_all',
    'ceiling30_scorepass',
    'ceiling30_all',
    'ceiling50_scorepass',
    'ceiling50_all',
    'ceiling70_scorepass',
    'ceiling70_all',
)
IOU_THRESHOLDS = (.3, .5, .7)


def _load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _load_arm(path, source, base, target, expected_adapt):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if bool(state.get('adapt_detector')) != bool(expected_adapt):
        raise ValueError('Saved arm type does not match requested oracle arm: ' + str(path))
    arm = AdaptationArm(source, base, expected_adapt).to(target)
    arm.load_state_dict(state['arm'], strict=True)
    return arm.eval()


def _torch_boxes(array):
    return torch.as_tensor(np.asarray(array), dtype=torch.float32)


def _original_from_trace(trace):
    gt = _torch_boxes(trace['gt'])
    ids = np.asarray(trace['ids']['range'], dtype=np.int64)
    if not len(trace['ids']['score']):
        return None, None, gt
    boxes = _torch_boxes(trace['corners'][ids])
    scores = torch.as_tensor(trace['scores'][ids], dtype=torch.float32)
    return boxes, scores, gt


def _all_geometry_ids(trace):
    corners = _torch_boxes(trace['corners'])
    if not len(corners):
        return np.empty((0,), dtype=np.int64)
    keep = torch.logical_and(
        box_utils.remove_large_pred_bbx(corners),
        box_utils.remove_bbx_abnormal_z(corners),
    )
    return torch.nonzero(keep, as_tuple=False).reshape(-1).cpu().numpy().astype(np.int64)


def _max_gt_iou(trace):
    ious = np.asarray(trace['ious'], dtype=np.float32)
    if ious.ndim != 2:
        raise ValueError('Trace IoU matrix must be two-dimensional')
    if not ious.shape[1]:
        return np.zeros((ious.shape[0],), dtype=np.float32)
    quality = ious.max(axis=1)
    if not np.isfinite(quality).all() or (quality < 0).any() or (quality > 1 + 1e-5).any():
        raise ValueError('Invalid GT-IoU quality values')
    return np.clip(quality, 0., 1.)


def _oracle_postprocess(trace, post_processor, pool, minimum_quality):
    """Run original NMS/range filtering after replacing score by true max GT IoU."""
    if pool == 'scorepass':
        candidate_ids = np.asarray(trace['ids']['geometry'], dtype=np.int64)
    elif pool == 'all':
        candidate_ids = _all_geometry_ids(trace)
    else:
        raise ValueError('Unknown oracle candidate pool: ' + str(pool))

    quality = _max_gt_iou(trace)
    if minimum_quality <= 0:
        candidate_ids = candidate_ids[quality[candidate_ids] > 0]
    else:
        candidate_ids = candidate_ids[quality[candidate_ids] >= float(minimum_quality)]

    gt = _torch_boxes(trace['gt'])
    if not len(candidate_ids):
        return None, None, gt

    boxes = _torch_boxes(trace['corners'][candidate_ids])
    scores = torch.as_tensor(quality[candidate_ids], dtype=torch.float32)

    keep = box_utils.nms_rotated(boxes, scores, post_processor.params['nms_thresh'])
    keep = torch.as_tensor(keep, dtype=torch.long)
    boxes = boxes[keep]
    scores = scores[keep]

    within = box_utils.get_mask_for_boxes_within_range_torch(boxes)
    boxes = boxes[within]
    scores = scores[within]
    if not len(boxes):
        return boxes, scores, gt
    return boxes, scores, gt


def _coverage(trace, candidate_ids):
    quality = _max_gt_iou(trace)
    gt_count = int(np.asarray(trace['gt']).shape[0])
    result = {'gt': gt_count, 'candidates': int(len(candidate_ids))}
    if not gt_count:
        for threshold in IOU_THRESHOLDS:
            result[f'covered_{int(threshold * 100)}'] = 0
        return result
    ious = np.asarray(trace['ious'], dtype=np.float32)
    subset = ious[np.asarray(candidate_ids, dtype=np.int64)]
    for threshold in IOU_THRESHOLDS:
        covered = 0 if not len(subset) else int((subset.max(axis=0) >= threshold).sum())
        result[f'covered_{int(threshold * 100)}'] = covered
    return result


def _add_counts(total, row):
    for key, value in row.items():
        total[key] = total.get(key, 0) + int(value)


def _frame_methods(trace, pp):
    outputs = {'original': _original_from_trace(trace)}
    for pool in ('scorepass', 'all'):
        outputs[f'quality_{pool}'] = _oracle_postprocess(trace, pp, pool, 0.)
        for threshold in IOU_THRESHOLDS:
            outputs[f'ceiling{int(threshold * 100)}_{pool}'] = _oracle_postprocess(
                trace, pp, pool, threshold)
    return outputs


def _assert_original_reproduction(condition, observed, saved, tolerance):
    for source in SOURCES:
        for metric in ('ap30', 'ap50', 'ap70'):
            left = float(observed[source]['original'][metric])
            right = float(saved['conditions'][condition]['results'][source][metric])
            if abs(left - right) > tolerance:
                raise RuntimeError(
                    f'{condition} reproduction mismatch {source}.{metric}: '
                    f'{left} versus saved {right}')


def _summarize(results, coverage):
    out = {}
    for source in SOURCES:
        original = results[source]['original']
        row = {
            'original': original,
            'quality_scorepass': results[source]['quality_scorepass'],
            'quality_all': results[source]['quality_all'],
            'ap70_ceiling_scorepass': results[source]['ceiling70_scorepass']['ap70'],
            'ap70_ceiling_all': results[source]['ceiling70_all']['ap70'],
            'ap70_gain_quality_scorepass_pp': 100.0 * (
                results[source]['quality_scorepass']['ap70'] - original['ap70']),
            'ap70_gain_quality_all_pp': 100.0 * (
                results[source]['quality_all']['ap70'] - original['ap70']),
            'ap70_ceiling_gain_scorepass_pp': 100.0 * (
                results[source]['ceiling70_scorepass']['ap70'] - original['ap70']),
            'ap70_ceiling_gain_all_pp': 100.0 * (
                results[source]['ceiling70_all']['ap70'] - original['ap70']),
            'candidate_coverage': coverage[source],
        }
        out[source] = row
    return out


def evaluate_condition(model, arms, dataset, loader, indices, branch, target, folder):
    stats = {
        source: {method: empty_stats() for method in METHODS}
        for source in SOURCES
    }
    coverage = {
        source: {'scorepass': {}, 'all': {}}
        for source in SOURCES
    }
    count = 0

    with torch.no_grad(), (folder / 'frames.jsonl').open('w', encoding='utf-8') as stream:
        for batch in loader:
            batch = to_device(batch, target)
            context = v3rt.context(model, batch['ego'], branch, verify=count == 0)
            pred_f, _ = arms['F'].predict(model.engine.base, context['levels'])
            pred_fd, _ = arms['F+D'].predict(model.engine.base, context['levels'])
            predictions = {
                'baseline': context['baseline_prediction'],
                'F': pred_f,
                'F+D': pred_fd,
            }
            frame = {
                'sample_index': int(batch['ego']['communication_sample_index'][0]),
                'sources': {},
            }

            for source, prediction in predictions.items():
                trace = trace_branch(dataset, batch, prediction)
                all_ids = _all_geometry_ids(trace)
                scorepass_ids = np.asarray(trace['ids']['geometry'], dtype=np.int64)
                frame_coverage = {
                    'scorepass': _coverage(trace, scorepass_ids),
                    'all': _coverage(trace, all_ids),
                }
                for pool in ('scorepass', 'all'):
                    _add_counts(coverage[source][pool], frame_coverage[pool])

                outputs = _frame_methods(trace, dataset.post_processor)
                for method, output in outputs.items():
                    for threshold in stats[source][method]:
                        eval_utils.caluclate_tp_fp(
                            *output, stats[source][method], threshold)

                frame['sources'][source] = {
                    'coverage': frame_coverage,
                    'original_final_boxes': (
                        0 if outputs['original'][0] is None else int(len(outputs['original'][0]))),
                    'quality_scorepass_final_boxes': (
                        0 if outputs['quality_scorepass'][0] is None
                        else int(len(outputs['quality_scorepass'][0]))),
                    'quality_all_final_boxes': (
                        0 if outputs['quality_all'][0] is None
                        else int(len(outputs['quality_all'][0]))),
                }

            stream.write(json.dumps(frame, ensure_ascii=False) + '\n')
            count += 1
            if count == 1 or count % 20 == 0:
                print(f'quality oracle {count}/{len(indices)}', flush=True)

    if count != len(indices) or not count:
        raise RuntimeError('Incomplete quality oracle evaluation')

    results = {
        source: {
            method: ap_values(value, eval_utils)
            for method, value in methods.items()
        }
        for source, methods in stats.items()
    }
    return {
        'frames': count,
        'results': results,
        'candidate_coverage_counts': coverage,
        'summary': _summarize(results, coverage),
        'global_sort': False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True,
                        help='Completed local_fusion_detector_adaptation run')
    parser.add_argument('--v3-config', default='local_fusion_v3/experiment.yaml')
    parser.add_argument('--frontend-config', required=True)
    parser.add_argument('--frontend-checkpoint', required=True)
    parser.add_argument('--v3-checkpoint', required=True)
    parser.add_argument('--output', default=None)
    parser.add_argument('--reproduction-tolerance', type=float, default=1e-6)
    args = parser.parse_args()

    verify_frozen()
    run = Path(args.run).resolve()
    protocol = _load_json(run / 'protocol.json')
    saved = _load_json(run / 'decision_results.json')
    output = Path(args.output).resolve() if args.output else run / 'quality_oracle'
    output.mkdir(parents=True, exist_ok=False)

    options, hypes = er.load_config(args.v3_config, args.frontend_config)
    target = device()
    model, frontend_digest = er.load_model(
        hypes, options, args.frontend_checkpoint, target)
    model.requires_grad_(False).eval()
    if frontend_digest != protocol['frontend_sha256']:
        raise ValueError('Frontend checkpoint differs from completed pilot')
    contract = v3rt.contract(options, args.frontend_config, frontend_digest)
    if contract != protocol['v3_contract']:
        raise ValueError('v3 config/source contract differs from completed pilot')
    if sha256(args.v3_checkpoint) != protocol['v3_checkpoint_sha256']:
        raise ValueError('v3 checkpoint differs from completed pilot')

    source, checkpoint = v3rt.load(args.v3_checkpoint, contract, target)
    if checkpoint['variant'] != 'residual':
        raise ValueError('Expected residual v3 checkpoint')
    arms = {
        'F': _load_arm(run / 'F.pth', source, model.engine.base, target, False),
        'F+D': _load_arm(run / 'F+D.pth', source, model.engine.base, target, True),
    }
    del source

    indices = [int(x) for x in protocol['validation_indices']]
    implementation = {
        'oracle_quality.py': sha256(Path(__file__).resolve()),
        'stage3_trace.py': sha256(Path(__file__).parents[1] / 'gspr_evidence' / 'stage3_trace.py'),
        'voxel_postprocessor.py': sha256(
            Path(__file__).parents[1] / 'opencood' / 'data_utils' / 'post_processor'
            / 'voxel_postprocessor.py'),
        'box_utils.py': sha256(Path(__file__).parents[1] / 'opencood' / 'utils' / 'box_utils.py'),
    }
    metadata = {
        'scope': 'GT-IoU oracle on exact completed F/F+D pilot validation indices; no retraining',
        'test_data_used': False,
        'validation_indices': indices,
        'definitions': {
            'quality': 'maximum BEV polygon IoU between a decoded candidate and any GT in the same frame',
            'quality_scorepass': (
                'original score-threshold + geometry candidate pool; replace score with true quality; '
                'same NMS and range filter'),
            'quality_all': (
                'all decoded geometry-valid candidates before original score threshold; replace score '
                'with true quality; exact-zero quality is discarded; same NMS and range filter'),
            'ceilingXX': (
                'per-metric stronger ceiling: additionally discard candidates with true quality below '
                'XX/100 before NMS; interpret only APXX as the corresponding ceiling'),
            'boundary': (
                'Oracle has GT access. It upper-bounds score/selection/NMS repair for the fixed decoded '
                'box geometry. It is not an upper bound for methods that alter regression/fusion geometry '
                'or generate new candidates.'),
        },
        'implementation_sha256': implementation,
    }
    write_json(output / 'protocol.json', metadata)

    conditions = {}
    for condition in ('clean', 'fog', 'rain', 'snow'):
        seed_all(int(protocol['pilot']['seed']) + 1)
        dataset, loader = selected_loader(
            hypes, options, 'validation', condition, indices)
        folder = output / condition
        folder.mkdir(parents=True)
        summary = evaluate_condition(
            model, arms, dataset, loader, indices,
            'clean' if condition == 'clean' else 'weather',
            target, folder)
        _assert_original_reproduction(
            condition, summary['results'], saved, args.reproduction_tolerance)
        summary.update(
            root=hypes['validate_dir'],
            online_weather=condition != 'clean',
        )
        conditions[condition] = summary
        write_json(folder / 'summary.json', summary)
        del dataset, loader

    report = {
        **metadata,
        'conditions': conditions,
        'reading_guide': {
            'first': (
                'Check quality_scorepass versus original. A large gain means perfect box-quality '
                'ranking/NMS among already admitted candidates has substantial room.'),
            'second': (
                'Check quality_all versus quality_scorepass. A large extra gain means many useful '
                'boxes exist before the current classification threshold and could be rescued by a '
                'quality-aware score.'),
            'third': (
                'Check ap70_ceiling_all. This is the strongest AP70 ceiling reported here for fixed '
                'decoded geometry under the original NMS/range logic.'),
        },
    }
    write_json(output / 'quality_oracle.json', report)
    verify_frozen()
    for name, expected in implementation.items():
        if name == 'oracle_quality.py':
            current = sha256(Path(__file__).resolve())
        elif name == 'stage3_trace.py':
            current = sha256(Path(__file__).parents[1] / 'gspr_evidence' / 'stage3_trace.py')
        elif name == 'voxel_postprocessor.py':
            current = sha256(
                Path(__file__).parents[1] / 'opencood' / 'data_utils' / 'post_processor'
                / 'voxel_postprocessor.py')
        else:
            current = sha256(Path(__file__).parents[1] / 'opencood' / 'utils' / 'box_utils.py')
        if current != expected:
            raise RuntimeError('Oracle implementation changed while running: ' + name)
    print('QUALITY ORACLE COMPLETE:', output / 'quality_oracle.json', flush=True)


if __name__ == '__main__':
    main()
