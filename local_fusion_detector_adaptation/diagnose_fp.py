"""Post-hoc output decomposition for the completed F versus F+D pilot.

This diagnostic does not retrain anything.  On the exact saved validation
indices it evaluates four prediction paths:
  F                = F classification + F regression
  F+D              = F+D classification + F+D regression
  FD_cls_F_reg     = F+D classification + F regression
  F_cls_FD_reg     = F classification + F+D regression

The two hybrids isolate whether post-NMS changes are driven by the final
classification tensor (psm) or the final regression tensor (rm).  Because the
F and F+D arms also learned different fusion weights, this is an output-level
causal decomposition; it does not attribute changes to cls_head/reg_head
parameters alone.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from ceif_audit.scoring import ap_values, empty_stats
from gspr_communication.runtime import device, seed_all, sha256, verify_frozen, write_json
from gspr_evidence import runtime as er
from local_fusion_utility_v2.outcomes import compare_predictions, detection_outcome
from local_fusion_v3 import runtime as v3rt
from opencood.tools.train_utils import to_device
from opencood.utils import common_utils as cu
from opencood.utils import eval_utils

from .model import AdaptationArm
from .pipeline import selected_loader


NAMES = ('baseline', 'F', 'F+D', 'FD_cls_F_reg', 'F_cls_FD_reg')
HYBRIDS = ('FD_cls_F_reg', 'F_cls_FD_reg')


def _load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _load_arm(path, source, base, target, expected_adapt):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if bool(state.get('adapt_detector')) != bool(expected_adapt):
        raise ValueError('Saved arm type does not match requested diagnostic arm: ' + str(path))
    arm = AdaptationArm(source, base, expected_adapt).to(target)
    arm.load_state_dict(state['arm'], strict=True)
    return arm.eval()


def _new_fp_boxes(action_fp_boxes, baseline_fp_boxes, identity_iou=.7):
    """Return action FP boxes that have no equivalent baseline FP."""
    action = np.asarray(action_fp_boxes)
    baseline = np.asarray(baseline_fp_boxes)
    if not len(action):
        shape = tuple(action.shape[1:]) if action.ndim > 1 else (8, 3)
        return np.empty((0, *shape), dtype=np.float32)
    if not len(baseline):
        return np.array(action, copy=True)
    old = list(cu.convert_format(baseline))
    new = list(cu.convert_format(action))
    keep = []
    for index, polygon in enumerate(new):
        ious = cu.compute_iou(polygon, old)
        if not len(ious) or float(np.max(ious)) < float(identity_iou):
            keep.append(index)
    return action[keep] if keep else np.empty((0, *action.shape[1:]), dtype=action.dtype)


def _presence(query_boxes, reference_boxes, identity_iou=.7):
    """For each query box, whether an equivalent reference box is present."""
    query = np.asarray(query_boxes)
    reference = np.asarray(reference_boxes)
    if not len(query):
        return np.zeros((0,), dtype=bool)
    if not len(reference):
        return np.zeros((len(query),), dtype=bool)
    refs = list(cu.convert_format(reference))
    flags = []
    for polygon in cu.convert_format(query):
        ious = cu.compute_iou(polygon, refs)
        flags.append(bool(len(ious) and float(np.max(ious)) >= float(identity_iou)))
    return np.asarray(flags, dtype=bool)


def _attribution(fd_new, cls_new, reg_new, identity_iou=.7):
    """Classify each F+D new FP by which one-output hybrid reproduces it."""
    in_cls = _presence(fd_new, cls_new, identity_iou)
    in_reg = _presence(fd_new, reg_new, identity_iou)
    return {
        'fd_new_fp': int(len(fd_new)),
        'classification_only': int(np.sum(in_cls & ~in_reg)),
        'regression_only': int(np.sum(~in_cls & in_reg)),
        'reproduced_by_both_hybrids': int(np.sum(in_cls & in_reg)),
        'joint_or_nms_interaction': int(np.sum(~in_cls & ~in_reg)),
        'reproduced_with_fd_classification': int(np.sum(in_cls)),
        'reproduced_with_fd_regression': int(np.sum(in_reg)),
    }


def _add_counts(total, value):
    for key, number in value.items():
        total[key] = total.get(key, 0) + int(number)


def _assert_reproduction(weather, observed, saved, tolerance):
    for name in ('baseline', 'F', 'F+D'):
        for metric in ('ap30', 'ap50', 'ap70'):
            left = float(observed[name][metric])
            right = float(saved[name][metric])
            if abs(left - right) > tolerance:
                raise RuntimeError(
                    f'{weather} reproduction mismatch {name}.{metric}: '
                    f'{left} versus saved {right}')


def evaluate_weather(model, arms, dataset, loader, indices, branch, target, folder):
    stats = {name: empty_stats() for name in NAMES}
    vs_f = {name: dict(recovered=0, lost=0, new_fp=0) for name in NAMES if name != 'F'}
    attribution = {}
    rows = []
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
                'FD_cls_F_reg': {'psm': pred_fd['psm'], 'rm': pred_f['rm']},
                'F_cls_FD_reg': {'psm': pred_f['psm'], 'rm': pred_fd['rm']},
            }
            post = {
                name: dataset.post_process(batch, {'ego': prediction})
                for name, prediction in predictions.items()
            }
            for name in NAMES:
                boxes, scores, gt = post[name]
                for threshold in stats[name]:
                    eval_utils.caluclate_tp_fp(boxes, scores, gt, stats[name], threshold)

            for name in vs_f:
                difference = compare_predictions(post['F'], post[name], .7, .7)
                for key in vs_f[name]:
                    vs_f[name][key] += int(difference[key])

            _, f_fp = detection_outcome(*post['F'], .7)
            _, fd_fp = detection_outcome(*post['F+D'], .7)
            _, cls_fp = detection_outcome(*post['FD_cls_F_reg'], .7)
            _, reg_fp = detection_outcome(*post['F_cls_FD_reg'], .7)
            fd_new = _new_fp_boxes(fd_fp, f_fp)
            cls_new = _new_fp_boxes(cls_fp, f_fp)
            reg_new = _new_fp_boxes(reg_fp, f_fp)
            frame_attr = _attribution(fd_new, cls_new, reg_new)
            _add_counts(attribution, frame_attr)
            row = {
                'sample_index': int(batch['ego']['communication_sample_index'][0]),
                'fd_new_fp': int(len(fd_new)),
                'classification_hybrid_new_fp': int(len(cls_new)),
                'regression_hybrid_new_fp': int(len(reg_new)),
                **frame_attr,
            }
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            count += 1
            if count == 1 or count % 20 == 0:
                print(f'diagnosis {count}/{len(indices)}', flush=True)

    if count != len(indices) or not count:
        raise RuntimeError('Incomplete FP diagnosis')
    results = {name: ap_values(value, eval_utils) for name, value in stats.items()}
    return {
        'frames': count,
        'results': results,
        'diagnostics_vs_F': vs_f,
        'fd_new_fp_attribution': attribution,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, help='Completed fusion_detector_adaptation run')
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
    saved_decision = _load_json(run / 'decision_results.json')
    output = Path(args.output).resolve() if args.output else run / 'fp_diagnosis'
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
    conditions = {}
    for weather in ('clean', 'fog', 'rain', 'snow'):
        seed_all(int(protocol['pilot']['seed']) + 1)
        dataset, loader = selected_loader(
            hypes, options, 'validation', weather, indices)
        folder = output / weather
        folder.mkdir(parents=True)
        summary = evaluate_weather(
            model, arms, dataset, loader, indices,
            'clean' if weather == 'clean' else 'weather',
            target, folder)
        _assert_reproduction(
            weather, summary['results'],
            saved_decision['conditions'][weather]['results'],
            args.reproduction_tolerance)
        conditions[weather] = summary
        write_json(folder / 'summary.json', summary)
        del dataset, loader

    totals = {}
    for weather in conditions:
        _add_counts(totals, conditions[weather]['fd_new_fp_attribution'])
    report = {
        'scope': 'post-hoc output-level decomposition on exact completed-pilot validation indices; no retraining',
        'interpretation': {
            'FD_cls_F_reg': 'F+D final classification tensor psm with F final regression tensor rm',
            'F_cls_FD_reg': 'F final classification tensor psm with F+D final regression tensor rm',
            'classification_only': 'F+D new FP reproduced only by the classification hybrid',
            'regression_only': 'F+D new FP reproduced only by the regression hybrid',
            'reproduced_by_both_hybrids': 'geometrically equivalent FP appears in both one-output hybrids',
            'joint_or_nms_interaction': 'neither one-output hybrid reproduces the F+D new FP; joint output/NMS interaction is required',
            'boundary': 'This isolates final psm versus rm effects, not cls_head/reg_head parameters alone, because F and F+D also learned different fusion weights.',
        },
        'conditions': conditions,
        'all_conditions_attribution': totals,
    }
    write_json(output / 'fp_diagnosis.json', report)
    verify_frozen()
    print('FP DIAGNOSIS COMPLETE:', output / 'fp_diagnosis.json', flush=True)


if __name__ == '__main__':
    main()
