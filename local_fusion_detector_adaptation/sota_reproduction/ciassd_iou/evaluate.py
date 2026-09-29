"""Fixed top256 replay, with exact original NMS/range/budget/AP and quality diagnostics."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from local_fusion_detector_adaptation.candidate_hypothesis_replay import (
    _ap, _evaluate_frame, _nms_and_budget, _stats)
from local_fusion_detector_adaptation.candidate_nms_scope import (
    _load, _resolve_root, _sha256)

from .extract import WEATHERS
from .model import make_head, mapped_iou, patch_logits, rectified_score


METHODS = ('original_f', 'top256_fused', 'direct_iou_regression',
           'ciassd_score', 'predicted_iou_only', 'gt_iou_oracle')


def _identity(rows, selected, scores, gt):
    """Eval-utils matching order, preserving candidate and GT identities at IoU .7."""
    from opencood.utils import common_utils
    if not selected:
        return set(), set()
    polygons = common_utils.convert_format(np.stack([row['corners'] for row in rows]))
    gt_polygons = list(common_utils.convert_format(gt))
    remaining = list(range(len(gt)))
    matched, false_positives = set(), set()
    chosen_scores = np.asarray([scores[index] for index in selected], dtype=np.float32)
    for rank in np.argsort(-chosen_scores):
        index = int(selected[int(rank)])
        if remaining:
            overlap = common_utils.compute_iou(
                polygons[index], [gt_polygons[j] for j in remaining])
        else:
            overlap = []
        if len(overlap) and float(np.max(overlap)) >= .7:
            matched.add(remaining.pop(int(np.argmax(overlap))))
        else:
            false_positives.add(int(rows[index]['candidate_id']))
    return matched, false_positives


def _quality_metrics(scores, qualities):
    scores = np.asarray(scores, dtype=np.float64)
    qualities = np.asarray(qualities, dtype=np.float64)
    labels = qualities >= .7
    correlation = spearmanr(scores, qualities).statistic if len(scores) > 1 else np.nan
    return {'candidates': len(scores), 'iou70_positive': int(labels.sum()),
            'score_gt_bev_iou_spearman': (float(correlation)
                                          if np.isfinite(correlation) else None),
            'iou70_roc_auc': (float(roc_auc_score(labels, scores))
                              if len(np.unique(labels)) == 2 else None),
            'iou70_pr_auc': (float(average_precision_score(labels, scores))
                             if len(np.unique(labels)) == 2 else None)}


def _paper_predictions(rows, patch_path, head, target_device):
    import torch
    path = Path(patch_path)
    with np.load(path, allow_pickle=False) as packed:
        ids = np.asarray(packed['candidate_id'], dtype=np.int64)
        patches = np.asarray(packed['patch'], dtype=np.float32)
    expected = np.asarray([row['candidate_id'] for row in rows], dtype=np.int64)
    if not np.array_equal(ids, expected):
        raise ValueError(f'{path}: candidate IDs differ from frozen top256')
    if patches.shape != (len(rows), head.in_channels, 3, 3):
        raise ValueError(f'{path}: wrong 3x3 patch shape')
    with torch.no_grad():
        x = torch.as_tensor(patches, device=target_device)
        anchors = torch.as_tensor(ids % head.out_channels, device=target_device)
        raw = patch_logits(head, x, anchors).cpu().numpy()
    return raw


def _direct_checkpoint(path, expected_f_hash):
    """The already trained S0 direct-IoU control; never fit it on validation."""
    import torch
    from local_fusion_detector_adaptation.candidate_s0_s6 import _network
    state = torch.load(path, map_location='cpu', weights_only=True)
    if (state['method'] != 'iou_regression'
            or state['f_checkpoint_sha256'] != expected_f_hash):
        raise ValueError('S0 direct IoU control does not match frozen F')
    mean, std = state['mean'].numpy(), state['std'].numpy()
    width = int(state['state']['0.weight'].shape[0])
    network = _network(len(mean), width, 1)
    network.load_state_dict(state['state'], strict=True)
    return network.eval(), mean, std


def _raw_direct_rows(root, weather):
    by_frame = {}
    with (root / weather / 'candidate_rows.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            raw = json.loads(line)
            if raw['arm'] == 'F':
                sample = int(raw['sample_index'])
                by_frame.setdefault(sample, {})[int(raw['candidate_id'])] = {
                    key: raw[key] for key in ('score', 'fused_logit',
                                             'fused_regression_deltas',
                                             'fused_decoded_box', 'source_count')}
    return by_frame


def _direct_predictions(rows, root, weather, sample, control, raw_rows):
    import torch
    from local_fusion_detector_adaptation.candidate_intervention_probe import _candidate_features
    network, mean, std = control
    # The S0 control needs its original 16+feature input, not source/GT fields.
    with np.load(root / weather / f'F_features_{sample}.npz',
                 allow_pickle=False) as packed:
        ids = np.asarray(packed['candidate_id'], dtype=np.int64)
        features = np.asarray(packed['feature'], dtype=np.float32)
    if not np.array_equal(ids, [row['candidate_id'] for row in rows]):
        raise ValueError('S0 control candidate feature IDs differ')
    inputs = np.stack([_candidate_features(raw_rows[int(cid)], vector)
                       for cid, vector in zip(ids, features)]).astype(np.float32)
    inputs = np.concatenate((inputs, np.zeros((len(inputs), 4), dtype=np.float32)), 1)
    if inputs.shape[1] != len(mean):
        raise ValueError('S0 direct IoU input dimension differs')
    inputs = np.clip((inputs-mean)/std, -10., 10.)
    inputs[:, -4:] = 0.
    with torch.no_grad():
        predictions = torch.sigmoid(network(torch.from_numpy(inputs))).flatten().numpy()
    return np.asarray([float(predictions[i]) if raw_rows[int(cid)]['source_count'] >= 2
                       else rows[i]['score'] for i, cid in enumerate(ids)])


def _render(report):
    lines = ['# CIA-SSD-style IoU quality: frozen F, fixed top256', '',
             'Development validation only. The 9 scenes have been reused in earlier work.',
             'All methods use the same top256 pool, original rotated NMS, range filter,',
             'and the original scorepass output count as each frame\'s budget.', '',
             '| Weather | Method | AP70 frame | AP70 global | Oracle recovery frame | Oracle recovery global | Spearman | IoU70 AUC | PR-AUC | Recovered GT | Lost GT | New FP |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for weather in WEATHERS:
        for method, row in report['conditions'][weather]['methods'].items():
            def number(value):
                return 'NA' if value is None else f'{value:.4f}'
            q = row['candidate_quality']
            lines.append('| ' + ' | '.join((weather, method,
                number(row['ap_frame_order']['ap70']),
                number(row['ap_global_sort']['ap70']),
                number(row['oracle_recovery_ratio_frame_order']),
                number(row['oracle_recovery_ratio_global_sort']),
                number(q['score_gt_bev_iou_spearman']),
                number(q['iou70_roc_auc']), number(q['iou70_pr_auc']),
                str(row['recovered_matched_gt_vs_original']),
                str(row['lost_original_matched_gt']), str(row['new_fp_vs_original']))) + ' |')
    lines += ['', 'AP30/AP50/AP70 for both ordering rules and all candidate diagnostics '
               'are in `results.json`.',
              'A negative recovery ratio means AP70 fell below original F. '
               'The denominator is the same fixed-budget GT-IoU Oracle AP70 minus original F AP70.',
              'The IoU-only score is diagnostic; `ciassd_score` is the main CIA-SSD confidence function.',
              'The oracle uses GT only after freezing the candidate pool and is not deployable.', '']
    return '\n'.join(lines)


def run(args):
    import torch

    root = _resolve_root(Path(args.eval_root))
    metadata, conditions, provenance = _load(root, 'F')
    if metadata.get('split') == 'train':
        raise ValueError('Evaluation input must be development validation')
    if metadata.get('scope') != 'same saved validation frames; online synthetic weather; no training or OPV2V-W test':
        raise ValueError('Evaluation requires the frozen development validation audit')
    patch_root = None
    head = None
    target_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not args.method_disabled:
        if not args.checkpoint or not args.patch_root:
            raise ValueError('Enabled method requires --checkpoint and --patch-root')
        patch_root = Path(args.patch_root).resolve()
        patch_meta = json.loads((patch_root / 'manifest.json').read_text(encoding='utf-8'))
        if (patch_meta['split'] != 'validation'
                or patch_meta['arm_sha256'] != metadata['arm_sha256']['F']
                or patch_meta['candidate_audit_sha256'] != _sha256(root / 'candidate_audit.json')):
            raise ValueError('Validation patch cache differs from frozen candidate extraction')
        state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        if state['arm_sha256'] != metadata['arm_sha256']['F']:
            raise ValueError('IoU head trained on another F checkpoint')
        head = make_head(int(state['channels']), int(state['anchors']))
        head.load_state_dict(state['state'], strict=True)
        head.to(target_device).eval()
    direct = (_direct_checkpoint(args.direct_iou_checkpoint,
                                 metadata['arm_sha256']['F'])
              if args.direct_iou_checkpoint and not args.method_disabled else None)
    report = {'protocol': {
        'split': 'development_validation', 'candidate_root': str(root),
        'candidate_audit_sha256': _sha256(root / 'candidate_audit.json'),
        'f_checkpoint_sha256': metadata['arm_sha256']['F'],
        'method_disabled': bool(args.method_disabled),
        'ciassd_checkpoint_sha256': _sha256(Path(args.checkpoint)) if head else None,
        'direct_iou_checkpoint_sha256': (_sha256(Path(args.direct_iou_checkpoint))
                                         if direct else None),
        'score_equation': 'sigmoid(cls_logit) * ((raw_iou+1)/2)^4',
        'oracle': 'top256 max GT BEV IoU > 0, then original NMS/range/budget',
        'budget': 'original scorepass final count per frame',
        'ap_modes': ['frame_order', 'global_sort'],
        'no_validation_fitting': True}, 'conditions': {}}
    frame_rows = []
    for weather in WEATHERS:
        data = conditions[weather]
        direct_rows = _raw_direct_rows(root, weather) if direct else None
        threshold = float(data['nms_iou_threshold'])
        method_names = (('original_f', 'top256_fused') if args.method_disabled else
                        tuple(name for name in METHODS
                              if name != 'direct_iou_regression' or direct))
        stats = {name: _stats() for name in method_names}
        candidate_scores = {name: [] for name in method_names}
        candidate_quality = {name: [] for name in method_names}
        counts = {name: Counter() for name in method_names}
        original_selected_by_sample = {}
        # Scorepass replay is the required regression gate. Run it before any quality head.
        for sample in metadata['validation_indices']:
            sample = int(sample)
            rows = data['rows'][sample]
            gt = data['targets'][sample]
            original_score = np.asarray([row['score'] for row in rows], dtype=np.float32)
            original_indices = [i for i, score in enumerate(original_score) if score > .2]
            original_rows = [rows[i] for i in original_indices]
            selected_local = _nms_and_budget(
                original_rows, original_score[original_indices], threshold)
            original = [original_indices[i] for i in selected_local]
            budget = len(original)
            original_selected_by_sample[sample] = (original, budget)
            _evaluate_frame(stats['original_f'], rows, original, original_score, gt)
            counts['original_f']['output'] += len(original)
            counts['original_f']['budget'] += budget
        observed = _ap(stats['original_f'], False)
        expected = metadata['conditions'][weather]['swap_ap']['F_fusion_F_detector']
        for metric in ('ap30', 'ap50', 'ap70'):
            if abs(observed[metric] - float(expected[metric])) > args.reproduction_tolerance:
                raise RuntimeError(f'{weather} disabled-method original {metric} reproduction failed: '
                                   f'{observed[metric]} vs {expected[metric]}')
        # Continue with the fixed candidate pool only after the regression gate.
        for sample in metadata['validation_indices']:
            sample = int(sample)
            rows = data['rows'][sample]
            gt = data['targets'][sample]
            original, budget = original_selected_by_sample[sample]
            score = np.asarray([row['score'] for row in rows], dtype=np.float32)
            quality = np.asarray([row['quality'] for row in rows], dtype=np.float32)
            within = np.asarray([row['within_range'] for row in rows], dtype=bool)
            scores = {'original_f': score, 'top256_fused': score}
            if head:
                raw = _paper_predictions(rows, patch_root / weather /
                                         f'F_patches_{sample}.npz', head, target_device)
                scores['ciassd_score'] = np.asarray(rectified_score(score, raw),
                                                    dtype=np.float32)
                scores['predicted_iou_only'] = np.asarray(mapped_iou(raw),
                                                           dtype=np.float32)
                counts['ciassd_score']['mapped_iou_below_zero'] += int((mapped_iou(raw) < 0).sum())
                counts['ciassd_score']['mapped_iou_above_one'] += int((mapped_iou(raw) > 1).sum())
            if direct:
                scores['direct_iou_regression'] = _direct_predictions(
                    rows, root, weather, sample, direct,
                    direct_rows[sample]).astype(np.float32)
            if not args.method_disabled:
                scores['gt_iou_oracle'] = quality
            original_matched, original_fp = _identity(rows, original, score, gt)
            for name in method_names:
                values = scores[name]
                if not np.isfinite(values).all():
                    raise ValueError(f'{weather}/{sample}/{name}: non-finite score')
                if name == 'original_f':
                    selected = original
                elif name == 'gt_iou_oracle':
                    usable = np.flatnonzero(quality > 0).tolist()
                    local = _nms_and_budget([rows[i] for i in usable],
                                            values[usable], threshold, budget)
                    selected = [usable[i] for i in local]
                else:
                    selected = _nms_and_budget(rows, values, threshold, budget)
                if name != 'original_f':
                    _evaluate_frame(stats[name], rows, selected, values, gt)
                matched, fp = _identity(rows, selected, values, gt)
                counts[name]['output'] += len(selected) if name != 'original_f' else 0
                counts[name]['budget'] += budget if name != 'original_f' else 0
                counts[name]['recovered'] += len(matched-original_matched)
                counts[name]['lost'] += len(original_matched-matched)
                counts[name]['new_fp'] += len(fp-original_fp)
                counts[name]['tp70'] += len(matched)
                counts[name]['fp70'] += len(fp)
                candidate_scores[name].extend(values[within].tolist())
                candidate_quality[name].extend(quality[within].tolist())
                frame_rows.append({'weather': weather, 'sample_index': sample,
                                   'method': name, 'budget': budget,
                                   'output': len(selected),
                                   'selected_candidate_ids': [int(rows[i]['candidate_id'])
                                                              for i in selected],
                                   'recovered_matched_gt': len(matched-original_matched),
                                   'lost_original_matched_gt': len(original_matched-matched),
                                   'new_fp': len(fp-original_fp)})
        result = {}
        original_frame = _ap(stats['original_f'], False)['ap70']
        original_global = _ap(stats['original_f'], True)['ap70']
        oracle_frame = (_ap(stats['gt_iou_oracle'], False)['ap70']
                        if 'gt_iou_oracle' in stats else None)
        oracle_global = (_ap(stats['gt_iou_oracle'], True)['ap70']
                         if 'gt_iou_oracle' in stats else None)
        for name in method_names:
            frame_ap = _ap(stats[name], False)
            global_ap = _ap(stats[name], True)
            def recovery(ap, original_ap, oracle_ap):
                return ((ap-original_ap)/(oracle_ap-original_ap)
                        if oracle_ap is not None and oracle_ap > original_ap+1e-12 else None)
            result[name] = {
                'ap_frame_order': frame_ap, 'ap_global_sort': global_ap,
                'candidate_quality': _quality_metrics(candidate_scores[name],
                                                      candidate_quality[name]),
                'oracle_recovery_ratio_frame_order': recovery(
                    frame_ap['ap70'], original_frame, oracle_frame),
                'oracle_recovery_ratio_global_sort': recovery(
                    global_ap['ap70'], original_global, oracle_global),
                'recovered_matched_gt_vs_original': counts[name]['recovered'],
                'lost_original_matched_gt': counts[name]['lost'],
                'new_fp_vs_original': counts[name]['new_fp'],
                'tp70': counts[name]['tp70'], 'fp70': counts[name]['fp70'],
                'output_boxes': counts[name]['output'],
                'budget_fill_rate': counts[name]['output']/max(1, counts[name]['budget'])}
            if name == 'ciassd_score':
                result[name]['mapped_iou_out_of_range'] = {
                    'below_zero': counts[name]['mapped_iou_below_zero'],
                    'above_one': counts[name]['mapped_iou_above_one']}
            if result[name]['tp70'] + result[name]['fp70'] != result[name]['output_boxes']:
                raise AssertionError(f'{weather}/{name}: matched/FP count differs from output')
            if result[name]['tp70'] != int(sum(stats[name][.7]['tp'])):
                raise AssertionError(f'{weather}/{name}: GT identity differs from AP matcher')
            if result[name]['fp70'] != int(sum(stats[name][.7]['fp'])):
                raise AssertionError(f'{weather}/{name}: FP identity differs from AP matcher')
        for metric in ('ap30', 'ap50', 'ap70'):
            if abs(result['top256_fused']['ap_frame_order'][metric] -
                   result['original_f']['ap_frame_order'][metric]) > args.reproduction_tolerance:
                raise RuntimeError(f'{weather}: disabled-method top256 {metric} differs '
                                   'from the saved fixed-budget original pipeline')
        report['conditions'][weather] = {
            'original_reproduction_passed': True,
            'original_reproduction_tolerance': args.reproduction_tolerance,
            'methods': result}
        print(f'{weather}: original reproduced; fixed-budget methods replayed', flush=True)
    # The top256 baseline must also agree with any previously saved fixed-budget replay.
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (output / 'results.md').write_text(_render(report), encoding='utf-8')
    with (output / 'frames.jsonl').open('w', encoding='utf-8') as stream:
        for row in frame_rows:
            stream.write(json.dumps(row) + '\n')
    print('CIA-SSD QUALITY REPLAY COMPLETE:', output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval-root', required=True)
    parser.add_argument('--patch-root')
    parser.add_argument('--checkpoint')
    parser.add_argument('--direct-iou-checkpoint')
    parser.add_argument('--method-disabled', action='store_true')
    parser.add_argument('--reproduction-tolerance', type=float, default=1e-6)
    parser.add_argument('--output', required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
