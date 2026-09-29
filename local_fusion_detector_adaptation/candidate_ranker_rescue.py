"""One bounded rescue test for score-inverted, directly suppressed candidates.

The detector and candidate extraction stay frozen. GT is used for TRAIN pair
targets and for evaluation diagnostics; the inference swap uses predictions
and original NMS relationships only. Thresholds and the primary gate are fixed
in this file before inspecting this run's evaluation results.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from .candidate_hypothesis_replay import WEATHERS, _evaluate_frame, _merge_stats, _stats
from .candidate_nms_scope import _matched_gt, _nms_ordered
from .candidate_ranker_pilot import (
    _feature_code_hashes, _load, _load_checkpoint, _method_summary,
    _model_mask, _network, _normalized, _parse_seeds, _predict,
    _prepare_baselines, _save_checkpoint, _scaler, _sha256,
    _training_arrays, _validate_frozen_contract,
)


METHODS = ('candidate_only', 'candidate_plus_simple',
           'candidate_plus_source', 'candidate_plus_distortion')
THRESHOLDS = (.70, .80, .90, .95)
PRIMARY_THRESHOLD = .90


def _pair_probability(low_probability, high_probability):
    """Probability proxy for low candidate outranking its direct suppressor."""
    low, high = np.clip([low_probability, high_probability], 1e-6, 1 - 1e-6)
    delta = np.log(low / (1 - low)) - np.log(high / (1 - high))
    return float(1 / (1 + np.exp(-np.clip(delta, -40, 40))))


def _eligible_edges(rows, fixed):
    """GT-free low-score to selected high-score direct NMS edges."""
    selected = set(fixed['selected']['top256_fused'])
    return [(low, high) for low, (high, _) in fixed['suppressors'].items()
            if high in selected and rows[low]['within_range']
            and rows[high]['within_range']
            and rows[low]['source_count'] >= 2
            and rows[high]['source_count'] >= 2
            and rows[low]['score'] <= .2 < rows[high]['score']]


def _single_swap(original_order, edges, probabilities, threshold):
    """At most one direct edge per frame; all other rank slots stay fixed."""
    choices = [(low, high, _pair_probability(probabilities[low],
                                            probabilities[high]))
               for low, high in edges]
    eligible = [item for item in choices if item[2] >= threshold]
    order = np.asarray(original_order, dtype=np.int64).copy()
    if not eligible:
        return order, None, choices
    low, high, confidence = max(eligible, key=lambda item: item[2])
    positions = {int(candidate): index for index, candidate in enumerate(order)}
    a, b = positions[low], positions[high]
    if a <= b:
        raise AssertionError('A suppressed low-score candidate must follow its winner')
    order[a], order[b] = order[b], order[a]
    return order, (low, high, confidence), choices


def _fit_hard_pairs(matrix, point_indices, labels, pairs, hard_mask,
                    feature_mask, seed, args):
    """Equal hard/easy pair draws per step; point loss is an auxiliary anchor."""
    import torch
    from torch.nn import functional as functional

    hard = np.flatnonzero(hard_mask)
    easy = np.flatnonzero(~hard_mask)
    if not len(hard) or not len(easy):
        raise ValueError('Both score-inverted and ordinary competition pairs are required')
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network = _network(matrix.shape[1], args.width).to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    rng = np.random.default_rng(seed)
    features = torch.as_tensor(matrix * feature_mask, dtype=torch.float32,
                               device=device)
    target = torch.as_tensor(labels, dtype=torch.float32, device=device)
    pair_index = torch.as_tensor(pairs, dtype=torch.long, device=device)
    good = float(labels[point_indices].sum())
    positive_weight = min(20., (len(point_indices) - good) / good)
    batches = int(np.ceil(len(point_indices) / args.batch_size))
    history = []
    for epoch in range(args.epochs):
        order = rng.permutation(point_indices)
        total = Counter()
        network.train()
        for step in range(batches):
            subset = order[step * args.batch_size:(step + 1) * args.batch_size]
            if not len(subset):
                continue
            hard_ids = rng.choice(hard, size=args.pair_batch, replace=True)
            easy_ids = rng.choice(easy, size=args.pair_batch, replace=True)
            optimizer.zero_grad(set_to_none=True)
            point_logits = network(features[torch.as_tensor(
                subset, dtype=torch.long, device=device)]).reshape(-1)
            point_loss = functional.binary_cross_entropy_with_logits(
                point_logits, target[torch.as_tensor(
                    subset, dtype=torch.long, device=device)],
                pos_weight=torch.tensor(positive_weight, device=device))

            def pair_loss(ids):
                chosen = pair_index[torch.as_tensor(ids, dtype=torch.long,
                                                    device=device)]
                logits = network(features[chosen.reshape(-1)]).reshape(-1, 2)
                return functional.softplus(logits[:, 1] - logits[:, 0]).mean()

            hard_loss = pair_loss(hard_ids)
            easy_loss = pair_loss(easy_ids)
            loss = args.point_weight * point_loss + .5 * (hard_loss + easy_loss)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite rescue loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 5.)
            optimizer.step()
            for key, value in (('point', point_loss), ('hard', hard_loss),
                               ('easy', easy_loss)):
                total[key] += float(value.detach())
            total['steps'] += 1
        summary = {'epoch': epoch + 1,
                   **{key: total[key] / total['steps']
                      for key in ('point', 'hard', 'easy')}}
        history.append(summary)
        print(f'rescue seed={seed} epoch={epoch + 1}/{args.epochs} '
              f'hard={summary["hard"]:.5f} easy={summary["easy"]:.5f}',
              flush=True)
    return network.cpu().eval(), history


def _edge_diagnostics(rows, choices):
    positive = negative = ambiguous = positive_above_half = 0
    values, labels = [], []
    for low, high, confidence in choices:
        low_quality, high_quality = rows[low]['quality'], rows[high]['quality']
        if low_quality >= .7 and high_quality < .5:
            positive += 1
            positive_above_half += int(confidence > .5)
            labels.append(1)
            values.append(confidence)
        elif low_quality < .5 and high_quality >= .7:
            negative += 1
            labels.append(0)
            values.append(confidence)
        else:
            ambiguous += 1
    return positive, negative, ambiguous, positive_above_half, labels, values


def _evaluate(data, baselines, predictions, seed):
    from opencood.utils import common_utils
    from sklearn.metrics import roc_auc_score

    result, frame_rows, edge_rows = {}, [], []
    for weather in WEATHERS:
        condition = data['rows'][weather]
        reference = baselines[weather]
        stats = {(method, t): _stats() for method in METHODS for t in THRESHOLDS}
        scenes = {(method, t): defaultdict(_stats)
                  for method in METHODS for t in THRESHOLDS}
        counts = {(method, t): Counter() for method in METHODS for t in THRESHOLDS}
        edge_counts = {method: Counter() for method in METHODS}
        edge_labels = {method: [] for method in METHODS}
        edge_values = {method: [] for method in METHODS}
        for sample in data['indices']:
            rows = condition['rows'][sample]
            fixed = reference['frames'][sample]
            gt = data['targets'][weather][sample]
            scene = str(data['scene_map'][str(sample)])
            edges = _eligible_edges(rows, fixed)
            top = fixed['selected']['top256_fused']
            polygons = common_utils.convert_format(np.stack(
                [row['corners'] for row in rows]))
            overlap = lambda candidate, others: common_utils.compute_iou(
                polygons[candidate], polygons[others])
            within = np.asarray([row['within_range'] for row in rows], dtype=bool)
            scores = [row['score'] for row in rows]
            for method in METHODS:
                probabilities = [float(predictions[method][row['vector_index']])
                                 if row['source_count'] >= 2 else row['score']
                                 for row in rows]
                all_choices = [(low, high, _pair_probability(
                    probabilities[low], probabilities[high]))
                    for low, high in edges]
                pos, neg, amb, correct, labels, values = _edge_diagnostics(
                    rows, all_choices)
                edge_counts[method].update(eligible=len(edges), positive=pos,
                                           negative=neg, ambiguous=amb,
                                           positive_above_half=correct)
                edge_labels[method].extend(labels)
                edge_values[method].extend(values)
                for low, high, confidence in all_choices:
                    edge_rows.append({
                        'seed': seed, 'weather': weather,
                        'sample_index': sample, 'scene': scene,
                        'method': method,
                        'low_candidate_id': rows[low]['candidate_id'],
                        'high_candidate_id': rows[high]['candidate_id'],
                        'pair_probability': confidence,
                        'low_quality_eval_only': rows[low]['quality'],
                        'high_quality_eval_only': rows[high]['quality'],
                    })
                for threshold in THRESHOLDS:
                    order, edit, _ = _single_swap(
                        fixed['original_order'], edges, probabilities,
                        threshold)
                    if edit is None:
                        selected = top
                    else:
                        picked, _ = _nms_ordered(order, overlap,
                                                 condition['threshold'])
                        selected = [i for i in picked if within[i]][:fixed['budget']]
                    matched = _matched_gt(rows, selected, gt)
                    gained = len(matched - fixed['matched_top'])
                    lost = len(fixed['matched_top'] - matched)
                    key = method, threshold
                    frame_stats = _stats()
                    _evaluate_frame(frame_stats, rows, selected, scores, gt)
                    _merge_stats(stats[key], frame_stats)
                    _merge_stats(scenes[key][scene], frame_stats)
                    counts[key].update(output=len(selected),
                                       budget=fixed['budget'], new=gained,
                                       lost=lost, edits=int(edit is not None))
                    if edit is not None:
                        low, high, confidence = edit
                        if rows[low]['quality'] >= .7 and rows[high]['quality'] < .5:
                            counts[key]['true_inversion_edits'] += 1
                        elif rows[low]['quality'] < .5 and rows[high]['quality'] >= .7:
                            counts[key]['opposite_quality_edits'] += 1
                        else:
                            counts[key]['ambiguous_edits'] += 1
                    if threshold == PRIMARY_THRESHOLD:
                        frame_rows.append({
                            'seed': seed, 'weather': weather,
                            'sample_index': sample, 'scene': scene,
                            'method': method, 'eligible_edges': len(edges),
                            'edit': None if edit is None else {
                                'low_candidate_id': rows[edit[0]]['candidate_id'],
                                'high_candidate_id': rows[edit[1]]['candidate_id'],
                                'pair_probability': edit[2],
                                'low_quality_eval_only': rows[edit[0]]['quality'],
                                'high_quality_eval_only': rows[edit[1]]['quality'],
                            }, 'budget': fixed['budget'],
                            'output_boxes': len(selected),
                            'new_gt': gained, 'lost_gt': lost,
                        })
        methods = {}
        baseline_tp = int(sum(reference['stats']['top256_fused'][.7]['tp']))
        for method in METHODS:
            by_threshold = {}
            for threshold in THRESHOLDS:
                key = method, threshold
                count = counts[key]
                summary = _method_summary(
                    stats[key], scenes[key], count['output'],
                    count['output'] / max(1, count['budget']),
                    count['new'], count['lost'], 0,
                    reference['target_direct_edges'])
                if summary['tp70'] - baseline_tp != count['new'] - count['lost']:
                    raise AssertionError(f'{weather}/{method}: TP identity mismatch')
                if summary['tp70'] + summary['fp70'] != count['output']:
                    raise AssertionError(f'{weather}/{method}: TP/FP mismatch')
                summary.pop('direct_edge_good_ranked_higher')
                summary.pop('direct_edge_total')
                summary['edits'] = count['edits']
                summary['true_inversion_edits'] = count['true_inversion_edits']
                summary['opposite_quality_edits'] = count['opposite_quality_edits']
                summary['ambiguous_edits'] = count['ambiguous_edits']
                by_threshold[f'{threshold:.2f}'] = summary
            labels = edge_labels[method]
            methods[method] = {
                'eligible_direct_edges': dict(edge_counts[method]),
                'edge_auc_pos_vs_opposite': (float(roc_auc_score(
                    labels, edge_values[method]))
                    if len(set(labels)) == 2 else None),
                'thresholds': by_threshold,
            }
        result[weather] = {'methods': methods}
        print(f'seed={seed} {weather}: bounded swaps evaluated', flush=True)
    return result, frame_rows, edge_rows


def _gate(report):
    """A fixed investment screen on the primary threshold only."""
    gains, vs_controls, exclusions, net = {}, {}, {}, {}
    for weather in WEATHERS:
        proposed_gain, control_gain, excluded_gain, net_gain = [], [], [], []
        original = report['baselines'][weather]['scorepass']
        for seed in report['protocol']['seeds']:
            methods = report['seeds'][str(seed)][weather]['methods']
            selected = {name: methods[name]['thresholds'][f'{PRIMARY_THRESHOLD:.2f}']
                        for name in METHODS}
            chosen = selected['candidate_plus_distortion']
            controls = [selected[name] for name in METHODS[:-1]]
            proposed_gain.append(chosen['ap_frame_order']['ap70'] -
                                 original['ap_frame_order']['ap70'])
            control_gain.append(chosen['ap_frame_order']['ap70'] - max(
                row['ap_frame_order']['ap70'] for row in controls))
            excluded_gain.append(min(
                chosen['leave_one_scene_out_ap70_frame'][scene] - max(
                    [original['leave_one_scene_out_ap70_frame'][scene]] +
                    [row['leave_one_scene_out_ap70_frame'][scene]
                     for row in controls])
                for scene in chosen['leave_one_scene_out_ap70_frame']))
            net_gain.append(chosen['new_matched_gt_vs_top256'] -
                            chosen['lost_matched_gt_vs_top256'])
        gains[weather] = float(np.mean(proposed_gain))
        vs_controls[weather] = float(np.mean(control_gain))
        exclusions[weather] = float(np.mean(excluded_gain))
        net[weather] = float(np.mean(net_gain))
    adverse = [weather for weather in WEATHERS if weather != 'clean']
    passes = sum(gains[w] >= .005 and vs_controls[w] >= .005
                 and exclusions[w] > 0 and net[w] > 0 for w in adverse)
    clean_ok = gains['clean'] >= -.001 and vs_controls['clean'] >= -.001
    return {
        'rule': ('At threshold 0.90: distortion must gain >=0.005 frame-order '
                 'AP70 over original and best equal-capacity control in at '
                 'least two adverse weathers; net GT gain and each leave-one-'
                 'scene-out advantage must be positive there; Clean loss vs '
                 'both references <=0.001. Exploratory screen only.'),
        'primary_threshold': PRIMARY_THRESHOLD,
        'mean_ap70_gain_vs_original': gains,
        'mean_ap70_gain_vs_best_control': vs_controls,
        'mean_worst_leave_one_scene_out_gain': exclusions,
        'mean_new_minus_lost_gt': net,
        'adverse_weather_pass_count': passes,
        'clean_preserved': bool(clean_ok),
        'rescue_gate_pass': bool(passes >= 2 and clean_ok),
        'evidence_status': 'reused exploratory validation; fresh scenes needed',
    }


def _markdown(report):
    lines = [
        '# 倒挂候选保守纠错：一次限定范围的挽救验证', '',
        '仅使用冻结检测器和官方 train 场景训练排序头。每帧最多交换一个原 NMS 直接抑制对。',
        '主结果使用原融合分数计算帧顺序 AP70；0.90 是预先固定的主阈值。',
        'GT 只用于训练和评价，不进入推理决策。', '',
        '| 天气 | 原 AP70 | 候选自身 | 简单一致性 | 来源基础量 | distortion | '
        'distortion 编辑数 | 净找回 GT |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for weather in WEATHERS:
        averages = {}
        for method in METHODS:
            records = [report['seeds'][str(seed)][weather]['methods'][method]
                       ['thresholds'][f'{PRIMARY_THRESHOLD:.2f}']
                       for seed in report['protocol']['seeds']]
            averages[method] = {
                'ap': np.mean([row['ap_frame_order']['ap70'] for row in records]),
                'edits': np.mean([row['edits'] for row in records]),
                'net': np.mean([row['new_matched_gt_vs_top256'] -
                                row['lost_matched_gt_vs_top256']
                                for row in records]),
            }
        original = report['baselines'][weather]['scorepass'][
            'ap_frame_order']['ap70']
        proposed = averages['candidate_plus_distortion']
        lines.append(
            f'| {weather} | {original:.4f} | '
            f'{averages["candidate_only"]["ap"]:.4f} | '
            f'{averages["candidate_plus_simple"]["ap"]:.4f} | '
            f'{averages["candidate_plus_source"]["ap"]:.4f} | '
            f'{proposed["ap"]:.4f} | {proposed["edits"]:.1f} | '
            f'{proposed["net"]:+.1f} |')
    gate = report['investment_gate']
    lines += ['', f'预设门槛：**{"通过" if gate["rescue_gate_pass"] else "未通过"}**。',
              'JSON 另含全部固定阈值、真实倒挂边覆盖、误交换数和逐场景剔除结果。',
              '本验证集已参与此前开发；任何通过结果仍需独立场景复验。', '']
    return '\n'.join(lines)


def run(args):
    evaluation = _load(args.eval_root, 'validation')
    training = _load(args.train_root, 'train') if args.mode == 'train-eval' else None
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    if training is not None:
        if training['root'] == evaluation['root']:
            raise ValueError('Training/evaluation roots must differ')
        _validate_frozen_contract(training['metadata'], evaluation['metadata'])
        if training['layout'] != evaluation['layout']:
            raise ValueError('Train/evaluation feature layout differs')
        point, labels, pairs, weights, counts = _training_arrays(
            training, args.pair_limit)
        hard_mask = weights > 1
        mean, std = _scaler(training['matrix'], training['records'])
        train_matrix = _normalized(training['matrix'], mean, std)
        seeds = _parse_seeds(args.seeds)
        train_hash = training['provenance'][str(training['root'] /
                                                'candidate_audit.json')]
        manifest = {
            'seeds': seeds, 'models': METHODS, 'layout': training['layout'],
            'training_audit_sha256': train_hash,
            'feature_code_sha256': _feature_code_hashes(),
            'rescue_code_sha256': _sha256(Path(__file__)),
            'frozen_model_sha256': {
                key: training['metadata'][key] for key in
                ('frontend_sha256', 'v3_checkpoint_sha256', 'arm_sha256')},
            'training_counts': counts,
            'training_settings': {
                'epochs': args.epochs, 'width': args.width,
                'batch_size': args.batch_size, 'pair_batch': args.pair_batch,
                'pair_limit': args.pair_limit,
                'point_weight': args.point_weight,
                'learning_rate': args.learning_rate,
                'weight_decay': args.weight_decay,
                'hard_easy_sampling': 'equal pair batches with replacement',
                'checkpoint_selection': 'fixed last epoch; no validation selection',
            },
        }
        checkpoint_folder = output / 'checkpoints'
    else:
        checkpoint_folder = Path(args.checkpoint_dir).resolve()
        manifest = json.loads((checkpoint_folder / 'manifest.json').read_text(
            encoding='utf-8'))
        if tuple(manifest['models']) != METHODS:
            raise ValueError('Checkpoint method set differs')
        if manifest['feature_code_sha256'] != _feature_code_hashes():
            raise ValueError('Feature code differs from trained checkpoint')
        if manifest['rescue_code_sha256'] != _sha256(Path(__file__)):
            raise ValueError('Rescue policy differs from trained checkpoint')
        if manifest['layout'] != evaluation['layout']:
            raise ValueError('Feature layout differs')
        frozen = manifest['frozen_model_sha256']
        for key in ('frontend_sha256', 'v3_checkpoint_sha256'):
            if frozen[key] != evaluation['metadata'][key]:
                raise ValueError(f'Frozen model differs: {key}')
        if frozen['arm_sha256']['F'] != evaluation['metadata']['arm_sha256']['F']:
            raise ValueError('Frozen F checkpoint differs')
        if evaluation['metadata'].get('row_builder_sha256',
               evaluation['metadata'].get('implementation_sha256')) != _sha256(
               Path(__file__).with_name('candidate_audit.py')):
            raise ValueError('Evaluation row builder differs')
        seeds = manifest['seeds']
        train_hash = manifest['training_audit_sha256']
    output.mkdir(parents=True, exist_ok=False)
    if training is not None:
        checkpoint_folder.mkdir()
        (checkpoint_folder / 'manifest.json').write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8')
    baselines = _prepare_baselines(evaluation, args.reproduction_tolerance)
    from .candidate_ranker_pilot import _baseline_summaries
    baseline_summary = _baseline_summaries(evaluation, baselines)
    report = {
        'protocol': {
            'mode': args.mode,
            'train_root': str(training['root']) if training else None,
            'eval_root': str(evaluation['root']),
            'eval_status': (
                'reused_exploratory_validation'
                if evaluation['metadata'].get('split') is None else
                'saved_evaluation_split; novelty must be verified externally'),
            'seeds': seeds, 'methods': METHODS,
            'thresholds': THRESHOLDS, 'primary_threshold': PRIMARY_THRESHOLD,
            'policy': 'one direct low-score/high-score NMS swap per frame',
            'primary_ap': 'original fused scores, frame order, same output budget',
            'gt_use': 'train labels and evaluation only',
            'frozen_model_sha256': manifest['frozen_model_sha256'],
            'training_audit_sha256': train_hash,
            'evaluation_audit_sha256': evaluation['provenance'][str(
                evaluation['root'] / 'candidate_audit.json')],
            'implementation_sha256': _sha256(Path(__file__)),
            'checkpoint_manifest_sha256': _sha256(checkpoint_folder /
                                                   'manifest.json'),
        },
        'training': manifest['training_counts'],
        'training_settings': manifest['training_settings'],
        'baselines': baseline_summary, 'seeds': {},
    }
    histories, frame_rows, edge_rows = {}, [], []
    for seed in seeds:
        predictions = {}
        histories[str(seed)] = {}
        for method in METHODS:
            mask = _model_mask(evaluation['layout'], method)
            if training is not None:
                network, history = _fit_hard_pairs(
                    train_matrix, point, labels, pairs, hard_mask,
                    mask, seed, args)
                _save_checkpoint(checkpoint_folder, seed, method, network,
                                 mean, std, training['layout'], train_hash,
                                 args.width)
                histories[str(seed)][method] = history
            else:
                network, mean, std = _load_checkpoint(
                    checkpoint_folder, seed, method, evaluation['layout'],
                    train_hash)
            eval_matrix = _normalized(evaluation['matrix'], mean, std)
            predictions[method] = _predict(network, eval_matrix, mask)
            del network, eval_matrix
        weather_result, frames, edges = _evaluate(evaluation, baselines,
                                                  predictions, seed)
        report['seeds'][str(seed)] = weather_result
        frame_rows.extend(frames)
        edge_rows.extend(edges)
    if training is not None:
        report['training_history'] = histories
    report['investment_gate'] = _gate(report)
    if _sha256(Path(__file__)) != report['protocol']['implementation_sha256']:
        raise RuntimeError('Rescue source changed during execution')
    (output / 'candidate_ranker_rescue.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8')
    (output / 'candidate_ranker_rescue.md').write_text(
        _markdown(report), encoding='utf-8')
    with (output / 'candidate_ranker_rescue_frames.jsonl').open(
            'w', encoding='utf-8') as stream:
        for row in frame_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    with (output / 'candidate_ranker_rescue_edges.jsonl').open(
            'w', encoding='utf-8') as stream:
        for row in edge_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f'CANDIDATE RANKER RESCUE COMPLETE: {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('train-eval', 'evaluate-only'),
                        default='train-eval')
    parser.add_argument('--train-root')
    parser.add_argument('--eval-root', required=True)
    parser.add_argument('--checkpoint-dir')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--seeds', default='20260929,20260930,20260931')
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--pair-batch', type=int, default=256)
    parser.add_argument('--pair-limit', type=int, default=128)
    parser.add_argument('--point-weight', type=float, default=.25)
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--reproduction-tolerance', type=float, default=1e-6)
    args = parser.parse_args()
    if args.mode == 'train-eval' and not args.train_root:
        parser.error('--train-root is required for train-eval')
    if args.mode == 'evaluate-only' and not args.checkpoint_dir:
        parser.error('--checkpoint-dir is required for evaluate-only')
    if (args.epochs < 1 or args.width < 4 or args.batch_size < 1
            or args.pair_batch < 1 or args.pair_limit < 1
            or args.point_weight < 0 or args.learning_rate <= 0
            or args.weight_decay < 0 or args.reproduction_tolerance <= 0):
        parser.error('Invalid rescue setting')
    run(args)


if __name__ == '__main__':
    main()
