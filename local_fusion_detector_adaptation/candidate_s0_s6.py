"""Server-only S0 ordinal-quality and S6 frozen-head stability screen.

Uses the existing official-train and validation top256 exports. No detector
training, validation-based checkpoint selection, GT inference feature, or new
candidate geometry is involved. This is an exploratory validation screen.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from .candidate_hypothesis_replay import WEATHERS, _ap, _evaluate_frame, _merge_stats, _stats
from .candidate_nms_scope import _matched_gt, _nms_ordered, _oracle_order
from .candidate_ranker_pilot import (
    _baseline_summaries, _load, _method_summary, _normalized, _parse_seeds,
    _prepare_baselines, _sha256, _validate_frozen_contract,
)


METHODS = ('binary70', 'iou_regression', 'ordinal', 's6_cls', 's6_geom',
           's6_both', 's6_shuffled')
S6_NAMES = ('cls_mean_abs', 'cls_std', 'geometry_mean', 'geometry_std')
THRESHOLDS = (.3, .5, .7)
S6_DRAWS = 8
S6_DROPOUT = .10


def _head_weights(path, expected_hash, feature_width):
    """Read only the frozen F 1x1 heads; verify their checkpoint identity."""
    import torch

    path = Path(path).resolve()
    if _sha256(path) != expected_hash:
        raise ValueError('F checkpoint SHA256 differs from candidate extraction')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if bool(checkpoint.get('adapt_detector')):
        raise ValueError('S6 requires the frozen-detector F checkpoint')
    state = checkpoint['arm']
    cls = state['detector.cls_head.weight'].detach().numpy()
    reg = state['detector.reg_head.weight'].detach().numpy()
    cb = state['detector.cls_head.bias'].detach().numpy()
    rb = state['detector.reg_head.bias'].detach().numpy()
    if (cls.ndim != 4 or reg.ndim != 4 or cls.shape[2:] != (1, 1)
            or reg.shape[2:] != (1, 1) or cls.shape[1] != feature_width
            or reg.shape != (cls.shape[0] * 7, feature_width, 1, 1)
            or cb.shape != (cls.shape[0],) or rb.shape != (reg.shape[0],)):
        raise ValueError('Frozen detector head does not match cached features')
    return (cls[:, :, 0, 0].astype(np.float32), cb.astype(np.float32),
            reg[:, :, 0, 0].reshape(cls.shape[0], 7, feature_width).astype(np.float32),
            rb.reshape(cls.shape[0], 7).astype(np.float32))


def _s6_features(data, heads, seed=20260929, batch_size=4096):
    """Deterministic shared dropout; geometry measured in box-relative units."""
    import torch

    cls_w, cls_b, reg_w, reg_b = heads
    candidates = data['matrix'][:, data['layout']['candidate'][0]:
                                   data['layout']['candidate'][1]]
    joined = candidates[:, 16:]
    anchor = np.asarray([row['candidate_id'] % len(cls_w)
                         for row in data['records']], dtype=np.int64)
    if joined.shape[1] != cls_w.shape[1]:
        raise ValueError('Candidate feature width differs from frozen head')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights_c = torch.as_tensor(cls_w, device=device)
    bias_c = torch.as_tensor(cls_b, device=device)
    weights_r = torch.as_tensor(reg_w, device=device)
    bias_r = torch.as_tensor(reg_b, device=device)
    rng = np.random.default_rng(seed)
    features = np.empty((len(joined), 4), dtype=np.float32)
    max_cls_error = max_reg_error = 0.
    with torch.no_grad():
        for start in range(0, len(joined), batch_size):
            stop = min(start + batch_size, len(joined))
            a = torch.as_tensor(anchor[start:stop], device=device)
            x = torch.as_tensor(joined[start:stop], device=device)
            cw, rw = weights_c[a], weights_r[a]
            original_cls = (cw * x).sum(-1) + bias_c[a]
            original_reg = torch.einsum('bd,bkd->bk', x, rw) + bias_r[a]
            cls_error = torch.max(torch.abs(original_cls - torch.as_tensor(
                candidates[start:stop, 1], device=device))).item()
            reg_error = torch.max(torch.abs(original_reg - torch.as_tensor(
                candidates[start:stop, 2:9], device=device))).item()
            max_cls_error = max(max_cls_error, cls_error)
            max_reg_error = max(max_reg_error, reg_error)
            if cls_error > 1e-4 or reg_error > 1e-4:
                raise RuntimeError('Frozen heads do not reproduce original candidate logits/deltas')
            mask = (rng.random((stop-start, S6_DRAWS, x.shape[1])) >=
                    S6_DROPOUT).astype(np.float32) / (1. - S6_DROPOUT)
            delta_x = x[:, None, :] * (torch.as_tensor(mask, device=device) - 1.)
            cls_change = torch.einsum('bsd,bd->bs', delta_x, cw)
            reg_change = torch.einsum('bsd,bkd->bsk', delta_x, rw)
            original_delta = candidates[start:stop, 2:9]
            box = candidates[start:stop, 9:16]
            anchor_h = box[:, 3] / np.exp(np.clip(original_delta[:, 3], -10., 10.))
            anchor_w = box[:, 4] / np.exp(np.clip(original_delta[:, 4], -10., 10.))
            anchor_l = box[:, 5] / np.exp(np.clip(original_delta[:, 5], -10., 10.))
            if np.any(np.minimum.reduce((anchor_h, anchor_w, anchor_l)) <= 0):
                raise ValueError('Cannot reconstruct positive anchor dimensions')
            diagonal = torch.as_tensor(np.sqrt(anchor_w**2 + anchor_l**2), device=device)
            bw = torch.as_tensor(box[:, 4], device=device).clamp_min(.1)
            bl = torch.as_tensor(box[:, 5], device=device).clamp_min(.1)
            geometry = torch.sqrt(
                (reg_change[:, :, 0] * diagonal[:, None] / bw[:, None])**2 +
                (reg_change[:, :, 1] * diagonal[:, None] / bl[:, None])**2 +
                reg_change[:, :, 3]**2 + reg_change[:, :, 4]**2 +
                torch.sin(reg_change[:, :, 6])**2 + 1e-12)
            block = torch.stack((cls_change.abs().mean(1),
                                 cls_change.std(1, unbiased=False),
                                 geometry.mean(1),
                                 geometry.std(1, unbiased=False)), dim=1)
            features[start:stop] = block.cpu().numpy()
    if not np.isfinite(features).all():
        raise ValueError('Non-finite S6 stability feature')
    return features, {'max_abs_cls_replay_error': max_cls_error,
                      'max_abs_reg_replay_error': max_reg_error,
                      'draws': S6_DRAWS, 'dropout': S6_DROPOUT,
                      'features': S6_NAMES,
                      'feature_interpretation': 'deterministic transform of cached candidate feature'}


def _shuffled(features, records, seed):
    """Within-weather and within-score-band negative control, no GT access."""
    rng = np.random.default_rng(seed)
    result = features.copy()
    for weather in WEATHERS:
        for low in (True, False):
            for multi_source in (True, False):
                indices = np.asarray([
                    index for index, row in enumerate(records)
                    if row['weather'] == weather and (row['score'] <= .2) == low
                    and (row['source_count'] >= 2) == multi_source],
                    dtype=np.int64)
                if len(indices) > 1:
                    result[indices] = features[rng.permutation(indices)]
    return result


def _input(data, s6, shuffled):
    candidate = data['matrix'][:, slice(*data['layout']['candidate'])]
    base = np.concatenate((candidate, s6), axis=1).astype(np.float32)
    negative = np.concatenate((candidate, shuffled), axis=1).astype(np.float32)
    return base, negative


def _mask(method, dimension):
    mask = np.zeros(dimension, dtype=np.float32)
    mask[:-4] = 1.
    if method == 's6_cls':
        mask[-4:-2] = 1.
    elif method == 's6_geom':
        mask[-2:] = 1.
    elif method in ('s6_both', 's6_shuffled'):
        mask[-4:] = 1.
    return mask


def _network(dimension, width, outputs):
    from torch import nn
    return nn.Sequential(nn.Linear(dimension, width), nn.ReLU(),
                         nn.Linear(width, width//2), nn.ReLU(),
                         nn.Linear(width//2, outputs))


def _fit(matrix, records, method, seed, args):
    import torch
    from torch.nn import functional as F

    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    selected = np.asarray([row['vector_index'] for row in records
                           if row['within_range'] and row['source_count'] >= 2],
                          dtype=np.int64)
    quality = np.asarray([row['quality'] for row in records], dtype=np.float32)
    if len(selected) == 0:
        raise ValueError('No training candidates inside evaluation range')
    targets = np.stack([quality >= cutoff for cutoff in THRESHOLDS], 1).astype(np.float32)
    positives = targets[selected].sum(axis=0)
    if np.any(positives == 0) or np.any(positives == len(selected)):
        raise ValueError('Ordinal training lacks both classes at some threshold')
    weight = np.minimum(20., (len(selected) - positives) / positives)
    output_size = 3 if method.startswith('ordinal') or method.startswith('s6_') else 1
    network = _network(matrix.shape[1], args.width, output_size).to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    mask = torch.as_tensor(_mask(method, matrix.shape[1]), device=device)
    x = torch.as_tensor(matrix, device=device) * mask
    q = torch.as_tensor(quality, device=device)
    y = torch.as_tensor(targets, device=device)
    w = torch.as_tensor(weight, device=device)
    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(args.epochs):
        network.train()
        losses = []
        for start in range(0, len(selected), args.batch_size):
            if start == 0:
                order = rng.permutation(selected)
            ids = torch.as_tensor(order[start:start+args.batch_size], device=device)
            prediction = network(x[ids])
            if method == 'binary70':
                loss = F.binary_cross_entropy_with_logits(
                    prediction[:, 0], y[ids, 2], pos_weight=w[2])
            elif method == 'iou_regression':
                loss = F.smooth_l1_loss(torch.sigmoid(prediction[:, 0]), q[ids])
            else:
                bce = F.binary_cross_entropy_with_logits(
                    prediction, y[ids], reduction='none')
                loss = (bce * torch.where(y[ids] > 0, w[None, :], 1.)).mean()
                probabilities = torch.sigmoid(prediction)
                loss = loss + .1 * (F.relu(probabilities[:, 1]-probabilities[:, 0]) +
                                    F.relu(probabilities[:, 2]-probabilities[:, 1])).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 5.)
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
        print(f'{method} seed={seed} epoch={epoch+1}/{args.epochs} loss={history[-1]:.5f}',
              flush=True)
    return network.cpu().eval(), history


def _predict(network, matrix, method, batch_size=8192):
    import torch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network = network.to(device).eval()
    mask = torch.as_tensor(_mask(method, matrix.shape[1]), device=device)
    result = np.empty(len(matrix), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(matrix), batch_size):
            x = torch.as_tensor(matrix[start:start+batch_size], device=device) * mask
            values = torch.sigmoid(network(x)).cpu().numpy()
            if method == 'iou_regression' or method == 'binary70':
                score = values[:, 0]
            else:
                values[:, 1] = np.minimum(values[:, 0], values[:, 1])
                values[:, 2] = np.minimum(values[:, 1], values[:, 2])
                score = .2*values[:, 0] + .3*values[:, 1] + .5*values[:, 2]
            result[start:start+len(score)] = score
    return result


def _evaluate(data, baselines, predictions):
    from opencood.utils import common_utils
    from sklearn.metrics import roc_auc_score
    result = {}
    for weather in WEATHERS:
        condition = data['rows'][weather]
        stats = {name: _stats() for name in METHODS}
        global_stats = {name: _stats() for name in METHODS}
        scenes = {name: defaultdict(_stats) for name in METHODS}
        counts = {name: Counter() for name in METHODS}
        for sample in data['indices']:
            rows = condition['rows'][sample]
            gt = data['targets'][weather][sample]
            fixed = baselines[weather]['frames'][sample]
            polygons = common_utils.convert_format(np.stack([r['corners'] for r in rows]))
            overlap = lambda candidate, others: common_utils.compute_iou(
                polygons[candidate], polygons[others])
            within = np.asarray([r['within_range'] for r in rows], dtype=bool)
            original_scores = [r['score'] for r in rows]
            scene = str(data['scene_map'][str(sample)])
            for name in METHODS:
                values = [float(predictions[name][r['vector_index']])
                          if r['source_count'] >= 2 else r['score'] for r in rows]
                order, _, _ = _oracle_order(fixed['original_order'],
                                            fixed['suppressors'], values,
                                            np.ones(len(rows), dtype=bool))
                kept, _ = _nms_ordered(order, overlap, condition['threshold'])
                selected = [index for index in kept if within[index]][:fixed['budget']]
                frame = _stats()
                _evaluate_frame(frame, rows, selected, original_scores, gt)
                _merge_stats(stats[name], frame)
                _merge_stats(scenes[name][scene], frame)
                _evaluate_frame(global_stats[name], rows, selected, values, gt)
                matched = _matched_gt(rows, selected, gt)
                counts[name]['new'] += len(matched-fixed['matched_top'])
                counts[name]['lost'] += len(fixed['matched_top']-matched)
                counts[name]['output'] += len(selected)
                counts[name]['budget'] += fixed['budget']
                counts[name]['edges'] += sum(values[low] > values[high]
                                             for low, high in fixed['target_edges'])
        weather_result = {}
        for name in METHODS:
            count = counts[name]
            summary = _method_summary(
                stats[name], scenes[name], count['output'],
                count['output']/max(1, count['budget']), count['new'],
                count['lost'], count['edges'], baselines[weather]['target_direct_edges'])
            summary['ap_using_learned_scores'] = {
                'frame_order': _ap(global_stats[name], False),
                'global_sort': _ap(global_stats[name], True)}
            top_tp = sum(baselines[weather]['stats']['top256_fused'][.7]['tp'])
            if (summary['tp70']-top_tp != count['new']-count['lost'] or
                    summary['tp70']+summary['fp70'] != count['output']):
                raise AssertionError(f'{weather}/{name}: TP/FP identity mismatch')
            bands = {}
            for band, low in (('low_score', True), ('high_score', False)):
                chosen = [r for sample in data['indices']
                          for r in condition['rows'][sample]
                          if r['within_range'] and r['source_count'] >= 2
                          and (r['score'] <= .2) == low
                          and (r['quality'] >= .7 or r['quality'] < .5)]
                labels = np.asarray([r['quality'] >= .7 for r in chosen], dtype=np.int8)
                scores = np.asarray([predictions[name][r['vector_index']]
                                     for r in chosen], dtype=np.float32)
                bands[band] = {
                    'good': int(labels.sum()), 'bad': int(len(labels)-labels.sum()),
                    'auc': float(roc_auc_score(labels, scores))
                           if len(np.unique(labels)) == 2 else None}
            summary['within_score_band_quality_auc'] = bands
            weather_result[name] = summary
        result[weather] = weather_result
        print(f'{weather}: exact NMS and AP replay complete', flush=True)
    return result


def _write_markdown(report, path):
    lines = ['# S0 / S6 候选可靠性验证', '',
             '开发集结果；9 个 validation 场景已经多次参与研究，不构成独立测试。',
             '主 AP 沿用原融合分数，学习值只改变原 NMS 抑制组内顺序。',
             'GT 仅用于训练标签与事后评价。S6 是原缓存特征的确定性变换。', '',
             '| 天气 | 原流程 AP70 | 二分类 | IoU 回归 | S0 有序 | +cls 稳定性 | +几何稳定性 | S6 两者 | 打乱特征 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for weather in WEATHERS:
        base = report['baselines'][weather]['scorepass']['ap_frame_order']['ap70']
        means = [float(np.mean([report['seeds'][str(seed)][weather][method]
                                ['ap_frame_order']['ap70']
                                for seed in report['protocol']['seeds']]))
                 for method in METHODS]
        lines.append('| '+weather+' | '+f'{base:.4f}'+' | '+' | '.join(
            f'{value:.4f}' for value in means)+' |')
    screen = report['investment_screen']
    lines += ['', '## 预定继续门槛', '',
              f"S0：{'通过' if screen['s0']['pilot_gate_pass'] else '未通过'}；"
              f"S6：{'通过' if screen['s6']['pilot_gate_pass'] else '未通过'}。",
              screen['rule'], '']
    lines += ['', 'JSON 含每个种子、天气、方法的 AP30/50/70、TP/FP、GT 新增/丢失、'
              '直接抑制边及逐场景剔除结果。先比较 S0 与二分类/IoU 回归，'
              '再比较 S6 两者与 S0、单头及打乱特征对照。',
              '旧 validation 上的任何提升仍需新场景复验。', '']
    path.write_text('\n'.join(lines), encoding='utf-8')


def _screen(report):
    """Fixed investment screen; reused validation remains exploratory."""
    comparisons = {'s0': ('ordinal', ('binary70', 'iou_regression')),
                   's6': ('s6_both', ('ordinal', 's6_cls', 's6_geom',
                                      's6_shuffled'))}
    outcome = {}
    for hypothesis, (proposed, controls) in comparisons.items():
        weather_rows = {}
        for weather in WEATHERS:
            deltas, net, scene_worst = [], [], []
            for seed in report['protocol']['seeds']:
                methods = report['seeds'][str(seed)][weather]
                candidate = methods[proposed]
                base = report['baselines'][weather]['scorepass']
                references = [base] + [methods[name] for name in controls]
                ap = candidate['ap_frame_order']['ap70']
                deltas.append(ap-max(row['ap_frame_order']['ap70']
                                     for row in references))
                net.append(candidate['new_matched_gt_vs_top256']-
                           candidate['lost_matched_gt_vs_top256'])
                scene_worst.append(min(
                    candidate['leave_one_scene_out_ap70_frame'][scene] -
                    max(row['leave_one_scene_out_ap70_frame'][scene]
                        for row in references)
                    for scene in candidate['leave_one_scene_out_ap70_frame']))
            weather_rows[weather] = {
                'mean_ap70_delta_vs_best_reference': float(np.mean(deltas)),
                'mean_net_new_gt_vs_top256': float(np.mean(net)),
                'mean_worst_leave_one_scene_delta': float(np.mean(scene_worst))}
        good_weather = sum(
            weather_rows[w]['mean_ap70_delta_vs_best_reference'] >= .005 and
            weather_rows[w]['mean_net_new_gt_vs_top256'] > 0 and
            weather_rows[w]['mean_worst_leave_one_scene_delta'] > 0
            for w in ('fog', 'rain', 'snow'))
        clean = weather_rows['clean']['mean_ap70_delta_vs_best_reference'] >= -.001
        outcome[hypothesis] = {
            'by_weather': weather_rows, 'adverse_weather_pass_count': good_weather,
            'clean_preserved': bool(clean),
            'pilot_gate_pass': bool(good_weather >= 2 and clean)}
    outcome['rule'] = ('At least two adverse weathers: mean original-frame-order '
                       'AP70 gain >=0.005 versus original detector and strongest '
                       'control, net new GT >0, and positive worst leave-one-scene '
                       'margin. Clean decrease <=0.001. Passing is only an '
                       'investment screen on repeatedly used validation scenes.')
    return outcome


def run(args):
    import torch

    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    train = _load(args.train_root, 'train')
    evaluation = _load(args.eval_root, 'validation')
    if train['root'] == evaluation['root']:
        raise ValueError('Train and evaluation roots must differ')
    _validate_frozen_contract(train['metadata'], evaluation['metadata'])
    if train['layout'] != evaluation['layout']:
        raise ValueError('Training/evaluation candidate features differ')
    expected = train['metadata']['arm_sha256']['F']
    heads = _head_weights(args.f_checkpoint, expected,
                          train['layout']['candidate'][1]-16)
    train_s6, train_replay = _s6_features(train, heads)
    eval_s6, eval_replay = _s6_features(evaluation, heads)
    train_matrix, train_negative = _input(
        train, train_s6, _shuffled(train_s6, train['records'], 20260930))
    eval_matrix, eval_negative = _input(
        evaluation, eval_s6, _shuffled(eval_s6, evaluation['records'], 20260931))
    valid = [r['vector_index'] for r in train['records']
             if r['within_range'] and r['source_count'] >= 2]
    mean = train_matrix[valid].mean(0, dtype=np.float64).astype(np.float32)
    std = np.maximum(train_matrix[valid].std(0, dtype=np.float64).astype(np.float32), 1e-4)
    matrices = (_normalized(train_matrix, mean, std),
                _normalized(eval_matrix, mean, std),
                _normalized(train_negative, mean, std),
                _normalized(eval_negative, mean, std))
    del train_matrix, eval_matrix, train_negative, eval_negative
    baselines = _prepare_baselines(evaluation, args.reproduction_tolerance)
    baseline_report = _baseline_summaries(evaluation, baselines)
    source_hash = _sha256(Path(__file__))
    seeds = _parse_seeds(args.seeds)
    output.mkdir(parents=True, exist_ok=False)
    (output/'checkpoints').mkdir()
    report = {'protocol': {
        'train_root': str(train['root']), 'eval_root': str(evaluation['root']),
        'validation_status': 'reused_exploratory_validation_9_scenes',
        'f_checkpoint_sha256': expected,
        'train_audit_sha256': train['provenance'][str(train['root']/'candidate_audit.json')],
        'eval_audit_sha256': evaluation['provenance'][str(evaluation['root']/'candidate_audit.json')],
        'implementation_sha256': source_hash, 'seeds': seeds,
        'epochs': args.epochs, 'width': args.width, 'batch_size': args.batch_size,
        'learning_rate': args.learning_rate, 'weight_decay': args.weight_decay,
        'thresholds': THRESHOLDS, 'ordinal_score_weights': (.2, .3, .5),
        'fixed_last_epoch': True, 'selection': 'original NMS components; original final budget',
        'primary_ap': 'original fused score; frame order',
        'gt_use': 'train labels and evaluation only',
        's6_train_replay': train_replay, 's6_eval_replay': eval_replay,
        's6_shuffled_control': 'within weather and original score band'},
        'baselines': baseline_report, 'seeds': {}, 'training_history': {}}
    for seed in seeds:
        predictions = {}
        histories = {}
        for method in METHODS:
            train_x = matrices[2] if method == 's6_shuffled' else matrices[0]
            eval_x = matrices[3] if method == 's6_shuffled' else matrices[1]
            network, history = _fit(train_x, train['records'], method, seed, args)
            predictions[method] = _predict(network, eval_x, method)
            torch.save({'state': network.state_dict(), 'method': method,
                        'seed': seed, 'mean': torch.as_tensor(mean),
                        'std': torch.as_tensor(std), 'implementation_sha256': source_hash,
                        'f_checkpoint_sha256': expected,
                        'train_audit_sha256': report['protocol']['train_audit_sha256']},
                       output/'checkpoints'/f'{method}_seed{seed}.pt')
            histories[method] = history
            del network
        report['seeds'][str(seed)] = _evaluate(evaluation, baselines, predictions)
        report['training_history'][str(seed)] = histories
        (output/'candidate_s0_s6.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    report['investment_screen'] = _screen(report)
    (output/'candidate_s0_s6.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    _write_markdown(report, output/'candidate_s0_s6.md')
    if _sha256(Path(__file__)) != source_hash:
        raise RuntimeError('Source code changed during remote experiment')
    print(f'S0/S6 COMPLETE: {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-root', required=True)
    parser.add_argument('--eval-root', required=True)
    parser.add_argument('--f-checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--seeds', default='20260929,20260930,20260931')
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--reproduction-tolerance', type=float, default=1e-6)
    args = parser.parse_args()
    if (args.epochs < 1 or args.width < 4 or args.batch_size < 1 or
            args.learning_rate <= 0 or args.weight_decay < 0 or
            args.reproduction_tolerance <= 0):
        parser.error('Invalid training settings')
    run(args)


if __name__ == '__main__':
    main()
