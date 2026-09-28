"""Scheme A: fixed-anchor, leave-one-peer-out candidate audit.

The frozen detector is not trained here. A regularized logistic regression is
only an out-of-fold information probe. GT IoU defines labels after extraction;
it is never included in a model feature. Leave-one-out is a conditional model
response, not a unique/additive causal contribution of one vehicle.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


WEATHERS = ('clean', 'fog', 'rain', 'snow')
GROUPS = ('low_good', 'low_bad', 'high_good', 'high_bad')
MODELS = ('candidate_only', 'candidate_plus_proxy',
          'candidate_plus_intervention', 'candidate_plus_proxy_intervention')
COMPARISONS = (('candidate_plus_proxy_intervention', 'candidate_only'),
               ('candidate_plus_proxy_intervention', 'candidate_plus_proxy'))
PROXY_NAMES = ('source_count', 'ego_score', 'max_peer_score',
               'mean_source_score', 'std_source_score', 'source_agreement',
               'mean_source_box_iou', 'min_source_box_iou',
               'max_source_box_iou', 'score_geometry_source_mismatch')
INTERVENTION_NAMES = (
    'max_logit_drop', 'min_logit_drop', 'mean_logit_drop', 'std_logit_drop',
    'max_score_drop', 'mean_score_drop',
    'min_box_iou_to_full', 'mean_box_iou_to_full', 'std_box_iou_to_full',
    'max_center_shift', 'mean_center_shift',
    'max_size_shift_l1', 'mean_size_shift_l1',
    'max_yaw_shift_abs', 'mean_yaw_shift_abs',
    'max_positive_logit_drop_stable_box',
    'max_positive_logit_drop_changed_box',
    'largest_logit_drop_box_iou',
    'largest_box_change_logit_drop',
    'score_geometry_response_source_mismatch',
    'positive_score_response_exists', 'geometry_response_exists',
)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _row_file(root, weather):
    choices = (root / weather / 'candidate_rows.jsonl',
               root / f'{weather}_candidate_rows.jsonl')
    found = [path for path in choices if path.is_file()]
    if len(found) != 1:
        raise ValueError(f'Expected exactly one {weather} candidate row file')
    return found[0]


def _finite(values, label, length=None):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or (length is not None and len(array) != length) or not np.isfinite(array).all():
        raise ValueError(f'Invalid {label}: expected finite one-dimensional values')
    return array


def _label(score, quality):
    if quality >= .7:
        return 'low_good' if score <= .2 else 'high_good'
    if quality < .5:
        return 'low_bad' if score <= .2 else 'high_bad'
    return 'middle'


def _proxy_features(row):
    count = int(row['source_count'])
    scores = _finite(row['source_scores_same_anchor'], 'source scores', count)
    ious = _finite(row['source_box_iou_same_anchor'], 'source box IoU', count)
    if count < 2 or (scores < 0).any() or (scores > 1).any() or (ious < 0).any() or (ious > 1).any():
        raise ValueError('Scheme A requires at least two valid sources')
    score_source, box_source = int(np.argmax(scores)), int(np.argmax(ious))
    if (int(row['score_source']) != score_source or int(row['geometry_source']) != box_source
            or bool(row['source_proxy_disagreement']) != (score_source != box_source)):
        raise ValueError('Stored source proxy identities disagree with their arrays')
    return np.asarray((count, scores[0], scores[1:].max(), scores.mean(),
                       scores.std(), (scores * ious).max(), ious.mean(),
                       ious.min(), ious.max(), float(score_source != box_source)),
                      dtype=np.float64)


def _intervention_features(row):
    count = int(row['source_count'])
    effects = row.get('leave_one_source_out')
    if effects is None or set(effects) != {str(i) for i in range(1, count)}:
        raise ValueError('Ablation row must contain every peer and must keep ego')
    ordered = [effects[str(i)] for i in range(1, count)]
    def values(name):
        return _finite([item[name] for item in ordered], name, count-1)
    logits = values('logit_drop')
    scores = values('score_drop')
    box_iou = values('box_iou_to_full')
    center = values('center_shift')
    size = values('size_shift_l1')
    yaw = values('yaw_shift_abs')
    if ((box_iou < 0).any() or (box_iou > 1).any() or (center < 0).any()
            or (size < 0).any() or (yaw < 0).any()
            or (yaw > np.pi + 1e-6).any()):
        raise ValueError('Invalid leave-one-peer-out box response')
    positive = np.maximum(logits, 0.)
    change = 1. - box_iou
    strongest_score = int(np.argmax(logits))
    strongest_geometry = int(np.argmax(change))
    has_score_response = bool(logits[strongest_score] > 0)
    has_geometry_response = bool(change[strongest_geometry] > 1e-4)
    return np.asarray((
        logits.max(), logits.min(), logits.mean(), logits.std(),
        scores.max(), scores.mean(), box_iou.min(), box_iou.mean(),
        box_iou.std(), center.max(), center.mean(), size.max(), size.mean(),
        yaw.max(), yaw.mean(), (positive * box_iou).max(),
        (positive * change).max(), box_iou[strongest_score],
        logits[strongest_geometry],
        float(has_score_response and has_geometry_response and
              strongest_score != strongest_geometry),
        float(has_score_response), float(has_geometry_response),
    ), dtype=np.float64)


def _candidate_features(row, joined):
    # The joined detector input belongs to the original fused candidate cell.
    # Classification and regression remain anchor-specific.
    score = float(row['score'])
    logit = float(row['fused_logit'])
    regression = _finite(row['fused_regression_deltas'], 'fused regression', 7)
    decoded = _finite(row['fused_decoded_box'], 'fused decoded box', 7)
    joined = _finite(joined, 'fused detector input')
    if not 0 <= score <= 1 or abs(1 / (1 + np.exp(-np.clip(logit, -60, 60))) - score) > 1e-5:
        raise ValueError('Fused score and logit disagree')
    return np.concatenate(([score, logit], regression, decoded, joined))


def load_rows(root, metadata, arm):
    if metadata.get('audit_pool') != 'top_256' or metadata.get('stage0_only'):
        raise ValueError('Scheme A requires top_256 candidate-source rows')
    if not metadata.get('candidate_feature_cache'):
        raise ValueError('Input was not generated with --candidate-features')
    if metadata.get('ablation_arm') not in (arm, 'both'):
        raise ValueError(f'Input has no leave-one-out data for {arm}')
    selected_by_weather = {}
    provenance = {}
    excluded = Counter()
    for weather in WEATHERS:
        path = _row_file(root, weather)
        provenance[str(path)] = _sha256(path)
        cache = {}
        seen = set()
        rows = []
        with path.open(encoding='utf-8') as stream:
            for line_no, line in enumerate(stream, 1):
                raw = json.loads(line)
                if raw['arm'] != arm:
                    continue
                if raw['weather'] != weather:
                    raise ValueError(f'{path}:{line_no}: weather mismatch')
                sample, cid = int(raw['sample_index']), int(raw['candidate_id'])
                if (sample, cid) in seen:
                    raise ValueError(f'{path}:{line_no}: duplicate candidate')
                seen.add((sample, cid))
                if not raw['within_range']:
                    excluded['out_of_range'] += 1
                    continue
                if int(raw['source_count']) < 2:
                    excluded['single_source'] += 1
                    continue
                if 'leave_one_source_out' not in raw:
                    excluded['not_in_ablation_frames'] += 1
                    continue
                name = raw.get('fused_feature_cache')
                expected = f'{arm}_features_{sample}.npz'
                if name != expected:
                    raise ValueError(f'{path}:{line_no}: feature cache name differs from fixed frame')
                if sample not in cache:
                    feature_path = root / weather / name
                    if not feature_path.is_file():
                        raise FileNotFoundError(feature_path)
                    provenance[str(feature_path)] = _sha256(feature_path)
                    with np.load(feature_path, allow_pickle=False) as packed:
                        ids = np.asarray(packed['candidate_id'], dtype=np.int64)
                        vectors = np.asarray(packed['feature'], dtype=np.float64)
                    if vectors.ndim != 2 or len(ids) != len(vectors) or len(np.unique(ids)) != len(ids):
                        raise ValueError(f'{feature_path}: invalid candidate feature matrix')
                    cache[sample] = {int(i): vector for i, vector in zip(ids, vectors)}
                if cid not in cache[sample]:
                    raise ValueError(f'{path}:{line_no}: candidate absent from feature cache')
                score = float(raw['score'])
                quality = float(raw['max_gt_iou'])
                if not (0 <= quality <= 1):
                    raise ValueError('Invalid post-hoc GT quality')
                group = _label(score, quality)
                if group == 'middle':
                    excluded['middle_quality'] += 1
                    continue
                rows.append({
                    'weather': weather, 'arm': arm, 'sample_index': sample,
                    'candidate_id': cid, 'group': group, 'quality': quality,
                    'candidate': _candidate_features(raw, cache[sample][cid]),
                    'proxy': _proxy_features(raw),
                    'intervention': _intervention_features(raw),
                    'posthoc': {
                        'score_raised_without_gt_iou_gain': any(
                            effect['score_drop'] > .01 and effect['gt_quality_change'] <= 0
                            for effect in raw['leave_one_source_out'].values()),
                        # This is a source-score proxy, not proof of a unique peer TP.
                        'peer_only_score_support': (
                            raw['source_scores_same_anchor'][0] <= .2 and
                            sum(score > .2 for score in
                                raw['source_scores_same_anchor'][1:]) == 1),
                    },
                })
        expected_frames = set(map(int, metadata['conditions'][weather]['ablation_frames']))
        actual_frames = set(cache)
        # A frame with no geometry-valid top256 candidates needs no feature file.
        if not actual_frames <= expected_frames or not rows:
            raise ValueError(f'{weather}: ablation cache has wrong frames or no labelled rows')
        selected_by_weather[weather] = rows
    return selected_by_weather, {'input_sha256': provenance, 'excluded': dict(excluded)}


def _profiles(rows):
    result = {}
    for group in GROUPS:
        chosen = [row for row in rows if row['group'] == group]
        result[group] = {'candidates': len(chosen),
                         'frames': len({row['sample_index'] for row in chosen})}
        if chosen:
            matrix = np.stack([row['intervention'] for row in chosen])
            result[group]['score_raised_without_gt_iou_gain_count'] = sum(
                row['posthoc']['score_raised_without_gt_iou_gain'] for row in chosen)
            result[group]['peer_only_score_support_count'] = sum(
                row['posthoc']['peer_only_score_support'] for row in chosen)
            result[group]['intervention'] = {
                name: {'median': float(np.median(matrix[:, i])),
                       'q25': float(np.percentile(matrix[:, i], 25)),
                       'q75': float(np.percentile(matrix[:, i], 75))}
                for i, name in enumerate(INTERVENTION_NAMES)}
    return result


def _feature_effect_sizes(rows, band):
    wanted = ('low_good', 'low_bad') if band == 'low' else ('high_good', 'high_bad')
    selected = [row for row in rows if row['group'] in wanted]
    y = np.asarray([int(row['group'].endswith('good')) for row in selected], dtype=np.int8)
    if len(np.unique(y)) != 2:
        return {}
    matrix = np.stack([row['intervention'] for row in selected])
    return {name: {'feature_auc': float(roc_auc_score(y, matrix[:, index])),
                   'rank_biserial_effect': float(2*roc_auc_score(y, matrix[:, index])-1)}
            for index, name in enumerate(INTERVENTION_NAMES)}


def _bootstrap(y, groups, predictions, repetitions, seed):
    unique = np.unique(groups)
    members = {key: np.flatnonzero(groups == key) for key in unique}
    rng = np.random.default_rng(seed)
    out = {}
    for enhanced, baseline in COMPARISONS:
        for metric, measure in (('roc_auc', roc_auc_score),
                                ('pr_auc', average_precision_score)):
            observed = float(measure(y, predictions[enhanced])
                             - measure(y, predictions[baseline]))
            draws = []
            for _ in range(repetitions):
                selected = rng.choice(unique, size=len(unique), replace=True)
                ids = np.concatenate([members[key] for key in selected])
                if len(np.unique(y[ids])) == 2:
                    draws.append(float(measure(y[ids], predictions[enhanced][ids])
                                       - measure(y[ids], predictions[baseline][ids])))
            out[f'{enhanced}_minus_{baseline}/{metric}'] = {
                'difference': observed,
                'bootstrap_95pct': [float(x) for x in np.percentile(draws, [2.5, 97.5])]
                                   if draws else None,
                'valid_draws': len(draws),
            }
    return out


def probe(rows, band, group_map, folds, bootstrap, seed):
    wanted = ('low_good', 'low_bad') if band == 'low' else ('high_good', 'high_bad')
    chosen = [row for row in rows if row['group'] in wanted]
    y = np.asarray([int(row['group'].endswith('good')) for row in chosen], dtype=np.int8)
    groups = np.asarray([str(group_map.get(str(row['sample_index']), row['sample_index']))
                         for row in chosen])
    summary = {'candidates': len(chosen), 'good': int(y.sum()),
               'bad': int(len(y)-y.sum()), 'groups': len(np.unique(groups))}
    if len(np.unique(groups)) < folds or len(np.unique(y)) != 2:
        return dict(summary, status='insufficient_groups_or_classes')
    splits = list(GroupKFold(n_splits=folds).split(np.zeros(len(y)), y, groups))
    if any(len(np.unique(y[train])) != 2 or len(np.unique(y[test])) != 2
           for train, test in splits):
        return dict(summary, status='one_fold_lacks_a_class')
    matrices = {}
    for name in MODELS:
        matrices[name] = np.stack([
            np.concatenate((row['candidate'],
                            row['proxy'] if name in ('candidate_plus_proxy',
                                                      'candidate_plus_proxy_intervention') else (),
                            row['intervention'] if name in ('candidate_plus_intervention',
                                                             'candidate_plus_proxy_intervention') else ()))
            for row in chosen])
        if not np.isfinite(matrices[name]).all():
            raise ValueError(f'{name} contains non-finite probe features')
    predictions = {}
    metrics = {}
    for name, matrix in matrices.items():
        oof = np.full(len(chosen), np.nan)
        for train, test in splits:
            classifier = make_pipeline(StandardScaler(), LogisticRegression(
                C=.1, max_iter=1000, solver='liblinear', class_weight='balanced'))
            classifier.fit(matrix[train], y[train])
            oof[test] = classifier.predict_proba(matrix[test])[:, 1]
        if not np.isfinite(oof).all():
            raise AssertionError('Missing out-of-fold predictions')
        predictions[name] = oof
        metrics[name] = {'roc_auc': float(roc_auc_score(y, oof)),
                         'pr_auc': float(average_precision_score(y, oof)),
                         'feature_count': matrix.shape[1]}
    return dict(summary, status='available', models=metrics,
                paired_group_bootstrap=_bootstrap(y, groups, predictions, bootstrap, seed),
                test_groups_by_fold=[sorted(map(str, np.unique(groups[test])))
                                     for _, test in splits])


def _gate(conditions):
    # Development-only decision screen, fixed before reading Scheme A results.
    signals = {}
    for weather in WEATHERS:
        signals[weather] = {}
        for band in ('low', 'high'):
            result = conditions[weather]['discrimination'][band]
            comparisons = result.get('paired_group_bootstrap', {})
            if not comparisons:
                signals[weather][band] = None
                continue
            signals[weather][band] = {}
            for baseline in ('candidate_only', 'candidate_plus_proxy'):
                prefix = f'candidate_plus_proxy_intervention_minus_{baseline}/'
                auc, pr = comparisons[prefix + 'roc_auc'], comparisons[prefix + 'pr_auc']
                signals[weather][band][baseline] = {
                    'delta_auc': auc['difference'],
                    'lower_95pct': auc['bootstrap_95pct'][0]
                    if auc['bootstrap_95pct'] else None,
                    'delta_pr_auc': pr['difference'],
                }
    weather_passes = {weather: all(
        signals[weather][band] is not None
        and all(value['delta_auc'] >= .03
                and value['lower_95pct'] is not None
                and value['lower_95pct'] > 0
                and value['delta_pr_auc'] > 0
                for value in signals[weather][band].values())
        for band in ('low', 'high')) for weather in WEATHERS[1:]}
    clean_safe = all(signals['clean'][band] is not None
                     and all(value['delta_auc'] >= -.01
                             for value in signals['clean'][band].values())
                     for band in ('low', 'high'))
    return {'scope': 'exploratory development screen, not model or AP success',
            'weather_passes': weather_passes, 'clean_delta_auc_at_least_minus_0.01': clean_safe,
            'continue_to_model_design': sum(weather_passes.values()) >= 2 and clean_safe,
            'signals': signals,
            'note': 'Formal method still needs fixed-pool NMS/AP and held-out evaluation.'}


def render_markdown(report):
    lines = ['# 方案 A：固定候选的逐来源干预审计', '',
             '每辆邻车依次移除后重新前向；候选池、anchor 和完整模型框始终固定。',
             'GT IoU 仅给候选贴离线标签；探针输入不含 GT。',
             '跨折 AUC 只检验信息存在，尚不代表最终 AP。', '',
             '每种条件的 `distributions_*.svg` 展示三项主要干预量的好坏框分布。',
             'JSON 中的 `rank_biserial_effect` 是单个特征的排序效应量；正值表示好框通常更高。', '',
             '| 条件 | 分数段 | 好/坏框 | 候选自身 AUC | 加来源代理 AUC | 加干预 AUC | 全部 AUC | 全部－来源代理 AUC [95%区间] |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for weather in WEATHERS:
        for band in ('low', 'high'):
            item = report['conditions'][weather]['discrimination'][band]
            if item['status'] != 'available':
                lines.append(f"| {weather} | {band} | {item['good']}/{item['bad']} | "
                             f"{item['status']} | | | | |")
                continue
            models = item['models']
            delta = item['paired_group_bootstrap'][
                'candidate_plus_proxy_intervention_minus_candidate_plus_proxy/roc_auc']
            interval = delta['bootstrap_95pct']
            interval_text = (f"{delta['difference']:+.3f} [{interval[0]:+.3f}, "
                             f"{interval[1]:+.3f}]") if interval else '无有效区间'
            lines.append(f"| {weather} | {band} | {item['good']}/{item['bad']} | "
                         f"{models['candidate_only']['roc_auc']:.3f} | "
                         f"{models['candidate_plus_proxy']['roc_auc']:.3f} | "
                         f"{models['candidate_plus_intervention']['roc_auc']:.3f} | "
                         f"{models['candidate_plus_proxy_intervention']['roc_auc']:.3f} | "
                         f"{interval_text} |")
    gate = report['go_no_go']
    lines += ['', '## 开发门槛', '',
              f"- 至少两种天气的低/高分两组同时过门槛：{sum(gate['weather_passes'].values())}/3。",
              f"- Clean 无明显退化：{gate['clean_delta_auc_at_least_minus_0.01']}。",
              f"- 是否进入模型设计：{gate['continue_to_model_design']}。", '',
              '## 解释边界', '',
              '- 移除某邻车会改变其余来源的归一化权重，因此这些数值是条件干预响应，不可相加为来源贡献。',
              '- `gt_quality_change` 仅存于原始行作事后核对，未进入任何探针。',
              '- `peer_only_score_support` 只是同 anchor 单车分数代理，不证明目标仅一辆邻车可见。',
              '- F+D 的单车预测代理仍使用冻结基础检测头；F 与 F+D 必须分别解释。',
              '- 若没有场景映射，交叉验证仅按帧隔离，同场景泄漏仍可能存在。',
              '- 当前是重复使用过的 validation 小样本；通过门槛只允许下一步建模。',
              '- 最终是否同时救回低分好框、压制高分坏框，须在固定输出预算下重新运行 NMS/AP。', '']
    return '\n'.join(lines)


def _distribution_svg(rows, weather):
    """Dependency-free small multiples; per-group bars show candidate fractions."""
    names = ('max_logit_drop', 'min_box_iou_to_full',
             'max_positive_logit_drop_stable_box')
    width, height = 990, 590
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="30" y="32" font-size="20" font-family="Arial">'
             f'{weather}: fixed-candidate peer-removal responses</text>',
             '<rect x="30" y="44" width="12" height="12" fill="#2b76bd" opacity=".7"/>',
             '<text x="48" y="55" font-size="13" font-family="Arial">good geometry</text>',
             '<rect x="190" y="44" width="12" height="12" fill="#d85b38" opacity=".7"/>',
             '<text x="208" y="55" font-size="13" font-family="Arial">bad geometry</text>']
    for row_index, band in enumerate(('low', 'high')):
        for col_index, name in enumerate(names):
            index = INTERVENTION_NAMES.index(name)
            left = [r['intervention'][index] for r in rows
                    if r['group'] == f'{band}_good']
            right = [r['intervention'][index] for r in rows
                     if r['group'] == f'{band}_bad']
            x, y = 30 + col_index*320, 90 + row_index*250
            parts.append(f'<text x="{x}" y="{y}" font-size="14" '
                         f'font-family="Arial">{band}: {name}</text>')
            parts.append(f'<text x="{x}" y="{y+18}" font-size="11" '
                         f'font-family="Arial">good n={len(left)}; bad n={len(right)}</text>')
            if not left or not right:
                parts.append(f'<text x="{x}" y="{y+90}" font-size="13" '
                             f'font-family="Arial">insufficient candidates</text>')
                continue
            all_values = np.asarray(left + right, dtype=np.float64)
            lower, upper = np.percentile(all_values, [1, 99])
            if upper <= lower:
                upper = lower + 1e-6
            bins = np.linspace(lower, upper, 19)
            left_hist = np.histogram(np.clip(left, lower, upper), bins=bins)[0] / len(left)
            right_hist = np.histogram(np.clip(right, lower, upper), bins=bins)[0] / len(right)
            maximum = max(float(left_hist.max()), float(right_hist.max()), 1e-9)
            base_y, plot_h, bar_w = y + 170, 130, 280/18
            parts.append(f'<line x1="{x}" y1="{base_y}" x2="{x+280}" '
                         f'y2="{base_y}" stroke="#555"/>')
            for bin_index in range(18):
                for offset, fraction, color in ((0, left_hist[bin_index], '#2b76bd'),
                                                 (bar_w*.43, right_hist[bin_index], '#d85b38')):
                    bar_h = plot_h * float(fraction) / maximum
                    parts.append(f'<rect x="{x+bin_index*bar_w+offset:.2f}" '
                                 f'y="{base_y-bar_h:.2f}" width="{bar_w*.47:.2f}" '
                                 f'height="{bar_h:.2f}" fill="{color}" opacity=".75"/>')
            parts.append(f'<text x="{x}" y="{base_y+16}" font-size="11" '
                         f'font-family="Arial">{lower:.3g}</text>')
            parts.append(f'<text x="{x+245}" y="{base_y+16}" font-size="11" '
                         f'font-family="Arial">{upper:.3g}</text>')
    parts.append('</svg>')
    return '\n'.join(parts) + '\n'


def run(args):
    root = Path(args.input_root).resolve()
    meta_path = root / 'candidate_audit.json'
    metadata = json.loads(meta_path.read_text(encoding='utf-8'))
    group_map = metadata.get('validation_scene_map') or {}
    grouping = 'scene' if group_map else 'frame'
    if args.group_map:
        group_map = json.loads(Path(args.group_map).read_text(encoding='utf-8'))
        grouping = 'scene'
    if group_map:
        missing = set(map(str, metadata['validation_indices'])) - set(group_map)
        if missing:
            raise ValueError(f'Group map misses {len(missing)} validation frame IDs')
    rows, provenance = load_rows(root, metadata, args.arm)
    report = {'protocol': {
        'scope': 'fixed top256, frozen F/F+D, validation-only information probe',
        'arm': args.arm, 'weathers': list(WEATHERS), 'input_root': str(root),
        'grouping': grouping,
        'group_map': (str(Path(args.group_map).resolve()) if args.group_map else
                      'candidate_audit.validation_scene_map' if group_map else None),
        'folds': args.folds, 'bootstrap': args.bootstrap, 'seed': args.seed,
        'candidate_audit_sha256': _sha256(meta_path),
        'implementation_sha256': _sha256(Path(__file__)),
        'candidate_feature_precision': metadata.get('feature_precision'),
        'proxy_feature_names': PROXY_NAMES,
        'intervention_feature_names': INTERVENTION_NAMES,
        'labels': {'low_good': 'score <= .2 and GT BEV IoU >= .7',
                   'low_bad': 'score <= .2 and GT BEV IoU < .5',
                   'high_good': 'score > .2 and GT BEV IoU >= .7',
                   'high_bad': 'score > .2 and GT BEV IoU < .5'},
    }, 'provenance': provenance, 'conditions': {}}
    for index, weather in enumerate(WEATHERS):
        selected = rows[weather]
        report['conditions'][weather] = {
            'candidates': len(selected), 'profiles': _profiles(selected),
            'feature_effect_sizes': {band: _feature_effect_sizes(selected, band)
                                     for band in ('low', 'high')},
            'discrimination': {
                band: probe(selected, band, group_map, args.folds, args.bootstrap,
                            args.seed + 10*index + j)
                for j, band in enumerate(('low', 'high'))},
        }
    report['go_no_go'] = _gate(report['conditions'])
    report['go_no_go']['scene_grouping_satisfied'] = grouping == 'scene'
    if grouping != 'scene':
        report['go_no_go']['continue_to_model_design'] = False
        report['go_no_go']['note'] += ' Frame-only grouping cannot pass this gate.'
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    for weather in WEATHERS:
        (output / f'distributions_{weather}.svg').write_text(
            _distribution_svg(rows[weather], weather), encoding='utf-8')
    (output / 'scheme_a_intervention.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (output / 'scheme_a_intervention.md').write_text(
        render_markdown(report), encoding='utf-8')
    print(f'SCHEME A INTERVENTION PROBE COMPLETE: {output}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--arm', choices=('F', 'F+D'), default='F')
    parser.add_argument('--group-map', help='JSON: validation frame ID -> scene ID')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--bootstrap', type=int, default=200)
    parser.add_argument('--seed', type=int, default=20260928)
    args = parser.parse_args()
    if args.folds < 2 or args.bootstrap < 1:
        parser.error('Require folds >= 2 and bootstrap >= 1')
    run(args)


if __name__ == '__main__':
    main()
