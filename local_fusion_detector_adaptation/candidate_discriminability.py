"""Small, offline top256 candidate discriminability audit.

GT IoU is used only to define evaluation labels. Every model input is available
at inference. Source-only detections are proxies, not causal fusion attribution.
"""
import argparse
from collections import Counter, defaultdict
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
ARMS = ('F', 'F+D')
PROFILE_METRICS = (
    'fused_score', 'ego_score', 'max_peer_score', 'delta_fused_ego',
    'delta_fused_max_peer', 'delta_fused_max_source',
    'source_fused_iou_mean', 'source_fused_iou_max', 'source_fused_iou_spread',
    'score_source_geometry_gap', 'score_geometry_source_mismatch',
    'amplification_weak_geometry', 'absolute_change_weak_geometry',
)
SIMPLE = ('fused_score', 'source_agreement', 'source_fused_iou_mean')
MARGINALS = SIMPLE + (
    'source_count', 'ego_score', 'max_peer_score', 'max_source_score',
    'mean_source_score', 'std_source_score', 'source_fused_iou_max',
    'source_fused_iou_min', 'source_fused_iou_std', 'source_fused_iou_spread',
)
DISTORTION = MARGINALS + (
    'score_geometry_source_mismatch', 'score_source_geometry_gap',
    'delta_ego_weak_mean', 'delta_max_weak_mean',
    'delta_ego_weak_best', 'delta_max_weak_best',
    'delta_ego_iou_spread', 'delta_max_iou_spread',
    'amplification_weak_geometry',
)
MODELS = {
    'fused_score': ('fused_score',),
    'simple_agreement': SIMPLE,
    'source_marginals': MARGINALS,
    'score_geometry_distortion': DISTORTION,
}
MODEL_COMPARISONS = (
    ('source_marginals', 'simple_agreement'),
    ('score_geometry_distortion', 'source_marginals'),
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
        raise ValueError(f'Expected exactly one {weather} candidate_rows.jsonl under {root}')
    return found[0]


def _finite_unit(values, name):
    array = np.asarray(values, dtype=np.float64)
    if not array.size or not np.isfinite(array).all() or (array < 0).any() or (array > 1).any():
        raise ValueError(f'{name} must be finite and in [0, 1]')
    return array


def _features(row):
    score = float(_finite_unit([row['score']], 'score')[0])
    quality = float(_finite_unit([row['max_gt_iou']], 'max_gt_iou')[0])
    scores = _finite_unit(row['source_scores_same_anchor'], 'source_scores')
    ious = _finite_unit(row['source_box_iou_same_anchor'], 'source_box_iou')
    count = int(row['source_count'])
    if len(scores) != count or len(ious) != count:
        raise ValueError('source_count differs from source arrays')
    if count < 2:
        return None, quality
    score_source = int(np.argmax(scores))
    geometry_source = int(np.argmax(ious))
    if (row['score_source'] != score_source or row['geometry_source'] != geometry_source
            or bool(row['source_proxy_disagreement']) != (score_source != geometry_source)):
        raise ValueError('Source identity fields disagree with score/geometry arrays')
    ego = float(scores[0])
    max_peer = float(scores[1:].max())
    max_source = float(scores.max())
    mean_iou = float(ious.mean())
    max_iou = float(ious.max())
    spread = max_iou - float(ious.min())
    delta_ego = score - ego
    delta_max = score - max_source
    values = {
        'fused_score': score, 'source_count': count,
        'ego_score': ego, 'max_peer_score': max_peer,
        'max_source_score': max_source,
        'mean_source_score': float(scores.mean()),
        'std_source_score': float(scores.std()),
        'source_agreement': float(np.max(scores * ious)),
        'delta_fused_ego': delta_ego,
        'delta_fused_max_peer': score - max_peer,
        'delta_fused_max_source': delta_max,
        'source_fused_iou_mean': mean_iou,
        'source_fused_iou_max': max_iou,
        'source_fused_iou_min': float(ious.min()),
        'source_fused_iou_std': float(ious.std()),
        'source_fused_iou_spread': spread,
        'score_source_geometry_gap': max_iou - float(ious[score_source]),
        'score_geometry_source_mismatch': int(score_source != geometry_source),
        'amplification_weak_geometry': max(0., delta_max) * (1. - mean_iou),
        'absolute_change_weak_geometry': abs(delta_max) * (1. - mean_iou),
        'delta_ego_weak_mean': delta_ego * (1. - mean_iou),
        'delta_max_weak_mean': delta_max * (1. - mean_iou),
        'delta_ego_weak_best': delta_ego * (1. - max_iou),
        'delta_max_weak_best': delta_max * (1. - max_iou),
        'delta_ego_iou_spread': delta_ego * spread,
        'delta_max_iou_spread': delta_max * spread,
    }
    return values, quality


def _group(score, quality):
    if quality >= .7:
        return 'low_good' if score <= .2 else 'high_good'
    if quality < .5:
        return 'low_bad' if score <= .2 else 'high_bad'
    return 'middle'


def load_rows(root, metadata):
    rows = {weather: {arm: [] for arm in ARMS} for weather in WEATHERS}
    sources = {}
    frame_counts = Counter()
    seen = set()
    excluded = Counter()
    for weather in WEATHERS:
        path = _row_file(root, weather)
        sources[str(path)] = _sha256(path)
        with path.open(encoding='utf-8') as stream:
            for line_no, line in enumerate(stream, 1):
                raw = json.loads(line)
                arm = raw['arm']
                if raw['weather'] != weather or arm not in ARMS:
                    raise ValueError(f'{path}:{line_no}: weather/arm mismatch')
                sample = int(raw['sample_index'])
                key = (weather, arm, sample, int(raw['candidate_id']))
                if key in seen:
                    raise ValueError(f'{path}:{line_no}: duplicate candidate {key}')
                seen.add(key)
                frame_counts[(weather, arm, sample)] += 1
                if frame_counts[(weather, arm, sample)] > 256:
                    raise ValueError(f'{path}:{line_no}: more than 256 candidates in one frame')
                if not raw['within_range']:
                    excluded['out_of_range'] += 1
                    continue
                features, quality = _features(raw)
                if features is None:
                    excluded['single_source'] += 1
                    continue
                corners = raw.get('fused_bev_corners')
                if corners is not None:
                    corners = np.asarray(corners, dtype=np.float64)
                    if corners.shape != (4, 2) or not np.isfinite(corners).all():
                        raise ValueError(f'{path}:{line_no}: invalid fused_bev_corners')
                rows[weather][arm].append({
                    'key': key, 'sample_index': sample, 'candidate_id': key[-1],
                    'quality': quality, 'group': _group(features['fused_score'], quality),
                    'features': features, 'corners': corners,
                })
    expected = set(metadata['validation_indices'])
    for weather in WEATHERS:
        for arm in ARMS:
            actual = {sample for condition, branch, sample in frame_counts
                      if condition == weather and branch == arm}
            if actual != expected:
                raise ValueError(f'{weather}/{arm}: frame IDs differ from candidate_audit.json')
    return rows, {'source_sha256': sources,
                  'raw_frame_counts': {f'{w}/{a}': [frame_counts[(w, a, i)]
                                                        for i in metadata['validation_indices']]
                                       for w in WEATHERS for a in ARMS},
                  'excluded': dict(excluded)}


def _quantiles(values):
    array = np.asarray(values, dtype=np.float64)
    return {'p10': float(np.percentile(array, 10)),
            'median': float(np.median(array)),
            'p90': float(np.percentile(array, 90))}


def profiles(rows):
    result = {}
    for group in ('low_good', 'low_bad', 'high_bad', 'high_good', 'middle'):
        chosen = [row for row in rows if row['group'] == group]
        result[group] = {'candidates': len(chosen),
                         'frames': len({row['sample_index'] for row in chosen})}
        if chosen:
            result[group]['features'] = {
                name: _quantiles([row['features'][name] for row in chosen])
                for name in PROFILE_METRICS}
    return result


def _bootstrap_deltas(y, groups, predictions, repetitions, seed):
    unique = np.unique(groups)
    members = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    result = {}
    for better, baseline in MODEL_COMPARISONS:
        observed = roc_auc_score(y, predictions[better]) - roc_auc_score(y, predictions[baseline])
        draws = []
        for _ in range(repetitions):
            chosen = rng.choice(unique, size=len(unique), replace=True)
            ids = np.concatenate([members[group] for group in chosen])
            if len(np.unique(y[ids])) == 2:
                draws.append(roc_auc_score(y[ids], predictions[better][ids])
                             - roc_auc_score(y[ids], predictions[baseline][ids]))
        result[f'{better}_minus_{baseline}'] = {
            'oof_auc_difference': float(observed),
            'bootstrap_95pct': [float(x) for x in np.percentile(draws, [2.5, 97.5])]
                               if draws else None,
            'bootstrap_valid_repetitions': len(draws),
        }
    return result


def probe(rows, band, group_map, folds, bootstrap, seed):
    chosen = [row for row in rows if row['group'] in
              (('low_good', 'low_bad') if band == 'low' else
               ('high_good', 'high_bad') if band == 'high' else
               ('low_good', 'low_bad', 'high_good', 'high_bad'))]
    y = np.asarray([int(row['quality'] >= .7) for row in chosen], dtype=np.int8)
    groups = np.asarray([group_map.get(str(row['sample_index']), str(row['sample_index']))
                         for row in chosen])
    if len(np.unique(groups)) < folds or len(np.unique(y)) < 2:
        raise ValueError(f'{band}: insufficient groups or classes for {folds} folds')
    predictions = {}
    metrics = {}
    splits = list(GroupKFold(n_splits=folds).split(np.zeros(len(y)), y, groups))
    fold_assignments = []
    for train, test in splits:
        train_groups = set(groups[train])
        test_groups = set(groups[test])
        if train_groups & test_groups:
            raise AssertionError('Training and test groups overlap')
        fold_assignments.append(sorted(map(str, test_groups)))
    for name, columns in MODELS.items():
        matrix = np.asarray([[row['features'][column] for column in columns]
                             for row in chosen], dtype=np.float64)
        oof = np.full(len(y), np.nan)
        fold_metrics = []
        for train, test in splits:
            if len(np.unique(y[train])) < 2 or len(np.unique(y[test])) < 2:
                raise ValueError(f'{band}: one fold lacks good or bad candidates')
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
            model.fit(matrix[train], y[train])
            oof[test] = model.predict_proba(matrix[test])[:, 1]
            fold_metrics.append({'auc': float(roc_auc_score(y[test], oof[test])),
                                 'average_precision': float(average_precision_score(y[test], oof[test])),
                                 'test_candidates': len(test),
                                 'test_groups': len(np.unique(groups[test]))})
        if not np.isfinite(oof).all():
            raise AssertionError('Missing out-of-fold predictions')
        predictions[name] = oof
        metrics[name] = {
            'mean_fold_auc': float(np.mean([fold['auc'] for fold in fold_metrics])),
            'pooled_oof_auc': float(roc_auc_score(y, oof)),
            'pooled_oof_average_precision': float(average_precision_score(y, oof)),
            'folds': fold_metrics,
        }
    return ({'candidates': len(chosen), 'good': int(y.sum()),
             'bad': int(len(y) - y.sum()), 'groups': len(np.unique(groups)),
             'test_groups_by_fold': fold_assignments,
             'models': metrics,
             'paired_group_bootstrap': _bootstrap_deltas(y, groups, predictions,
                                                         bootstrap, seed)},
            {row['key']: {name: float(predictions[name][i]) for name in MODELS}
             for i, row in enumerate(chosen)})


def probe_or_insufficient(rows, band, group_map, folds, bootstrap, seed):
    try:
        report, scores = probe(rows, band, group_map, folds, bootstrap, seed)
        report['status'] = 'available'
        return report, scores
    except ValueError as error:
        reason = str(error)
        if ('insufficient groups or classes' not in reason
                and 'one fold lacks good or bad candidates' not in reason):
            raise
        chosen = [row for row in rows if row['group'] in
                  (('low_good', 'low_bad') if band == 'low' else
                   ('high_good', 'high_bad') if band == 'high' else
                   ('low_good', 'low_bad', 'high_good', 'high_bad'))]
        good = sum(row['quality'] >= .7 for row in chosen)
        return ({'status': 'insufficient', 'reason': reason,
                 'candidates': len(chosen), 'good': good,
                 'bad': len(chosen) - good, 'models': {}}, None)


def _paired_scene_median_interval(values, scenes, repetitions, seed):
    unique = np.unique(scenes)
    members = {scene: np.flatnonzero(scenes == scene) for scene in unique}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repetitions):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([members[scene] for scene in chosen])
        draws.append(float(np.median(values[indices])))
    return [float(value) for value in np.percentile(draws, [2.5, 97.5])]


def paired_weather(rows, arm, weather, group_map, repetitions, seed):
    clean = {row['key'][2:]: row for row in rows['clean'][arm]}
    adverse = {row['key'][2:]: row for row in rows[weather][arm]}
    shared = sorted(clean.keys() & adverse.keys())
    common = [key for key in shared if clean[key]['features']['source_count'] ==
              adverse[key]['features']['source_count']]
    fields = ('fused_score', 'source_fused_iou_mean',
              'absolute_change_weak_geometry', 'amplification_weak_geometry')
    result = {'matched_candidates': len(common),
              'shared_anchor_candidates': len(shared),
              'different_source_count_excluded': len(shared) - len(common),
              'clean_candidates': len(clean), 'weather_candidates': len(adverse),
              'selection_note': ('Only anchors in both top256 pools, within range, '
                                 'and with equal source counts are paired.')}
    if common:
        scenes = np.asarray([str(group_map.get(str(key[0]), key[0])) for key in common])
        result['weather_minus_clean'] = {}
        for index, field in enumerate(fields):
            values = np.asarray([adverse[key]['features'][field] -
                                 clean[key]['features'][field] for key in common])
            result['weather_minus_clean'][field] = {
                **_quantiles(values),
                'paired_scene_bootstrap_median_95pct': _paired_scene_median_interval(
                    values, scenes, repetitions, seed + index),
            }
        result['gt_iou_weather_minus_clean'] = _quantiles(
            [adverse[key]['quality'] - clean[key]['quality'] for key in common])
    return result


def _signed_area(polygon):
    return .5 * sum(float(polygon[i, 0] * polygon[(i + 1) % len(polygon), 1]
                            - polygon[(i + 1) % len(polygon), 0] * polygon[i, 1])
                    for i in range(len(polygon)))


def _cross(left, right):
    return float(left[0] * right[1] - left[1] * right[0])


def _bev_iou(first, second):
    """Convex quadrilateral BEV IoU; same polygon geometry as rotated NMS."""
    orientation = 1. if _signed_area(second) >= 0 else -1.
    clipped = [point for point in first]
    for edge in range(4):
        start = second[edge]
        end = second[(edge + 1) % 4]
        direction = end - start
        previous = clipped
        clipped = []
        if not previous:
            return 0.
        for index, current in enumerate(previous):
            before = previous[index - 1]
            before_side = orientation * _cross(direction, before - start)
            current_side = orientation * _cross(direction, current - start)
            before_inside = before_side >= -1e-10
            current_inside = current_side >= -1e-10
            if before_inside != current_inside:
                fraction = before_side / (before_side - current_side)
                clipped.append(before + fraction * (current - before))
            if current_inside:
                clipped.append(current)
    if len(clipped) < 3:
        return 0.
    intersection = abs(_signed_area(np.asarray(clipped)))
    union = abs(_signed_area(first)) + abs(_signed_area(second)) - intersection
    return float(intersection / union) if union > 0 else 0.


def competition(rows, oof, threshold):
    if not rows or all(row['corners'] is None for row in rows):
        return {'status': 'unavailable',
                'reason': 'Input candidate rows do not contain fused_bev_corners.'}
    if any(row['corners'] is None for row in rows):
        raise ValueError('Only some candidate rows contain fused_bev_corners')
    by_frame = defaultdict(list)
    for row in rows:
        by_frame[row['sample_index']].append(row)
    counts = Counter()
    correct = Counter()
    examples = []
    candidate_competitors = Counter()
    for frame, frame_rows in by_frame.items():
        boxes = np.asarray([row['corners'] for row in frame_rows])
        lo, hi = boxes.min(axis=1), boxes.max(axis=1)
        for i, left in enumerate(frame_rows):
            nearby = np.flatnonzero(((hi[i] > lo) & (lo[i] < hi)).all(axis=1))
            for j in nearby:
                if j <= i:
                    continue
                overlap = _bev_iou(boxes[i], boxes[j])
                if overlap <= threshold:
                    continue
                right = frame_rows[j]
                candidate_competitors[left['key']] += 1
                candidate_competitors[right['key']] += 1
                counts['all_competing_pairs'] += 1
                if {left['group'], right['group']} <= {'low_good', 'low_bad',
                                                        'high_good', 'high_bad'}:
                    good, bad = (left, right) if left['quality'] >= .7 else (right, left)
                    if good['quality'] < .7 or bad['quality'] >= .5:
                        continue
                    kind = ('low_good_high_bad' if good['group'] == 'low_good'
                            and bad['group'] == 'high_bad' else 'other_good_bad')
                    counts[kind] += 1
                    correct[f'{kind}/fused_score'] += int(
                        good['features']['fused_score'] > bad['features']['fused_score'])
                    correct[f'{kind}/source_agreement'] += int(
                        good['features']['source_agreement'] > bad['features']['source_agreement'])
                    if oof is not None:
                        for name in MODELS:
                            correct[f'{kind}/{name}_oof'] += int(oof[good['key']][name]
                                                                 > oof[bad['key']][name])
                    if kind == 'low_good_high_bad' and len(examples) < 10:
                        examples.append({'sample_index': frame,
                                         'good_candidate_id': good['candidate_id'],
                                         'bad_candidate_id': bad['candidate_id'],
                                         'bev_iou': overlap})
    pair_rates = {}
    for kind in ('low_good_high_bad', 'other_good_bad'):
        if counts[kind]:
            pair_rates[kind] = {name.split('/', 1)[1]: value / counts[kind]
                                for name, value in correct.items() if name.startswith(kind + '/')}
    return {'status': 'available', 'nms_iou_threshold': threshold,
            'pair_counts': dict(counts), 'correct_order_rate': pair_rates,
            'competitor_count_by_group': {
                group: _quantiles([candidate_competitors[row['key']] for row in rows
                                   if row['group'] == group])
                for group in ('low_good', 'low_bad', 'high_bad', 'high_good')
                if any(row['group'] == group for row in rows)},
            'low_good_high_bad_examples': examples,
            'note': ('Pairs are among within-range candidates with at least two sources. '
                     'Out-of-range candidates can also suppress boxes in the original '
                     'pre-range NMS. This does not replay full greedy NMS or AP.')}


def render_markdown(report):
    lines = [
        '# top256 候选区分性审计', '',
        'GT IoU 只用于事后标注好坏框；模型输入全部来自推理时可见的候选和来源代理。',
        '低分：融合分数 ≤0.2；高分：>0.2；好框：GT BEV IoU ≥0.7；坏框：<0.5。',
        'AUC 越高越能把好框排在坏框前；下表汇总按组交叉验证的折外预测，不代表最终 AP。', '',
        '| 天气 | 组别 | 分数段 | 简单一致性 AUC | 来源基础量 AUC | 加入分数—几何变化 AUC | 最后一步差值及 95% 区间 |',
        '|---|---|---|---:|---:|---:|---:|',
    ]
    for weather in WEATHERS:
        for arm in ARMS:
            for band in ('low', 'high'):
                item = report['conditions'][weather][arm]['discrimination'][band]
                if item['status'] != 'available':
                    lines.append(f"| {weather} | {arm} | {band} | "
                                 f"样本不足（好/坏 {item['good']}/{item['bad']}） | | | |")
                    continue
                auc = {name: item['models'][name]['pooled_oof_auc'] for name in MODELS}
                delta = item['paired_group_bootstrap'][
                    'score_geometry_distortion_minus_source_marginals']
                interval = delta['bootstrap_95pct']
                interval_text = (f"{delta['oof_auc_difference']:+.3f} "
                                 f"[{interval[0]:+.3f}, {interval[1]:+.3f}]") if interval else '无'
                lines.append(f"| {weather} | {arm} | {band} | "
                             f"{auc['simple_agreement']:.3f} | "
                             f"{auc['source_marginals']:.3f} | "
                             f"{auc['score_geometry_distortion']:.3f} | {interval_text} |")
    lines += ['', '## 候选组画像', '',
              '| 天气 | 组别 | 候选类型 | 数量 | 融合分数－最高来源分数（中位数） | 平均来源/融合框 IoU（中位数） | 分类/几何首位来源不同 |',
              '|---|---|---|---:|---:|---:|---:|']
    for weather in WEATHERS:
        for arm in ARMS:
            for group in ('low_good', 'low_bad', 'high_bad', 'high_good'):
                item = report['conditions'][weather][arm]['profiles'][group]
                if not item['candidates']:
                    continue
                features = item['features']
                # Binary median hides prevalence, so use the recorded count.
                mismatch_count = report['conditions'][weather][arm][
                    'group_mismatch_counts'].get(group, 0)
                mismatch_rate = mismatch_count / item['candidates']
                lines.append(f"| {weather} | {arm} | {group} | {item['candidates']} | "
                             f"{features['delta_fused_max_source']['median']:+.3f} | "
                             f"{features['source_fused_iou_mean']['median']:.3f} | "
                             f"{mismatch_rate:.1%} |")
    lines += ['', '`source_fused_iou_spread` 是各来源相对融合框的 IoU 范围，'
              '不能解释为来源框之间的两两离散度。其完整分位数在 JSON 中。', '',
              '## Clean 与天气的同 anchor 配对', '',
              '| 天气 | 组别 | 双方均入池候选 | 融合分数差（中位数） | 来源/融合 IoU 差（中位数） |',
              '|---|---|---:|---:|---:|']
    for weather in WEATHERS[1:]:
        for arm in ARMS:
            item = report['clean_weather_pairs'][weather][arm]
            if item['matched_candidates']:
                change = item['weather_minus_clean']
                lines.append(f"| {weather} | {arm} | {item['matched_candidates']} | "
                             f"{change['fused_score']['median']:+.4f} | "
                             f"{change['source_fused_iou_mean']['median']:+.4f} |")
    lines += ['', '## NMS 竞争', '']
    for weather in WEATHERS:
        for arm in ARMS:
            item = report['conditions'][weather][arm]['nms_competition']
            if item['status'] == 'unavailable':
                lines.append(f'- {weather}/{arm}：未记录融合框坐标，现有 JSONL 无法计算真实候选重叠。')
            else:
                lines.append(f"- {weather}/{arm}：IoU>{item['nms_iou_threshold']}; "
                             f"低分好框与高分坏框竞争对 "
                             f"{item['pair_counts'].get('low_good_high_bad', 0)} 个。")
    lines += ['', '## 解释边界', '',
              '- 来源单独预测是关系代理；它不能证明某来源对融合输出的因果贡献。',
              '- 帧分组交叉验证仍可能共享场景；提供 `--group-map` 可改为场景分组。',
              '- Clean/天气配对只包含双方都进入 top256 的 anchor，存在选池偏差。',
              '- 当前数据来自已多次分析的固定 validation 帧，不能作为论文最终独立验证。',
              '- F+D 同时调整融合与检测头，不能单独归因于融合。',
              '- 95% 区间只对已拟合的跨折预测按帧或场景重抽样；未覆盖模型重训和特征选择的不确定性。',
              '- 候选级 AUC 和成对顺序需要进一步以相同 top256 池、完整 NMS 与相同 AP 口径验证。', '']
    return '\n'.join(lines)


def run(args):
    root = Path(args.input_root).resolve()
    meta_path = root / 'candidate_audit.json'
    if not meta_path.is_file():
        raise FileNotFoundError(f'Missing candidate_audit.json: {meta_path}')
    metadata = json.loads(meta_path.read_text(encoding='utf-8'))
    if metadata.get('audit_pool') != 'top_256' or metadata.get('stage0_only'):
        raise ValueError('Input candidate audit must use audit_pool=top_256 with source rows')
    if not metadata.get('validation_indices'):
        raise ValueError('Input audit has no validation_indices')
    for weather in WEATHERS:
        stored = metadata.get('conditions', {}).get(weather, {}).get('nms_iou_threshold')
        if stored is not None and abs(float(stored) - args.nms_iou) > 1e-8:
            raise ValueError(f'{weather}: --nms-iou differs from saved NMS threshold')
    group_map = metadata.get('validation_scene_map') or {}
    if args.group_map:
        group_map = json.loads(Path(args.group_map).read_text(encoding='utf-8'))
    if group_map:
        missing = set(map(str, metadata['validation_indices'])) - set(group_map)
        if missing:
            raise ValueError(f'Group map omits {len(missing)} validation frame IDs')
    rows, provenance = load_rows(root, metadata)
    report = {
        'protocol': {
            'scope': 'offline exploratory audit on saved top256 candidate rows',
            'input_root': str(root), 'audit_pool': 'top_256',
            'validation_indices': metadata['validation_indices'],
            'arms': list(ARMS), 'weathers': list(WEATHERS),
            'score_cutoff': .2, 'good_gt_bev_iou_at_least': .7,
            'bad_gt_bev_iou_below': .5, 'in_range_only': True,
            'min_sources': 2, 'folds': args.folds,
            'bootstrap_repetitions': args.bootstrap,
            'bootstrap_group': 'scene' if group_map else 'frame',
            'group_map': (str(Path(args.group_map).resolve()) if args.group_map else
                          'candidate_audit.validation_scene_map' if group_map else None),
            'seed': args.seed, 'model_features': {k: list(v) for k, v in MODELS.items()},
            'nms_iou_threshold': args.nms_iou,
            'candidate_audit_sha256': _sha256(meta_path),
            'implementation_sha256': _sha256(Path(__file__)),
            'source_interpretation': metadata.get('source_interpretation'),
        },
        'provenance': provenance, 'conditions': {}, 'clean_weather_pairs': {},
    }
    for weather_index, weather in enumerate(WEATHERS):
        report['conditions'][weather] = {}
        for arm_index, arm in enumerate(ARMS):
            selected = rows[weather][arm]
            condition = {'candidates': len(selected), 'profiles': profiles(selected),
                         'group_mismatch_counts': dict(Counter(
                             row['group'] for row in selected
                             if row['features']['score_geometry_source_mismatch'])),
                         'discrimination': {}}
            oof_all = None
            for band_index, band in enumerate(('low', 'high')):
                result, _ = probe_or_insufficient(
                    selected, band, group_map, args.folds, args.bootstrap,
                    args.seed + 100 * weather_index + 10 * arm_index + band_index)
                condition['discrimination'][band] = result
            if selected and selected[0]['corners'] is not None:
                result, oof_all = probe_or_insufficient(
                    selected, 'all', group_map, args.folds,
                    args.bootstrap, args.seed + 1000 + 10 * weather_index + arm_index)
                condition['discrimination']['all_for_nms'] = result
            condition['nms_competition'] = competition(selected, oof_all, args.nms_iou)
            report['conditions'][weather][arm] = condition
        if weather != 'clean':
            report['clean_weather_pairs'][weather] = {
                arm: paired_weather(rows, arm, weather, group_map,
                                    args.bootstrap, args.seed + 10000 + weather_index * 10
                                    + arm_index)
                for arm_index, arm in enumerate(ARMS)}
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'top256_discriminability.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (output / 'top256_discriminability.md').write_text(
        render_markdown(report), encoding='utf-8')
    print(f'TOP256 DISCRIMINABILITY AUDIT COMPLETE: {output}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', required=True,
                        help='Root with candidate_audit.json and weather JSONL rows')
    parser.add_argument('--output-dir', required=True, help='New output directory')
    parser.add_argument('--group-map', help='Optional JSON: sample_index -> scene ID')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--bootstrap', type=int, default=200)
    parser.add_argument('--seed', type=int, default=20260928)
    parser.add_argument('--nms-iou', type=float, default=.15)
    args = parser.parse_args()
    if args.folds < 2 or args.bootstrap < 1 or not 0 < args.nms_iou < 1:
        parser.error('Require folds >= 2, bootstrap >= 1, and 0 < nms-iou < 1')
    run(args)


if __name__ == '__main__':
    main()
