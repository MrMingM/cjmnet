"""Server-side, scene-held-out top256 score/geometry hypothesis replay.

This module consumes a NEW candidate_audit extraction with all validation
frames ablated, candidate feature caches, fused BEV corners and frame targets.
It trains fixed linear information probes, scores every top256 candidate out of
fold, then replays the original rotated NMS and AP with a fixed frame budget.
GT is used only for training labels and evaluation, never as a score feature.
"""
import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .candidate_discriminability import _bev_iou, _features
from .candidate_intervention_probe import (
    INTERVENTION_NAMES, PROXY_NAMES, _candidate_features, _intervention_features,
    _proxy_features,
)


WEATHERS = ('clean', 'fog', 'rain', 'snow')
PASSIVE_NAMES = (
    'score_source_geometry_gap', 'delta_ego_weak_mean',
    'delta_max_weak_mean', 'delta_ego_weak_best',
    'delta_max_weak_best', 'delta_ego_iou_spread',
    'delta_max_iou_spread', 'amplification_weak_geometry',
)
SOURCE_GEOMETRY_NAMES = (
    'mean_source_center_shift_to_fused', 'max_source_center_shift_to_fused',
    'std_source_center_shift_to_fused', 'mean_source_pairwise_box_iou',
)
DISTORTION_GEOMETRY_NAMES = ('score_gain_times_center_shift',)
MODELS = (
    'candidate_only', 'candidate_plus_simple_agreement',
    'candidate_plus_source', 'candidate_plus_passive_distortion',
    'candidate_plus_intervention', 'candidate_plus_both',
)
METHODS = ('fused_score', 'source_agreement_raw') + MODELS
THRESHOLDS = (.3, .5, .7)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _model_vector(item, name):
    parts = [item['candidate']]
    if name == 'candidate_plus_simple_agreement':
        parts.append(np.asarray([item['simple_agreement']], dtype=np.float32))
    elif name != 'candidate_only':
        parts.append(item['proxy'])
        parts.append(item['source_geometry'])
        if name in ('candidate_plus_passive_distortion', 'candidate_plus_both'):
            parts.append(item['passive'])
            parts.append(item['distortion_geometry'])
        if name in ('candidate_plus_intervention', 'candidate_plus_both'):
            parts.append(item['intervention'])
    return np.concatenate(parts)


def _source_geometry_features(row):
    count = int(row['source_count'])
    shifts = np.asarray(row['source_box_center_shift_to_fused'], dtype=np.float64)
    pair_iou = float(row['source_box_pairwise_iou_mean'])
    if (count < 2 or shifts.shape != (count,) or not np.isfinite(shifts).all()
            or (shifts < 0).any() or not np.isfinite(pair_iou)
            or not 0 <= pair_iou <= 1):
        raise ValueError('Missing or invalid source-to-fused geometry trajectory')
    source_scores = np.asarray(row['source_scores_same_anchor'], dtype=np.float64)
    if source_scores.shape != (count,) or not np.isfinite(source_scores).all():
        raise ValueError('Invalid source scores for geometry trajectory')
    score_gain = float(row['score']) - float(source_scores.max())
    return (np.asarray((shifts.mean(), shifts.max(), shifts.std(), pair_iou),
                       dtype=np.float32),
            np.asarray((score_gain * shifts.mean(),), dtype=np.float32))


def _load_targets(root, weather, expected, provenance):
    path = root / weather / 'frame_targets.jsonl'
    if not path.is_file():
        raise FileNotFoundError(f'Missing frame GT log; re-export candidates: {path}')
    provenance[str(path)] = _sha256(path)
    targets = {}
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            raw = json.loads(line)
            sample = int(raw['sample_index'])
            if raw['weather'] != weather or sample in targets:
                raise ValueError(f'{path}: duplicate or wrong-weather target row')
            corners = np.asarray(raw['gt_bev_corners'], dtype=np.float32)
            if corners.size == 0:
                corners = corners.reshape(0, 4, 2)
            if corners.ndim != 3 or corners.shape[1:] != (4, 2) or not np.isfinite(corners).all():
                raise ValueError(f'{path}: invalid GT BEV corners')
            targets[sample] = corners
    if set(targets) != expected:
        raise ValueError(f'{weather}: GT frames differ from validation_indices')
    return targets


def load_extraction(root, arm):
    meta_path = root / 'candidate_audit.json'
    metadata = json.loads(meta_path.read_text(encoding='utf-8'))
    if metadata.get('audit_pool') != 'top_256' or metadata.get('stage0_only'):
        raise ValueError('Replay requires complete top_256 candidate-source extraction')
    if not metadata.get('candidate_feature_cache') or metadata.get('ablation_arm') not in (arm, 'both'):
        raise ValueError('Replay requires candidate features and all-peer ablations for selected arm')
    indices = [int(value) for value in metadata['validation_indices']]
    if len(indices) != len(set(indices)) or not indices:
        raise ValueError('Invalid validation_indices')
    scene_map = metadata.get('validation_scene_map') or {}
    if set(scene_map) != set(map(str, indices)):
        raise ValueError('Replay requires a complete validation_scene_map for scene-held-out folds')
    provenance = {str(meta_path): _sha256(meta_path)}
    frames = {}
    items = []
    for weather in WEATHERS:
        condition = metadata['conditions'][weather]
        if abs(float(condition['original_score_threshold']) - .2) > 1e-8:
            raise ValueError(f'{weather}: original score threshold differs from replay contract')
        if set(map(int, condition.get('ablation_frames', ()))) != set(indices):
            raise ValueError(f'{weather}: every validation frame must have peer ablation')
        threshold = float(condition['nms_iou_threshold'])
        targets = _load_targets(root, weather, set(indices), provenance)
        rows_path = root / weather / 'candidate_rows.jsonl'
        provenance[str(rows_path)] = _sha256(rows_path)
        by_frame = {sample: [] for sample in indices}
        cache = {}
        seen = set()
        with rows_path.open(encoding='utf-8') as stream:
            for line_no, line in enumerate(stream, 1):
                raw = json.loads(line)
                if raw['arm'] != arm:
                    continue
                sample, cid = int(raw['sample_index']), int(raw['candidate_id'])
                key = (weather, sample, cid)
                if raw['weather'] != weather or sample not in by_frame or key in seen:
                    raise ValueError(f'{rows_path}:{line_no}: wrong or duplicate candidate')
                seen.add(key)
                corners = np.asarray(raw.get('fused_bev_corners'), dtype=np.float32)
                if corners.shape != (4, 2) or not np.isfinite(corners).all():
                    raise ValueError(f'{rows_path}:{line_no}: missing fused BEV corners')
                score = float(raw['score'])
                quality = float(raw['max_gt_iou'])
                if not 0 <= score <= 1 or not 0 <= quality <= 1:
                    raise ValueError(f'{rows_path}:{line_no}: score/GT quality outside [0,1]')
                count = int(raw['source_count'])
                item = {'key': key, 'weather': weather, 'sample_index': sample,
                        'candidate_id': cid, 'corners': corners,
                        'within_range': bool(raw['within_range']),
                        'score': score, 'quality': quality,
                        'source_count': count}
                if count >= 2:
                    name = raw.get('fused_feature_cache')
                    expected_name = f'{arm}_features_{sample}.npz'
                    if name != expected_name or 'leave_one_source_out' not in raw:
                        raise ValueError(f'{rows_path}:{line_no}: missing fixed-candidate intervention')
                    if sample not in cache:
                        feature_path = root / weather / name
                        provenance[str(feature_path)] = _sha256(feature_path)
                        with np.load(feature_path, allow_pickle=False) as packed:
                            ids = np.asarray(packed['candidate_id'], dtype=np.int64)
                            vectors = np.asarray(packed['feature'], dtype=np.float32)
                        if vectors.ndim != 2 or len(ids) != len(vectors) or len(np.unique(ids)) != len(ids):
                            raise ValueError(f'{feature_path}: invalid candidate feature cache')
                        cache[sample] = {int(index): vector for index, vector in zip(ids, vectors)}
                    if cid not in cache[sample]:
                        raise ValueError(f'{rows_path}:{line_no}: candidate absent from feature cache')
                    passive, _ = _features(raw)
                    if passive is None:
                        raise AssertionError('Multi-source candidate has no passive features')
                    source_geometry, distortion_geometry = _source_geometry_features(raw)
                    item.update({
                        'candidate': _candidate_features(raw, cache[sample][cid]).astype(np.float32),
                        'proxy': _proxy_features(raw).astype(np.float32),
                        'passive': np.asarray([passive[name] for name in PASSIVE_NAMES],
                                              dtype=np.float32),
                        'intervention': _intervention_features(raw).astype(np.float32),
                        'source_geometry': source_geometry,
                        'distortion_geometry': distortion_geometry,
                        'simple_agreement': passive['source_agreement'],
                    })
                else:
                    # Ego-only candidates receive the same original score in
                    # every method. They remain in the NMS/AP replay.
                    item['simple_agreement'] = score
                by_frame[sample].append(item)
                items.append(item)
        if any(len(rows) > 256 for rows in by_frame.values()):
            raise ValueError(f'{weather}: top256 frame contains more than 256 candidates')
        expected_top = int(condition['pools'][arm]['top_256']['total_candidates'])
        expected_original = int(condition['pools'][arm]['original']['total_candidates'])
        actual_top = sum(map(len, by_frame.values()))
        actual_original = sum(row['score'] > .2 for frame_rows in by_frame.values()
                              for row in frame_rows)
        if actual_top != expected_top or actual_original != expected_original:
            raise ValueError(f'{weather}: saved top256 rows do not contain the full '
                             'original scorepass pool')
        frames[weather] = {'rows': by_frame, 'targets': targets,
                           'nms_iou_threshold': threshold}
    return metadata, frames, items, provenance


def cross_fit(items, scene_map, folds, seed):
    selected = [item for item in items if item['source_count'] >= 2]
    if not selected:
        raise ValueError('No multi-source candidates')
    groups = np.asarray([str(scene_map[str(item['sample_index'])]) for item in selected])
    label = np.asarray([int(item['quality'] >= .7) for item in selected], dtype=np.int8)
    trainable = np.asarray([item['within_range'] and
                            (item['quality'] >= .7 or item['quality'] < .5)
                            for item in selected], dtype=bool)
    if len(np.unique(groups[trainable])) < folds:
        raise ValueError('Insufficient independent scenes for cross-fitting')
    splits = list(GroupKFold(n_splits=folds).split(
        np.zeros(int(trainable.sum())), label[trainable], groups[trainable]))
    trainable_indices = np.flatnonzero(trainable)
    fold_groups = []
    for train, test in splits:
        test_groups = set(groups[trainable_indices[test]])
        train_groups = set(groups[trainable_indices[train]])
        if test_groups & train_groups:
            raise AssertionError('Scene leakage in cross-fitting')
        fold_groups.append(sorted(test_groups))
    if set(groups) != {group for fold in fold_groups for group in fold}:
        raise ValueError('Some scene has no labelled candidate and cannot be scored out of fold')
    predictions = {name: np.full(len(selected), np.nan, dtype=np.float32) for name in MODELS}
    for name in MODELS:
        matrix = np.stack([_model_vector(item, name) for item in selected]).astype(np.float32)
        if not np.isfinite(matrix).all():
            raise ValueError(f'{name}: non-finite inference features')
        for fold, (train_local, _) in enumerate(splits):
            train = trainable_indices[train_local]
            test = np.flatnonzero(np.isin(groups, fold_groups[fold]))
            if len(np.unique(label[train])) != 2:
                raise ValueError('Training scene fold has only one quality class')
            classifier = make_pipeline(StandardScaler(), LogisticRegression(
                C=.1, max_iter=1000, solver='liblinear', random_state=seed))
            classifier.fit(matrix[train], label[train])
            predictions[name][test] = classifier.predict_proba(matrix[test])[:, 1]
        if not np.isfinite(predictions[name]).all():
            raise AssertionError(f'{name}: missing out-of-fold score')
    for index, item in enumerate(selected):
        item['oof_scores'] = {name: float(predictions[name][index]) for name in MODELS}
    return {'folds': folds, 'test_scenes_by_fold': fold_groups,
            'trainable_candidates': int(trainable.sum()),
            'all_scored_candidates': len(selected),
            'single_source_fallback_candidates': len(items) - len(selected),
            'model_features': {
                name: int(len(_model_vector(selected[0], name))) for name in MODELS},
            'regularization_C': .1, 'solver': 'liblinear', 'seed': seed}


def _score(item, method):
    if method == 'fused_score' or item['source_count'] < 2:
        return item['score']
    if method == 'source_agreement_raw':
        return item['simple_agreement']
    return item['oof_scores'][method]


def _nms_and_budget(rows, scores, threshold, budget=None):
    """Run the repository's rotated NMS, then its range filter, then budget."""
    import torch
    from opencood.utils import box_utils
    if not rows:
        return []
    corners = torch.as_tensor(np.stack([row['corners'] for row in rows]),
                              dtype=torch.float32)
    values = torch.as_tensor(np.asarray(scores, dtype=np.float32))
    keep = box_utils.nms_rotated(corners, values, threshold)
    selected = [int(index) for index in keep if rows[int(index)]['within_range']]
    return selected if budget is None else selected[:budget]


def _stats():
    return {value: {'tp': [], 'fp': [], 'gt': 0, 'score': []} for value in THRESHOLDS}


def _evaluate_frame(stats, rows, selected, scores, gt):
    import torch
    from opencood.utils import eval_utils
    boxes = torch.as_tensor(np.asarray([rows[i]['corners'] for i in selected],
                                       dtype=np.float32).reshape(-1, 4, 2))
    values = torch.as_tensor(np.asarray([scores[i] for i in selected], dtype=np.float32))
    target = torch.as_tensor(gt, dtype=torch.float32)
    for threshold in THRESHOLDS:
        eval_utils.caluclate_tp_fp(boxes, values, target, stats, threshold)


def _ap(stats, global_sort):
    from opencood.utils import eval_utils
    return {f'ap{int(value * 100)}': float(eval_utils.calculate_ap(
        copy.deepcopy(stats), value, global_sort)[0]) for value in THRESHOLDS}


def _merge_stats(total, part):
    for threshold in THRESHOLDS:
        total[threshold]['gt'] += part[threshold]['gt']
        for name in ('tp', 'fp', 'score'):
            total[threshold][name].extend(part[threshold][name])


def _ap70_scene_bootstrap(scene_stats, repetitions, seed):
    from opencood.utils import eval_utils
    scenes = sorted(scene_stats)
    rng = np.random.default_rng(seed)
    comparisons = (('candidate_plus_both', 'candidate_only'),
                   ('candidate_plus_both', 'candidate_plus_simple_agreement'),
                   ('candidate_plus_both', 'candidate_plus_source'),
                   ('candidate_plus_passive_distortion',
                    'candidate_plus_simple_agreement'),
                   ('candidate_plus_passive_distortion', 'candidate_plus_source'),
                   ('candidate_plus_intervention', 'candidate_plus_source'))
    results = {}
    for enhanced, baseline in comparisons:
        full_enhanced, full_baseline = _stats(), _stats()
        for scene in scenes:
            _merge_stats(full_enhanced, scene_stats[scene][enhanced])
            _merge_stats(full_baseline, scene_stats[scene][baseline])
        observed = (eval_utils.calculate_ap(copy.deepcopy(full_enhanced), .7, True)[0]
                    - eval_utils.calculate_ap(copy.deepcopy(full_baseline), .7, True)[0])
        draws = []
        for _ in range(repetitions):
            sampled = rng.choice(scenes, size=len(scenes), replace=True)
            left, right = _stats(), _stats()
            for scene in sampled:
                _merge_stats(left, scene_stats[scene][enhanced])
                _merge_stats(right, scene_stats[scene][baseline])
            if left[.7]['gt'] == 0:
                continue
            draws.append(eval_utils.calculate_ap(copy.deepcopy(left), .7, True)[0]
                         - eval_utils.calculate_ap(copy.deepcopy(right), .7, True)[0])
        results[f'{enhanced}_minus_{baseline}'] = {
            'ap70_difference': float(observed),
            'scene_bootstrap_95pct': [float(value)
                                      for value in np.percentile(draws, [2.5, 97.5])]
                                     if draws else None,
            'valid_repetitions': len(draws),
        }
    return results


def _competing_pairs(rows, threshold, exact=False):
    valid = [row for row in rows if row['within_range'] and
             (row['quality'] >= .7 or row['quality'] < .5)]
    if len(valid) < 2:
        return []
    boxes = np.stack([row['corners'] for row in valid])
    lo, hi = boxes.min(axis=1), boxes.max(axis=1)
    if exact:
        from opencood.utils import common_utils
        polygons = common_utils.convert_format(boxes)
    pairs = []
    for index, first in enumerate(valid):
        nearby = np.flatnonzero(((hi[index] > lo) & (lo[index] < hi)).all(1))
        for other in nearby:
            if other <= index:
                continue
            second = valid[other]
            good, bad = (first, second) if first['quality'] >= .7 else (second, first)
            if good['quality'] < .7 or bad['quality'] >= .5:
                continue
            overlap = (float(common_utils.compute_iou(
                polygons[index], [polygons[other]])[0]) if exact
                       else _bev_iou(boxes[index], boxes[other]))
            if overlap > threshold:
                pairs.append((good, bad))
    return pairs


def replay(metadata, frames, arm, tolerance, bootstrap, seed):
    baseline = defaultdict(_stats)
    original_output_counts = Counter()
    budgeted = {weather: {name: _stats() for name in METHODS} for weather in WEATHERS}
    unbudgeted = {weather: {name: _stats() for name in METHODS} for weather in WEATHERS}
    pair_tallies = {weather: {name: Counter() for name in METHODS} for weather in WEATHERS}
    counts = {weather: {name: Counter() for name in METHODS} for weather in WEATHERS}
    scene_stats = {weather: defaultdict(lambda: {name: _stats() for name in METHODS})
                   for weather in WEATHERS}
    for weather in WEATHERS:
        condition = frames[weather]
        nms_threshold = condition['nms_iou_threshold']
        for sample in metadata['validation_indices']:
            sample = int(sample)
            rows = condition['rows'][sample]
            gt = condition['targets'][sample]
            original_scores = [row['score'] for row in rows]
            original_pool = [row for row in rows if row['score'] > .2]
            original_indices = [index for index, row in enumerate(rows) if row['score'] > .2]
            original_selected_local = _nms_and_budget(
                original_pool, [row['score'] for row in original_pool], nms_threshold)
            original_selected = [original_indices[i] for i in original_selected_local]
            budget = len(original_selected)
            original_output_counts[weather] += budget
            _evaluate_frame(baseline[weather], rows, original_selected, original_scores, gt)
            pairs = _competing_pairs(rows, nms_threshold, exact=True)
            for name in METHODS:
                scores = [_score(row, name) for row in rows]
                all_selected = _nms_and_budget(rows, scores, nms_threshold)
                fixed_selected = all_selected[:budget]
                _evaluate_frame(unbudgeted[weather][name], rows, all_selected, scores, gt)
                frame_stats = _stats()
                _evaluate_frame(frame_stats, rows, fixed_selected, scores, gt)
                _merge_stats(budgeted[weather][name], frame_stats)
                scene = str(metadata['validation_scene_map'][str(sample)])
                _merge_stats(scene_stats[weather][scene][name], frame_stats)
                counts[weather][name]['unbudgeted'] += len(all_selected)
                counts[weather][name]['budgeted'] += len(fixed_selected)
                kept = {rows[i]['key'] for i in fixed_selected}
                for good, bad in pairs:
                    if good['score'] <= .2 and bad['score'] > .2:
                        pair_tallies[weather][name]['low_good_high_bad_pairs'] += 1
                        state = ('good_kept_bad_dropped' if good['key'] in kept and bad['key'] not in kept
                                 else 'bad_kept_good_dropped' if bad['key'] in kept and good['key'] not in kept
                                 else 'both_kept' if good['key'] in kept and bad['key'] in kept
                                 else 'both_dropped')
                        pair_tallies[weather][name][state] += 1
    result = {}
    swap = 'F_fusion_F_detector' if arm == 'F' else 'FD_fusion_FD_detector'
    for weather in WEATHERS:
        observed = _ap(baseline[weather], False)
        expected = metadata['conditions'][weather]['swap_ap'][swap]
        for metric in ('ap30', 'ap50', 'ap70'):
            if abs(observed[metric] - expected[metric]) > tolerance:
                raise RuntimeError(f'{weather} {arm} scorepass {metric} replay differs: '
                                   f'{observed[metric]} versus {expected[metric]}')
        result[weather] = {
            'scorepass_reproduction': observed,
            'budgeted_ap_frame_order': {name: _ap(stats, False)
                                        for name, stats in budgeted[weather].items()},
            'budgeted_ap_global_sort': {name: _ap(stats, True)
                                       for name, stats in budgeted[weather].items()},
            'unbudgeted_ap_frame_order': {name: _ap(stats, False)
                                          for name, stats in unbudgeted[weather].items()},
            'unbudgeted_ap_global_sort': {name: _ap(stats, True)
                                         for name, stats in unbudgeted[weather].items()},
            'output_counts': {name: dict(value) for name, value in counts[weather].items()},
            'original_total_outputs': original_output_counts[weather],
            'budget_fill_rate': {
                name: value['budgeted'] / max(1, original_output_counts[weather])
                for name, value in counts[weather].items()},
            'low_good_high_bad_nms_pairs_budgeted': {
                name: dict(value) for name, value in pair_tallies[weather].items()},
            'paired_scene_bootstrap_ap70_global_budgeted': _ap70_scene_bootstrap(
                scene_stats[weather], bootstrap, seed + WEATHERS.index(weather)),
        }
    return result


def _auc_scene_bootstrap(rows, labels, scene_map, repetitions, seed):
    groups = np.asarray([str(scene_map[str(row['sample_index'])]) for row in rows])
    unique = np.unique(groups)
    members = {scene: np.flatnonzero(groups == scene) for scene in unique}
    comparisons = (
        ('candidate_plus_passive_distortion', 'candidate_plus_simple_agreement'),
        ('candidate_plus_passive_distortion', 'candidate_plus_source'),
        ('candidate_plus_both', 'candidate_only'),
        ('candidate_plus_both', 'candidate_plus_simple_agreement'),
        ('candidate_plus_both', 'candidate_plus_source'),
    )
    rng = np.random.default_rng(seed)
    predictions = {name: np.asarray([_score(row, name) for row in rows])
                   for name in {name for pair in comparisons for name in pair}}
    result = {}
    for enhanced, baseline in comparisons:
        observed = (roc_auc_score(labels, predictions[enhanced])
                    - roc_auc_score(labels, predictions[baseline]))
        draws = []
        for _ in range(repetitions):
            sampled = rng.choice(unique, size=len(unique), replace=True)
            ids = np.concatenate([members[scene] for scene in sampled])
            if len(np.unique(labels[ids])) != 2:
                continue
            draws.append(roc_auc_score(labels[ids], predictions[enhanced][ids])
                         - roc_auc_score(labels[ids], predictions[baseline][ids]))
        result[f'{enhanced}_minus_{baseline}'] = {
            'auc_difference': float(observed),
            'scene_bootstrap_95pct': [float(value) for value in
                                      np.percentile(draws, [2.5, 97.5])]
                                     if draws else None,
            'valid_repetitions': len(draws),
        }
    return result


def discrimination(items, scene_map=None, bootstrap=200, seed=20260928):
    report = {}
    for weather in WEATHERS:
        report[weather] = {}
        for band in ('low', 'high'):
            chosen = [row for row in items if row['weather'] == weather and
                      row['within_range'] and row['source_count'] >= 2 and
                      (row['score'] <= .2 if band == 'low' else row['score'] > .2) and
                      (row['quality'] >= .7 or row['quality'] < .5)]
            y = np.asarray([int(row['quality'] >= .7) for row in chosen])
            entry = {
                'candidates': len(chosen), 'good': int(y.sum()),
                'bad': int(len(y) - y.sum()),
            }
            if len(np.unique(y)) != 2:
                entry.update(status='insufficient_classes', models={})
            else:
                entry.update(status='available', models={name: {'auc': float(roc_auc_score(
                    y, [_score(row, name) for row in chosen])),
                                  'average_precision': float(average_precision_score(
                    y, [_score(row, name) for row in chosen]))}
                           for name in METHODS})
                if scene_map is not None:
                    entry['paired_scene_bootstrap_auc'] = _auc_scene_bootstrap(
                        chosen, y, scene_map, bootstrap,
                        seed + 100 * WEATHERS.index(weather) + (band == 'high'))
            report[weather][band] = entry
    return report


def _paired_median_interval(values, scenes, repetitions, seed):
    unique = np.unique(scenes)
    members = {scene: np.flatnonzero(scenes == scene) for scene in unique}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repetitions):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        ids = np.concatenate([members[scene] for scene in chosen])
        draws.append(float(np.median(values[ids])))
    return [float(value) for value in np.percentile(draws, [2.5, 97.5])]


def _paired_rate_interval(flags, scenes, repetitions, seed):
    unique = np.unique(scenes)
    members = {scene: np.flatnonzero(scenes == scene) for scene in unique}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repetitions):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        ids = np.concatenate([members[scene] for scene in chosen])
        draws.append(float(flags[ids].mean()))
    return [float(value) for value in np.percentile(draws, [2.5, 97.5])]


def paired_intervention_weather(items, scene_map, repetitions, seed):
    """Same-anchor Clean/weather change in conditional peer-removal responses."""
    metric_indices = {name: INTERVENTION_NAMES.index(name) for name in (
        'max_score_drop', 'min_box_iou_to_full',
        'max_positive_logit_drop_stable_box',
        'max_positive_logit_drop_changed_box',
        'score_geometry_response_source_mismatch')}
    clean = {(row['sample_index'], row['candidate_id']): row for row in items
             if row['weather'] == 'clean' and row['within_range'] and row['source_count'] >= 2}
    result = {}
    for offset, weather in enumerate(WEATHERS[1:]):
        adverse = {(row['sample_index'], row['candidate_id']): row for row in items
                   if row['weather'] == weather and row['within_range']
                   and row['source_count'] >= 2}
        shared = clean.keys() & adverse.keys()
        pairs = [(clean[key], adverse[key]) for key in sorted(shared)
                 if clean[key]['source_count'] == adverse[key]['source_count']]
        condition = {'shared_anchors': len(shared), 'matched_equal_source_count': len(pairs),
                     'source_count_mismatch_excluded': len(shared) - len(pairs),
                     'selection_note': 'Both conditions entered top256 and range.'}
        by_group = {}
        for group in ('all', 'low_good', 'low_bad', 'high_good', 'high_bad'):
            def label(row):
                good = row['quality'] >= .7
                bad = row['quality'] < .5
                band = 'low' if row['score'] <= .2 else 'high'
                return f'{band}_good' if good else f'{band}_bad' if bad else 'middle'
            chosen = pairs if group == 'all' else [pair for pair in pairs
                                                    if label(pair[1]) == group]
            entry = {'matched_candidates': len(chosen),
                     'frames': len({pair[1]['sample_index'] for pair in chosen}),
                     'at_least_two_peers': sum(pair[1]['source_count'] >= 3
                                               for pair in chosen)}
            if chosen:
                scenes = np.asarray([str(scene_map[str(pair[1]['sample_index'])])
                                     for pair in chosen])
                score_delta = np.asarray([pair[1]['score'] - pair[0]['score']
                                          for pair in chosen], dtype=np.float64)
                iou_position = PROXY_NAMES.index('mean_source_box_iou')
                geometry_delta = np.asarray([
                    pair[1]['proxy'][iou_position] - pair[0]['proxy'][iou_position]
                    for pair in chosen], dtype=np.float64)
                patterns = {
                    'score_drop_geometry_stable': ((score_delta <= -.05)
                                                   & (np.abs(geometry_delta) <= .05)),
                    'score_gain_without_geometry_gain': ((score_delta >= .05)
                                                         & (geometry_delta <= 0)),
                    'geometry_gain_without_score_gain': ((geometry_delta >= .05)
                                                         & (score_delta <= 0)),
                }
                entry['cross_weather_asynchrony'] = {
                    'score_change_threshold': .05,
                    'source_fused_mean_iou_change_threshold': .05,
                    'median_score_weather_minus_clean': float(np.median(score_delta)),
                    'median_geometry_support_weather_minus_clean': float(
                        np.median(geometry_delta)),
                    'patterns': {name: {
                        'count': int(flags.sum()), 'rate': float(flags.mean()),
                        'paired_scene_bootstrap_rate_95pct': _paired_rate_interval(
                            flags, scenes, repetitions, seed + 1000 * offset + index),
                    } for index, (name, flags) in enumerate(patterns.items())},
                }
                fields = {}
                for index, (name, position) in enumerate(metric_indices.items()):
                    values = np.asarray([pair[1]['intervention'][position]
                                         - pair[0]['intervention'][position]
                                         for pair in chosen], dtype=np.float64)
                    fields[name] = {
                        'median_weather_minus_clean': float(np.median(values)),
                        'q10': float(np.percentile(values, 10)),
                        'q90': float(np.percentile(values, 90)),
                        'paired_scene_bootstrap_median_95pct': _paired_median_interval(
                            values, scenes, repetitions, seed + 100 * offset + 10 * index),
                    }
                entry['response_changes'] = fields
            by_group[group] = entry
        condition['groups_by_weather_quality'] = by_group
        result[weather] = condition
    return result


def render_markdown(report):
    lines = [
        '# top256 假设检验：固定候选池与真实后处理', '',
        '所有学习分数均按场景隔离生成折外预测；GT 只用于训练标签和事后评价。',
        '下表的 AUC 在相同分数段内区分好框和坏框；AP70 在相同 top256 池、'
        '原旋转 NMS 和逐帧固定输出数量下计算。', '',
        '| 天气 | 低分 AUC：候选自身 / 简单一致性 / 来源 / 变化+干预 | '
        '高分 AUC：同序 | 固定数量 AP70：候选自身 / 简单一致性 / 来源 / 变化+干预 |',
        '|---|---|---|---|',
    ]
    for weather in WEATHERS:
        condition = report['conditions'][weather]
        def auc(band, name):
            entry = condition['discrimination'][band]
            return (entry['models'][name]['auc'] if entry['status'] == 'available'
                    else None)
        def ap(name):
            return condition['replay']['budgeted_ap_global_sort'][name]['ap70']
        names = ('candidate_only', 'candidate_plus_simple_agreement',
                 'candidate_plus_source', 'candidate_plus_both')
        low = ' / '.join(f'{value:.3f}' if value is not None else '样本不足'
                         for value in (auc('low', name) for name in names))
        high = ' / '.join(f'{value:.3f}' if value is not None else '样本不足'
                          for value in (auc('high', name) for name in names))
        ap_values = ' / '.join(f'{100*ap(name):.2f}' for name in names)
        lines.append(f'| {weather} | {low} | {high} | {ap_values} |')
    lines += ['', '## 相同分数段内的折外 AUC 增量', '',
              '| 天气 | 低分：变化+干预相对来源 | 高分：变化+干预相对来源 |',
              '|---|---:|---:|']
    for weather in WEATHERS:
        values = []
        for band in ('low', 'high'):
            entry = report['conditions'][weather]['discrimination'][band]
            if entry['status'] != 'available':
                values.append('样本不足')
                continue
            delta = entry['paired_scene_bootstrap_auc'][
                'candidate_plus_both_minus_candidate_plus_source']
            ci = delta['scene_bootstrap_95pct']
            values.append(f"{delta['auc_difference']:+.3f} "
                          f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else '无有效区间')
        lines.append(f'| {weather} | {values[0]} | {values[1]} |')
    lines += ['', '各方法的帧顺序 AP、跨帧排序 AP、输出数和 NMS 竞争对去留见 JSON。',
              '原 score>0.2 的 AP 已与保存的 pilot 逐天气复现；不一致时程序直接报错。', '',
              '## 固定输出数量后的 AP70 增量', '',
              '| 天气 | 变化+干预相对简单一致性 | 变化+干预相对来源基础量 |',
              '|---|---:|---:|']
    for weather in WEATHERS:
        intervals = report['conditions'][weather]['replay'][
            'paired_scene_bootstrap_ap70_global_budgeted']
        values = []
        for baseline in ('candidate_plus_simple_agreement', 'candidate_plus_source'):
            item = intervals[f'candidate_plus_both_minus_{baseline}']
            ci = item['scene_bootstrap_95pct']
            values.append(f"{100*item['ap70_difference']:+.2f} pp "
                          f"[{100*ci[0]:+.2f}, {100*ci[1]:+.2f}]" if ci else '无有效区间')
        lines.append(f'| {weather} | {values[0]} | {values[1]} |')
    lines += ['', '区间按场景重抽样，固定已经训练出的折外分数；'
              '没有包含重新训练和特征选择的不确定性。', '',
              '## 同 anchor 的 Clean/天气逐车干预变化', '',
              '| 天气 | 匹配候选 | 低分好框：最大分数响应差 | 高分坏框：最大分数响应差 |',
              '|---|---:|---:|---:|']
    for weather in WEATHERS[1:]:
        item = report['paired_clean_weather_intervention'][weather]
        groups = item['groups_by_weather_quality']
        def change(group):
            values = groups[group].get('response_changes')
            return (f"{values['max_score_drop']['median_weather_minus_clean']:+.4f}"
                    if values else '无')
        lines.append(f"| {weather} | {item['matched_equal_source_count']} | "
                     f"{change('low_good')} | {change('high_bad')} |")
    lines += ['', '## Clean 到天气的分数—几何不同步', '',
              '| 天气 | 配对候选 | 分数下降而几何稳定 | 分数上升而几何未增强 |',
              '|---|---:|---:|---:|']
    for weather in WEATHERS[1:]:
        group = report['paired_clean_weather_intervention'][weather][
            'groups_by_weather_quality']['all']
        patterns = group.get('cross_weather_asynchrony', {}).get('patterns', {})
        def rate(name):
            item = patterns.get(name)
            return f"{item['count']} ({item['rate']:.1%})" if item else '无'
        lines.append(f"| {weather} | {group['matched_candidates']} | "
                     f"{rate('score_drop_geometry_stable')} | "
                     f"{rate('score_gain_without_geometry_gain')} |")
    lines += ['', '阈值和按场景重抽样区间见 JSON；几何支持是各来源框与融合框的平均 BEV IoU。',
              '完整的分数响应、框变化与来源错位分布及区间见 JSON。'
              '配对仅覆盖两种条件都进入 top256 的候选。', '',
              '## 解释边界', '',
              '- 这是固定 validation 场景上的探索性验证；正式论文结论还需独立场景。',
              '- F 是主分析。F+D 同时改变检测头，其结果须单独解释。',
              '- 逐车移除后的变化是条件响应，剩余来源的融合权重也会变化。',
              '- 固定输出数量取每帧原 score>0.2 流程的最终框数。',
              '- 好坏框成对统计只计数范围内且 BEV IoU 超过原 NMS 阈值的候选。', '']
    return '\n'.join(lines)


def run(args):
    root = Path(args.input_root).resolve()
    metadata, frames, items, provenance = load_extraction(root, args.arm)
    crossfit = cross_fit(items, metadata['validation_scene_map'], args.folds, args.seed)
    scores = discrimination(items, metadata['validation_scene_map'],
                            args.bootstrap, args.seed)
    replayed = replay(metadata, frames, args.arm, args.reproduction_tolerance,
                      args.bootstrap, args.seed)
    paired = paired_intervention_weather(items, metadata['validation_scene_map'],
                                         args.bootstrap, args.seed)
    report = {
        'protocol': {'input_root': str(root), 'arm': args.arm,
                     'validation_indices': metadata['validation_indices'],
                     'scene_map': metadata['validation_scene_map'],
                     'candidate_audit_sha256': _sha256(root / 'candidate_audit.json'),
                     'implementation_sha256': _sha256(Path(__file__)),
                     'fixed_pool': 'top_256', 'good_gt_iou_at_least': .7,
                     'bad_gt_iou_below': .5, 'original_score_threshold': .2,
                     'inference_feature_sets': list(MODELS),
                     'candidate_feature_contract': ('fused score and logit, seven regression '
                                                    'deltas, seven decoded box values, and '
                                                    'fixed-anchor fused detector-input vector'),
                     'proxy_feature_names': list(PROXY_NAMES),
                     'source_geometry_feature_names': list(SOURCE_GEOMETRY_NAMES),
                     'passive_distortion_feature_names': list(PASSIVE_NAMES),
                     'distortion_geometry_feature_names': list(DISTORTION_GEOMETRY_NAMES),
                     'intervention_feature_names': list(INTERVENTION_NAMES),
                     'scene_bootstrap_repetitions': args.bootstrap,
                     'reproduction_tolerance': args.reproduction_tolerance},
        'provenance': provenance, 'cross_fit': crossfit,
        'paired_clean_weather_intervention': paired,
        'conditions': {weather: {'discrimination': scores[weather],
                                'replay': replayed[weather]} for weather in WEATHERS},
    }
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'candidate_hypothesis_replay.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (output / 'candidate_hypothesis_replay.md').write_text(
        render_markdown(report), encoding='utf-8')
    print(f'CANDIDATE HYPOTHESIS REPLAY COMPLETE: {output}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--arm', choices=('F', 'F+D'), default='F')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=20260928)
    parser.add_argument('--bootstrap', type=int, default=200)
    parser.add_argument('--reproduction-tolerance', type=float, default=1e-6)
    args = parser.parse_args()
    if args.folds < 2 or args.bootstrap < 1 or args.reproduction_tolerance <= 0:
        parser.error('Require folds >= 2, bootstrap >= 1, and a positive reproduction tolerance')
    run(args)


if __name__ == '__main__':
    main()
