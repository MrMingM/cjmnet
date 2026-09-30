"""Offline R1/R2 cheap gate on frozen top256 exports; run on the remote server.

Official train scenes provide all labels and fitted parameters. Scene-held-out
train folds are a signal screen; the previously reused validation scenes are
an exploratory AP replay only. GT never enters an inference feature.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from .candidate_hypothesis_replay import (_ap, _evaluate_frame, _merge_stats,
                                          _nms_and_budget, _stats)
from .candidate_nms_scope import _matched_gt, _nms_ordered, _oracle_order
from .candidate_ranker_pilot import (_baseline_summaries, _load,
                                     _method_summary, _prepare_baselines,
                                     _sha256, _validate_frozen_contract)
from .candidate_r1_r2_extract import R1_MATCH_WIDTH, R1_NAMES, R2_NAMES, WEATHERS


METHODS = ('candidate_only', 'candidate_source', 'r1_opportunity_source',
           'r1', 'r1_source',
           'r1_shuffled', 'r2_score_source', 'r2_geometry_source',
           'r2', 'r2_source', 'r2_shuffled', 'r1_r2_source')
POLICIES = ('local_nms_order', 'full_rescore')
GROUPS = ('candidate', 'simple', 'proxy', 'source_geometry',
          'r1_matches', 'r1_opportunity', 'r2_score', 'r2_geometry')
ACTIVE = {
    'candidate_only': ('candidate',),
    'candidate_source': ('candidate', 'simple', 'proxy', 'source_geometry'),
    'r1_opportunity_source': ('candidate', 'simple', 'proxy',
                              'source_geometry', 'r1_opportunity'),
    'r1': ('candidate', 'r1_matches', 'r1_opportunity'),
    'r1_source': ('candidate', 'simple', 'proxy', 'source_geometry',
                  'r1_matches', 'r1_opportunity'),
    'r1_shuffled': ('candidate', 'simple', 'proxy', 'source_geometry',
                    'r1_matches', 'r1_opportunity'),
    'r2_score_source': ('candidate', 'simple', 'proxy', 'source_geometry',
                         'r2_score'),
    'r2_geometry_source': ('candidate', 'simple', 'proxy', 'source_geometry',
                            'r2_geometry'),
    'r2': ('candidate', 'r2_score', 'r2_geometry'),
    'r2_source': ('candidate', 'simple', 'proxy', 'source_geometry',
                  'r2_score', 'r2_geometry'),
    'r2_shuffled': ('candidate', 'simple', 'proxy', 'source_geometry',
                    'r2_score', 'r2_geometry'),
    'r1_r2_source': GROUPS,
}


def _load_overlay(data, path):
    root = Path(path).resolve()
    meta_path = root / 'overlay.json'
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    base = data['root'] / 'candidate_audit.json'
    split = 'train' if data['metadata'].get('split') == 'train' else 'validation'
    if (meta['split'] != split or meta['base_audit_sha256'] != _sha256(base)
            or meta['sample_indices'] != data['indices']
            or {str(k): int(v) for k, v in meta['scene_map'].items()}
               != {str(k): int(v) for k, v in data['scene_map'].items()}
            or tuple(meta['r1_names']) != R1_NAMES or
            tuple(meta['r2_names']) != R2_NAMES):
        raise ValueError('R1/R2 overlay does not match base candidate extraction')
    identity = {(row['weather'], row['sample_index'], row['candidate_id']): i
                for i, row in enumerate(data['records'])}
    r1 = np.empty((len(identity), len(R1_NAMES)), dtype=np.float32)
    r2 = np.empty((len(identity), len(R2_NAMES)), dtype=np.float32)
    filled = set()
    hashes = {str(meta_path): _sha256(meta_path)}
    for weather in WEATHERS:
        path = root / weather / 'overlay_rows.jsonl'
        hashes[str(path)] = _sha256(path)
        if (meta['conditions'][weather]['base_rows_sha256'] !=
                _sha256(data['root'] / weather / 'candidate_rows.jsonl')):
            raise ValueError(f'{weather}: base candidate rows changed after overlay export')
        count = 0
        with path.open(encoding='utf-8') as stream:
            for line_no, line in enumerate(stream, 1):
                raw = json.loads(line)
                key = (raw['weather'], int(raw['sample_index']),
                       int(raw['candidate_id']))
                if raw['weather'] != weather or key not in identity or key in filled:
                    raise ValueError(f'{path}:{line_no}: duplicate or unrelated candidate')
                index = identity[key]
                if (abs(float(raw['f_score_check']) - data['records'][index]['score']) > 1e-5
                        or int(raw['source_count']) != data['records'][index]['source_count']):
                    raise ValueError(f'{path}:{line_no}: F candidate contract differs')
                a = np.asarray(raw['r1'], dtype=np.float32)
                b = np.asarray(raw['r2'], dtype=np.float32)
                if (a.shape != (len(R1_NAMES),) or b.shape != (len(R2_NAMES),)
                        or not np.isfinite(a).all() or not np.isfinite(b).all()):
                    raise ValueError(f'{path}:{line_no}: invalid R1/R2 vector')
                r1[index], r2[index] = a, b
                filled.add(key)
                count += 1
        if count != int(meta['conditions'][weather]['candidates']):
            raise ValueError(f'{weather}: overlay count differs from metadata')
    if filled != set(identity):
        raise ValueError('R1/R2 overlay omits base F candidates')
    return r1, r2, meta, hashes


def _layout(data, r1, r2):
    old = data['layout']
    original = data['matrix']
    pieces = [original[:, slice(*old[name])] for name in GROUPS[:4]]
    pieces.extend((r1[:, :R1_MATCH_WIDTH], r1[:, R1_MATCH_WIDTH:],
                   r2[:, :3], r2[:, 3:]))
    layout, offset = {}, 0
    for name, part in zip(GROUPS, pieces):
        layout[name] = (offset, offset + part.shape[1])
        offset += part.shape[1]
    matrix = np.concatenate(pieces, axis=1).astype(np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError('Non-finite combined candidate feature matrix')
    return matrix, layout


def _shuffled(matrix, layout, records, seed):
    """GT-free within-weather/score-band/source-count negative controls."""
    rng = np.random.default_rng(seed)
    result = matrix.copy()
    buckets = defaultdict(list)
    for index, row in enumerate(records):
        buckets[(row['weather'], row['score'] <= .2,
                 min(row['source_count'], 5))].append(index)
    for indices in buckets.values():
        if len(indices) < 2:
            continue
        chosen = np.asarray(indices, dtype=np.int64)
        for name in ('r1_matches', 'r1_opportunity', 'r2_score', 'r2_geometry'):
            block = slice(*layout[name])
            result[chosen, block] = matrix[rng.permutation(chosen), block]
    return result


def _mask(layout, method):
    width = max(stop for _, stop in layout.values())
    mask = np.zeros(width, dtype=np.float32)
    for name in ACTIVE[method]:
        start, stop = layout[name]
        mask[start:stop] = 1.
    return mask


def _usable_indices(data, limit, seed):
    rng = np.random.default_rng(seed)
    selected = []
    labels = np.asarray([row['quality'] >= .7 for row in data['records']], dtype=np.int8)
    for weather in WEATHERS:
        for positive in (False, True):
            candidates = np.asarray([
                i for i, row in enumerate(data['records'])
                if row['weather'] == weather and row['within_range']
                and row['source_count'] >= 2
                and (row['quality'] >= .7 if positive else row['quality'] < .5)],
                dtype=np.int64)
            if not len(candidates):
                raise ValueError(f'{weather}: training has no quality class {positive}')
            selected.extend(rng.choice(candidates, min(len(candidates), limit),
                                       replace=False).tolist())
    return np.asarray(sorted(selected), dtype=np.int64), labels


def _fit_predict(train, labels, selected, evaluation, mask, seed):
    mean = train[selected].mean(0, dtype=np.float64)
    std = np.maximum(train[selected].std(0, dtype=np.float64), 1e-4)
    x = np.clip((train[selected] - mean) / std, -10, 10) * mask
    y = labels[selected]
    if len(np.unique(y)) != 2:
        raise ValueError('Training fold lacks a quality class')
    model = LogisticRegression(solver='liblinear', C=1., max_iter=150,
                               class_weight='balanced', random_state=seed)
    model.fit(x, y)
    result = np.empty(len(evaluation), dtype=np.float32)
    for first in range(0, len(result), 8192):
        block = np.clip((evaluation[first:first+8192] - mean) / std,
                        -10, 10) * mask
        result[first:first+8192] = model.predict_proba(block)[:, 1]
    if not np.isfinite(result).all():
        raise ValueError('Non-finite quality probability')
    return result


def _quality_metrics(rows, scores):
    labels = np.asarray([row['quality'] >= .7 for row in rows], dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float32)
    result = {'good': int(labels.sum()), 'bad': int(len(labels)-labels.sum()),
              'auc': None, 'pr_auc': None}
    if len(np.unique(labels)) == 2:
        result['auc'] = float(roc_auc_score(labels, scores))
        result['pr_auc'] = float(average_precision_score(labels, scores))
    return result


def _raw_audit(data, r1, r2, baselines):
    """Untrained, fixed-orientation feature audit on the reused validation set."""
    names = (('R1:' + name) for name in R1_NAMES)
    feature_names = tuple(names) + tuple('R2:' + name for name in R2_NAMES)
    features = np.concatenate((r1, r2), axis=1)
    chosen_features = ('R1:peer_support_03', 'R1:peer_postnms_support_03',
                       'R1:peer_max_points_near_center',
                       'R2:fd_minus_f_score', 'R2:abs_score_gap',
                       'R2:fd_f_bev_iou', 'R2:fd_f_center_shift')
    diagnostic = {'feature_names': feature_names, 'strata': {},
                  'direct_nms_edges': {}, 'missing_match_with_nearby_points': {}}
    for weather in WEATHERS:
        indices = [i for i, row in enumerate(data['records'])
                   if row['weather'] == weather and row['within_range']
                   and row['source_count'] >= 2 and
                   (row['quality'] >= .7 or row['quality'] < .5)]
        strata = {}
        for band, band_check in (('low', lambda score: score <= .2),
                                 ('high', lambda score: score > .2)):
            for distance, distance_check in (
                    ('all_distances', lambda value: True),
                    ('near_0_30m', lambda value: value < 30),
                    ('middle_30_60m', lambda value: 30 <= value < 60),
                    ('far_60m_plus', lambda value: value >= 60)):
                for vehicles, source_check in (
                        ('all_sources', lambda value: True),
                        ('two_sources', lambda value: value == 2),
                        ('three_plus_sources', lambda value: value >= 3)):
                    selected = [i for i in indices
                        if band_check(data['records'][i]['score']) and
                        distance_check(float(np.linalg.norm(
                            data['records'][i]['corners'].mean(0)))) and
                        source_check(data['records'][i]['source_count'])]
                    rows = [data['records'][i] for i in selected]
                    strata[f'{band}/{distance}/{vehicles}'] = {
                        name: _quality_metrics(rows, features[selected, offset])
                        for offset, name in enumerate(feature_names)
                        if name in chosen_features}
        diagnostic['strata'][weather] = strata
        edge_pairs = []
        for sample in data['indices']:
            rows = data['rows'][weather]['rows'][sample]
            for good, bad in baselines[weather]['frames'][sample]['target_edges']:
                edge_pairs.append((rows[good]['vector_index'],
                                   rows[bad]['vector_index']))
        direct = {'total': len(edge_pairs)}
        for name in chosen_features:
            column = features[:, feature_names.index(name)]
            direct[name] = {
                'good_higher': int(sum(bool(column[good] > column[bad])
                                       for good, bad in edge_pairs)),
                'equal': int(sum(bool(column[good] == column[bad])
                                 for good, bad in edge_pairs)),
            }
        diagnostic['direct_nms_edges'][weather] = direct
        support = r1[:, R1_NAMES.index('peer_support_03')]
        points = r1[:, R1_NAMES.index('peer_max_points_near_center')]
        no_match = [i for i in indices if support[i] == 0]
        diagnostic['missing_match_with_nearby_points'][weather] = {
            'no_match': len(no_match),
            'nearby_points_at_least_3': int(sum(bool(points[i] >= 3)
                                                for i in no_match)),
            'nearby_points_at_least_10': int(sum(bool(points[i] >= 10)
                                                 for i in no_match)),
        }
    return diagnostic


def _source_match_audit(data, overlay_root, limit=10000):
    """Reservoir-sampled GT check of peer boxes that overlap fused candidates."""
    from opencood.utils import common_utils

    output = {}
    for weather in WEATHERS:
        identity = {(row['sample_index'], row['candidate_id']): row
                    for row in data['records'] if row['weather'] == weather}
        rng = np.random.default_rng(20260929 + WEATHERS.index(weather))
        reservoir, total = [], 0
        path = Path(overlay_root) / weather / 'overlay_rows.jsonl'
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                raw = json.loads(line)
                sample = int(raw['sample_index'])
                row = identity[(sample, int(raw['candidate_id']))]
                if not row['within_range'] or .5 <= row['quality'] < .7:
                    continue
                for peer in raw['matches'][1:]:
                    if peer['iou'] < .3:
                        continue
                    corners = np.asarray(peer['best_source_bev_corners'],
                                         dtype=np.float32)
                    if corners.shape != (4, 2) or not np.isfinite(corners).all():
                        raise ValueError(f'{path}: matched source box missing')
                    item = (sample, corners, row['quality'] >= .7,
                            row['score'] <= .2)
                    total += 1
                    if len(reservoir) < limit:
                        reservoir.append(item)
                    else:
                        replace = int(rng.integers(total))
                        if replace < limit:
                            reservoir[replace] = item
        gt_polygons = {sample: common_utils.convert_format(corners)
                       for sample, corners in data['targets'][weather].items()
                       if len(corners)}
        counts = Counter()
        for sample, corners, fused_good, low_score in reservoir:
            polygon = common_utils.convert_format(corners[None])[0]
            others = gt_polygons.get(sample)
            quality = (float(np.max(common_utils.compute_iou(polygon, others)))
                       if others is not None else 0.)
            counts['sampled_links'] += 1
            counts['fused_good_links'] += int(fused_good)
            counts['fused_bad_links'] += int(not fused_good)
            counts['good_fused_bad_source_iou_lt_05'] += int(fused_good and quality < .5)
            counts['bad_fused_good_source_iou_ge_07'] += int(not fused_good and quality >= .7)
            counts['low_score_good_fused_bad_source'] += int(
                low_score and fused_good and quality < .5)
        output[weather] = {'total_matched_peer_links': total,
                           'reservoir_limit': limit, **dict(counts)}
    return output


def _train_crossfit(data, matrix, shuffled, layout, selected, labels, seed):
    groups = np.asarray([data['records'][i]['scene'] for i in selected])
    folds = min(5, len(set(groups)))
    if folds < 2:
        raise ValueError('Scene-held-out training screen needs >=2 scenes')
    result = {}
    for method in METHODS:
        source = shuffled if method.endswith('shuffled') else matrix
        scores = np.full(len(matrix), np.nan, dtype=np.float32)
        for train_pos, held_pos in GroupKFold(folds).split(selected, labels[selected], groups):
            fitted = selected[train_pos]
            held = selected[held_pos]
            prediction = _fit_predict(source, labels, fitted, source[held],
                                      _mask(layout, method), seed)
            scores[held] = prediction
        if not np.isfinite(scores[selected]).all():
            raise AssertionError('Incomplete out-of-fold prediction')
        bands = {}
        for weather in WEATHERS:
            for band, check in (('low', lambda value: value <= .2),
                                ('high', lambda value: value > .2)):
                chosen = [i for i in selected if
                          data['records'][i]['weather'] == weather and
                          check(data['records'][i]['score'])]
                bands[f'{weather}_{band}'] = _quality_metrics(
                    [data['records'][i] for i in chosen], scores[chosen])
        result[method] = bands
        print(f'train scene-held-out: {method}', flush=True)
    return result


def _replay(data, baselines, predictions, seed):
    from opencood.utils import common_utils

    output, frame_rows = {}, []
    for weather in WEATHERS:
        condition = data['rows'][weather]
        reference = baselines[weather]
        stats = {(method, policy): _stats() for method in METHODS for policy in POLICIES}
        global_scores = {(method, policy): _stats() for method in METHODS for policy in POLICIES}
        scenes = {(method, policy): defaultdict(_stats)
                  for method in METHODS for policy in POLICIES}
        counts = {(method, policy): Counter() for method in METHODS for policy in POLICIES}
        point = {}
        for method in METHODS:
            for band, check in (('low', lambda value: value <= .2),
                                ('high', lambda value: value > .2)):
                chosen = [row for sample in data['indices']
                          for row in condition['rows'][sample]
                          if row['within_range'] and row['source_count'] >= 2
                          and check(row['score']) and
                          (row['quality'] >= .7 or row['quality'] < .5)]
                point[(method, band)] = _quality_metrics(
                    chosen, [predictions[method][row['vector_index']]
                             for row in chosen])
        for sample in data['indices']:
            rows = condition['rows'][sample]
            frame = reference['frames'][sample]
            gt = data['targets'][weather][sample]
            scene = str(data['scene_map'][str(sample)])
            original_scores = np.asarray([row['score'] for row in rows])
            within = np.asarray([row['within_range'] for row in rows], dtype=bool)
            polygons = common_utils.convert_format(np.stack(
                [row['corners'] for row in rows]))
            overlap = lambda candidate, others: common_utils.compute_iou(
                polygons[candidate], polygons[others])
            for method in METHODS:
                values = np.asarray([
                    predictions[method][row['vector_index']]
                    if row['source_count'] >= 2 else row['score'] for row in rows],
                    dtype=np.float32)
                for policy in POLICIES:
                    if policy == 'local_nms_order':
                        order, _, _ = _oracle_order(frame['original_order'],
                            frame['suppressors'], values,
                            np.ones(len(rows), dtype=bool))
                        picked, _ = _nms_ordered(order, overlap,
                                                  condition['threshold'])
                        selected = [i for i in picked if within[i]][:frame['budget']]
                    else:
                        selected = _nms_and_budget(rows, values,
                                        condition['threshold'], frame['budget'])
                    key = method, policy
                    part = _stats()
                    _evaluate_frame(part, rows, selected, original_scores, gt)
                    _merge_stats(stats[key], part)
                    _merge_stats(scenes[key][scene], part)
                    _evaluate_frame(global_scores[key], rows, selected, values, gt)
                    matched = _matched_gt(rows, selected, gt)
                    gained = len(matched - frame['matched_top'])
                    lost = len(frame['matched_top'] - matched)
                    counts[key].update(output=len(selected), budget=frame['budget'],
                                       new=gained, lost=lost,
                                       direct_correct=sum(values[low] > values[high]
                                          for low, high in frame['target_edges']))
                    frame_rows.append({'seed': seed, 'weather': weather,
                        'sample_index': sample, 'scene': scene,
                        'method': method, 'policy': policy,
                        'budget': frame['budget'], 'output_boxes': len(selected),
                        'new_gt': gained, 'lost_gt': lost,
                        'direct_edges': len(frame['target_edges'])})
        methods = {}
        top_fp = int(sum(reference['stats']['top256_fused'][.7]['fp']))
        top_tp = int(sum(reference['stats']['top256_fused'][.7]['tp']))
        for method in METHODS:
            policies = {}
            for policy in POLICIES:
                key = method, policy
                count = counts[key]
                summary = _method_summary(stats[key], scenes[key], count['output'],
                    count['output'] / max(1, count['budget']), count['new'],
                    count['lost'], count['direct_correct'],
                    reference['target_direct_edges'])
                if (summary['tp70'] - top_tp != count['new'] - count['lost'] or
                        summary['tp70'] + summary['fp70'] != count['output']):
                    raise AssertionError(f'{weather}/{method}/{policy}: TP/FP mismatch')
                summary['new_fp_vs_top256'] = summary['fp70'] - top_fp
                summary['ap_using_learned_output_score'] = {
                    'frame_order': _ap(global_scores[key], False),
                    'global_sort': _ap(global_scores[key], True)}
                policies[policy] = summary
            methods[method] = {'same_score_band_quality': {
                band: point[(method, band)] for band in ('low', 'high')},
                'policies': policies}
        output[weather] = {'methods': methods}
        print(f'eval {weather}: R1/R2 fixed-budget replay complete', flush=True)
    return output, frame_rows


def _gate(report, proposed, controls, policy):
    if policy not in POLICIES:
        raise ValueError(f'Unknown replay policy: {policy}')
    deltas, excluded = {}, {}
    for weather in WEATHERS:
        direct = report['baselines'][weather]['scorepass']
        gains, margins, worst, nets = [], [], [], []
        for seed in report['protocol']['seeds']:
            methods = report['seeds'][str(seed)][weather]['methods']
            chosen = methods[proposed]['policies'][policy]
            compared = [methods[name]['policies'][policy]
                        for name in controls]
            ap = chosen['ap_frame_order']['ap70']
            gains.append(ap - direct['ap_frame_order']['ap70'])
            margins.append(ap - max(x['ap_frame_order']['ap70'] for x in compared))
            nets.append(chosen['new_matched_gt_vs_top256'] -
                        chosen['lost_matched_gt_vs_top256'])
            worst.append(min(chosen['leave_one_scene_out_ap70_frame'][scene] -
                max([direct['leave_one_scene_out_ap70_frame'][scene]] +
                    [x['leave_one_scene_out_ap70_frame'][scene] for x in compared])
                for scene in chosen['leave_one_scene_out_ap70_frame']))
        deltas[weather] = {'vs_original': float(np.mean(gains)),
                           'vs_best_control': float(np.mean(margins)),
                           'net_gt': float(np.mean(nets))}
        excluded[weather] = float(np.mean(worst))
    passed = sum(deltas[w]['vs_original'] >= .005 and
                 deltas[w]['vs_best_control'] >= .005 and
                 deltas[w]['net_gt'] > 0 and excluded[w] > 0
                 for w in ('fog', 'rain', 'snow'))
    clean = (deltas['clean']['vs_original'] >= -.001 and
             deltas['clean']['vs_best_control'] >= -.001)
    return {'proposed': proposed, 'controls': controls, 'policy': policy,
            'weather_pass_count': passed,
            'clean_preserved': bool(clean),
            'pilot_gate_pass': bool(passed >= 2 and clean),
            'delta': deltas, 'worst_leave_one_scene_margin': excluded,
            'status': 'exploratory reused validation; independent scenes required'}


def run(args):
    evaluation = _load(args.eval_root, 'validation')
    er1, er2, emeta, ehashes = _load_overlay(evaluation, args.eval_overlay)
    if args.audit_only:
        output = Path(args.output).resolve()
        output.mkdir(parents=True, exist_ok=False)
        baselines = _prepare_baselines(evaluation, args.reproduction_tolerance)
        baseline_summary = _baseline_summaries(evaluation, baselines)
        report = {
            'protocol': {'audit_only': True,
                'eval_root': str(evaluation['root']),
                'eval_overlay': str(args.eval_overlay),
                'validation_status': 'reused development scenes, not independent test',
                'source_sha256': _sha256(Path(__file__).resolve()),
                'eval_input_sha256': {**evaluation['provenance'], **ehashes}},
            'baselines': baseline_summary,
            'untrained_validation_audit': _raw_audit(
                evaluation, er1, er2, baselines),
            'source_match_gt_audit': _source_match_audit(
                evaluation, args.eval_overlay),
            'extra_compute_seconds_by_weather': {
                weather: emeta['conditions'][weather]['extra_compute_seconds']
                for weather in WEATHERS},
        }
        if _sha256(Path(__file__).resolve()) != report['protocol']['source_sha256']:
            raise RuntimeError('Gate source changed while running')
        with (output / 'candidate_r1_r2.json').open('w', encoding='utf-8') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        lines = ['# R1/R2 不训练审计', '',
                 '独立车辆候选匹配和 F/F+D 同 anchor 分歧的完整统计见 JSON。',
                 '旧 validation 场景仅可用于探索性筛选。', '',
                 '| 天气 | R1 匹配邻车候选 / 几何核对抽样 | 原 F AP70 |',
                 '|---|---:|---:|']
        for weather in WEATHERS:
            match = report['source_match_gt_audit'][weather]
            ap = baseline_summary[weather]['scorepass']['ap_frame_order']['ap70']
            lines.append(f'| {weather} | {match["total_matched_peer_links"]}/'
                         f'{match.get("sampled_links", 0)} | {ap:.4f} |')
        (output / 'candidate_r1_r2.md').write_text(
            '\n'.join(lines) + '\n', encoding='utf-8')
        print('R1/R2 UNTRAINED AUDIT COMPLETE:', output / 'candidate_r1_r2.json',
              flush=True)
        return
    train = _load(args.train_root, 'train')
    if train['root'] == evaluation['root']:
        raise ValueError('Train/evaluation candidate roots must differ')
    _validate_frozen_contract(train['metadata'], evaluation['metadata'])
    tr1, tr2, tmeta, thashes = _load_overlay(train, args.train_overlay)
    for field in ('frontend_sha256', 'v3_checkpoint_sha256', 'arm_sha256',
                  'source_topk', 'source_preselect', 'r1_names', 'r2_names'):
        if tmeta[field] != emeta[field]:
            raise ValueError(f'Train/evaluation overlay contract differs: {field}')
    training, layout = _layout(train, tr1, tr2)
    eval_matrix, eval_layout = _layout(evaluation, er1, er2)
    if layout != eval_layout:
        raise ValueError('Train/evaluation feature layout differs')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    baselines = _prepare_baselines(evaluation, args.reproduction_tolerance)
    baseline_summary = _baseline_summaries(evaluation, baselines)
    seeds = [int(x) for x in args.seeds.split(',')]
    if len(seeds) != len(set(seeds)) or not seeds:
        raise ValueError('Seeds must be unique')
    report = {'protocol': {'train_root': str(train['root']),
        'eval_root': str(evaluation['root']), 'train_overlay': str(args.train_overlay),
        'eval_overlay': str(args.eval_overlay), 'seeds': seeds,
        'feature_layout': layout, 'methods': METHODS, 'policies': POLICIES,
        'max_per_class_weather': args.max_per_class_weather,
        'label': 'good>=0.7 versus bad<0.5; middle unused for training',
        'model': 'same-width masked L2 logistic regression, fixed C=1',
        'validation_status': 'reused development scenes, not independent test',
        'source_sha256': _sha256(Path(__file__).resolve()),
        'train_input_sha256': {**train['provenance'], **thashes},
        'eval_input_sha256': {**evaluation['provenance'], **ehashes}},
        'baselines': baseline_summary,
        'untrained_validation_audit': _raw_audit(evaluation, er1, er2, baselines),
        'source_match_gt_audit': _source_match_audit(
            evaluation, args.eval_overlay),
        'extra_compute_seconds_by_weather': {
            weather: emeta['conditions'][weather]['extra_compute_seconds']
            for weather in WEATHERS},
        'train_crossfit': {}, 'seeds': {}}
    frame_rows = []
    for seed in seeds:
        selected, labels = _usable_indices(train, args.max_per_class_weather, seed)
        shuffled_train = _shuffled(training, layout, train['records'], seed)
        shuffled_eval = _shuffled(eval_matrix, layout, evaluation['records'], seed)
        if seed == seeds[0]:
            report['train_crossfit'] = _train_crossfit(
                train, training, shuffled_train, layout, selected, labels, seed)
            report['protocol']['train_labelled_selected'] = int(len(selected))
        predictions = {}
        for method in METHODS:
            source = shuffled_train if method.endswith('shuffled') else training
            target = shuffled_eval if method.endswith('shuffled') else eval_matrix
            predictions[method] = _fit_predict(source, labels, selected, target,
                                               _mask(layout, method), seed)
        replay, frames = _replay(evaluation, baselines, predictions, seed)
        report['seeds'][str(seed)] = replay
        frame_rows.extend(frames)
    gate_specs = {
        'R1': ('r1_source', ('candidate_only', 'candidate_source',
                             'r1_opportunity_source', 'r1_shuffled')),
        'R2': ('r2_source', ('candidate_only', 'candidate_source',
                             'r2_shuffled', 'r2_score_source',
                             'r2_geometry_source')),
        'R2_score_only': ('r2_score_source',
                          ('candidate_only', 'candidate_source', 'r2_shuffled')),
        'R2_geometry_only': ('r2_geometry_source',
                             ('candidate_only', 'candidate_source', 'r2_shuffled')),
        'R1_R2_combined': ('r1_r2_source',
                           ('candidate_only', 'candidate_source',
                            'r1_source', 'r2_source')),
    }
    report['gates'] = {
        f'{name}/{policy}': _gate(report, proposed, controls, policy)
        for name, (proposed, controls) in gate_specs.items()
        for policy in POLICIES
    }
    if _sha256(Path(__file__).resolve()) != report['protocol']['source_sha256']:
        raise RuntimeError('Gate source changed while running')
    with (output / 'candidate_r1_r2_frames.jsonl').open('w', encoding='utf-8') as stream:
        for row in frame_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    with (output / 'candidate_r1_r2.json').open('w', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    lines = ['# R1/R2 候选可靠性 Cheap Gate', '',
             '旧 validation 场景已反复参与开发；本结果仅是探索性筛选。',
             '主结果沿用原融合分数，只改变 NMS 排序和框的去留。',
             'GT 仅用于官方 train 标签与事后评价。']
    for policy, title in (('local_nms_order', '只改原 NMS 组顺序'),
                          ('full_rescore', '完整 top256 重评分')):
        lines.extend(['', f'## {title}：原 F 分数计算的帧顺序 AP70', '',
            '| 天气 | 原 F | 候选自身 | 来源基础量 | R1 观测机会 | R1+来源 | R2 分数 | R2 几何 | R2 合用 | R1+R2 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|'])
        for weather in WEATHERS:
            baseline = baseline_summary[weather]['scorepass']['ap_frame_order']['ap70']
            def mean(method):
                return float(np.mean([report['seeds'][str(seed)][weather]['methods']
                    [method]['policies'][policy]['ap_frame_order']['ap70']
                    for seed in seeds]))
            lines.append(f'| {weather} | {baseline:.4f} | '
                f'{mean("candidate_only"):.4f} | {mean("candidate_source"):.4f} | '
                f'{mean("r1_opportunity_source"):.4f} | '
                f'{mean("r1_source"):.4f} | {mean("r2_score_source"):.4f} | '
                f'{mean("r2_geometry_source"):.4f} | {mean("r2_source"):.4f} | '
                f'{mean("r1_r2_source"):.4f} |')
    lines.extend(['', '## R1 匹配几何核对与额外计算', '',
                  '| 天气 | 邻车匹配总数 / 抽样数 | 好融合框对应差来源框 | R1 秒/帧 | R2 秒/帧 |',
                  '|---|---:|---:|---:|---:|'])
    for weather in WEATHERS:
        match = report['source_match_gt_audit'][weather]
        clock = report['extra_compute_seconds_by_weather'][weather]
        frames = emeta['conditions'][weather]['frames']
        good = match.get('fused_good_links', 0)
        bad_source = match.get('good_fused_bad_source_iou_lt_05', 0)
        fraction = f'{bad_source}/{good}' if good else '无好框样本'
        lines.append(f'| {weather} | {match["total_matched_peer_links"]}/'
            f'{match.get("sampled_links", 0)} | {fraction} | '
            f'{clock["r1_source_head_pool_match"]/frames:.3f} | '
            f'{clock["r2_fd_head_decode"]/frames:.3f} |')
    lines.extend(['', '## 预定继续门槛', ''])
    for name, gate in report['gates'].items():
        lines.append(f'- {name}：{"通过" if gate["pilot_gate_pass"] else "未通过"}；'
                     f'恶劣天气通过 {gate["weather_pass_count"]}/3，'
                     f'Clean 保持={gate["clean_preserved"]}。')
    lines.extend(['', 'JSON 含每种子、天气、策略的 AP30/50/70、AUC/PR-AUC、'
                  '真实 NMS 边、TP/FP、新增/丢失 GT、逐场景剔除以及训练折外结果。',
                  '正结果仍须在新独立场景复验。', ''])
    (output / 'candidate_r1_r2.md').write_text('\n'.join(lines), encoding='utf-8')
    print('R1/R2 CHEAP GATE COMPLETE:', output / 'candidate_r1_r2.json', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('train-root', 'eval-root', 'train-overlay', 'eval-overlay', 'output'):
        parser.add_argument('--' + field, required=True)
    parser.add_argument('--seeds', default='20260929,20260930,20260931')
    parser.add_argument('--max-per-class-weather', type=int, default=5000)
    parser.add_argument('--reproduction-tolerance', type=float, default=1e-6)
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    if args.max_per_class_weather < 1:
        parser.error('--max-per-class-weather must be positive')
    run(args)


if __name__ == '__main__':
    main()
