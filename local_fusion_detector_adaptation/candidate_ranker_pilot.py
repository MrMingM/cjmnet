"""Frozen-detector, scene-separated top256 candidate reliability pilot.

Train labels and pair targets use GT only in official TRAIN scenes. Validation
and later evaluation load saved candidate extractions and never update weights.
The primary AP comparison uses the original fused scores after candidate
selection; learned scores are reported separately as a deployment diagnostic.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .candidate_discriminability import _features
from .candidate_hypothesis_replay import (
    PASSIVE_NAMES, WEATHERS, _ap, _evaluate_frame, _merge_stats,
    _nms_and_budget, _source_geometry_features, _stats,
)
from .candidate_intervention_probe import _candidate_features, _proxy_features
from .candidate_nms_scope import _matched_gt, _nms_ordered, _oracle_order


MODELS = ('candidate_only', 'candidate_plus_simple',
          'candidate_plus_source', 'candidate_plus_distortion')
BASELINES = ('scorepass', 'top256_fused', 'source_agreement_raw')
ALL_METHODS = BASELINES + MODELS
FEATURE_GROUPS = ('candidate', 'simple', 'proxy', 'source_geometry',
                  'passive', 'distortion_geometry')


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _feature_code_hashes():
    folder = Path(__file__).resolve().parent
    return {name: _sha256(folder / name) for name in (
        'candidate_ranker_pilot.py', 'candidate_intervention_probe.py',
        'candidate_discriminability.py', 'candidate_hypothesis_replay.py',
        'candidate_nms_scope.py')}


def _root(path):
    root = Path(path).resolve()
    if not (root / 'candidate_audit.json').is_file():
        root = root / 'extraction'
    if not (root / 'candidate_audit.json').is_file():
        raise FileNotFoundError(f'Missing candidate_audit.json: {path}')
    return root


def _feature_parts(raw, joined):
    """All inference inputs are built without reading the GT quality field."""
    candidate = _candidate_features(raw, joined).astype(np.float32)
    count = int(raw['source_count'])
    if count >= 2:
        # _features validates row quality but its feature dictionary uses only
        # source predictions, fused score and source-to-fused geometry.
        passive, _ = _features(dict(raw, max_gt_iou=0.))
        geometry, interaction = _source_geometry_features(raw)
        proxy = _proxy_features(raw).astype(np.float32)
        source_agreement = float(passive['source_agreement'])
        source = {
            'simple': np.asarray([source_agreement], dtype=np.float32),
            'proxy': proxy,
            'source_geometry': geometry,
            'passive': np.asarray([passive[name] for name in PASSIVE_NAMES],
                                  dtype=np.float32),
            'distortion_geometry': interaction,
        }
    else:
        source_agreement = float(raw['score'])
        source = {
            'simple': np.zeros(1, dtype=np.float32),
            'proxy': np.zeros(10, dtype=np.float32),
            'source_geometry': np.zeros(4, dtype=np.float32),
            'passive': np.zeros(len(PASSIVE_NAMES), dtype=np.float32),
            'distortion_geometry': np.zeros(1, dtype=np.float32),
        }
    parts = {'candidate': candidate, **source}
    for name, vector in parts.items():
        if vector.ndim != 1 or not np.isfinite(vector).all():
            raise ValueError(f'Invalid inference feature group: {name}')
    return parts, source_agreement


def _layout(parts):
    offset, layout = 0, {}
    for name in FEATURE_GROUPS:
        width = len(parts[name])
        layout[name] = [offset, offset + width]
        offset += width
    return layout


def _model_mask(layout, name):
    active = {'candidate_only': ('candidate',),
              'candidate_plus_simple': ('candidate', 'simple'),
              'candidate_plus_source': ('candidate', 'simple', 'proxy',
                                        'source_geometry'),
              'candidate_plus_distortion': FEATURE_GROUPS}[name]
    mask = np.zeros(max(stop for _, stop in layout.values()), dtype=np.float32)
    for group in active:
        start, stop = layout[group]
        mask[start:stop] = 1.
    return mask


def _load(root, split):
    root = _root(root)
    path = root / 'candidate_audit.json'
    meta = json.loads(path.read_text(encoding='utf-8'))
    if meta.get('audit_pool') != 'top_256' or meta.get('stage0_only'):
        raise ValueError('Ranker requires a complete top_256 candidate audit')
    if not meta.get('candidate_feature_cache'):
        raise ValueError('Ranker requires candidate feature caches')
    if split == 'train':
        if meta.get('split') != 'train':
            raise ValueError('Training input must be exported from official train scenes')
        indices = [int(index) for index in meta['sample_indices']]
        scene_map = meta['scene_map']
    else:
        if meta.get('split') == 'train':
            raise ValueError('Evaluation input is a training extraction')
        indices = [int(index) for index in meta['validation_indices']]
        scene_map = meta['validation_scene_map']
        if not all('swap_ap' in meta['conditions'][weather] for weather in WEATHERS):
            raise ValueError('Evaluation input lacks frozen original AP checks')
    if not indices or len(indices) != len(set(indices)):
        raise ValueError('Invalid frame indices')
    if set(scene_map) != set(map(str, indices)):
        raise ValueError('Incomplete scene map')
    if len(set(scene_map.values())) < 2:
        raise ValueError('Ranker requires at least two independent scenes per split')
    scene_order = [int(scene_map[str(index)]) for index in indices]
    if indices != sorted(indices) or scene_order != sorted(scene_order):
        raise ValueError('Frame and scene order must be monotonic for frame-order AP')
    hashes = {str(path): _sha256(path)}
    rows_by_weather, targets_by_weather = {}, {}
    feature_vectors, records = [], []
    layout = None
    for weather in WEATHERS:
        condition = meta['conditions'][weather]
        if abs(float(condition['original_score_threshold']) - .2) > 1e-8:
            raise ValueError(f'{weather}: unexpected score threshold')
        threshold = float(condition['nms_iou_threshold'])
        if not 0 < threshold < 1:
            raise ValueError(f'{weather}: invalid NMS threshold')
        folder = root / weather
        target_path = folder / 'frame_targets.jsonl'
        row_path = folder / 'candidate_rows.jsonl'
        hashes[str(target_path)] = _sha256(target_path)
        hashes[str(row_path)] = _sha256(row_path)
        targets = {}
        with target_path.open(encoding='utf-8') as stream:
            for line_no, line in enumerate(stream, 1):
                raw = json.loads(line)
                sample = int(raw['sample_index'])
                if raw['weather'] != weather or sample in targets:
                    raise ValueError(f'{target_path}:{line_no}: invalid target identity')
                gt = np.asarray(raw['gt_bev_corners'], dtype=np.float32)
                if gt.size == 0:
                    gt = gt.reshape(0, 4, 2)
                if gt.ndim != 3 or gt.shape[1:] != (4, 2) or not np.isfinite(gt).all():
                    raise ValueError(f'{target_path}:{line_no}: invalid GT corners')
                targets[sample] = gt
        if set(targets) != set(indices):
            raise ValueError(f'{weather}: target frames differ from metadata')
        by_frame = {sample: [] for sample in indices}
        seen = set()
        cache = {}
        cache_digest = hashlib.sha256()
        with row_path.open(encoding='utf-8') as stream:
            for line_no, line in enumerate(stream, 1):
                raw = json.loads(line)
                if raw['arm'] != 'F':
                    continue
                sample, cid = int(raw['sample_index']), int(raw['candidate_id'])
                if (raw['weather'] != weather or sample not in by_frame
                        or (sample, cid) in seen):
                    raise ValueError(f'{row_path}:{line_no}: invalid candidate identity')
                seen.add((sample, cid))
                expected = f'F_features_{sample}.npz'
                if raw.get('fused_feature_cache') != expected:
                    raise ValueError(f'{row_path}:{line_no}: missing feature cache')
                if sample not in cache:
                    feature_path = folder / expected
                    cache_digest.update(expected.encode('utf-8') + b'\0')
                    cache_digest.update(_sha256(feature_path).encode('ascii') + b'\n')
                    with np.load(feature_path, allow_pickle=False) as packed:
                        ids = np.asarray(packed['candidate_id'], dtype=np.int64)
                        vectors = np.asarray(packed['feature'], dtype=np.float32)
                    if (vectors.ndim != 2 or len(ids) != len(vectors)
                            or len(set(ids.tolist())) != len(ids)
                            or not np.isfinite(vectors).all()):
                        raise ValueError(f'{feature_path}: invalid feature cache')
                    cache[sample] = dict(zip(map(int, ids), vectors))
                if cid not in cache[sample]:
                    raise ValueError(f'{row_path}:{line_no}: missing candidate feature')
                score, quality = float(raw['score']), float(raw['max_gt_iou'])
                corners = np.asarray(raw['fused_bev_corners'], dtype=np.float32)
                if (not 0 <= score <= 1 or not 0 <= quality <= 1
                        or corners.shape != (4, 2) or not np.isfinite(corners).all()):
                    raise ValueError(f'{row_path}:{line_no}: invalid score/box/quality')
                parts, agreement = _feature_parts(raw, cache[sample][cid])
                current_layout = _layout(parts)
                if layout is None:
                    layout = current_layout
                elif layout != current_layout:
                    raise ValueError('Candidate feature dimension differs across frames')
                vector = np.concatenate([parts[name] for name in FEATURE_GROUPS])
                vector_index = len(feature_vectors)
                feature_vectors.append(vector)
                row = {
                    'weather': weather, 'sample_index': sample,
                    'scene': str(scene_map[str(sample)]), 'candidate_id': cid,
                    'score': score, 'quality': quality, 'corners': corners,
                    'within_range': bool(raw['within_range']),
                    'source_count': int(raw['source_count']),
                    'simple_agreement': agreement,
                    'vector_index': vector_index,
                }
                by_frame[sample].append(row)
                records.append(row)
        if any(not 1 <= len(rows) <= 256 for rows in by_frame.values()):
            raise ValueError(f'{weather}: each frame needs 1–256 candidates')
        pools = condition['pools']['F']
        count = sum(map(len, by_frame.values()))
        if count != int(pools['top_256']['total_candidates']):
            raise ValueError(f'{weather}: candidate count differs from metadata')
        if split != 'train':
            original = sum(row['score'] > .2 for rows in by_frame.values()
                           for row in rows)
            if original != int(pools['original']['total_candidates']):
                raise ValueError(f'{weather}: top256 omits original scorepass pool')
        rows_by_weather[weather] = {'rows': by_frame, 'threshold': threshold}
        targets_by_weather[weather] = targets
        hashes[str(folder / 'F_feature_cache_manifest')] = cache_digest.hexdigest()
        print(f'{split} loaded {weather}: {count} F candidates', flush=True)
    for sample in indices:
        clean_gt = targets_by_weather['clean'][sample]
        if any(not np.array_equal(clean_gt, targets_by_weather[weather][sample])
               for weather in WEATHERS[1:]):
            raise ValueError(f'Weather datasets disagree on frame {sample} GT')
    matrix = np.stack(feature_vectors).astype(np.float32)
    return {'root': root, 'metadata': meta, 'indices': indices,
            'scene_map': scene_map, 'rows': rows_by_weather,
            'targets': targets_by_weather, 'records': records,
            'matrix': matrix, 'layout': layout, 'provenance': hashes}


def _competition_pairs(rows, threshold, limit):
    """Exact BEV overlapping good/bad pairs; prioritize score-inverted pairs."""
    from opencood.utils import common_utils

    good = [row for row in rows if row['within_range']
            and row['source_count'] >= 2 and row['quality'] >= .7]
    bad = [row for row in rows if row['within_range']
           and row['source_count'] >= 2 and row['quality'] < .5]
    if not good or not bad:
        return []
    bad_boxes = np.stack([row['corners'] for row in bad])
    bad_low, bad_high = bad_boxes.min(axis=1), bad_boxes.max(axis=1)
    bad_polygons = common_utils.convert_format(bad_boxes)
    found = []
    for winner in good:
        box = winner['corners']
        nearby = np.flatnonzero(((bad_high > box.min(axis=0)) &
                                 (bad_low < box.max(axis=0))).all(axis=1))
        if not len(nearby):
            continue
        polygon = common_utils.convert_format(box[None])[0]
        overlaps = common_utils.compute_iou(polygon, bad_polygons[nearby])
        for local, overlap in zip(nearby, overlaps):
            if overlap > threshold:
                loser = bad[int(local)]
                hard = winner['score'] <= .2 and loser['score'] > .2
                found.append((winner['vector_index'], loser['vector_index'],
                              hard, float(overlap)))
    found.sort(key=lambda pair: (-int(pair[2]), -pair[3], pair[0], pair[1]))
    return found[:limit]


def _training_arrays(data, pair_limit):
    labels = np.asarray([row['quality'] >= .7 for row in data['records']],
                        dtype=np.float32)
    usable = np.asarray([row['within_range'] and row['source_count'] >= 2 and
                         (row['quality'] >= .7 or row['quality'] < .5)
                         for row in data['records']], dtype=bool)
    point_indices = np.flatnonzero(usable)
    if not len(point_indices) or len(np.unique(labels[point_indices])) != 2:
        raise ValueError('Training pool needs both good and bad multi-source boxes')
    pairs = []
    weather_counts = {}
    for weather in WEATHERS:
        before = len(pairs)
        condition = data['rows'][weather]
        for sample in data['indices']:
            pairs.extend(_competition_pairs(condition['rows'][sample],
                                            condition['threshold'], pair_limit))
        weather_counts[weather] = len(pairs) - before
    if not pairs or not any(pair[2] for pair in pairs):
        raise ValueError('Training split lacks score-inverted NMS competition pairs')
    pair_index = np.asarray([(good, bad) for good, bad, _, _ in pairs],
                            dtype=np.int64)
    pair_weights = np.asarray([4. if hard else 1. for _, _, hard, _ in pairs],
                              dtype=np.float32)
    return point_indices, labels, pair_index, pair_weights, {
        'labelled_candidates': len(point_indices),
        'good_candidates': int(labels[point_indices].sum()),
        'bad_candidates': int(len(point_indices) - labels[point_indices].sum()),
        'competition_pairs': len(pairs),
        'score_inverted_pairs': sum(pair[2] for pair in pairs),
        'pairs_by_weather': weather_counts,
        'pair_limit_per_frame': pair_limit,
    }


def _scaler(matrix, records):
    indices = [row['vector_index'] for row in records
               if row['within_range'] and row['source_count'] >= 2]
    values = matrix[indices]
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = values.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-4)
    return mean, std


def _normalized(matrix, mean, std):
    if matrix.shape[1] != len(mean) or len(mean) != len(std):
        raise ValueError('Feature dimension differs from trained scaler')
    values = np.clip((matrix - mean) / std, -10., 10.).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError('Non-finite standardized features')
    return values


def _network(dimension, width):
    from torch import nn
    return nn.Sequential(nn.Linear(dimension, width), nn.ReLU(),
                         nn.Linear(width, width // 2), nn.ReLU(),
                         nn.Linear(width // 2, 1))


def _fit(matrix, point_indices, labels, pairs, pair_weights, mask, seed, args):
    import torch
    from torch.nn import functional as functional

    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network = _network(matrix.shape[1], args.width).to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    rng = np.random.default_rng(seed)
    usable_good = float(labels[point_indices].sum())
    positive_weight = min(20., (len(point_indices) - usable_good) / usable_good)
    features = torch.as_tensor(matrix * mask, dtype=torch.float32, device=device)
    target = torch.as_tensor(labels, dtype=torch.float32, device=device)
    pair_index = torch.as_tensor(pairs, dtype=torch.long, device=device)
    weights = torch.as_tensor(pair_weights, dtype=torch.float32, device=device)
    batches = max(1, int(np.ceil(len(point_indices) / args.batch_size)))
    history = []
    for epoch in range(args.epochs):
        order = rng.permutation(point_indices)
        pair_order = rng.permutation(len(pairs))
        network.train()
        totals = Counter()
        for step in range(batches):
            subset = order[step * args.batch_size:(step + 1) * args.batch_size]
            if not len(subset):
                continue
            sampled = rng.choice(pair_order, size=args.pair_batch,
                                 replace=len(pairs) < args.pair_batch)
            optimizer.zero_grad(set_to_none=True)
            indices = torch.as_tensor(subset, dtype=torch.long, device=device)
            logits = network(features[indices]).reshape(-1)
            point_loss = functional.binary_cross_entropy_with_logits(
                logits, target[indices], pos_weight=torch.tensor(
                    positive_weight, device=device))
            pair_ids = pair_index[torch.as_tensor(sampled, dtype=torch.long,
                                                  device=device)]
            pair_logits = network(features[pair_ids.reshape(-1)]).reshape(-1, 2)
            pair_loss = (functional.softplus(pair_logits[:, 1] - pair_logits[:, 0])
                         * weights[torch.as_tensor(sampled, dtype=torch.long,
                                                   device=device)]).mean()
            loss = point_loss + args.pair_weight * pair_loss
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite ranker loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 5.)
            optimizer.step()
            totals['point'] += float(point_loss.detach())
            totals['pair'] += float(pair_loss.detach())
            totals['steps'] += 1
        summary = {'epoch': epoch + 1,
                   'point_loss': totals['point'] / totals['steps'],
                   'pair_loss': totals['pair'] / totals['steps']}
        history.append(summary)
        print(f'ranker seed={seed} epoch={epoch+1}/{args.epochs} '
              f'point={summary["point_loss"]:.5f} '
              f'pair={summary["pair_loss"]:.5f}', flush=True)
    return network.cpu().eval(), history


def _predict(network, matrix, mask, batch_size=8192):
    import torch
    network.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network = network.to(device)
    result = np.empty(len(matrix), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(matrix), batch_size):
            values = torch.as_tensor(matrix[start:start + batch_size] * mask,
                                     dtype=torch.float32, device=device)
            result[start:start + batch_size] = torch.sigmoid(
                network(values).reshape(-1)).cpu().numpy()
    return result


def _checkpoint_path(folder, seed, method):
    return folder / f'{method}_seed{seed}.pt'


def _save_checkpoint(folder, seed, method, network, mean, std, layout,
                     training_hash, width):
    import torch
    checkpoint = {
        'state': network.state_dict(), 'mean': torch.as_tensor(mean),
        'std': torch.as_tensor(std), 'layout': layout,
        'training_audit_sha256': training_hash,
        'model_name': method, 'seed': seed, 'width': width,
    }
    torch.save(checkpoint, _checkpoint_path(folder, seed, method))


def _load_checkpoint(folder, seed, method, layout, expected_training_hash):
    import torch
    path = _checkpoint_path(folder, seed, method)
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if (checkpoint['model_name'] != method or checkpoint['seed'] != seed
            or checkpoint['layout'] != layout
            or checkpoint['training_audit_sha256'] != expected_training_hash):
        raise ValueError(f'Checkpoint contract differs: {path}')
    network = _network(max(stop for _, stop in layout.values()),
                       int(checkpoint['width']))
    network.load_state_dict(checkpoint['state'], strict=True)
    return network.eval(), checkpoint['mean'].numpy(), checkpoint['std'].numpy()


def _prepare_baselines(data, tolerance):
    """Cache immutable original selections and exact direct NMS suppressions."""
    from opencood.utils import common_utils

    result = {}
    for weather in WEATHERS:
        condition = data['rows'][weather]
        threshold = condition['threshold']
        originals = {method: _stats() for method in BASELINES}
        scene_stats = {method: defaultdict(_stats) for method in BASELINES}
        frames = {}
        for sample in data['indices']:
            rows = condition['rows'][sample]
            gt = data['targets'][weather][sample]
            scores = [row['score'] for row in rows]
            original_indices = [i for i, row in enumerate(rows) if row['score'] > .2]
            original_pool = [rows[i] for i in original_indices]
            original_local = _nms_and_budget(
                original_pool, [row['score'] for row in original_pool], threshold)
            original = [original_indices[index] for index in original_local]
            budget = len(original)
            top = _nms_and_budget(rows, scores, threshold, budget)
            agreement = [row['simple_agreement'] if row['source_count'] >= 2
                         else row['score'] for row in rows]
            agreement_selected = _nms_and_budget(rows, agreement, threshold,
                                                 budget)
            selections = {'scorepass': original, 'top256_fused': top,
                          'source_agreement_raw': agreement_selected}
            scene = str(data['scene_map'][str(sample)])
            for name, selected in selections.items():
                frame_stats = _stats()
                _evaluate_frame(frame_stats, rows, selected, scores, gt)
                _merge_stats(originals[name], frame_stats)
                _merge_stats(scene_stats[name][scene], frame_stats)
            polygons = common_utils.convert_format(
                np.stack([row['corners'] for row in rows]))
            overlap = lambda candidate, others: common_utils.compute_iou(
                polygons[candidate], polygons[others])
            original_order = np.argsort(np.asarray(scores, dtype=np.float32))[::-1]
            _, suppressed = _nms_ordered(original_order, overlap, threshold)
            top_set = set(top)
            target_edges = [(low, high) for low, (high, _) in suppressed.items()
                            if rows[low]['within_range'] and rows[high]['within_range']
                            and rows[low]['score'] <= .2 and rows[low]['quality'] >= .7
                            and rows[high]['score'] > .2 and rows[high]['quality'] < .5
                            and high in top_set]
            frames[sample] = {
                'budget': budget, 'selected': selections,
                'matched_top': _matched_gt(rows, top, gt),
                'target_edges': target_edges,
                'original_order': original_order,
                'suppressors': suppressed,
            }
        observed = _ap(originals['scorepass'], False)
        swap = data['metadata']['conditions'][weather]['swap_ap'][
            'F_fusion_F_detector']
        for metric in ('ap30', 'ap50', 'ap70'):
            if abs(observed[metric] - float(swap[metric])) > tolerance:
                raise RuntimeError(f'{weather}: original {metric} AP reproduction failed')
        result[weather] = {
            'frames': frames, 'stats': originals,
            'scene_stats': scene_stats, 'reproduced_ap': observed,
            'target_direct_edges': sum(len(value['target_edges'])
                                       for value in frames.values()),
        }
        print(f'{weather}: original AP reproduced; direct edge count '
              f'{result[weather]["target_direct_edges"]}', flush=True)
    return result


def _method_summary(stats, per_scene, output, filled, new, lost,
                    direct_correct, direct_total):
    ap_frame = _ap(stats, False)
    ap_global = _ap(stats, True)
    leave_one_scene_out = {}
    for excluded in sorted(per_scene, key=int):
        remaining = _stats()
        for scene in sorted(per_scene, key=int):
            if scene != excluded:
                _merge_stats(remaining, per_scene[scene])
        leave_one_scene_out[excluded] = _ap(remaining, False)['ap70']
    return {
        'ap_frame_order': ap_frame, 'ap_global_sort': ap_global,
        'tp70': int(sum(stats[.7]['tp'])),
        'fp70': int(sum(stats[.7]['fp'])),
        'output_boxes': output, 'budget_fill_rate': filled,
        'new_matched_gt_vs_top256': new,
        'lost_matched_gt_vs_top256': lost,
        'direct_edge_good_ranked_higher': direct_correct,
        'direct_edge_total': direct_total,
        'leave_one_scene_out_ap70_frame': leave_one_scene_out,
    }


def _evaluate_seed(data, baselines, probabilities, seed):
    from sklearn.metrics import roc_auc_score

    result, frame_rows = {}, []
    for weather in WEATHERS:
        condition = data['rows'][weather]
        threshold = condition['threshold']
        reference = baselines[weather]
        stats = {name: _stats() for name in MODELS}
        learned_stats = {name: _stats() for name in MODELS}
        scenes = {name: defaultdict(_stats) for name in MODELS}
        counts = {name: Counter() for name in MODELS}
        correct = Counter()
        direct_total = reference['target_direct_edges']
        for sample in data['indices']:
            rows = condition['rows'][sample]
            gt = data['targets'][weather][sample]
            fixed = reference['frames'][sample]
            budget = fixed['budget']
            scene = str(data['scene_map'][str(sample)])
            top_matched = fixed['matched_top']
            from opencood.utils import common_utils
            polygons = common_utils.convert_format(
                np.stack([row['corners'] for row in rows]))
            overlap = lambda candidate, others: common_utils.compute_iou(
                polygons[candidate], polygons[others])
            within = np.asarray([row['within_range'] for row in rows], dtype=bool)
            for name in MODELS:
                values = [float(probabilities[name][row['vector_index']])
                          if row['source_count'] >= 2 else row['score']
                          for row in rows]
                order, _, _ = _oracle_order(
                    fixed['original_order'], fixed['suppressors'], values,
                    np.ones(len(rows), dtype=bool))
                picked, _ = _nms_ordered(order, overlap, threshold)
                selected = [index for index in picked if within[index]][:budget]
                original_scores = [row['score'] for row in rows]
                frame_stats = _stats()
                _evaluate_frame(frame_stats, rows, selected,
                                original_scores, gt)
                _merge_stats(stats[name], frame_stats)
                _merge_stats(scenes[name][scene], frame_stats)
                _evaluate_frame(learned_stats[name], rows, selected, values, gt)
                matched = _matched_gt(rows, selected, gt)
                gained, lost = len(matched - top_matched), len(top_matched - matched)
                counts[name]['output'] += len(selected)
                counts[name]['budget'] += budget
                counts[name]['new'] += gained
                counts[name]['lost'] += lost
                for low, high in fixed['target_edges']:
                    correct[name] += int(values[low] > values[high])
                frame_rows.append({
                    'seed': seed, 'weather': weather, 'sample_index': sample,
                    'scene': scene, 'method': name, 'budget': budget,
                    'output_boxes': len(selected), 'new_gt': gained,
                    'lost_gt': lost,
                    'direct_edges': len(fixed['target_edges']),
                    'direct_correct': sum(values[low] > values[high]
                                          for low, high in fixed['target_edges']),
                })
        methods = {}
        baseline_tp = int(sum(reference['stats']['top256_fused'][.7]['tp']))
        for name in MODELS:
            count = counts[name]
            summary = _method_summary(
                stats[name], scenes[name], count['output'],
                count['output'] / max(1, count['budget']),
                count['new'], count['lost'], correct[name], direct_total)
            if summary['tp70'] - baseline_tp != count['new'] - count['lost']:
                raise AssertionError(f'{weather}/{name}: TP identity mismatch')
            if summary['tp70'] + summary['fp70'] != count['output']:
                raise AssertionError(f'{weather}/{name}: TP/FP count mismatch')
            summary['ap_using_learned_output_score'] = {
                'frame_order': _ap(learned_stats[name], False),
                'global_sort': _ap(learned_stats[name], True),
            }
            bands = {}
            for band, inside in (('low', lambda score: score <= .2),
                                 ('high', lambda score: score > .2)):
                chosen = [row for sample in data['indices']
                          for row in condition['rows'][sample]
                          if row['within_range'] and row['source_count'] >= 2
                          and inside(row['score'])
                          and (row['quality'] >= .7 or row['quality'] < .5)]
                labels = np.asarray([row['quality'] >= .7 for row in chosen],
                                    dtype=np.int8)
                scores = [probabilities[name][row['vector_index']]
                          for row in chosen]
                bands[band] = {
                    'good': int(labels.sum()),
                    'bad': int(len(labels) - labels.sum()),
                    'auc': float(roc_auc_score(labels, scores))
                           if len(np.unique(labels)) == 2 else None,
                }
            summary['same_score_band_quality_auc'] = bands
            methods[name] = summary
        result[weather] = {'methods': methods}
        print(f'seed={seed} {weather}: NMS/AP evaluation complete', flush=True)
    return result, frame_rows


def _baseline_summaries(data, baselines):
    result = {}
    for weather in WEATHERS:
        condition = data['rows'][weather]
        reference = baselines[weather]
        methods = {}
        for name in BASELINES:
            output = budget = new = lost = correct = 0
            for sample in data['indices']:
                rows = condition['rows'][sample]
                frame = reference['frames'][sample]
                selected = frame['selected'][name]
                output += len(selected)
                budget += frame['budget']
                matched = _matched_gt(rows, selected,
                                      data['targets'][weather][sample])
                new += len(matched - frame['matched_top'])
                lost += len(frame['matched_top'] - matched)
                if name == 'source_agreement_raw':
                    correct += sum(rows[low]['simple_agreement'] >
                                   rows[high]['simple_agreement']
                                   for low, high in frame['target_edges'])
            methods[name] = _method_summary(
                reference['stats'][name], reference['scene_stats'][name],
                output, output / max(1, budget), new, lost, correct,
                reference['target_direct_edges'])
            if methods[name]['tp70'] + methods[name]['fp70'] != output:
                raise AssertionError(f'{weather}/{name}: invalid baseline TP/FP')
        result[weather] = methods
    return result


def _investment_gate(report):
    """Fixed pilot screen; reused validation can never certify a final method."""
    differences, original_differences, excluded_differences = {}, {}, {}
    for weather in WEATHERS:
        changes, original_changes, worst_without_scene = [], [], []
        original = report['baselines'][weather]['scorepass']
        for seed in report['protocol']['seeds']:
            methods = report['seeds'][str(seed)][weather]['methods']
            proposed = methods['candidate_plus_distortion']
            controls = [methods[name] for name in MODELS[:-1]]
            changes.append(proposed['ap_frame_order']['ap70'] -
                           max(row['ap_frame_order']['ap70'] for row in controls))
            original_changes.append(proposed['ap_frame_order']['ap70'] -
                                    original['ap_frame_order']['ap70'])
            excluded = proposed['leave_one_scene_out_ap70_frame']
            worst_without_scene.append(min(
                excluded[scene] - max(
                    [original['leave_one_scene_out_ap70_frame'][scene]] +
                    [row['leave_one_scene_out_ap70_frame'][scene]
                     for row in controls])
                for scene in excluded))
        differences[weather] = float(np.mean(changes))
        original_differences[weather] = float(np.mean(original_changes))
        excluded_differences[weather] = float(np.mean(worst_without_scene))
    weather_passes = sum(differences[weather] >= .005
                         and original_differences[weather] >= .005
                         and excluded_differences[weather] > 0
                         for weather in ('fog', 'rain', 'snow'))
    clean_passes = (differences['clean'] >= -.001
                    and original_differences['clean'] >= -.001)
    return {
        'rule': ('Invest further only if mean frame-order AP70 gain over both '
                 'the original detector and the best same-capacity control is '
                 '>=0.005 in at least two adverse weathers, remains positive '
                 'after removing any one scene there, and Clean loss versus '
                 'both references is <=0.001. This is an investment screen, '
                 'not a significance test or independent test-set success.'),
        'mean_ap70_delta_vs_best_control': differences,
        'mean_ap70_delta_vs_original': original_differences,
        'mean_worst_delta_excluding_one_scene': excluded_differences,
        'weather_pass_count': weather_passes,
        'clean_preserved': bool(clean_passes),
        'pilot_gate_pass': bool(weather_passes >= 2 and clean_passes),
        'final_evidence_status': 'exploratory validation; fresh scenes required',
    }


def _markdown(report):
    lines = ['# 冻结检测器候选排序头试验', '',
             '主 AP 使用原融合分数，仅检验候选进入 NMS 的排序与去留。',
             'learned-output-score AP 单独保存在 JSON；它同时改变排序和跨帧置信度。',
             'GT 只提供训练标签和验证评价，不是推理输入。', '',
             '| 天气 | 原流程 AP70 | 候选自身 | +简单一致性 | +来源基础量 | '
             '+score–geometry distortion | 真正坏压好边 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for weather in WEATHERS:
        original = report['baselines'][weather]['scorepass']['ap_frame_order']['ap70']
        average = {}
        for name in MODELS:
            average[name] = float(np.mean([
                report['seeds'][str(seed)][weather]['methods'][name]
                ['ap_frame_order']['ap70']
                for seed in report['protocol']['seeds']]))
        lines.append(
            f"| {weather} | {original:.4f} | "
            f"{average['candidate_only']:.4f} | "
            f"{average['candidate_plus_simple']:.4f} | "
            f"{average['candidate_plus_source']:.4f} | "
            f"{average['candidate_plus_distortion']:.4f} | "
            f"{report['baselines'][weather]['top256_fused']['direct_edge_total']} |")
    gate = report['investment_gate']
    lines += ['', '## 投入门槛', '',
              f"预先固定的试验门槛：{'通过' if gate['pilot_gate_pass'] else '未通过'}。",
              '这批 validation 场景已被反复用于开发；通过仅表示值得做独立场景复验。',
              'JSON 含每个种子、逐天气、逐模型的 TP/FP、找回/丢失 GT、两种 AP、'
              '真实抑制边排序与逐场景剔除结果。', '']
    return '\n'.join(lines)


def _validate_frozen_contract(train_meta, evaluation_meta):
    row_builder = _sha256(Path(__file__).with_name('candidate_audit.py'))
    if (train_meta.get('row_builder_sha256') != row_builder
            or evaluation_meta.get('row_builder_sha256',
                                   evaluation_meta.get('implementation_sha256'))
            != row_builder):
        raise ValueError('Candidate row builder differs between train/evaluation')
    for field in ('frontend_sha256', 'v3_checkpoint_sha256'):
        if train_meta.get(field) != evaluation_meta.get(field):
            raise ValueError(f'Train/evaluation frozen model differs: {field}')
    if train_meta['arm_sha256']['F'] != evaluation_meta['arm_sha256']['F']:
        raise ValueError('Train/evaluation F checkpoint differs')


def _parse_seeds(text):
    seeds = [int(value) for value in text.split(',')]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError('Seeds must be a nonempty unique comma-separated list')
    return seeds


def run(args):
    evaluation = _load(args.eval_root, 'validation')
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    checkpoint_folder = output / 'checkpoints'
    training = None
    if args.mode == 'train-eval':
        training = _load(args.train_root, 'train')
        if training['root'] == evaluation['root']:
            raise ValueError('Training and evaluation roots must differ')
        _validate_frozen_contract(training['metadata'], evaluation['metadata'])
        if training['layout'] != evaluation['layout']:
            raise ValueError('Train/evaluation feature layout differs')
        point, labels, pairs, weights, counts = _training_arrays(
            training, args.pair_limit)
        mean, std = _scaler(training['matrix'], training['records'])
        train_matrix = _normalized(training['matrix'], mean, std)
        seeds = _parse_seeds(args.seeds)
        train_hash = training['provenance'][str(training['root'] /
                                                'candidate_audit.json')]
        manifest = {
            'seeds': seeds, 'models': MODELS, 'layout': training['layout'],
            'training_audit_sha256': train_hash,
            'feature_code_sha256': _feature_code_hashes(),
            'frozen_model_sha256': {
                key: training['metadata'][key]
                for key in ('frontend_sha256', 'v3_checkpoint_sha256', 'arm_sha256')},
            'training_counts': counts,
            'training_settings': {
                'epochs': args.epochs, 'width': args.width,
                'batch_size': args.batch_size, 'pair_batch': args.pair_batch,
                'pair_weight': args.pair_weight,
                'learning_rate': args.learning_rate,
                'weight_decay': args.weight_decay,
                'checkpoint_selection': 'fixed last epoch; no validation selection',
            },
        }
    else:
        checkpoint_folder = Path(args.checkpoint_dir).resolve()
        manifest = json.loads((checkpoint_folder / 'manifest.json').read_text(
            encoding='utf-8'))
        if tuple(manifest['models']) != MODELS:
            raise ValueError('Checkpoint method set differs from current code')
        if manifest['feature_code_sha256'] != _feature_code_hashes():
            raise ValueError('Feature/replay code differs from trained checkpoint')
        seeds = manifest['seeds']
        if manifest['layout'] != evaluation['layout']:
            raise ValueError('Evaluation feature layout differs from checkpoint')
        frozen = manifest['frozen_model_sha256']
        for key in ('frontend_sha256', 'v3_checkpoint_sha256'):
            if frozen[key] != evaluation['metadata'][key]:
                raise ValueError(f'Frozen model mismatch: {key}')
        if frozen['arm_sha256']['F'] != evaluation['metadata']['arm_sha256']['F']:
            raise ValueError('Frozen F checkpoint mismatch')
        if evaluation['metadata'].get('row_builder_sha256',
                                      evaluation['metadata'].get('implementation_sha256')) != _sha256(
                Path(__file__).with_name('candidate_audit.py')):
            raise ValueError('Evaluation candidate row builder differs from code')
        train_hash = manifest['training_audit_sha256']
    output.mkdir(parents=True, exist_ok=False)
    if training is not None:
        checkpoint_folder.mkdir()
        (checkpoint_folder / 'manifest.json').write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8')
    baseline = _prepare_baselines(evaluation, args.reproduction_tolerance)
    baseline_summary = _baseline_summaries(evaluation, baseline)
    report = {
        'protocol': {
            'mode': args.mode, 'train_root': str(training['root']) if training else None,
            'eval_root': str(evaluation['root']), 'eval_status':
            'reused_exploratory_validation' if evaluation['metadata'].get('split') is None
            else 'saved_evaluation_split; novelty must be verified externally',
            'seeds': seeds, 'models': MODELS,
            'primary_ap_scores': 'original fused scores after learned NMS selection',
            'secondary_ap_scores': 'learned reliability probabilities',
            'selection_policy': ('permute only baseline NMS suppression-component '
                                 'rank slots using predicted reliability; rerun exact NMS'),
            'score_threshold': .2, 'nms_pool': 'top_256',
            'output_budget': 'original scorepass final boxes per frame',
            'gt_use': 'train labels/pair targets and evaluation only',
            'frozen_model_sha256': manifest['frozen_model_sha256'],
            'training_audit_sha256': train_hash,
            'evaluation_audit_sha256': evaluation['provenance'][str(
                evaluation['root'] / 'candidate_audit.json')],
            'implementation_sha256': _sha256(Path(__file__)),
            'checkpoint_manifest_sha256': _sha256(checkpoint_folder /
                                                   'manifest.json'),
        },
        'training': manifest.get('training_counts'),
        'training_settings': manifest['training_settings'],
        'baselines': baseline_summary,
        'seeds': {}, 'investment_gate': None,
    }
    frame_log = []
    histories = {}
    for seed in seeds:
        probabilities = {}
        histories[str(seed)] = {}
        for method in MODELS:
            mask = _model_mask(evaluation['layout'], method)
            if training is not None:
                network, history = _fit(train_matrix, point, labels, pairs,
                                        weights, mask, seed, args)
                _save_checkpoint(checkpoint_folder, seed, method, network,
                                 mean, std, training['layout'], train_hash,
                                 args.width)
                histories[str(seed)][method] = history
            else:
                network, mean, std = _load_checkpoint(
                    checkpoint_folder, seed, method, evaluation['layout'],
                    train_hash)
            eval_matrix = _normalized(evaluation['matrix'], mean, std)
            probabilities[method] = _predict(network, eval_matrix, mask)
            del network, eval_matrix
        weather_result, frame_result = _evaluate_seed(
            evaluation, baseline, probabilities, seed)
        report['seeds'][str(seed)] = weather_result
        frame_log.extend(frame_result)
    if training is not None:
        report['training_history'] = histories
    report['investment_gate'] = _investment_gate(report)
    if _sha256(Path(__file__)) != report['protocol']['implementation_sha256']:
        raise RuntimeError('Ranker source changed during execution')
    (output / 'candidate_ranker_pilot.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (output / 'candidate_ranker_pilot.md').write_text(
        _markdown(report), encoding='utf-8')
    with (output / 'candidate_ranker_frames.jsonl').open(
            'w', encoding='utf-8') as stream:
        for row in frame_log:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f'CANDIDATE RANKER PILOT COMPLETE: {output}', flush=True)


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
    parser.add_argument('--pair-batch', type=int, default=512)
    parser.add_argument('--pair-limit', type=int, default=128)
    parser.add_argument('--pair-weight', type=float, default=.5)
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
            or args.pair_weight < 0 or args.learning_rate <= 0
            or args.weight_decay < 0 or args.reproduction_tolerance <= 0):
        parser.error('Invalid training or reproduction setting')
    run(args)


if __name__ == '__main__':
    main()
