"""Server-only, GT-assisted scope check for actual top256 NMS suppression.

This is an oracle diagnostic, not an inference method or a mathematical AP
upper bound. Candidate geometry and the original output scores stay fixed.
Within stars of *actual baseline NMS suppression edges*, GT quality may only
permute the occupied rank positions. NMS is then replayed in that order.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .candidate_hypothesis_replay import _ap, _evaluate_frame, _stats


WEATHERS = ('clean', 'fog', 'rain', 'snow')
METHODS = ('scorepass', 'top256_fused', 'targeted_gt_order', 'all_conflicts_gt_order')


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _resolve_root(path):
    root = Path(path).resolve()
    if not (root / 'candidate_audit.json').is_file():
        root = root / 'extraction'
    if not (root / 'candidate_audit.json').is_file():
        raise FileNotFoundError(f'No candidate_audit.json in input or extraction/: {path}')
    return root


def _load(root, arm):
    meta_path = root / 'candidate_audit.json'
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    if meta.get('audit_pool') != 'top_256' or meta.get('stage0_only'):
        raise ValueError('The scope check requires a complete top_256 source audit')
    indices = [int(value) for value in meta['validation_indices']]
    if not indices or len(indices) != len(set(indices)):
        raise ValueError('Invalid validation frame indices')
    scenes = meta.get('validation_scene_map') or {}
    if set(scenes) != set(map(str, indices)):
        raise ValueError('A complete scene map is required')
    provenance = {str(meta_path): _sha256(meta_path)}
    conditions = {}
    for weather in WEATHERS:
        condition = meta['conditions'][weather]
        if abs(float(condition['original_score_threshold']) - .2) > 1e-8:
            raise ValueError(f'{weather}: original threshold is not 0.2')
        threshold = float(condition['nms_iou_threshold'])
        if not 0 < threshold < 1:
            raise ValueError(f'{weather}: invalid NMS IoU threshold')
        folder = root / weather
        target_path = folder / 'frame_targets.jsonl'
        row_path = folder / 'candidate_rows.jsonl'
        provenance[str(target_path)] = _sha256(target_path)
        provenance[str(row_path)] = _sha256(row_path)
        targets = {}
        with target_path.open(encoding='utf-8') as stream:
            for line_no, line in enumerate(stream, 1):
                raw = json.loads(line)
                sample = int(raw['sample_index'])
                if raw['weather'] != weather or sample in targets:
                    raise ValueError(f'{target_path}:{line_no}: duplicate or wrong weather')
                gt = np.asarray(raw['gt_bev_corners'], dtype=np.float32)
                if gt.size == 0:
                    gt = gt.reshape(0, 4, 2)
                if gt.ndim != 3 or gt.shape[1:] != (4, 2) or not np.isfinite(gt).all():
                    raise ValueError(f'{target_path}:{line_no}: invalid GT corners')
                targets[sample] = gt
        by_frame = {sample: [] for sample in indices}
        seen = set()
        with row_path.open(encoding='utf-8') as stream:
            for line_no, line in enumerate(stream, 1):
                raw = json.loads(line)
                if raw['arm'] != arm:
                    continue
                sample = int(raw['sample_index'])
                cid = int(raw['candidate_id'])
                if (raw['weather'] != weather or sample not in by_frame
                        or (sample, cid) in seen):
                    raise ValueError(f'{row_path}:{line_no}: invalid candidate identity')
                seen.add((sample, cid))
                corners = np.asarray(raw['fused_bev_corners'], dtype=np.float32)
                score, quality = float(raw['score']), float(raw['max_gt_iou'])
                if (corners.shape != (4, 2) or not np.isfinite(corners).all()
                        or not np.isfinite([score, quality]).all()
                        or not 0 <= score <= 1 or not 0 <= quality <= 1):
                    raise ValueError(f'{row_path}:{line_no}: invalid box/score/quality')
                by_frame[sample].append({
                    'candidate_id': cid, 'corners': corners, 'score': score,
                    'quality': quality, 'within_range': bool(raw['within_range']),
                })
        if set(targets) != set(indices):
            raise ValueError(f'{weather}: target frames differ from validation indices')
        if any(not 1 <= len(rows) <= 256 for rows in by_frame.values()):
            raise ValueError(f'{weather}: each validation frame needs 1–256 candidates')
        pools = condition['pools'][arm]
        top_count = sum(len(rows) for rows in by_frame.values())
        original_count = sum(row['score'] > .2 for rows in by_frame.values()
                             for row in rows)
        if (top_count != int(pools['top_256']['total_candidates'])
                or original_count != int(pools['original']['total_candidates'])):
            raise ValueError(f'{weather}: top256 omits part of the original pool')
        conditions[weather] = {'rows': by_frame, 'targets': targets,
                               'nms_iou_threshold': threshold}
    return meta, conditions, provenance


def _nms_ordered(order, overlap, threshold):
    """Exact greedy NMS for a supplied rank order; returns first suppressors."""
    pool = np.asarray(order, dtype=np.int64).copy()
    if len(pool) != len(np.unique(pool)):
        raise ValueError('NMS order has duplicate candidates')
    picked, suppressors = [], {}
    while len(pool):
        winner = int(pool[0])
        picked.append(winner)
        if len(pool) == 1:
            break
        values = np.asarray(overlap(winner, pool[1:]), dtype=np.float64)
        if values.shape != (len(pool) - 1,) or not np.isfinite(values).all():
            raise ValueError('Invalid polygon overlap vector')
        suppressed = np.flatnonzero(values > threshold) + 1
        for position in suppressed:
            suppressors[int(pool[position])] = (winner, float(values[position - 1]))
        pool = np.delete(pool, np.r_[0, suppressed])
    return picked, suppressors


def _suppression_components(suppressors):
    """Baseline first-suppressor edges form stars rooted at picked boxes."""
    components = defaultdict(set)
    for loser, (winner, _) in suppressors.items():
        if winner in suppressors:
            raise AssertionError('A baseline suppressor was itself suppressed')
        components[winner].update((winner, loser))
    return {root: sorted(members) for root, members in components.items()}


def _oracle_order(original_order, suppressors, quality, within_range, roots=None):
    """Permute only the original rank slots occupied by allowed NMS stars."""
    original_order = np.asarray(original_order, dtype=np.int64)
    quality = np.asarray(quality, dtype=np.float64)
    within_range = np.asarray(within_range, dtype=bool)
    position = {int(candidate): offset for offset, candidate in enumerate(original_order)}
    revised = original_order.copy()
    used = 0
    for root, members in _suppression_components(suppressors).items():
        if roots is not None and root not in roots:
            continue
        slots = sorted(position[member] for member in members)
        # Out-of-range winners cannot contribute a final detection. GT is
        # otherwise used only for this counterfactual NMS ordering.
        ranking = sorted(members, key=lambda member: (
            -int(within_range[member]), -quality[member], position[member]))
        revised[slots] = ranking
        used += 1
    if set(revised) != set(original_order):
        raise AssertionError('Oracle altered candidate pool membership')
    return revised, used, int(np.count_nonzero(revised != original_order))


def _is_low_good(row):
    return row['within_range'] and row['score'] <= .2 and row['quality'] >= .7


def _is_high_bad(row):
    return row['within_range'] and row['score'] > .2 and row['quality'] < .5


def _matched_gt(rows, selected, gt):
    """Return GT identities using eval_utils' per-frame score-ordered match rule."""
    from opencood.utils import common_utils
    if not selected or not len(gt):
        return set()
    boxes = np.stack([row['corners'] for row in rows])
    polygons = common_utils.convert_format(boxes)
    gt_polygons = list(common_utils.convert_format(gt))
    remaining = list(range(len(gt)))
    values = np.asarray([rows[index]['score'] for index in selected], dtype=np.float32)
    matched = set()
    for local_index in np.argsort(-values):
        if not remaining:
            break
        candidate = selected[int(local_index)]
        ious = common_utils.compute_iou(
            polygons[candidate], [gt_polygons[index] for index in remaining])
        if len(ious) and float(np.max(ious)) >= .7:
            gt_index = remaining.pop(int(np.argmax(ious)))
            matched.add(gt_index)
    return matched


def _frame_scope(rows, gt, threshold):
    import torch
    from gspr_evidence.stage3_trace import polygon_ious
    from opencood.utils import box_utils, common_utils

    boxes = np.asarray([row['corners'] for row in rows], dtype=np.float32).reshape(-1, 4, 2)
    scores = np.asarray([row['score'] for row in rows], dtype=np.float32)
    quality = np.asarray([row['quality'] for row in rows], dtype=np.float64)
    within = np.asarray([row['within_range'] for row in rows], dtype=bool)
    polygons = common_utils.convert_format(boxes)
    overlap = lambda candidate, others: common_utils.compute_iou(
        polygons[candidate], polygons[others])
    original_order = np.argsort(scores)[::-1]
    top_picked, suppressors = _nms_ordered(original_order, overlap, threshold)
    official = box_utils.nms_rotated(torch.as_tensor(boxes),
                                     torch.as_tensor(scores), threshold)
    if not np.array_equal(top_picked, official):
        raise AssertionError('Original top256 NMS does not match repository NMS')
    scorepass_ids = np.flatnonzero(scores > .2)
    scorepass_order = scorepass_ids[np.argsort(scores[scorepass_ids])[::-1]]
    scorepass_picked, _ = _nms_ordered(scorepass_order, overlap, threshold)
    official_scorepass = box_utils.nms_rotated(
        torch.as_tensor(boxes[scorepass_ids]), torch.as_tensor(scores[scorepass_ids]),
        threshold)
    if not np.array_equal(scorepass_picked, scorepass_ids[official_scorepass]):
        raise AssertionError('Original scorepass NMS does not match repository NMS')
    original_final = [index for index in scorepass_picked if within[index]]
    budget = len(original_final)
    top_final = [index for index in top_picked if within[index]][:budget]
    components = _suppression_components(suppressors)
    target_edges = [(loser, winner, iou) for loser, (winner, iou) in suppressors.items()
                    if _is_low_good(rows[loser]) and _is_high_bad(rows[winner])]
    target_roots = {winner for _, winner, _ in target_edges}
    target_order, targeted_components, targeted_moved = _oracle_order(
        original_order, suppressors, quality, within, target_roots)
    all_order, all_components, all_moved = _oracle_order(
        original_order, suppressors, quality, within)
    target_picked, _ = _nms_ordered(target_order, overlap, threshold)
    all_picked, _ = _nms_ordered(all_order, overlap, threshold)
    selected = {
        'scorepass': original_final,
        'top256_fused': top_final,
        'targeted_gt_order': [index for index in target_picked if within[index]][:budget],
        'all_conflicts_gt_order': [index for index in all_picked if within[index]][:budget],
    }
    ious = polygon_ious(boxes, gt)
    actual_quality = ious.max(axis=1) if len(gt) else np.zeros(len(rows))
    if len(rows) and not np.allclose(actual_quality, quality, atol=1e-5, rtol=1e-5):
        raise AssertionError('Saved candidate quality differs from exact GT polygon IoU')
    matched = {method: _matched_gt(rows, indices, gt)
               for method, indices in selected.items()}
    top_missed = set(range(len(gt))) - matched['top256_fused']
    original_missed = set(range(len(gt))) - matched['scorepass']
    covered_by_target_edges = set()
    for loser, _, _ in target_edges:
        covered_by_target_edges.update(np.flatnonzero(ious[loser] >= .7).tolist())
    final_top_set = set(top_final)
    target_details = [{
        'low_good_candidate_id': rows[loser]['candidate_id'],
        'high_bad_suppressor_id': rows[winner]['candidate_id'],
        'nms_bev_iou': iou,
        'suppressor_score': rows[winner]['score'],
        'suppressor_quality': rows[winner]['quality'],
        'suppressor_in_fixed_output': winner in final_top_set,
    } for loser, winner, iou in target_edges]
    lowgood_status = Counter()
    for index, row in enumerate(rows):
        if not _is_low_good(row):
            continue
        if index in suppressors:
            winner = suppressors[index][0]
            if _is_high_bad(rows[winner]):
                label = 'suppressed_by_high_bad'
            elif not rows[winner]['within_range']:
                label = 'suppressed_by_out_of_range'
            else:
                label = 'suppressed_by_other'
            lowgood_status[label] += 1
        elif index in final_top_set:
            lowgood_status['in_fixed_output'] += 1
        else:
            lowgood_status['survived_nms_beyond_budget'] += 1
    return selected, matched, {
        'budget': budget, 'candidate_count': len(rows), 'gt_count': len(gt),
        'nms_suppression_edges': len(suppressors),
        'nms_suppression_components': len(components),
        'targeted_components': targeted_components,
        'targeted_rank_positions_moved': targeted_moved,
        'all_components_reordered': all_components,
        'all_rank_positions_moved': all_moved,
        'direct_low_good_suppressed_by_high_bad': len(target_edges),
        'direct_edge_suppressor_in_fixed_output': sum(
            item['suppressor_in_fixed_output'] for item in target_details),
        'eligible_missed_gt_vs_top256': len(covered_by_target_edges & top_missed),
        'eligible_missed_gt_vs_scorepass': len(covered_by_target_edges & original_missed),
        'low_good_status': dict(lowgood_status),
        'target_edges': target_details,
    }


def _markdown(report):
    lines = ['# top256 真实 NMS 竞争范围验证', '',
             'GT 仅用于受限候选排序反事实及事后评价；没有训练可部署方法。',
             '候选框、原始输出分数和每帧原流程输出预算保持固定。',
             'targeted 只重排原 NMS 中确实由高分坏框直接压掉低分好框的抑制组。',
             'all_conflicts 重排所有原 NMS 抑制组，作为更宽的参考。',
             '两者只改变各组原来占据的排名位置，随后重新运行精确 NMS。', '',
             '| 天气 | 真正的坏压好边 | 涉及组 | 可覆盖但原 top256 未命中的 GT | '
             '受限反事实新获/丢失 GT | 帧顺序 AP70：原流程/top256/受限/全部组 | '
             '跨帧 AP70：原流程/top256/受限/全部组 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for weather in WEATHERS:
        condition = report['conditions'][weather]
        scope = condition['scope']
        target = condition['methods']['targeted_gt_order']
        top = condition['methods']['top256_fused']
        all_conflicts = condition['methods']['all_conflicts_gt_order']
        original = condition['methods']['scorepass']
        lines.append(
            f"| {weather} | {scope['direct_low_good_suppressed_by_high_bad']} | "
            f"{scope['targeted_components']} | {scope['eligible_missed_gt_vs_top256']} | "
            f"{target['new_matched_gt_vs_top256']}/{target['lost_matched_gt_vs_top256']} | "
            f"{original['ap_frame_order']['ap70']:.4f}/"
            f"{top['ap_frame_order']['ap70']:.4f}/"
            f"{target['ap_frame_order']['ap70']:.4f}/"
            f"{all_conflicts['ap_frame_order']['ap70']:.4f} | "
            f"{original['ap_global_sort']['ap70']:.4f}/"
            f"{top['ap_global_sort']['ap70']:.4f}/"
            f"{target['ap_global_sort']['ap70']:.4f}/"
            f"{all_conflicts['ap_global_sort']['ap70']:.4f} |")
    lines += ['', '## 判读', '',
              '- `eligible_missed_gt_vs_top256` 是候选覆盖机会，可能无法同时实现，也不等于 AP 上界。',
              '- 反事实使用 GT 选 NMS 顺序，是诊断，不代表推理时能学到相同排序。',
              '- AP 使用原分类分数评价；没有把 GT 质量写成输出置信度。',
              '- 如果某方法的预算填满率小于 1，AP 差异需同时考虑输出数量。',
              '- 输入仍是既有的 9 个 validation 场景，不能据此宣称独立泛化。', '']
    return '\n'.join(lines)


def run(args):
    root = _resolve_root(args.input_root)
    meta, conditions, provenance = _load(root, args.arm)
    report = {
        'protocol': {
            'scope': 'server-only GT-assisted fixed-pool NMS opportunity diagnostic',
            'input_root': str(root), 'arm': args.arm,
            'validation_indices': meta['validation_indices'],
            'scene_map': meta['validation_scene_map'],
            'score_threshold': .2, 'low_good_gt_iou_at_least': .7,
            'high_bad_gt_iou_below': .5,
            'output_scores': 'original fused scores for every method',
            'candidate_audit_sha256': provenance[str(root / 'candidate_audit.json')],
            'implementation_sha256': _sha256(Path(__file__)),
            'reproduction_tolerance': args.reproduction_tolerance,
        }, 'provenance': provenance, 'conditions': {},
    }
    frame_log = []
    for weather in WEATHERS:
        rows_by_frame = conditions[weather]['rows']
        targets = conditions[weather]['targets']
        threshold = conditions[weather]['nms_iou_threshold']
        statistics = {method: _stats() for method in METHODS}
        scope = Counter()
        matched_new = {method: Counter() for method in METHODS}
        counts = {method: Counter() for method in METHODS}
        for frame_number, sample in enumerate(meta['validation_indices'], 1):
            sample = int(sample)
            rows, gt = rows_by_frame[sample], targets[sample]
            selected, matched, detail = _frame_scope(rows, gt, threshold)
            for method in METHODS:
                _evaluate_frame(statistics[method], rows, selected[method],
                                [row['score'] for row in rows], gt)
                counts[method]['output_boxes'] += len(selected[method])
                counts[method]['budget_boxes'] += detail['budget']
                matched_new[method]['new_vs_top256'] += len(
                    matched[method] - matched['top256_fused'])
                matched_new[method]['lost_vs_top256'] += len(
                    matched['top256_fused'] - matched[method])
                matched_new[method]['new_vs_scorepass'] += len(
                    matched[method] - matched['scorepass'])
                matched_new[method]['lost_vs_scorepass'] += len(
                    matched['scorepass'] - matched[method])
            for key, value in detail.items():
                if isinstance(value, int):
                    scope[key] += value
            scope.update(detail['low_good_status'])
            frame_log.append({'weather': weather, 'sample_index': sample,
                              'scene': meta['validation_scene_map'][str(sample)],
                              'scope': detail,
                              'matched_gt': {name: sorted(ids)
                                             for name, ids in matched.items()},
                              'output_candidate_ids': {
                                  name: [rows[index]['candidate_id'] for index in ids]
                                  for name, ids in selected.items()}})
            if frame_number == 1 or frame_number % 20 == 0:
                print(f'{weather} NMS scope {frame_number}/{len(meta["validation_indices"])}',
                      flush=True)
        swap = ('F_fusion_F_detector' if args.arm == 'F'
                else 'FD_fusion_FD_detector')
        observed = _ap(statistics['scorepass'], False)
        expected = meta['conditions'][weather]['swap_ap'][swap]
        for metric in ('ap30', 'ap50', 'ap70'):
            if abs(observed[metric] - float(expected[metric])) > args.reproduction_tolerance:
                raise RuntimeError(f'{weather}: original {metric} AP reproduction failed: '
                                   f'{observed[metric]} versus {expected[metric]}')
        methods = {}
        for method in METHODS:
            methods[method] = {
                'ap_frame_order': _ap(statistics[method], False),
                'ap_global_sort': _ap(statistics[method], True),
                'output_boxes': counts[method]['output_boxes'],
                'budget_fill_rate': counts[method]['output_boxes'] /
                                    max(1, counts[method]['budget_boxes']),
                'tp70': int(sum(statistics[method][.7]['tp'])),
                'fp70': int(sum(statistics[method][.7]['fp'])),
                'new_matched_gt_vs_top256': matched_new[method]['new_vs_top256'],
                'lost_matched_gt_vs_top256': matched_new[method]['lost_vs_top256'],
                'new_matched_gt_vs_scorepass': matched_new[method]['new_vs_scorepass'],
                'lost_matched_gt_vs_scorepass': matched_new[method]['lost_vs_scorepass'],
            }
        for method in METHODS:
            assigned = sum(len(item['matched_gt'][method]) for item in frame_log
                           if item['weather'] == weather)
            if methods[method]['tp70'] != assigned:
                raise AssertionError(f'{weather}/{method}: GT identity match differs from TP')
            if (methods[method]['tp70'] + methods[method]['fp70']
                    != methods[method]['output_boxes']):
                raise AssertionError(f'{weather}/{method}: TP/FP count differs from boxes')
        report['conditions'][weather] = {
            'nms_iou_threshold': threshold,
            'frames': len(meta['validation_indices']),
            'independent_scenes': len(set(meta['validation_scene_map'].values())),
            'scorepass_reproduction': observed,
            'scope': dict(scope), 'methods': methods,
        }
        print(f'{weather}: scope complete', flush=True)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'candidate_nms_scope.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (output / 'candidate_nms_scope.md').write_text(
        _markdown(report), encoding='utf-8')
    with (output / 'candidate_nms_scope_frames.jsonl').open('w', encoding='utf-8') as stream:
        for row in frame_log:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f'CANDIDATE NMS SCOPE COMPLETE: {output}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', required=True,
                        help='Existing candidate_audit directory or its parent with extraction/')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--arm', choices=('F', 'F+D'), default='F')
    parser.add_argument('--reproduction-tolerance', type=float, default=1e-6)
    args = parser.parse_args()
    if args.reproduction_tolerance <= 0:
        parser.error('Reproduction tolerance must be positive')
    run(args)


if __name__ == '__main__':
    main()
