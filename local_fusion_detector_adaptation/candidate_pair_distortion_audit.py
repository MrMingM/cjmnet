"""Direct low-score-good versus high-score-bad distortion audit.

Run on the SERVER against an existing, complete top256 extraction. This file
only reads saved candidate rows; it never loads or forwards the detector.
Every metric is available at inference. GT IoU defines the two groups only.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from .candidate_discriminability import _features
from .candidate_intervention_probe import INTERVENTION_NAMES, _intervention_features


WEATHERS = ('clean', 'fog', 'rain', 'snow')
METRICS = (
    ('fused_score', 'control'),
    ('source_agreement', 'simple_source_control'),
    ('fused_minus_ego_score', 'score_change'),
    ('fused_minus_max_source_score', 'score_change'),
    ('mean_source_fused_iou', 'geometry'),
    ('mean_source_pairwise_iou', 'geometry'),
    ('mean_source_fused_center_shift', 'geometry'),
    ('score_source_geometry_gap', 'score_geometry'),
    ('score_geometry_source_mismatch', 'score_geometry'),
    ('score_gain_weak_geometry', 'score_geometry'),
    ('positive_score_gain_weak_geometry', 'score_geometry'),
    ('score_gain_times_center_shift', 'score_geometry'),
    ('max_peer_removal_score_drop', 'peer_intervention'),
    ('min_peer_removal_box_iou', 'peer_intervention'),
    ('peer_response_source_mismatch', 'peer_intervention'),
)
METRIC_NAMES = tuple(name for name, _ in METRICS)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _metrics(raw):
    passive, _ = _features(raw)
    if passive is None:
        raise ValueError('Direct audit requires at least two sources')
    count = int(raw['source_count'])
    shifts = np.asarray(raw['source_box_center_shift_to_fused'], dtype=np.float64)
    pair_iou = float(raw['source_box_pairwise_iou_mean'])
    if (shifts.shape != (count,) or not np.isfinite(shifts).all()
            or (shifts < 0).any() or not np.isfinite(pair_iou)
            or not 0 <= pair_iou <= 1):
        raise ValueError('Invalid source-to-fused geometry trajectory')
    intervention = _intervention_features(raw)
    gain = passive['delta_fused_max_source']
    result = {
        'fused_score': passive['fused_score'],
        'source_agreement': passive['source_agreement'],
        'fused_minus_ego_score': passive['delta_fused_ego'],
        'fused_minus_max_source_score': gain,
        'mean_source_fused_iou': passive['source_fused_iou_mean'],
        'mean_source_pairwise_iou': pair_iou,
        'mean_source_fused_center_shift': float(shifts.mean()),
        'score_source_geometry_gap': passive['score_source_geometry_gap'],
        'score_geometry_source_mismatch': passive['score_geometry_source_mismatch'],
        'score_gain_weak_geometry': gain * (1. - passive['source_fused_iou_mean']),
        'positive_score_gain_weak_geometry': passive['amplification_weak_geometry'],
        'score_gain_times_center_shift': gain * float(shifts.mean()),
        'max_peer_removal_score_drop': intervention[
            INTERVENTION_NAMES.index('max_score_drop')],
        'min_peer_removal_box_iou': intervention[
            INTERVENTION_NAMES.index('min_box_iou_to_full')],
        'peer_response_source_mismatch': intervention[
            INTERVENTION_NAMES.index('score_geometry_response_source_mismatch')],
    }
    values = np.asarray([result[name] for name in METRIC_NAMES], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('Non-finite direct-comparison metric')
    return result


def _group(raw):
    score, quality = float(raw['score']), float(raw['max_gt_iou'])
    if not (0 <= score <= 1 and 0 <= quality <= 1):
        raise ValueError('Score or GT quality outside [0,1]')
    if score <= .2 and quality >= .7:
        return 'low_good'
    if score > .2 and quality < .5:
        return 'high_bad'
    return None


def _overlap_pairs(by_frame, threshold):
    """Potential NMS competitors: same-frame boxes with IoU > NMS threshold."""
    from opencood.utils import common_utils
    pairs = []
    for sample, rows in by_frame.items():
        good = [row for row in rows if row['group'] == 'low_good']
        bad = [row for row in rows if row['group'] == 'high_bad']
        if not good or not bad:
            continue
        good_boxes = np.stack([row['corners'] for row in good])
        bad_boxes = np.stack([row['corners'] for row in bad])
        good_lo, good_hi = good_boxes.min(axis=1), good_boxes.max(axis=1)
        bad_lo, bad_hi = bad_boxes.min(axis=1), bad_boxes.max(axis=1)
        good_poly = common_utils.convert_format(good_boxes)
        bad_poly = common_utils.convert_format(bad_boxes)
        for index, left in enumerate(good):
            nearby = np.flatnonzero(((good_hi[index] > bad_lo)
                                      & (good_lo[index] < bad_hi)).all(axis=1))
            for other in nearby:
                iou = float(common_utils.compute_iou(
                    good_poly[index], [bad_poly[other]])[0])
                if iou > threshold:
                    pairs.append((left, bad[other], iou))
    return pairs


def _scene_interval(values, scenes, statistic, repetitions, seed):
    unique = np.unique(scenes)
    members = {scene: np.flatnonzero(scenes == scene) for scene in unique}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repetitions):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        ids = np.concatenate([members[scene] for scene in chosen])
        if len(ids):
            draws.append(statistic(values[ids]))
    return ([float(value) for value in np.percentile(draws, [2.5, 97.5])]
            if draws else None, len(draws))


def _summarize(rows, pairs, scene_map, repetitions, seed):
    good = [row for row in rows if row['group'] == 'low_good']
    bad = [row for row in rows if row['group'] == 'high_bad']
    report = {
        'low_good_candidates': len(good), 'high_bad_candidates': len(bad),
        'overlapping_pairs': len(pairs),
        'pair_frames': len({left['sample_index'] for left, _, _ in pairs}),
        'pair_scenes': len({str(scene_map[str(left['sample_index'])])
                            for left, _, _ in pairs}),
        'metrics': {},
    }
    for offset, name in enumerate(METRIC_NAMES):
        left = np.asarray([row['metrics'][name] for row in good], dtype=np.float64)
        right = np.asarray([row['metrics'][name] for row in bad], dtype=np.float64)
        entry = {
            'family': dict(METRICS)[name],
            'low_good_median': float(np.median(left)) if len(left) else None,
            'high_bad_median': float(np.median(right)) if len(right) else None,
            'unpaired_auc_good_higher': (float(roc_auc_score(
                np.r_[np.ones(len(left)), np.zeros(len(right))], np.r_[left, right]))
                if len(left) and len(right) else None),
        }
        if pairs:
            differences = np.asarray([a['metrics'][name] - b['metrics'][name]
                                      for a, b, _ in pairs], dtype=np.float64)
            scenes = np.asarray([str(scene_map[str(a['sample_index'])])
                                 for a, _, _ in pairs])
            # Ties get half credit; no direction is chosen using the GT labels.
            good_higher = (differences > 0).astype(float) + .5 * (differences == 0)
            rate_ci, rate_valid = _scene_interval(
                good_higher, scenes, np.mean, repetitions, seed + offset * 2)
            median_ci, median_valid = _scene_interval(
                differences, scenes, np.median, repetitions, seed + offset * 2 + 1)
            entry['overlap_pair_good_minus_bad_median'] = float(np.median(differences))
            entry['overlap_pair_median_scene_bootstrap_95pct'] = median_ci
            entry['overlap_pair_good_higher_rate_ties_half'] = float(good_higher.mean())
            entry['overlap_pair_rate_scene_bootstrap_95pct'] = rate_ci
            entry['overlap_pair_bootstrap_valid_repetitions'] = min(rate_valid, median_valid)
        else:
            entry.update(overlap_pair_good_minus_bad_median=None,
                         overlap_pair_median_scene_bootstrap_95pct=None,
                         overlap_pair_good_higher_rate_ties_half=None,
                         overlap_pair_rate_scene_bootstrap_95pct=None,
                         overlap_pair_bootstrap_valid_repetitions=0)
        report['metrics'][name] = entry
    return report


def _read_weather(root, weather, arm, metadata, provenance):
    path = root / weather / 'candidate_rows.jsonl'
    provenance[str(path)] = _sha256(path)
    expected = set(map(int, metadata['validation_indices']))
    scene_map = metadata['validation_scene_map']
    by_frame = defaultdict(list)
    counts = defaultdict(int)
    seen = set()
    with path.open(encoding='utf-8') as stream:
        for line_no, line in enumerate(stream, 1):
            raw = json.loads(line)
            if raw['arm'] != arm:
                continue
            sample, cid = int(raw['sample_index']), int(raw['candidate_id'])
            if (raw['weather'] != weather or sample not in expected
                    or (sample, cid) in seen or str(sample) not in scene_map):
                raise ValueError(f'{path}:{line_no}: invalid weather/frame/anchor identity')
            seen.add((sample, cid))
            counts[sample] += 1
            if counts[sample] > 256:
                raise ValueError(f'{path}:{line_no}: over 256 rows in a frame')
            if not raw['within_range'] or int(raw['source_count']) < 2:
                continue
            group = _group(raw)
            if group is None:
                continue
            corners = np.asarray(raw['fused_bev_corners'], dtype=np.float32)
            if corners.shape != (4, 2) or not np.isfinite(corners).all():
                raise ValueError(f'{path}:{line_no}: invalid fused box')
            by_frame[sample].append({'sample_index': sample, 'candidate_id': cid,
                                     'group': group, 'corners': corners,
                                     'metrics': _metrics(raw)})
    if set(counts) != expected:
        raise ValueError(f'{weather}: candidate rows omit a validation frame')
    total = sum(counts.values())
    saved = int(metadata['conditions'][weather]['pools'][arm]['top_256']['total_candidates'])
    if total != saved:
        raise ValueError(f'{weather}: {total} candidate rows versus {saved} in summary')
    return [row for rows in by_frame.values() for row in rows], by_frame


def _markdown(report):
    lines = [
        '# 低分好框 vs 高分坏框：逐指标直接审计', '',
        '只读取已保存的 top256 候选。GT IoU 只用于分组；下列指标均可在推理时计算。',
        '低分好框：原分数≤0.2 且 GT BEV IoU≥0.7；高分坏框：原分数>0.2 且 GT BEV IoU<0.5。',
        '成对比较只使用同帧 BEV IoU 大于原 NMS 阈值的候选对。',
        '“好框更高比例”中相等计 0.5；低于 0.5 也可能说明反向区分，不能事后翻转方向宣称已验证排序器。', '',
    ]
    for weather in WEATHERS:
        item = report['conditions'][weather]
        lines += [f'## {weather}', '',
                  f"低分好框 {item['low_good_candidates']}；高分坏框 "
                  f"{item['high_bad_candidates']}；重叠对 {item['overlapping_pairs']}，"
                  f"来自 {item['pair_scenes']} 个场景。", '',
                  '| 指标 | 低分好框中位数 | 高分坏框中位数 | 全候选 AUC（好框值更高） | '
                  '重叠对好框−坏框中位数 [场景区间] | 重叠对好框更高比例 [场景区间] |',
                  '|---|---:|---:|---:|---:|---:|']
        for name in METRIC_NAMES:
            metric = item['metrics'][name]
            def number(value, digits=4):
                return f'{value:.{digits}f}' if value is not None else '无'
            median_ci = metric['overlap_pair_median_scene_bootstrap_95pct']
            rate_ci = metric['overlap_pair_rate_scene_bootstrap_95pct']
            median = number(metric['overlap_pair_good_minus_bad_median'])
            rate = number(metric['overlap_pair_good_higher_rate_ties_half'], 3)
            if median_ci:
                median += f' [{median_ci[0]:.4f}, {median_ci[1]:.4f}]'
            if rate_ci:
                rate += f' [{rate_ci[0]:.3f}, {rate_ci[1]:.3f}]'
            lines.append(f"| {name} | {number(metric['low_good_median'])} | "
                         f"{number(metric['high_bad_median'])} | "
                         f"{number(metric['unpaired_auc_good_higher'], 3)} | "
                         f'{median} | {rate} |')
        lines.append('')
    lines += [
        '## 判读边界', '',
        '- 全候选 AUC 是描述性结果；两组由原分数阈值定义，尤其不能用 fused_score 的 AUC 证明协同指标有效。',
        '- 重叠对表示可能竞争；第三个候选仍可能在真实 NMS 中压掉两者。本报告不代替已有完整 NMS/AP 回放。',
        '- 场景区间固定已经导出的候选；只有少量独立场景，且逐项比较没有多重检验修正。',
        '- 来源单独预测和逐车移除是推理时代理与条件响应，不能单独证明融合的因果贡献。',
        '- 看区分能力时同时对照 source_agreement，并检查效果是否在 Fog、Rain、Snow 中方向一致。', '',
    ]
    return '\n'.join(lines)


def run(args):
    root = Path(args.input_root).resolve()
    if not (root / 'candidate_audit.json').is_file() and \
            (root / 'extraction' / 'candidate_audit.json').is_file():
        root = root / 'extraction'
    meta_path = root / 'candidate_audit.json'
    metadata = json.loads(meta_path.read_text(encoding='utf-8'))
    if metadata.get('audit_pool') != 'top_256' or metadata.get('stage0_only'):
        raise ValueError('Requires a full top_256 extraction')
    if metadata.get('ablation_arm') not in (args.arm, 'both'):
        raise ValueError('Selected arm has no peer-removal data')
    scene_map = metadata.get('validation_scene_map') or {}
    indices = [int(index) for index in metadata['validation_indices']]
    if set(scene_map) != set(map(str, indices)):
        raise ValueError('Complete scene map required')
    provenance = {str(meta_path): _sha256(meta_path),
                  str(Path(__file__)): _sha256(Path(__file__))}
    report = {
        'protocol': {'input_root': str(root), 'arm': args.arm,
                     'weather_conditions': list(WEATHERS),
                     'low_good': 'score <= .2 and max_gt_iou >= .7',
                     'high_bad': 'score > .2 and max_gt_iou < .5',
                     'within_range_only': True, 'minimum_sources': 2,
                     'bootstrap_unit': 'scene', 'bootstrap_repetitions': args.bootstrap,
                     'seed': args.seed,
                     'metric_families': dict(METRICS)},
        'provenance': provenance, 'conditions': {},
    }
    for offset, weather in enumerate(WEATHERS):
        condition = metadata['conditions'][weather]
        if set(map(int, condition['ablation_frames'])) != set(indices):
            raise ValueError(f'{weather}: peer-removal audit did not cover every frame')
        rows, by_frame = _read_weather(root, weather, args.arm, metadata, provenance)
        pairs = _overlap_pairs(by_frame, float(condition['nms_iou_threshold']))
        report['conditions'][weather] = _summarize(
            rows, pairs, scene_map, args.bootstrap, args.seed + offset * 100)
        report['conditions'][weather]['nms_iou_threshold'] = float(
            condition['nms_iou_threshold'])
        print(f'{weather}: {len(rows)} selected candidates, {len(pairs)} overlap pairs',
              flush=True)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'direct_pair_distortion.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (output / 'direct_pair_distortion.md').write_text(_markdown(report), encoding='utf-8')
    print(f'DIRECT PAIR DISTORTION AUDIT COMPLETE: {output}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', required=True,
                        help='Extraction directory or its parent run directory')
    parser.add_argument('--output-dir', required=True, help='New output directory')
    parser.add_argument('--arm', choices=('F', 'F+D'), default='F')
    parser.add_argument('--bootstrap', type=int, default=500)
    parser.add_argument('--seed', type=int, default=20260928)
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error('--bootstrap must be positive')
    run(args)


if __name__ == '__main__':
    main()
