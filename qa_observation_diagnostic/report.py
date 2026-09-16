"""Offline report for Q-A hypotheses H-A1..H-A6 from collector JSONL files."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .metrics import (binary_auc, distance_bin, json_float, partial_corr, pearson,
                      safe_mean, safe_quantile, spearman)

WEATHERS = ('clean', 'fog', 'rain', 'snow')


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_root(root):
    root = Path(root)
    targets, frames, protocols = {}, {}, {}
    for weather in WEATHERS:
        folder = root / weather
        if not folder.is_dir():
            raise FileNotFoundError(f'Missing weather output: {folder}')
        targets[weather] = read_jsonl(folder / 'targets.jsonl')
        frames[weather] = read_jsonl(folder / 'frames.jsonl')
        protocols[weather] = json.loads((folder / 'protocol.json').read_text(encoding='utf-8'))
    expected = protocols['clean']['sample_indices']
    for weather in WEATHERS[1:]:
        if protocols[weather]['sample_indices'] != expected:
            raise ValueError(f'Frame queue mismatch clean vs {weather}')
    return targets, frames, protocols


def key(row):
    return int(row['sample_index']), int(row['target_index'])


def paired(targets):
    maps = {w: {key(r): r for r in targets[w]} for w in WEATHERS}
    common = set(maps['clean'])
    for w in WEATHERS[1:]:
        common &= set(maps[w])
    if not common:
        raise RuntimeError('No paired target keys across clean/fog/rain/snow')
    max_center_delta = 0.0
    for k in common:
        c = maps['clean'][k]
        for w in WEATHERS[1:]:
            r = maps[w][k]
            max_center_delta = max(max_center_delta,
                math.hypot(c['center_x'] - r['center_x'], c['center_y'] - r['center_y']))
    if max_center_delta > 1e-3:
        raise ValueError(f'Paired GT geometry changed across weather; max center delta {max_center_delta}')
    return maps, sorted(common), max_center_delta


def quantity_audit(rows):
    region = [r['ego']['region_neff'] for r in rows]
    reliable_count = [r['ego']['reliable_count'] for r in rows]
    box_neff = [r['ego']['box_neff'] for r in rows]
    controls = np.asarray([[math.log1p(r['distance']), math.log1p(r['box_area']), r['occlusion_proxy']] for r in rows])
    return dict(
        n=len(rows),
        region_vs_reliable_count_pearson=json_float(pearson(region, reliable_count)),
        region_vs_reliable_count_spearman=json_float(spearman(region, reliable_count)),
        region_vs_reliable_count_partial=json_float(partial_corr(region, reliable_count, controls)),
        region_vs_box_neff_pearson=json_float(pearson(region, box_neff)),
        region_vs_box_neff_spearman=json_float(spearman(region, box_neff)),
        region_vs_box_neff_partial=json_float(partial_corr(region, box_neff, controls)),
    )


def structure_audit(rows):
    if not rows:
        return {}
    detected = np.asarray([int(r['ego_detected']) for r in rows], dtype=np.int64)
    neff = np.asarray([r['ego']['box_neff'] for r in rows], dtype=float)
    coverage = np.asarray([r['ego']['coverage4x4'] for r in rows], dtype=float)
    entropy = np.asarray([r['ego']['entropy4x4'] for r in rows], dtype=float)
    span = np.asarray([r['ego']['span'] for r in rows], dtype=float)
    controls = np.asarray([[math.log1p(r['ego']['box_neff']), math.log1p(r['distance']),
                            math.log1p(r['box_area']), r['occlusion_proxy']] for r in rows], dtype=float)
    q_edges = np.unique(np.quantile(neff, np.linspace(0, 1, 11))) if len(neff) >= 20 else np.unique(neff)
    d_edges = np.asarray([0, 20, 40, 60, 80, 100, np.inf], dtype=float)
    deltas = []
    eligible_bins = 0
    if len(q_edges) >= 2:
        qbin = np.clip(np.digitize(neff, q_edges[1:-1], right=True), 0, max(len(q_edges)-2, 0))
        dbin = np.digitize([r['distance'] for r in rows], d_edges[1:-1], right=False)
        for qb in np.unique(qbin):
            for db in np.unique(dbin):
                m = (qbin == qb) & (dbin == db)
                pos = coverage[m & (detected == 1)]
                neg = coverage[m & (detected == 0)]
                if len(pos) >= 5 and len(neg) >= 5:
                    eligible_bins += 1
                    deltas.append(float(pos.mean() - neg.mean()))
    return dict(
        n=len(rows), detected=int(detected.sum()), missed=int((1-detected).sum()),
        quantity_auc=json_float(binary_auc(neff, detected)),
        coverage_auc=json_float(binary_auc(coverage, detected)),
        entropy_auc=json_float(binary_auc(entropy, detected)),
        span_auc=json_float(binary_auc(span, detected)),
        coverage_partial_corr_detection=json_float(partial_corr(coverage, detected, controls)),
        entropy_partial_corr_detection=json_float(partial_corr(entropy, detected, controls)),
        span_partial_corr_detection=json_float(partial_corr(span, detected, controls)),
        matched_bins=eligible_bins,
        matched_bin_mean_coverage_delta=json_float(safe_mean(deltas)),
        matched_bin_positive_fraction=(sum(x > 0 for x in deltas) / len(deltas)) if deltas else None,
    )


def build_clean_thresholds(clean_rows):
    buckets = defaultdict(list)
    for r in clean_rows:
        if r['ego_detected']:
            buckets[distance_bin(r['distance'])].append(r)
    global_rows = [r for r in clean_rows if r['ego_detected']]
    if not global_rows:
        raise RuntimeError('No clean ego-detected targets for empirical support thresholds')

    def threshold(rows):
        rows = rows or global_rows
        return dict(neff=safe_quantile((x['ego']['box_neff'] for x in rows), .25),
                    coverage=safe_quantile((x['ego']['coverage4x4'] for x in rows), .25),
                    confidence=safe_quantile((x['ego']['confidence_max'] for x in rows), .25),
                    semantic=safe_quantile((x['ego']['semantic_norm_mean'] for x in rows), .25))
    out = {name: threshold(rows) for name, rows in buckets.items()}
    out['global'] = threshold(global_rows)
    return out


def threshold_for(row, thresholds):
    return thresholds.get(distance_bin(row['distance']), thresholds['global'])


def support_strong(stats, th):
    return stats['box_neff'] >= th['neff'] and stats['coverage4x4'] >= th['coverage']


def failure_partition(clean_map, weather_rows, thresholds):
    counts = Counter()
    details = defaultdict(list)
    eligible = 0
    for r in weather_rows:
        c = clean_map.get(key(r))
        if c is None or not c['ego_detected'] or r['ego_detected']:
            continue
        eligible += 1
        th = threshold_for(r, thresholds)
        ego_strong = support_strong(r['ego'], th)
        peer_flags = [support_strong(p, th) for p in r['peers']]
        peer_strong = any(peer_flags)
        peer_conf_strong = any(flag and p['confidence_max'] >= th['confidence']
                               for flag, p in zip(peer_flags, r['peers']))
        if (not ego_strong) and (not peer_strong):
            name = 'all_source_weak'
        elif (not ego_strong) and peer_strong and r['full_detected']:
            name = 'ego_weak_peer_strong_full_recovers'
        elif (not ego_strong) and peer_strong and (not r['full_detected']):
            name = 'ego_weak_peer_strong_full_still_misses'
        elif ego_strong and (not r['full_detected']):
            name = 'ego_strong_full_miss_local_or_downstream'
        elif ego_strong and r['full_detected']:
            name = 'ego_strong_full_recovers'
        else:
            name = 'mixed_other'
        counts[name] += 1
        details[name].append(dict(peer_conf_strong=peer_conf_strong,
            ego_conf_strong=r['ego']['confidence_max'] >= th['confidence'],
            ego_semantic_strong=r['ego']['semantic_norm_mean'] >= th['semantic']))
    result = dict(eligible_weather_induced_ego_misses=eligible, counts=dict(counts), fractions={})
    for name, count in counts.items():
        result['fractions'][name] = count / eligible if eligible else None
    fusion_rows = details.get('ego_weak_peer_strong_full_still_misses', [])
    result['fusion_candidate_peer_conf_strong'] = sum(x['peer_conf_strong'] for x in fusion_rows)
    result['fusion_candidate_peer_conf_strong_fraction'] = (
        result['fusion_candidate_peer_conf_strong'] / len(fusion_rows) if fusion_rows else None)
    local_rows = details.get('ego_strong_full_miss_local_or_downstream', []) + details.get('ego_strong_full_recovers', [])
    result['ego_strong_weather_miss'] = len(local_rows)
    result['ego_strong_weather_miss_fraction'] = len(local_rows) / eligible if eligible else None
    result['local_candidate_conf_strong'] = sum(x['ego_conf_strong'] for x in local_rows)
    result['local_candidate_conf_strong_fraction'] = (
        result['local_candidate_conf_strong'] / len(local_rows) if local_rows else None)
    result['local_candidate_semantic_strong'] = sum(x['ego_semantic_strong'] for x in local_rows)
    result['local_candidate_semantic_strong_fraction'] = (
        result['local_candidate_semantic_strong'] / len(local_rows) if local_rows else None)
    return result


def occlusion_audit(clean_map, weather_rows):
    eligible = []
    for r in weather_rows:
        c = clean_map.get(key(r))
        if c is not None and c['ego_detected']:
            eligible.append((c, r))
    if not eligible:
        return {}
    high = [(c, r) for c, r in eligible if c['occlusion_proxy'] >= .25]
    low = [(c, r) for c, r in eligible if c['occlusion_proxy'] < .25]

    def stats(rows):
        if not rows:
            return dict(n=0, weather_miss_rate=None, mean_relative_neff_loss=None,
                        mean_relative_raw_loss=None)
        miss = [not r['ego_detected'] for _, r in rows]
        neff_loss = [(c['ego']['box_neff'] - r['ego']['box_neff']) / max(c['ego']['box_neff'], 1e-6)
                     for c, r in rows]
        raw_loss = [(c['ego']['raw_count'] - r['ego']['raw_count']) / max(c['ego']['raw_count'], 1)
                    for c, r in rows]
        return dict(n=len(rows), weather_miss_rate=float(np.mean(miss)),
                    mean_relative_neff_loss=safe_mean(neff_loss), mean_relative_raw_loss=safe_mean(raw_loss))
    hs, ls = stats(high), stats(low)
    risk_ratio = None
    if hs['weather_miss_rate'] is not None and ls['weather_miss_rate'] not in (None, 0):
        risk_ratio = hs['weather_miss_rate'] / ls['weather_miss_rate']
    return dict(high_occlusion=hs, low_occlusion=ls, miss_risk_ratio_high_vs_low=risk_ratio,
                threshold=.25)


def visibility_audit(frame_rows, target_rows):
    totals = {group: Counter() for group in ('background_zero', 'target_zero')}
    for f in frame_rows:
        for group in totals:
            totals[group].update(f['background'][group])
    out = {}
    for group, c in totals.items():
        n = c['count']
        out[group] = dict(c)
        out[group]['traversal_positive_fraction'] = c['traversal_positive'] / n if n else None
        out[group]['traversal_ge_050_fraction'] = c['traversal_ge_050'] / n if n else None
        out[group]['mean_traversal'] = c['traversal_sum'] / n if n else None
    low_known = [r for r in target_rows if r['ego']['known_fraction'] < .25]
    out['low_known_targets'] = dict(
        n=len(low_known),
        traversal_ge_050=sum(r['ego']['traversal_mean'] >= .5 for r in low_known),
        traversal_ge_050_fraction=(sum(r['ego']['traversal_mean'] >= .5 for r in low_known) / len(low_known)) if low_known else None,
        mean_traversal=safe_mean(r['ego']['traversal_mean'] for r in low_known),
    )
    bg = out['background_zero'].get('traversal_ge_050_fraction')
    tgt = out['target_zero'].get('traversal_ge_050_fraction')
    out['background_minus_target_high_traversal'] = (bg - tgt) if bg is not None and tgt is not None else None
    return out


def verdicts(results):
    q = results['E-A0_quantity_redundancy']
    stable_redundant = all(
        q[w].get('region_vs_reliable_count_pearson') is not None
        and q[w].get('region_vs_reliable_count_spearman') is not None
        and q[w]['region_vs_reliable_count_pearson'] > .85
        and q[w]['region_vs_reliable_count_spearman'] > .85 for w in WEATHERS)

    structure = results['E-A1_quantity_matched_structure']
    structural_support = sum(
        (structure[w].get('coverage_partial_corr_detection') or 0) > .10
        and (structure[w].get('matched_bin_positive_fraction') or 0) >= .65
        for w in WEATHERS[1:])

    partitions = results['E-A2_failure_stage_partition']
    fusion_frac = safe_mean((partitions[w]['fractions'].get('ego_weak_peer_strong_full_still_misses', 0.0)
                             for w in WEATHERS[1:]))
    recover_frac = safe_mean((partitions[w]['fractions'].get('ego_weak_peer_strong_full_recovers', 0.0)
                              for w in WEATHERS[1:]))
    local_frac = safe_mean((partitions[w].get('ego_strong_weather_miss_fraction', 0.0)
                            for w in WEATHERS[1:]))

    occ = results['E-A3_occlusion_weather']
    occ_support = sum((occ[w].get('miss_risk_ratio_high_vs_low') or 0) >= 1.5 for w in WEATHERS[1:])

    vis = results['E-A4_approx_visibility']
    vis_gaps = [vis[w].get('background_minus_target_high_traversal') for w in WEATHERS]
    vis_support = sum(v is not None and v >= .10 for v in vis_gaps)

    return {
        'H-A1_quantity_dominant': ('quantity proxies are redundant; quantity family retained as one hypothesis'
            if stable_redundant else 'quantity proxies are not uniformly redundant; inspect weather/definition before simplifying'),
        'H-A2_structure_dominant': ('independent structural signal observed in >=2 adverse weathers'
            if structural_support >= 2 else 'no stable first-round independent structural signal; weaken or keep unresolved'),
        'H-A3_visibility_free_space': ('approximate traversal separates zero-support background from target cells in multiple conditions'
            if vis_support >= 2 else 'approximate traversal not yet a reliable free-space discriminator; do not trigger precise replay yet'),
        'H-A4_occlusion_x_weather': ('high-occlusion targets show >=1.5x weather-induced miss risk in >=2 adverse weathers'
            if occ_support >= 2 else 'interaction not stable enough in first-round proxy; weaken or refine occlusion label'),
        'H-A5_local_downstream_bottleneck': (f'weather-induced ego-miss fraction with empirically strong local evidence: {local_frac:.3f}'
            if math.isfinite(local_frac) else 'insufficient eligible cases'),
        'H-A6_fusion_bottleneck': (f'ego-weak/peer-strong/full-still-miss mean fraction {fusion_frac:.3f}; '
                                  f'full-recover mean fraction {recover_frac:.3f}'
            if math.isfinite(fusion_frac) and math.isfinite(recover_frac) else 'insufficient eligible cases'),
        'gating_note': 'These are development diagnostics, not paper claims. Threshold-based partitions are empirical labels, not causal proof.'
    }


def markdown(results):
    lines = [
        '# Q-A observation insufficiency diagnostic report', '',
        '> Development OPV2V validation + online weather only. No OPV2V-W test data are used for hypothesis selection.', '',
        '## Executive summary', '',
    ]
    for name, text in results['verdicts'].items():
        lines.append(f'- **{name}**: {text}')
    lines += ['', '## E-A0 — Quantity proxy redundancy', '',
              '| Weather | Pearson(region N_eff, reliable count) | Spearman | Partial corr | Pearson(region N_eff, box N_eff) |',
              '|---|---:|---:|---:|---:|']
    for w in WEATHERS:
        r = results['E-A0_quantity_redundancy'][w]
        lines.append(f"| {w} | {r['region_vs_reliable_count_pearson']} | {r['region_vs_reliable_count_spearman']} | "
                     f"{r['region_vs_reliable_count_partial']} | {r['region_vs_box_neff_pearson']} |")
    lines += ['', '## E-A1 — Quantity-matched structural signal', '',
              '| Weather | Quantity AUC | Coverage AUC | Coverage partial corr | Matched bins | Positive-bin fraction |',
              '|---|---:|---:|---:|---:|---:|']
    for w in WEATHERS:
        r = results['E-A1_quantity_matched_structure'][w]
        lines.append(f"| {w} | {r.get('quantity_auc')} | {r.get('coverage_auc')} | "
                     f"{r.get('coverage_partial_corr_detection')} | {r.get('matched_bins')} | {r.get('matched_bin_positive_fraction')} |")
    lines += ['', '## E-A2 — Failure-stage partition', '']
    for w in WEATHERS[1:]:
        r = results['E-A2_failure_stage_partition'][w]
        lines += [f'### {w}', '', f"Eligible weather-induced ego misses: **{r['eligible_weather_induced_ego_misses']}**", '',
                  '```json', json.dumps(r, ensure_ascii=False, indent=2), '```', '']
    lines += ['## E-A3 — Occlusion × weather interaction', '',
              '| Weather | Low-occ miss rate | High-occ miss rate | Risk ratio | Low-occ N_eff loss | High-occ N_eff loss |',
              '|---|---:|---:|---:|---:|---:|']
    for w in WEATHERS[1:]:
        r = results['E-A3_occlusion_weather'][w]
        lo, hi = r.get('low_occlusion', {}), r.get('high_occlusion', {})
        lines.append(f"| {w} | {lo.get('weather_miss_rate')} | {hi.get('weather_miss_rate')} | "
                     f"{r.get('miss_risk_ratio_high_vs_low')} | {lo.get('mean_relative_neff_loss')} | {hi.get('mean_relative_neff_loss')} |")
    lines += ['', '## E-A4 — Approximate visibility/free-space gate', '']
    for w in WEATHERS:
        r = results['E-A4_approx_visibility'][w]
        lines += [f'### {w}', '', '```json', json.dumps(r, ensure_ascii=False, indent=2), '```', '']
    lines += ['## Interpretation boundaries', '',
              '- `N_eff` and reliability statistics use **retained voxel slots**, not all raw LiDAR returns.',
              '- E-A2 strong/weak labels use clean-detected Q25 thresholds within distance bins; they are diagnostic labels, not deployable rules.',
              '- E-A3 occlusion is a **GT angular-overlap proxy**, not a perfect CARLA occlusion annotation.',
              '- E-A4 uses the existing angular-depth traversal proxy. It is **not occupancy probability or real-sensor free-space truth**.',
              '- `ego weak + peer strong + full still misses` is a fusion-bottleneck candidate, not proof that attention is the unique cause.',
              '- Do not use this report to tune on OPV2V-W test. If a hypothesis survives, design Oracle/Cheap-Gate experiments next.', '']
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, help='Run root containing clean/fog/rain/snow collector folders')
    args = p.parse_args()
    root = Path(args.root)
    targets, frames, protocols = load_root(root)
    maps, common, max_delta = paired(targets)
    paired_targets = {w: [maps[w][k] for k in common] for w in WEATHERS}
    clean_map = maps['clean']
    thresholds = build_clean_thresholds(paired_targets['clean'])

    result = dict(
        schema=1, paired_targets=len(common), max_gt_center_delta=max_delta,
        clean_support_thresholds=thresholds,
        E_A0_quantity_redundancy={w: quantity_audit(paired_targets[w]) for w in WEATHERS},
        E_A1_quantity_matched_structure={w: structure_audit(paired_targets[w]) for w in WEATHERS},
        E_A2_failure_stage_partition={w: failure_partition(clean_map, paired_targets[w], thresholds) for w in WEATHERS[1:]},
        E_A3_occlusion_weather={w: occlusion_audit(clean_map, paired_targets[w]) for w in WEATHERS[1:]},
        E_A4_approx_visibility={w: visibility_audit(frames[w], paired_targets[w]) for w in WEATHERS},
        protocol={w: dict(data_root=protocols[w]['data_root'], online_weather=protocols[w]['online_weather'],
                          frontend_sha256=protocols[w]['frontend_sha256']) for w in WEATHERS},
    )
    result['E-A0_quantity_redundancy'] = result.pop('E_A0_quantity_redundancy')
    result['E-A1_quantity_matched_structure'] = result.pop('E_A1_quantity_matched_structure')
    result['E-A2_failure_stage_partition'] = result.pop('E_A2_failure_stage_partition')
    result['E-A3_occlusion_weather'] = result.pop('E_A3_occlusion_weather')
    result['E-A4_approx_visibility'] = result.pop('E_A4_approx_visibility')
    result['verdicts'] = verdicts(result)
    (root / 'hypothesis_results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (root / 'hypothesis_report.md').write_text(markdown(result) + '\n', encoding='utf-8')
    print(f'Wrote {root / "hypothesis_report.md"}')
    print(f'Wrote {root / "hypothesis_results.json"}')


if __name__ == '__main__':
    main()
