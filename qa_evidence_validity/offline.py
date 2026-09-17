"""Stage-2 offline evidence-validity audit from the completed Q-A Stage-1 logs.

No GPU/model inference is used here. The goal is to test whether empirical
"strong evidence" labels are stable, repair the Snow near-zero denominator
artifact, stratify the candidates, and count ego->full degradation cases.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from qa_observation_diagnostic.metrics import distance_bin, safe_quantile

WEATHERS = ("clean", "fog", "rain", "snow")
ADVERSE = WEATHERS[1:]
QUANTILES = (0.20, 0.25, 0.30, 0.40)


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def key(row):
    return int(row["sample_index"]), int(row["target_index"])


def load_stage1(root):
    root = Path(root)
    targets, protocols = {}, {}
    for weather in WEATHERS:
        folder = root / weather
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing Stage-1 weather folder: {folder}")
        targets[weather] = read_jsonl(folder / "targets.jsonl")
        protocols[weather] = json.loads((folder / "protocol.json").read_text(encoding="utf-8"))
    expected = protocols["clean"]["sample_indices"]
    for weather in ADVERSE:
        if protocols[weather]["sample_indices"] != expected:
            raise ValueError(f"Stage-1 frame queue differs for clean/{weather}")
        if not protocols[weather].get("development_only", False):
            raise ValueError("Evidence-validity audit must use development logs only")
        if protocols[weather].get("test_data_used", True):
            raise ValueError("Stage-1 protocol says test data were used; refuse hypothesis audit")
    maps = {w: {key(r): r for r in targets[w]} for w in WEATHERS}
    common = set(maps["clean"])
    for weather in ADVERSE:
        common &= set(maps[weather])
    if not common:
        raise RuntimeError("No paired Stage-1 targets")
    return targets, maps, sorted(common), protocols


def build_thresholds(clean_rows, quantile):
    buckets = defaultdict(list)
    eligible = [r for r in clean_rows if r["ego_detected"]]
    if not eligible:
        raise RuntimeError("No clean ego-detected targets for empirical thresholds")
    for r in eligible:
        buckets[distance_bin(r["distance"])].append(r)

    def threshold(rows):
        rows = rows or eligible
        return {
            "neff": safe_quantile((x["ego"]["box_neff"] for x in rows), quantile),
            "coverage": safe_quantile((x["ego"]["coverage4x4"] for x in rows), quantile),
            "confidence": safe_quantile((x["ego"]["confidence_max"] for x in rows), quantile),
            "semantic": safe_quantile((x["ego"]["semantic_norm_mean"] for x in rows), quantile),
        }

    out = {name: threshold(rows) for name, rows in buckets.items()}
    out["global"] = threshold(eligible)
    return out


def threshold_for(row, thresholds):
    return thresholds.get(distance_bin(row["distance"]), thresholds["global"])


def support_strong(stats, threshold):
    return stats["box_neff"] >= threshold["neff"] and stats["coverage4x4"] >= threshold["coverage"]


def eligible_weather_misses(clean_map, weather_rows):
    return [r for r in weather_rows
            if key(r) in clean_map and clean_map[key(r)]["ego_detected"] and not r["ego_detected"]]


def jaccard(a, b):
    a, b = set(a), set(b)
    union = a | b
    return len(a & b) / len(union) if union else None


def partition_for_quantile(clean_map, weather_rows, thresholds):
    counts = Counter()
    sets = defaultdict(set)
    eligible = eligible_weather_misses(clean_map, weather_rows)
    for r in eligible:
        k = key(r)
        th = threshold_for(r, thresholds)
        ego_strong = support_strong(r["ego"], th)
        peer_strong = any(support_strong(p, th) for p in r["peers"])
        if ego_strong:
            counts["ego_strong"] += 1
            sets["ego_strong"].add(k)
        if peer_strong:
            counts["any_peer_strong"] += 1
            sets["any_peer_strong"].add(k)
        if (not ego_strong) and peer_strong and r["full_detected"]:
            counts["ego_weak_peer_strong_full_recovers"] += 1
            sets["ego_weak_peer_strong_full_recovers"].add(k)
        if (not ego_strong) and peer_strong and (not r["full_detected"]):
            counts["ego_weak_peer_strong_full_still_miss"] += 1
            sets["ego_weak_peer_strong_full_still_miss"].add(k)
        if (not ego_strong) and (not peer_strong):
            counts["all_source_weak"] += 1
            sets["all_source_weak"].add(k)
    n = len(eligible)
    return {
        "eligible": n,
        "counts": dict(counts),
        "fractions": {name: count / n if n else None for name, count in counts.items()},
        "sets": {name: sorted([list(x) for x in values]) for name, values in sets.items()},
    }


def threshold_sensitivity(targets, maps):
    thresholds = {q: build_thresholds(targets["clean"], q) for q in QUANTILES}
    result = {}
    for weather in ADVERSE:
        by_q = {}
        raw_sets = {}
        for q in QUANTILES:
            p = partition_for_quantile(maps["clean"], targets[weather], thresholds[q])
            raw_sets[q] = {name: {tuple(x) for x in values} for name, values in p.pop("sets").items()}
            by_q[f"q{int(q * 100):02d}"] = p
        ref = raw_sets[0.25]
        stability = {}
        for name in ("ego_strong", "any_peer_strong", "ego_weak_peer_strong_full_still_miss"):
            stability[name] = {
                f"q{int(q * 100):02d}_vs_q25_jaccard": jaccard(raw_sets[q].get(name, set()), ref.get(name, set()))
                for q in QUANTILES
            }
        ego_fracs = [by_q[f"q{int(q*100):02d}"]["fractions"].get("ego_strong", 0.0) for q in QUANTILES]
        peer_fracs = [by_q[f"q{int(q*100):02d}"]["fractions"].get("any_peer_strong", 0.0) for q in QUANTILES]
        result[weather] = {
            "by_quantile": by_q,
            "jaccard_vs_q25": stability,
            "ego_strong_fraction_range": max(ego_fracs) - min(ego_fracs),
            "peer_strong_fraction_range": max(peer_fracs) - min(peer_fracs),
        }
    return result, thresholds


def _group_neff_change(pairs):
    eps = 1e-6
    clean = np.asarray([c["ego"]["box_neff"] for c, _ in pairs], dtype=np.float64)
    weather = np.asarray([r["ego"]["box_neff"] for _, r in pairs], dtype=np.float64)
    if not len(clean):
        return {"n": 0}
    zero = clean <= eps
    positive = ~zero
    absolute = clean - weather
    bounded = (clean - weather) / (clean + weather + eps)
    log_delta = np.log1p(clean) - np.log1p(weather)
    relative = (clean[positive] - weather[positive]) / clean[positive] if positive.any() else np.asarray([])
    return {
        "n": int(len(clean)),
        "clean_neff_zero_n": int(zero.sum()),
        "clean_neff_zero_fraction": float(zero.mean()),
        "absolute_delta_mean": float(absolute.mean()),
        "absolute_delta_median": float(np.median(absolute)),
        "bounded_symmetric_delta_mean": float(bounded.mean()),
        "bounded_symmetric_delta_median": float(np.median(bounded)),
        "log1p_delta_mean": float(log_delta.mean()),
        "log1p_delta_median": float(np.median(log_delta)),
        "relative_loss_clean_positive_mean": float(relative.mean()) if len(relative) else None,
        "relative_loss_clean_positive_median": float(np.median(relative)) if len(relative) else None,
        "relative_loss_denominator_note": "Relative loss excludes clean box_neff<=1e-6; zero-clean cases are reported separately.",
    }


def corrected_neff_changes(targets, maps):
    result = {}
    for weather in ADVERSE:
        pairs = []
        for r in targets[weather]:
            c = maps["clean"].get(key(r))
            if c is not None and c["ego_detected"]:
                pairs.append((c, r))
        low = [(c, r) for c, r in pairs if c["occlusion_proxy"] < 0.25]
        high = [(c, r) for c, r in pairs if c["occlusion_proxy"] >= 0.25]
        result[weather] = {
            "all_clean_detected": _group_neff_change(pairs),
            "low_occlusion": _group_neff_change(low),
            "high_occlusion": _group_neff_change(high),
            "occlusion_threshold": 0.25,
        }
    return result


def _top_share(counter, k=3):
    values = sorted(counter.values(), reverse=True)
    total = sum(values)
    return sum(values[:k]) / total if total else None


def stratified_candidates(targets, maps, q25_thresholds):
    result = {}
    for weather in ADVERSE:
        eligible = eligible_weather_misses(maps["clean"], targets[weather])
        by_scene = defaultdict(lambda: Counter())
        by_distance = defaultdict(lambda: Counter())
        global_counts = Counter()
        for r in eligible:
            th = threshold_for(r, q25_thresholds)
            ego_strong = support_strong(r["ego"], th)
            peer_strong = any(support_strong(p, th) for p in r["peers"])
            labels = ["eligible"]
            if ego_strong:
                labels.append("ego_strong")
            if peer_strong:
                labels.append("any_peer_strong")
            if (not ego_strong) and peer_strong and not r["full_detected"]:
                labels.append("fusion_candidate_metric")
            if (not ego_strong) and peer_strong and r["full_detected"]:
                labels.append("metric_peer_full_recovers")
            for name in labels:
                global_counts[name] += 1
                by_scene[str(r["scene"])][name] += 1
                by_distance[distance_bin(r["distance"])][name] += 1

        scene_fusion = Counter({s: c["fusion_candidate_metric"] for s, c in by_scene.items()
                                if c["fusion_candidate_metric"]})
        scene_local = Counter({s: c["ego_strong"] for s, c in by_scene.items() if c["ego_strong"]})
        result[weather] = {
            "global": dict(global_counts),
            "scene": {s: dict(c) for s, c in sorted(by_scene.items(), key=lambda x: int(x[0]))},
            "distance": {d: dict(c) for d, c in by_distance.items()},
            "top3_scene_share_ego_strong": _top_share(scene_local, 3),
            "top3_scene_share_metric_fusion_candidates": _top_share(scene_fusion, 3),
        }
    return result


def ego_full_transitions(targets):
    result = {}
    for weather in WEATHERS:
        counts = Counter()
        scene_degrade = Counter()
        for r in targets[weather]:
            e, f = bool(r["ego_detected"]), bool(r["full_detected"])
            if e and f:
                name = "ego_yes_full_yes"
            elif e and not f:
                name = "ego_yes_full_no"
                scene_degrade[str(r["scene"])] += 1
            elif (not e) and f:
                name = "ego_no_full_yes"
            else:
                name = "ego_no_full_no"
            counts[name] += 1
        ego_yes = counts["ego_yes_full_yes"] + counts["ego_yes_full_no"]
        ego_no = counts["ego_no_full_yes"] + counts["ego_no_full_no"]
        result[weather] = {
            "counts": dict(counts),
            "ego_to_full_degradation_fraction_given_ego_detected":
                counts["ego_yes_full_no"] / ego_yes if ego_yes else None,
            "full_recovery_fraction_given_ego_miss":
                counts["ego_no_full_yes"] / ego_no if ego_no else None,
            "top3_scene_share_ego_to_full_degradation": _top_share(scene_degrade, 3),
            "ego_to_full_degradation_by_scene": dict(scene_degrade),
        }
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1-root", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    targets, maps, common, protocols = load_stage1(args.stage1_root)
    sensitivity, thresholds = threshold_sensitivity(targets, maps)
    result = {
        "schema": 1,
        "purpose": "Stage-2 evidence validity audit; offline portion",
        "stage1_root": str(Path(args.stage1_root).resolve()),
        "paired_target_keys": len(common),
        "threshold_quantiles": list(QUANTILES),
        "threshold_sensitivity": sensitivity,
        "corrected_neff_change": corrected_neff_changes(targets, maps),
        "stratified_q25": stratified_candidates(targets, maps, thresholds[0.25]),
        "ego_full_transitions": ego_full_transitions(targets),
        "boundaries": [
            "Strong/weak labels are empirical clean-detected quantile labels, not causal truth.",
            "bounded_symmetric_delta is a new bounded change metric, not the old relative-loss rate.",
            "clean box_neff<=1e-6 cases are reported separately rather than hidden by an epsilon denominator.",
            "Scene/distance stratification tests concentration; it does not by itself establish mechanism.",
        ],
    }
    (out / "offline_results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                                               encoding="utf-8")
    print(f"Wrote {out/'offline_results.json'}")


if __name__ == "__main__":
    main()
