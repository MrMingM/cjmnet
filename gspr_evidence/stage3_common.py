"""Shared pure-Python helpers for Stage-3 H-A6 diagnostics."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ADVERSE = ("fog", "rain", "snow")


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


def load_stage1_map(stage1_root, weather):
    rows = read_jsonl(Path(stage1_root) / weather / "targets.jsonl")
    return {key(r): r for r in rows}


def load_stage2_rows(stage2_root, weather):
    path = Path(stage2_root) / "peer" / weather / "peer_targets.jsonl"
    rows = read_jsonl(path)
    if not rows:
        raise RuntimeError(f"No Stage-2 peer rows: {path}")
    return rows


def source_valid_full_miss(row):
    return bool(row.get("any_peer_alone_detected")) and not bool(row["full"]["matched"])


def candidate_rows(stage2_root, weather):
    return [r for r in load_stage2_rows(stage2_root, weather) if source_valid_full_miss(r)]


def failure_stage(stage_flags):
    """Locate where IoU>=0.7 support first disappears after decoding.

    This is a disappearance location, not a causal root-cause label.
    """
    ordered = (
        ("decoded", "no_iou70_decoded"),
        ("score", "score_threshold"),
        ("geometry", "geometry_sanity"),
        ("nms_top1000", "nms_top1000"),
        ("nms", "nms_suppression"),
        ("range", "range_filter"),
    )
    previous = True
    for name, label in ordered:
        current = bool(stage_flags.get(name, False))
        if previous and not current:
            return label
        previous = current
    if bool(stage_flags.get("range", False)) and not bool(stage_flags.get("final_matched", False)):
        return "final_matching_competition"
    if bool(stage_flags.get("final_matched", False)):
        return "final_detected"
    return "unclassified"


def rate(num, den):
    return num / den if den else None


def scene_failure_table(stage1_map, stage2_rows):
    """Per-scene denominator-aware source-valid/full-miss statistics."""
    by_scene = defaultdict(lambda: {
        "source_valid_target_occurrences": 0,
        "failure_target_occurrences": 0,
        "source_valid_frames": set(),
        "failure_frames": set(),
    })
    by_distance = defaultdict(lambda: {
        "source_valid_target_occurrences": 0,
        "failure_target_occurrences": 0,
        "source_valid_frames": set(),
        "failure_frames": set(),
    })
    from qa_observation_diagnostic.metrics import distance_bin

    total_source_valid = total_failure = 0
    source_valid_frames, failure_frames = set(), set()
    for audit in stage2_rows:
        k = key(audit)
        stage = stage1_map.get(k)
        if stage is None:
            raise KeyError(f"Stage-2 target missing from Stage-1: {k}")
        if not bool(audit.get("any_peer_alone_detected")):
            continue
        frame = int(audit["sample_index"])
        scene = str(stage["scene"])
        dbin = distance_bin(float(stage["distance"]))
        failed = not bool(audit["full"]["matched"])

        total_source_valid += 1
        source_valid_frames.add(frame)
        for bucket in (by_scene[scene], by_distance[dbin]):
            bucket["source_valid_target_occurrences"] += 1
            bucket["source_valid_frames"].add(frame)
        if failed:
            total_failure += 1
            failure_frames.add(frame)
            for bucket in (by_scene[scene], by_distance[dbin]):
                bucket["failure_target_occurrences"] += 1
                bucket["failure_frames"].add(frame)

    def finalize(mapping):
        out = {}
        for name, v in sorted(mapping.items()):
            den = int(v["source_valid_target_occurrences"])
            num = int(v["failure_target_occurrences"])
            sf = len(v["source_valid_frames"])
            ff = len(v["failure_frames"])
            out[name] = {
                "source_valid_target_occurrences": den,
                "failure_target_occurrences": num,
                "failure_rate_given_source_valid": rate(num, den),
                "source_valid_unique_frames": sf,
                "failure_unique_frames": ff,
                "failure_frame_share_among_source_valid_frames": rate(ff, sf),
            }
        return out

    return {
        "overall": {
            "source_valid_target_occurrences": total_source_valid,
            "failure_target_occurrences": total_failure,
            "failure_rate_given_source_valid": rate(total_failure, total_source_valid),
            "source_valid_unique_frames": len(source_valid_frames),
            "failure_unique_frames": len(failure_frames),
            "failure_frame_share_among_source_valid_frames":
                rate(len(failure_frames), len(source_valid_frames)),
        },
        "by_scene": finalize(by_scene),
        "by_distance": finalize(by_distance),
    }


def choose_frame_oracles(subset_results, candidate_targets, full_matched, full_fp):
    """Summarize target-wise opportunity and realizable one-subset frame Oracles."""
    candidate_targets = set(candidate_targets)
    full_matched = set(full_matched)
    rows = []
    target_recoverable = {int(t): False for t in candidate_targets}

    for result in subset_results:
        matched = set(result["matched"])
        recovered = sorted(candidate_targets & matched)
        for t in recovered:
            target_recoverable[int(t)] = True
        lost = sorted(full_matched - matched)
        gained = sorted(matched - full_matched)
        fp = int(result["fp"])
        rows.append({
            "subset": list(result["subset"]),
            "matched_count": len(matched),
            "recovered_candidate_targets": recovered,
            "recovered_candidate_count": len(recovered),
            "lost_full_targets": lost,
            "lost_full_count": len(lost),
            "gained_vs_full_targets": gained,
            "gained_vs_full_count": len(gained),
            "fp": fp,
            "fp_delta_vs_full": fp - int(full_fp),
        })

    best = min(
        rows,
        key=lambda r: (
            -r["recovered_candidate_count"],
            r["lost_full_count"],
            max(r["fp_delta_vs_full"], 0),
            -r["gained_vs_full_count"],
            len(r["subset"]),
            tuple(r["subset"]),
        ),
    ) if rows else None

    safe_rows = [
        r for r in rows
        if r["lost_full_count"] == 0 and r["fp_delta_vs_full"] <= 0
    ]
    safe_best = min(
        safe_rows,
        key=lambda r: (
            -r["recovered_candidate_count"],
            -r["gained_vs_full_count"],
            len(r["subset"]),
            tuple(r["subset"]),
        ),
    ) if safe_rows else None

    return {
        "target_wise_recoverable": target_recoverable,
        "target_wise_recoverable_count": sum(target_recoverable.values()),
        "candidate_target_count": len(candidate_targets),
        "best_frame_subset": best,
        "safe_best_frame_subset": safe_best,
        "all_subsets": rows,
    }
