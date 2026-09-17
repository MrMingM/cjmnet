"""Pure-CPU bookkeeping for the H-A5 local evidence audit."""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

QUANTILE_LABELS = ("q20", "q25", "q30", "q40")
LATE_FAILURES = {
    "score_filtered",
    "geometry_filtered",
    "nms_topk_filtered",
    "nms_suppressed",
    "range_filtered",
    "matching_competition",
    "other",
}


def classify(row):
    """Return a conservative descriptive A5 category, never a root-cause claim."""
    failure = row["weather_path"]["failure_stage"]
    same = row["clean_matched_anchor_in_weather"]
    if row["weather_path"]["matched"]:
        raise ValueError("H-A5 candidate unexpectedly detected in weather ego branch")
    if failure != "no_iou70_after_decode":
        return "late_detection_path_loss"
    if bool(same["score_passes_threshold"]):
        return "regression_or_localization_loss_with_cls_survival"
    return "joint_head_or_upstream_unresolved"


def summarize_rows(rows):
    rows = list(rows)
    counts = Counter(classify(r) for r in rows)
    failures = Counter(r["weather_path"]["failure_stage"] for r in rows)
    out = {
        "occurrences": len(rows),
        "unique_frames": len({int(r["sample_index"]) for r in rows}),
        "classification_counts": dict(counts),
        "failure_stage_counts": dict(failures),
        "same_anchor_geometry_survives_n": sum(
            bool(r["clean_matched_anchor_in_weather"]["iou70"]) for r in rows
        ),
        "same_anchor_cls_survives_n": sum(
            bool(r["clean_matched_anchor_in_weather"]["score_passes_threshold"]) for r in rows
        ),
        "any_decoded_iou70_n": sum(
            r["weather_path"]["stages"]["decoded"]["qualifying_count"] > 0 for r in rows
        ),
    }
    for q in QUANTILE_LABELS:
        subset = [r for r in rows if r["proxy_strong"][q]]
        denom = len(subset)
        sub_counts = Counter(classify(r) for r in subset)
        out[q] = {
            "strong_occurrences": denom,
            "strong_unique_frames": len({int(r["sample_index"]) for r in subset}),
            "classification_counts": dict(sub_counts),
            "late_detection_path_loss_fraction": (
                sub_counts["late_detection_path_loss"] / denom if denom else None
            ),
            "a5b_support_fraction": (
                (sub_counts["late_detection_path_loss"]
                 + sub_counts["regression_or_localization_loss_with_cls_survival"]) / denom
                if denom else None
            ),
            "upstream_unresolved_fraction": (
                sub_counts["joint_head_or_upstream_unresolved"] / denom if denom else None
            ),
        }
    by_scene = defaultdict(list)
    for r in rows:
        by_scene[str(r["scene"])].append(r)
    out["by_scene"] = {
        scene: {
            "occurrences": len(items),
            "q25_strong": sum(bool(x["proxy_strong"]["q25"]) for x in items),
            "a5b_support_q25": sum(
                bool(x["proxy_strong"]["q25"])
                and classify(x) != "joint_head_or_upstream_unresolved"
                for x in items
            ),
        }
        for scene, items in sorted(by_scene.items(), key=lambda kv: int(kv[0]))
    }
    same = [r["clean_matched_anchor_in_weather"] for r in rows]
    if same:
        out["same_anchor_delta"] = {
            "score_delta_median": float(np.median([x["score_delta"] for x in same])),
            "iou_delta_median": float(np.median([x["iou_delta"] for x in same])),
            "logit_delta_median": float(np.median([x["logit_delta"] for x in same])),
        }
    else:
        out["same_anchor_delta"] = None
    return out
