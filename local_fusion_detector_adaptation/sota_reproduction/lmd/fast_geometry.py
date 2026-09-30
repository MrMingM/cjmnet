"""Exact-score replay helpers with per-frame cached candidate/GT BEV IoU.

The evaluation order and greedy one-GT-per-detection matching are the same as
OpenCOOD's eval_utils.caluclate_tp_fp. This is an optimization, not an alternative
metric or a geometry repair.
"""
from __future__ import annotations

import numpy as np

def candidate_gt_iou(rows, gt, weather="", sample_index=-1):
    n, m = len(rows), len(gt)
    result = np.zeros((n, m), dtype=np.float32)
    if not n or not m:
        return result
    from gspr_evidence.stage3_trace import polygon_ious

    corners = np.asarray(
        [row["corners"] for row in rows], dtype=np.float32).reshape(n, 4, 2)
    try:
        result = polygon_ious(corners, np.asarray(gt, dtype=np.float32))
    except Exception as exc:
        failing_ids = []
        for i in range(n):
            try:
                polygon_ious(corners[i:i+1], np.asarray(gt, dtype=np.float32))
            except Exception:
                failing_ids.append(int(rows[i]["candidate_id"]))
                if len(failing_ids) == 5:
                    break
        raise ValueError(
            f"{weather}/{sample_index}: invalid candidate/GT polygon IoU; "
            f"candidate_ids={failing_ids}") from exc
    if result.shape != (n, m) or not np.isfinite(result).all():
        raise ValueError(
            f"{weather}/{sample_index}: non-finite candidate/GT IoU")
    return result


def evaluate_frame_cached(stats, selected, full_scores, gt_ious, gt_count):
    """Update the original three OpenCOOD AP statistics without redoing IoU."""
    indices = np.asarray(selected, dtype=np.int64)
    full_scores = np.asarray(full_scores, dtype=np.float32)
    selected_scores = full_scores[indices]
    order = np.argsort(-selected_scores)
    indices = indices[order]
    selected_scores = selected_scores[order].tolist()
    for threshold in stats:
        remaining = list(range(int(gt_count)))
        tp, fp = [], []
        for candidate in indices:
            if remaining:
                overlaps = gt_ious[int(candidate), remaining]
            else:
                overlaps = np.empty((0,), dtype=np.float32)
            if not len(overlaps) or np.max(overlaps) < threshold:
                tp.append(0)
                fp.append(1)
            else:
                tp.append(1)
                fp.append(0)
                remaining.pop(int(np.argmax(overlaps)))
        stats[threshold]["tp"].extend(tp)
        stats[threshold]["fp"].extend(fp)
        stats[threshold]["score"].extend(selected_scores)
        stats[threshold]["gt"] += int(gt_count)


def assign_cached(rows, selected, full_scores, gt_ious, gt_count,
                  threshold=0.7):
    """Mirror the existing recovered/lost GT diagnostic using cached IoU."""
    matched, fp = set(), set()
    remaining = list(range(int(gt_count)))
    for candidate in sorted(selected, key=lambda i: (
            -float(full_scores[i]), int(i))):
        overlaps = (gt_ious[int(candidate), remaining] if remaining
                    else np.empty((0,), dtype=np.float32))
        if len(overlaps) and float(np.max(overlaps)) >= threshold:
            matched.add(remaining.pop(int(np.argmax(overlaps))))
        else:
            fp.add(int(rows[candidate]["candidate_id"]))
    return matched, fp


def assert_same_as_opencood(stats, rows, selected, full_scores, gt):
    """Runtime parity check against OpenCOOD for a small number of frames."""
    from local_fusion_detector_adaptation.candidate_hypothesis_replay import (
        _evaluate_frame, _stats,
    )
    slow = _stats()
    _evaluate_frame(slow, rows, selected, full_scores, gt)
    for threshold in slow:
        actual, reference = stats[threshold], slow[threshold]
        if actual["gt"] != reference["gt"]:
            raise AssertionError("Cached GT count differs from OpenCOOD")
        for key in ("tp", "fp"):
            if actual[key] != reference[key]:
                raise AssertionError(
                    f"Cached {key} differs from OpenCOOD at IoU={threshold}")
        if not np.array_equal(np.asarray(actual["score"], dtype=np.float32),
                              np.asarray(reference["score"], dtype=np.float32)):
            raise AssertionError(
                f"Cached scores differ from OpenCOOD at IoU={threshold}")
