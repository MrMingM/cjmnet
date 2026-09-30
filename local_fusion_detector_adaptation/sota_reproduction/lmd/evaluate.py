"""Evaluate adapted LMD-core on frozen top256 candidates with the original NMS budget."""
from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import (average_precision_score, mean_absolute_error,
                             mean_squared_error, roc_auc_score)

from local_fusion_detector_adaptation.candidate_hypothesis_replay import (
    _ap, _evaluate_frame, _stats,
)
from local_fusion_detector_adaptation.candidate_nms_scope import _nms_ordered
from .features import build_frame_features, feature_names, iter_frames
from .fast_geometry import (candidate_gt_iou, evaluate_frame_cached,
                            assign_cached, assert_same_as_opencood)

WEATHERS = ("clean", "fog", "rain", "snow")
METHODS = ("original_f", "top256_fused", "lmd_regression",
           "lmd_classification", "gt_iou_oracle")


def _meta(root):
    root = Path(root).resolve()
    meta = json.loads((root / "candidate_audit.json").read_text(encoding="utf-8"))
    if meta.get("split") not in ("validation", "benchmark"):
        raise ValueError("Evaluation root must be validation or fixed benchmark extraction")
    return root, meta


def _targets(root, weather):
    path = root / weather / "frame_targets.jsonl"
    result = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            gt = np.asarray(row["gt_bev_corners"], dtype=np.float32)
            if gt.size == 0:
                gt = gt.reshape(0, 4, 2)
            result[int(row["sample_index"])] = gt
    return result


def _prepare_rows(rows):
    return [dict(row,
                 corners=np.asarray(row["fused_bev_corners"], dtype=np.float32),
                 quality=float(row["max_gt_iou"]))
            for row in rows]


def _nms(rows, values, threshold, budget=None, usable=None, overlap_matrix=None):
    from opencood.utils import common_utils
    if usable is None:
        usable = np.arange(len(rows), dtype=np.int64)
    else:
        usable = np.asarray(usable, dtype=np.int64)
    if not len(usable):
        return []
    values = np.asarray(values, dtype=np.float64)
    order = usable[np.argsort(values[usable])[::-1]]
    if overlap_matrix is None:
        corners = np.stack([rows[i]["corners"] for i in range(len(rows))])
        polygons = common_utils.convert_format(corners)
        overlap = lambda candidate, others: common_utils.compute_iou(
            polygons[int(candidate)], polygons[np.asarray(others, dtype=np.int64)])
    else:
        overlap_matrix = np.asarray(overlap_matrix, dtype=np.float32)
        if overlap_matrix.shape != (len(rows), len(rows)):
            raise ValueError("LMD cached NMS overlap shape mismatch")
        overlap = lambda candidate, others: overlap_matrix[
            int(candidate), np.asarray(others, dtype=np.int64)]
    picked, _ = _nms_ordered(order, overlap, threshold)
    selected = [int(i) for i in picked if rows[int(i)]["within_range"]]
    return selected if budget is None else selected[:int(budget)]


def _assign(rows, selected, gt, values, threshold=0.7):
    from opencood.utils import common_utils
    if not selected:
        return set(), set()
    corners = np.stack([row["corners"] for row in rows])
    polygons = common_utils.convert_format(corners)
    gt_polygons = list(common_utils.convert_format(gt)) if len(gt) else []
    remaining = list(range(len(gt_polygons)))
    matched, fp = set(), set()
    for candidate in sorted(selected, key=lambda i: (-float(values[i]), int(i))):
        if remaining:
            ious = common_utils.compute_iou(
                polygons[candidate], [gt_polygons[j] for j in remaining])
        else:
            ious = []
        if len(ious) and float(np.max(ious)) >= threshold:
            pos = int(np.argmax(ious))
            matched.add(remaining.pop(pos))
        else:
            fp.add(int(rows[candidate]["candidate_id"]))
    return matched, fp


def _rankdata(a):
    a = np.asarray(a)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    i = 0
    while i < len(a):
        j = i + 1
        while j < len(a) and a[order[j]] == a[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def _quality_metrics(scores, quality):
    s = np.asarray(scores, dtype=np.float64)
    q = np.asarray(quality, dtype=np.float64)
    if not len(s):
        return {}
    sr, qr = _rankdata(s), _rankdata(q)
    spearman = (float(np.corrcoef(sr, qr)[0, 1])
                if np.std(sr) and np.std(qr) else None)
    label = (q >= 0.7).astype(np.int8)
    return {
        "score_gt_bev_iou_spearman": spearman,
        "mae_to_gt_iou": float(mean_absolute_error(q, s)),
        "rmse_to_gt_iou": float(mean_squared_error(q, s) ** 0.5),
        "iou70_roc_auc": (float(roc_auc_score(label, s))
                          if len(np.unique(label)) == 2 else None),
        "iou70_pr_auc": (float(average_precision_score(label, s))
                         if label.sum() else None),
    }


def _recovery(ap, original, oracle):
    return ((ap - original) / (oracle - original)
            if oracle > original + 1e-12 else None)


def run(args):
    root, meta = _meta(args.eval_root)
    with Path(args.model).open("rb") as stream:
        package = pickle.load(stream)
    if package.get("method") != "adapted_lmd_core":
        raise ValueError("Unexpected model package")
    if package["feature_names"] != feature_names():
        raise ValueError("LMD feature schema differs from trained model")
    if package["f_checkpoint_sha256"] != meta["arm_sha256"]["F"]:
        raise ValueError("Evaluation uses another frozen F checkpoint")
    if (package["frontend_sha256"] != meta["frontend_sha256"]
            or package["v3_checkpoint_sha256"] != meta["v3_checkpoint_sha256"]):
        raise ValueError("Evaluation frontend/v3 differs from LMD training")

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"protocol": {
        "method": "adapted_lmd_core",
        "split": meta["split"],
        "candidate_pool": meta["candidate_pool"],
        "proposal_iou_threshold": package["proposal_iou_threshold"],
        "fixed_output_budget": "per-frame original score>0.2 post-NMS/range output count",
        "primary_score": "LMD meta-regression predicted GT BEV IoU clipped to [0,1]",
        "ap_modes": ["frame_order", "global_sort"],
        "gt_inference": False,
        "evaluation_engine": "cached exact BEV IoU with OpenCOOD parity checks",
        "max_frames_per_weather": int(args.max_frames_per_weather),
        "subset_only": bool(args.max_frames_per_weather),
    }, "conditions": {}}
    frame_log = []

    for weather in WEATHERS:
        condition = meta["conditions"][weather]
        threshold = float(condition["nms_iou_threshold"])
        box_order = condition["box_order"]
        targets = _targets(root, weather)
        active = [name for name in METHODS
                  if name != "lmd_classification" or package["classifier"] is not None]
        stats_method = {name: _stats() for name in active}
        stats_selection = {name: _stats() for name in active}
        candidate_scores = {name: [] for name in active}
        candidate_quality = []
        counts = {name: Counter() for name in active}
        frame_count = 0

        for sample, _, raw_rows in iter_frames(root, weather):
            frame_count += 1
            rows = _prepare_rows(raw_rows)
            gt = targets[sample]
            try:
                x, quality, _, overlap_matrix = build_frame_features(
                    raw_rows, box_order, package["proposal_iou_threshold"],
                    return_overlap=True)
                gt_ious = candidate_gt_iou(rows, gt, weather, sample)
            except Exception as exc:
                raise RuntimeError(
                    f"LMD geometry failed at weather={weather}, "
                    f"sample_index={sample}; cache has not been modified") from exc
            if len(x):
                scaled = package["scaler"].transform(x)
                reg = np.clip(package["regressor"].predict(scaled),
                              0.0, 1.0).astype(np.float32)
            else:
                scaled = x
                reg = np.empty((0,), dtype=np.float32)
            if gt_ious.shape[1] and len(quality) and not np.allclose(
                    quality, gt_ious.max(axis=1), atol=1e-5, rtol=1e-5):
                raise AssertionError(
                    f"{weather}/{sample}: cached GT IoU differs from extraction")
            original_scores = np.asarray(
                [row["score"] for row in rows], dtype=np.float32)
            methods = {
                "original_f": original_scores,
                "top256_fused": original_scores,
                "lmd_regression": reg,
                "gt_iou_oracle": quality,
            }
            if package["classifier"] is not None:
                methods["lmd_classification"] = (
                    package["classifier"].predict_proba(scaled)[:, 1].astype(
                        np.float32) if len(x) else np.empty((0,), dtype=np.float32))

            eligible = np.flatnonzero(original_scores > 0.2)
            original = _nms(
                rows, original_scores, threshold, usable=eligible,
                overlap_matrix=overlap_matrix)
            budget = len(original)
            original_match, original_fp = assign_cached(
                rows, original, original_scores, gt_ious, len(gt))
            selected_map = {
                "original_f": original,
                "top256_fused": _nms(
                    rows, original_scores, threshold, budget,
                    overlap_matrix=overlap_matrix),
                "lmd_regression": _nms(
                    rows, reg, threshold, budget,
                    overlap_matrix=overlap_matrix),
            }
            if "lmd_classification" in methods:
                selected_map["lmd_classification"] = _nms(
                    rows, methods["lmd_classification"], threshold, budget,
                    overlap_matrix=overlap_matrix)
            oracle_usable = np.flatnonzero(quality > 0)
            selected_map["gt_iou_oracle"] = _nms(
                rows, quality, threshold, budget, oracle_usable,
                overlap_matrix=overlap_matrix)
            if frame_count <= 2:
                # A small on-cache exactness check before trusting the fast path.
                reference = _nms(rows, original_scores, threshold,
                                 usable=eligible)
                if reference != original:
                    raise AssertionError(
                        f"{weather}/{sample}: cached NMS differs from OpenCOOD")

            if (condition["original_scorepass_exceeds_top256_frames"] == 0
                    and selected_map["top256_fused"] != original):
                raise AssertionError(
                    f"{weather}/{sample}: top256 fused score failed exact original replay")

            for name, selected in selected_map.items():
                values = methods[name]
                if frame_count <= 2:
                    # Compare both score conventions against the original
                    # evaluator without altering the aggregate statistics.
                    method_check, selection_check = _stats(), _stats()
                    evaluate_frame_cached(
                        method_check, selected, values, gt_ious, len(gt))
                    evaluate_frame_cached(
                        selection_check, selected, original_scores, gt_ious,
                        len(gt))
                    assert_same_as_opencood(
                        method_check, rows, selected, values, gt)
                    assert_same_as_opencood(
                        selection_check, rows, selected, original_scores, gt)
                evaluate_frame_cached(
                    stats_method[name], selected, values, gt_ious, len(gt))
                evaluate_frame_cached(
                    stats_selection[name], selected, original_scores, gt_ious,
                    len(gt))
                matched, fp = assign_cached(
                    rows, selected, values, gt_ious, len(gt))
                if frame_count <= 2:
                    reference_match, reference_fp = _assign(
                        rows, selected, gt, values)
                    if matched != reference_match or fp != reference_fp:
                        raise AssertionError(
                            f"{weather}/{sample}/{name}: GT assignment mismatch")
                counts[name]["recovered"] += len(matched - original_match)
                counts[name]["lost"] += len(original_match - matched)
                counts[name]["new_fp"] += len(fp - original_fp)
                counts[name]["output"] += len(selected)
                counts[name]["budget"] += budget
                candidate_scores[name].extend(values.tolist())
                frame_log.append({
                    "weather": weather, "sample_index": sample, "method": name,
                    "budget": budget, "output": len(selected),
                    "selected_candidate_ids": [
                        int(rows[i]["candidate_id"]) for i in selected],
                    "recovered_matched_gt": len(matched - original_match),
                    "lost_original_matched_gt": len(original_match - matched),
                    "new_fp": len(fp - original_fp),
                })
            candidate_quality.extend(quality.tolist())
            if frame_count == 1 or frame_count % 25 == 0:
                print(
                    f"LMD {weather} replay: {frame_count}/"
                    f"{condition['frames']} frames; last sample={sample}",
                    flush=True)
            if args.max_frames_per_weather and frame_count >= args.max_frames_per_weather:
                break

        results = {}
        original_frame = _ap(
            stats_method["original_f"], False)["ap70"]
        original_global = _ap(
            stats_method["original_f"], True)["ap70"]
        oracle_frame = _ap(
            stats_method["gt_iou_oracle"], False)["ap70"]
        oracle_global = _ap(
            stats_method["gt_iou_oracle"], True)["ap70"]

        for name in active:
            ap_frame = _ap(stats_method[name], False)
            ap_global = _ap(stats_method[name], True)
            selection_frame = _ap(stats_selection[name], False)
            selection_global = _ap(stats_selection[name], True)
            results[name] = {
                "ap_using_method_score_frame_order": ap_frame,
                "ap_using_method_score_global_sort": ap_global,
                "ap_selection_only_original_score_frame_order": selection_frame,
                "ap_selection_only_original_score_global_sort": selection_global,
                "oracle_recovery_ratio_frame_order": _recovery(
                    ap_frame["ap70"], original_frame, oracle_frame),
                "oracle_recovery_ratio_global_sort": _recovery(
                    ap_global["ap70"], original_global, oracle_global),
                "recovered_matched_gt_vs_original": counts[name]["recovered"],
                "lost_original_matched_gt": counts[name]["lost"],
                "new_fp_vs_original": counts[name]["new_fp"],
                "budget_fill_rate": (
                    counts[name]["output"] / max(1, counts[name]["budget"])),
                "candidate_quality": _quality_metrics(
                    candidate_scores[name], candidate_quality),
            }

        report["conditions"][weather] = {
            "frames": frame_count,
            "original_scorepass_exceeds_top256_frames": condition[
                "original_scorepass_exceeds_top256_frames"],
            "exact_original_replay_from_top256": condition[
                "original_scorepass_exceeds_top256_frames"] == 0,
            "methods": results,
        }
        (output / "results_partial.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
        print(f"{weather}: adapted LMD replay complete", flush=True)

    (output / "results.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    with (output / "frames.jsonl").open("w", encoding="utf-8") as stream:
        for row in frame_log:
            stream.write(json.dumps(row) + "\n")

    lines = [
        "# Adapted LMD-core benchmark", "",
        "主分数是 LMD meta-regression 预测的 GT BEV IoU；"
        "固定 decoded geometry、top256、原 NMS 和原输出预算。", "",
        "| 条件 | Original AP70 | LMD AP70 | Oracle AP70 | "
        "Oracle recovery | LMD global AP70 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for weather in WEATHERS:
        m = report["conditions"][weather]["methods"]
        ratio = m["lmd_regression"]["oracle_recovery_ratio_frame_order"]
        lines.append("| %s | %.4f | %.4f | %.4f | %s | %.4f |" % (
            weather,
            m["original_f"]["ap_using_method_score_frame_order"]["ap70"],
            m["lmd_regression"]["ap_using_method_score_frame_order"]["ap70"],
            m["gt_iou_oracle"]["ap_using_method_score_frame_order"]["ap70"],
            ("%.3f" % ratio if ratio is not None else "NA"),
            m["lmd_regression"]["ap_using_method_score_global_sort"]["ap70"]))
    lines += [
        "",
        "完整 AP30/AP50/AP70、质量指标、TP/FP 变化及 selection-only 诊断见 results.json。",
        "",
    ]
    (output / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print("LMD EVALUATION COMPLETE:", output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-frames-per-weather", type=int, default=0,
                        help="Offline smoke only; 0 means full evaluation")
    args = parser.parse_args()
    if args.max_frames_per_weather < 0:
        parser.error("--max-frames-per-weather must be >= 0")
    run(args)


if __name__ == "__main__":
    main()
