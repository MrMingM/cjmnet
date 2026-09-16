"""Analyze object-level counterfactual utility exported by oracle_eval.py.

The script deliberately treats a frame as the resampling unit.  Targets from
the same frame and the same coalition are correlated, so an event-wise
bootstrap would produce confidence intervals that are too optimistic.
"""

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose density/confidence alignment with Oracle utility")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--soft_eps", type=float, default=0.005,
                        help="Minimum negative IoU*score change called harmful")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--max_scatter", type=int, default=20000)
    return parser.parse_args()


def to_float(row, name):
    try:
        return float(row[name])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def to_int(row, name):
    try:
        return int(float(row[name]))
    except (KeyError, TypeError, ValueError):
        return -1


def average_ranks(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def correlation(x, y, rank=False):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if x.size < 3:
        return None
    if rank:
        x, y = average_ranks(x), average_ranks(y)
    if np.std(x) < 1.0e-12 or np.std(y) < 1.0e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def roc_auc(labels, scores):
    """Tie-aware binary ROC AUC; returns None for a single-class target."""
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    keep = np.isfinite(scores) & ((labels == 0) | (labels == 1))
    labels, scores = labels[keep], scores[keep]
    positive = labels == 1
    n_pos = int(positive.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = average_ranks(scores)
    value = (ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0)
    return float(value / (n_pos * n_neg))


def percentile_ci(values):
    values = [x for x in values if x is not None and math.isfinite(x)]
    if not values:
        return [None, None]
    return [float(x) for x in np.percentile(values, [2.5, 97.5])]


def cluster_bootstrap(rows, soft_eps, repetitions, seed):
    by_frame = defaultdict(list)
    for row in rows:
        by_frame[row["frame_index"]].append(row)
    frames = sorted(by_frame)
    if not frames or repetitions <= 0:
        return {}
    rng = np.random.default_rng(seed)
    estimates = defaultdict(list)
    for _ in range(repetitions):
        chosen = rng.choice(frames, size=len(frames), replace=True)
        sample = [event for frame in chosen for event in by_frame[int(frame)]]
        utility = np.asarray([x["utility"] for x in sample])
        estimates["mean_soft_utility"].append(float(np.mean(utility)))
        estimates["soft_harmful_eps_ratio"].append(
            float(np.mean(utility < -soft_eps)))
        estimates["hard_harmful_ratio"].append(
            float(np.mean([x["hard_harmful"] for x in sample])))
        density = np.asarray([x["density"] for x in sample])
        confidence = np.asarray([x["confidence"] for x in sample])
        harmful = utility < -soft_eps
        estimates["density_spearman"].append(
            correlation(density, utility, rank=True))
        estimates["confidence_spearman"].append(
            correlation(confidence, utility, rank=True))
        density_auc = roc_auc(harmful, -density)
        confidence_auc = roc_auc(harmful, -confidence)
        if density_auc is not None:
            estimates["low_density_harmful_auc"].append(density_auc)
        if confidence_auc is not None:
            estimates["low_confidence_harmful_auc"].append(confidence_auc)
        if density_auc is not None and confidence_auc is not None:
            estimates["confidence_minus_density_auc"].append(
                confidence_auc - density_auc)
    return {name: percentile_ci(values) for name, values in estimates.items()}


def quantile_bins(rows, key, bins, soft_eps):
    values = np.asarray([x[key] for x in rows], dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return []
    edges = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, bins + 1)))
    if edges.size < 2:
        return []
    output = []
    for index in range(edges.size - 1):
        lower, upper = edges[index], edges[index + 1]
        if index == edges.size - 2:
            selected = [x for x in rows if lower <= x[key] <= upper]
        else:
            selected = [x for x in rows if lower <= x[key] < upper]
        if not selected:
            continue
        utility = np.asarray([x["utility"] for x in selected])
        output.append({
            "bin": index + 1,
            "lower": float(lower),
            "upper": float(upper),
            "count": len(selected),
            "feature_mean": float(np.mean([x[key] for x in selected])),
            "utility_mean": float(np.mean(utility)),
            "utility_median": float(np.median(utility)),
            "soft_harmful_eps_ratio": float(np.mean(utility < -soft_eps)),
            "hard_harmful_ratio": float(np.mean(
                [x["hard_harmful"] for x in selected]))
        })
    return output


def write_table(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite_number(value):
    if value is None or not math.isfinite(value):
        return None
    return float(value)


def slot_summary(rows, soft_eps):
    result = {}
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["neighbor_index"]].append(row)
    for slot, selected in sorted(grouped.items()):
        utility = np.asarray([x["utility"] for x in selected])
        addition = np.asarray([x["addition_utility"] for x in selected])
        beneficial_alone = addition > soft_eps
        reversed_in_coalition = beneficial_alone & (utility < -soft_eps)
        result[str(slot)] = {
            "events": len(selected),
            "frames": len({x["frame_index"] for x in selected}),
            "mean_soft_utility": float(np.mean(utility)),
            "median_soft_utility": float(np.median(utility)),
            "soft_harmful_eps_ratio": float(np.mean(utility < -soft_eps)),
            "beneficial_alone_ratio": float(np.mean(beneficial_alone)),
            "beneficial_alone_harmful_in_coalition_ratio": float(
                np.mean(reversed_in_coalition)),
            "conditional_reversal_given_beneficial_alone": float(
                reversed_in_coalition.sum() / beneficial_alone.sum())
            if beneficial_alone.any() else None,
            "addition_coalition_spearman": correlation(
                addition, utility, rank=True),
            "mean_redundancy_gap": float(np.mean(addition - utility)),
            "hard_harmful_ratio": float(np.mean(
                [x["hard_harmful"] for x in selected])),
            "error_patterns": dict(Counter(
                x["error_pattern"] for x in selected))
        }
    return result


def object_level_oracle(rows):
    """Upper envelope over candidate coalitions for each GT target."""
    grouped = defaultdict(dict)
    for row in rows:
        key = (row["frame_index"], row["gt_key"])
        candidates = grouped[key]
        for name, state in row["candidate_states"].items():
            # ego/all repeat across neighbor events; identical values are kept
            # once while plus/minus names remain neighbor-specific.
            candidates[name] = state
    all_recalled = []
    oracle_recalled = []
    all_quality = []
    oracle_quality = []
    best_counts = Counter()
    corrected_targets = 0
    for candidates in grouped.values():
        baseline = candidates["all"]
        best_name, best = max(
            candidates.items(),
            key=lambda item: (
                int(item[1]["recalled"]), item[1]["quality"],
                item[1]["iou"], item[1]["score"]))
        all_recalled.append(int(baseline["recalled"]))
        oracle_recalled.append(int(best["recalled"]))
        all_quality.append(baseline["quality"])
        oracle_quality.append(best["quality"])
        best_counts[best_name] += 1
        corrected_targets += int(
            not baseline["recalled"] and best["recalled"])
    all_recall = float(np.mean(all_recalled)) if all_recalled else 0.0
    oracle_recall = float(np.mean(oracle_recalled)) if oracle_recalled else 0.0
    all_soft = float(np.mean(all_quality)) if all_quality else 0.0
    oracle_soft = float(np.mean(oracle_quality)) if oracle_quality else 0.0
    return {
        "targets": len(grouped),
        "all_recall_rate": all_recall,
        "oracle_recall_rate": oracle_recall,
        "recall_rate_gain": oracle_recall - all_recall,
        "corrected_targets": corrected_targets,
        "all_mean_iou_score": all_soft,
        "oracle_mean_iou_score": oracle_soft,
        "mean_iou_score_gain": oracle_soft - all_soft,
        "best_candidate_counts": dict(best_counts),
        "note": (
            "Per-target coalition upper envelope; it is not mathematical AP "
            "and cannot be deployed without a spatial selector.")
    }


def create_plot(path, rows, density_bins, confidence_bins, soft_eps,
                max_scatter, seed):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    rng = np.random.default_rng(seed)
    if len(rows) > max_scatter:
        indices = rng.choice(len(rows), max_scatter, replace=False)
        shown = [rows[int(i)] for i in indices]
    else:
        shown = rows

    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for ax, key, title in [
            (axes[0, 0], "density", "Neighbor point density"),
            (axes[0, 1], "confidence", "Where2comm confidence")]:
        x = np.asarray([r[key] for r in shown])
        y = np.asarray([r["utility"] for r in shown])
        keep = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[keep], y[keep], s=5, alpha=0.15, rasterized=True)
        ax.axhline(-soft_eps, color="tab:red", linestyle="--", linewidth=1)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel(title)
        ax.set_ylabel("Coalition marginal utility (IoU x score)")

    for ax, table, title in [
            (axes[1, 0], density_bins, "Density quantile calibration"),
            (axes[1, 1], confidence_bins, "Confidence quantile calibration")]:
        x = [r["bin"] for r in table]
        ax.plot(x, [r["utility_mean"] for r in table], marker="o",
                label="mean utility")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel("Quantile bin (low to high)")
        ax.set_ylabel("Mean utility")
        twin = ax.twinx()
        twin.plot(x, [r["soft_harmful_eps_ratio"] for r in table],
                  color="tab:red", marker="s", label="harmful ratio")
        twin.set_ylim(0.0, 1.0)
        twin.set_ylabel("P(utility < -eps)", color="tab:red")
        ax.set_title(title)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return True


def create_context_plot(path, rows, soft_eps, max_scatter, seed):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    rng = np.random.default_rng(seed)
    if len(rows) > max_scatter:
        indices = rng.choice(len(rows), max_scatter, replace=False)
        rows = [rows[int(i)] for i in indices]
    addition = np.asarray([r["addition_utility"] for r in rows])
    coalition = np.asarray([r["utility"] for r in rows])
    keep = np.isfinite(addition) & np.isfinite(coalition)
    addition, coalition = addition[keep], coalition[keep]
    limit = float(np.quantile(np.abs(np.r_[addition, coalition]), 0.99))
    limit = max(limit, soft_eps * 2.0, 1.0e-3)
    figure, ax = plt.subplots(figsize=(7.5, 7.0), constrained_layout=True)
    ax.scatter(addition, coalition, s=6, alpha=0.18, rasterized=True)
    ax.axvline(soft_eps, color="tab:green", linestyle="--", linewidth=1)
    ax.axhline(-soft_eps, color="tab:red", linestyle="--", linewidth=1)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.plot([-limit, limit], [-limit, limit], color="tab:gray",
            linestyle=":", linewidth=1, label="context invariant")
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_xlabel("Isolated addition utility: ego+i minus ego")
    ax.set_ylabel("Coalition marginal utility: all minus all-i")
    ax.set_title("Context-dependent neighbor utility")
    ax.legend(loc="upper left")
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return True


def main():
    args = parse_args()
    if args.soft_eps < 0.0:
        raise ValueError("--soft_eps must be non-negative")
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.input_csv)), "oracle_analysis")
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    with open(args.input_csv, newline="", encoding="utf-8") as stream:
        for raw in csv.DictReader(stream):
            row = {
                "frame_index": to_int(raw, "frame_index"),
                "gt_key": raw.get("gt_id") or raw.get("gt_index", "-1"),
                "neighbor_index": to_int(raw, "neighbor_index"),
                "density": to_float(raw, "point_density"),
                "confidence": to_float(raw, "where2comm_confidence"),
                "utility": to_float(raw, "coalition_soft_utility"),
                "addition_utility": to_float(raw, "addition_soft_utility"),
                "addition_recall_delta": to_int(raw, "addition_recall_delta"),
                "hard_harmful": to_int(raw, "coalition_hard_harmful"),
                "error_pattern": raw.get("coalition_error_pattern", "UNKNOWN"),
                "addition_error_pattern": raw.get(
                    "addition_error_pattern", "UNKNOWN"),
                "candidate_states": {}
            }
            neighbor_index = row["neighbor_index"]
            for name, prefix in [
                    ("ego", "ego"), ("all", "all"),
                    ("ego_plus_%d" % neighbor_index, "ego_plus_i"),
                    ("all_minus_%d" % neighbor_index, "all_minus_i")]:
                iou = to_float(raw, prefix + "_iou_with_pred")
                score = to_float(raw, prefix + "_classification_score")
                row["candidate_states"][name] = {
                    "recalled": to_int(raw, prefix + "_is_recalled") > 0,
                    "iou": iou,
                    "score": score,
                    "quality": iou * score
                }
            if row["frame_index"] >= 0 and math.isfinite(row["utility"]):
                rows.append(row)
    if not rows:
        raise RuntimeError("No valid Oracle events found in %s" % args.input_csv)

    utility = np.asarray([x["utility"] for x in rows])
    density = np.asarray([x["density"] for x in rows])
    confidence = np.asarray([x["confidence"] for x in rows])
    addition = np.asarray([x["addition_utility"] for x in rows])
    harmful = utility < -args.soft_eps
    beneficial_alone = addition > args.soft_eps
    reversed_in_coalition = beneficial_alone & harmful
    density_bins = quantile_bins(rows, "density", args.bins, args.soft_eps)
    confidence_bins = quantile_bins(
        rows, "confidence", args.bins, args.soft_eps)

    thresholds = [0.0, 0.001, 0.005, 0.01, 0.02]
    summary = {
        "input_csv": os.path.abspath(args.input_csv),
        "events": len(rows),
        "frames": len({x["frame_index"] for x in rows}),
        "soft_eps": args.soft_eps,
        "utility": {
            "mean": float(np.mean(utility)),
            "median": float(np.median(utility)),
            "quantiles": {
                str(q): float(np.quantile(utility, q))
                for q in [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
            },
            "harmful_ratio_by_epsilon": {
                str(eps): float(np.mean(utility < -eps)) for eps in thresholds
            },
            "hard_harmful_ratio": float(np.mean(
                [x["hard_harmful"] for x in rows])),
            "error_patterns": dict(Counter(
                x["error_pattern"] for x in rows))
        },
        "alignment": {
            "density_pearson": correlation(density, utility),
            "density_spearman": correlation(density, utility, rank=True),
            "confidence_pearson": correlation(confidence, utility),
            "confidence_spearman": correlation(confidence, utility, rank=True),
            "low_density_harmful_auc": roc_auc(harmful, -density),
            "low_confidence_harmful_auc": roc_auc(harmful, -confidence)
        },
        "context_dependence": {
            "addition_mean": float(np.nanmean(addition)),
            "addition_median": float(np.nanmedian(addition)),
            "addition_coalition_pearson": correlation(addition, utility),
            "addition_coalition_spearman": correlation(
                addition, utility, rank=True),
            "beneficial_alone_ratio": float(np.mean(beneficial_alone)),
            "beneficial_alone_harmful_in_coalition_ratio": float(
                np.mean(reversed_in_coalition)),
            "conditional_reversal_given_beneficial_alone": float(
                reversed_in_coalition.sum() / beneficial_alone.sum())
            if beneficial_alone.any() else None,
            "mean_redundancy_gap": float(np.nanmean(addition - utility)),
            "median_redundancy_gap": float(np.nanmedian(addition - utility)),
            "addition_error_patterns": dict(Counter(
                x["addition_error_pattern"] for x in rows))
        },
        "object_level_oracle": object_level_oracle(rows),
        "cluster_bootstrap_95_ci": cluster_bootstrap(
            rows, args.soft_eps, args.bootstrap, args.seed),
        "by_neighbor_slot": slot_summary(rows, args.soft_eps)
    }

    # JSON does not officially support NaN; sanitize correlations/AUCs.
    summary["alignment"] = {
        key: finite_number(value) for key, value in summary["alignment"].items()
    }
    summary_path = os.path.join(output_dir, "oracle_diagnostic_summary.json")
    with open(summary_path, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    write_table(os.path.join(output_dir, "density_calibration.csv"), density_bins)
    write_table(os.path.join(output_dir, "confidence_calibration.csv"),
                confidence_bins)
    plot_path = os.path.join(output_dir, "oracle_utility_diagnostics.png")
    plotted = create_plot(plot_path, rows, density_bins, confidence_bins,
                          args.soft_eps, args.max_scatter, args.seed)
    context_plot_path = os.path.join(
        output_dir, "oracle_context_interaction.png")
    context_plotted = create_context_plot(
        context_plot_path, rows, args.soft_eps, args.max_scatter, args.seed)

    print("Oracle diagnostic analysis finished")
    print("Events: %d, frames: %d" % (summary["events"], summary["frames"]))
    print("Mean/median utility: %.6f / %.6f" % (
        summary["utility"]["mean"], summary["utility"]["median"]))
    print("P(utility < -%.4f): %.4f" % (
        args.soft_eps, float(np.mean(harmful))))
    print("Alignment: %s" % json.dumps(summary["alignment"]))
    print("Context dependence: %s" % json.dumps(
        summary["context_dependence"]))
    print("Object-level Oracle: %s" % json.dumps(
        summary["object_level_oracle"]))
    print("Summary: %s" % summary_path)
    if plotted:
        print("Plot: %s" % plot_path)
    else:
        print("Plot skipped: matplotlib is not installed")
    if context_plotted:
        print("Context plot: %s" % context_plot_path)


if __name__ == "__main__":
    main()
