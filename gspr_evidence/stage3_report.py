"""Generate compact Stage-3A/3B reports from machine-readable summaries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

WEATHERS = ("fog", "rain", "snow")


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _pct(num, den):
    return 100.0 * num / den if den else 0.0


def _protocols(root):
    return {w: _load(Path(root) / w / "protocol.json") for w in WEATHERS}


def report_a(root):
    root = Path(root)
    strat = _load(root / "stratification" / "stage2_5_stratification.json")
    summaries = {w: _load(root / w / "summary.json") for w in WEATHERS}
    protocols = _protocols(root)
    smoke = any(bool(p.get("smoke_or_debug")) for p in protocols.values())
    lines = [
        "# Stage-3A H-A6 disappearance audit",
        "",
        "> Development OPV2V validation + online weather only. No OPV2V-W test. "
        "Counts are target-frame occurrences, not independent vehicles.",
    ]
    if smoke:
        lines += ["", "> **SMOKE/DEBUG RUN: this report cannot be used for the Stage-3 scientific gate.**"]
    lines += [
        "",
        "## 1. Stage-2.5 denominator-aware prevalence",
        "",
        "| Weather | Any peer-alone detects | Full still misses | Failure rate given source-valid | "
        "Source-valid frames | Failure frames |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for w in WEATHERS:
        r = strat[w]["overall"]
        lines.append(
            f"| {w} | {r['source_valid_target_occurrences']} | {r['failure_target_occurrences']} | "
            f"{100*r['failure_rate_given_source_valid']:.2f}% | "
            f"{r['source_valid_unique_frames']} | {r['failure_unique_frames']} |"
        )

    lines += [
        "",
        "## 2. Last observable disappearance stage",
        "",
        "| Weather | Candidates | decoded has no IoU70 | score threshold | geometry | top-1000 | "
        "NMS suppression | range | final matching |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    keys = (
        "no_iou70_decoded", "score_threshold", "geometry_sanity",
        "nms_top1000", "nms_suppression", "range_filter",
        "final_matching_competition",
    )
    for w in WEATHERS:
        s = summaries[w]
        c = s["last_observable_disappearance_stage"]
        n = int(s["candidate_target_occurrences"])
        vals = [int(c.get(k, 0)) for k in keys]
        lines.append(
            f"| {w} | {n} | {vals[0]} ({_pct(vals[0], n):.1f}%) | "
            f"{vals[1]} ({_pct(vals[1], n):.1f}%) | {vals[2]} ({_pct(vals[2], n):.1f}%) | "
            f"{vals[3]} ({_pct(vals[3], n):.1f}%) | {vals[4]} ({_pct(vals[4], n):.1f}%) | "
            f"{vals[5]} ({_pct(vals[5], n):.1f}%) | {vals[6]} ({_pct(vals[6], n):.1f}%) |"
        )

    lines += [
        "",
        "## 3. Interpretation boundary",
        "",
        "- A label says where IoU>=0.7 support is last observed to disappear in the frozen path.",
        "- `score_threshold` or `nms_suppression` does **not** prove score/NMS is the root cause; "
        "upstream fusion may have changed the feature, score, localization, or competing boxes.",
        "- `no_iou70_decoded` includes localization/representation outcomes and is not by itself a proof of an attention failure.",
        "- Scene-specific conclusions must use the denominator-aware Stage-2.5 table, not raw failure counts alone.",
        "- Stage-3B should be launched only after inspecting a **full, non-smoke** Stage-3A report.",
        "",
        "Full per-target details are in `{fog,rain,snow}/targets.jsonl`; denominator-aware scene/distance "
        "tables are in `stratification/stage2_5_stratification.md`.",
    ]
    result = {
        "schema": 1,
        "phase": "stage3a",
        "development_only": True,
        "test_data_used": False,
        "smoke_or_debug": smoke,
        "protocols": protocols,
        "stratification": {w: strat[w] for w in WEATHERS},
        "summaries": summaries,
    }
    return "\n".join(lines) + "\n", result


def report_b(root):
    root = Path(root)
    summaries = {w: _load(root / w / "summary.json") for w in WEATHERS}
    protocols = _protocols(root)
    smoke = any(bool(p.get("smoke_or_debug")) for p in protocols.values())
    lines = [
        "# Stage-3B limited source-subset Oracle",
        "",
        "> Development candidate frames only. Fixed encoding, fixed original AttFuse, ego always present, "
        "whole-peer inclusion/exclusion only. This is not AP and not a fusion-theory upper bound.",
    ]
    if smoke:
        lines += ["", "> **SMOKE/DEBUG RUN: do not promote these subset rates to a full Stage-3B conclusion.**"]
    lines += [
        "",
        "| Weather | Candidate targets | Target-wise recoverable | Target-wise rate | "
        "Best-frame recovered sum | Safe-frame recovered sum | Candidate frames |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for w in WEATHERS:
        s = summaries[w]
        n = int(s["candidate_target_occurrences"])
        r = int(s["target_wise_recoverable_occurrences"])
        lines.append(
            f"| {w} | {n} | {r} | {_pct(r, n):.2f}% | "
            f"{s['sum_best_frame_subset_recovered_occurrences']} | "
            f"{s['sum_safe_frame_subset_recovered_occurrences']} | "
            f"{s['candidate_unique_frames']} |"
        )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "- Target-wise recoverability allows a different hindsight subset for different targets; it is opportunity only.",
        "- `best_frame_subset` uses one subset for the whole frame and reports recovered candidates, lost full TPs, and FP change.",
        "- `safe_best_frame_subset` additionally requires zero lost full TPs and no FP increase.",
        "- All-subset failure only rules out this restricted whole-agent subset space with the original fusion operator and mandatory ego.",
        "- `peer-alone success + ego-containing subsets fail` does not by itself prove ego-query suppression.",
    ]
    result = {
        "schema": 1,
        "phase": "stage3b",
        "development_only": True,
        "test_data_used": False,
        "smoke_or_debug": smoke,
        "protocols": protocols,
        "summaries": summaries,
    }
    return "\n".join(lines) + "\n", result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", required=True, choices=("a", "b"))
    p.add_argument("--root", required=True)
    args = p.parse_args()
    root = Path(args.root)
    text, data = report_a(root) if args.phase == "a" else report_b(root)
    stem = "stage3a_report" if args.phase == "a" else "stage3b_report"
    (root / f"{stem}.md").write_text(text, encoding="utf-8")
    (root / f"{stem}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(root / f"{stem}.md")


if __name__ == "__main__":
    main()
