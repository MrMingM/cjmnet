"""Build a conservative H-A5 report from per-weather local evidence traces."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import QUANTILE_LABELS, summarize_rows

WEATHERS = ("fog", "rain", "snow")


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _pct(n, d):
    return "—" if not d else f"{100.0 * n / d:.2f}%"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audit-root", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    root = Path(args.audit_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results = {
        "schema": 1,
        "purpose": "H-A5 local evidence validity audit",
        "unit": "target-frame occurrence",
        "weather": {},
        "interpretation": {
            "late_detection_path_loss": "A correct IoU>=0.7 proposal existed after decode but final ego detection still failed.",
            "regression_or_localization_loss_with_cls_survival": "No IoU>=0.7 decoded box, but the clean-matched anchor still passed the weather classification threshold.",
            "joint_head_or_upstream_unresolved": "Neither a correct decoded box nor classification survival at the clean-matched anchor was observed; A5a remains compatible but is not proved.",
        },
    }
    lines = [
        "# H-A5 local evidence audit",
        "",
        "> Development OPV2V validation + online weather only. No OPV2V-W test data, no training, no fusion intervention.",
        "",
        "This audit asks whether Stage-1 metric-strong ego misses contain target-aligned detector evidence before the final local detection pipeline finishes.",
        "It is designed to separate a directly observable **late local downstream loss** from cases that remain **upstream/A5a-compatible but unresolved**.",
        "",
        "## 1. Main q25 result",
        "",
        "| Weather | All ego misses | q25 strong | Late detection-path loss | Regression/localization support | Upstream unresolved |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for weather in WEATHERS:
        folder = root / weather
        protocol = _read_json(folder / "protocol.json")
        summary = _read_json(folder / "summary.json")
        rows = _read_rows(folder / "targets.jsonl")
        if not summary.get("complete") or len(rows) != summary["candidate_targets"]:
            raise RuntimeError(f"Incomplete audit for {weather}")
        if protocol.get("test_data_used", True) or not protocol.get("development_only", False):
            raise ValueError(f"Non-development protocol for {weather}")
        stats = summarize_rows(rows)
        results["weather"][weather] = {
            "protocol": protocol,
            "summary": summary,
            "statistics": stats,
        }
        q25 = stats["q25"]
        cc = q25["classification_counts"]
        strong = q25["strong_occurrences"]
        late = cc.get("late_detection_path_loss", 0)
        reg = cc.get("regression_or_localization_loss_with_cls_survival", 0)
        unresolved = cc.get("joint_head_or_upstream_unresolved", 0)
        lines.append(
            f"| {weather} | {stats['occurrences']} | {strong} | "
            f"{late} ({_pct(late, strong)}) | {reg} ({_pct(reg, strong)}) | "
            f"{unresolved} ({_pct(unresolved, strong)}) |"
        )

    lines += [
        "",
        "## 2. Threshold sensitivity",
        "",
        "The old strong/weak ruler is not treated as truth. The table below only checks whether the A5 split changes drastically when the clean-detected quantile threshold changes.",
        "",
        "| Weather | Strong threshold | Strong n | A5b-support fraction | Upstream-unresolved fraction |",
        "|---|---|---:|---:|---:|",
    ]
    for weather in WEATHERS:
        stats = results["weather"][weather]["statistics"]
        for q in QUANTILE_LABELS:
            item = stats[q]
            support = item["a5b_support_fraction"]
            unresolved = item["upstream_unresolved_fraction"]
            lines.append(
                f"| {weather} | {q} | {item['strong_occurrences']} | "
                f"{'—' if support is None else f'{100*support:.2f}%'} | "
                f"{'—' if unresolved is None else f'{100*unresolved:.2f}%'} |"
            )

    lines += [
        "",
        "## 3. How to interpret the categories",
        "",
        "- **Late detection-path loss**: the weather ego branch already produced at least one decoded proposal with IoU>=0.7 for the GT, but it disappeared at score filtering / geometry filtering / NMS / range filtering / greedy matching. This is direct evidence for an A5b-type downstream failure in that occurrence.",
        "- **Regression/localization support**: no correct decoded box survived, but the exact anchor that detects the target in clean still passes the weather classification threshold. This supports surviving target-semantic activation with localization/regression failure, but it is not a complete root-cause proof.",
        "- **Upstream unresolved**: the proxy says strong, yet neither of the above task-oriented signals survives. These are the cases most compatible with A5a, but they can also arise because PillarVFE/backbone representations lost or distorted target semantics. A deeper feature-level intervention is required before declaring A5a.",
        "",
        "## 4. Decision boundary for H-A5",
        "",
        "This experiment can **positively identify a subset of A5b**. It cannot by itself positively prove A5a for the remaining cases. If most q25-strong misses fall into upstream unresolved, the next A5 experiment should focus only on that subset and intervene at PillarVFE/backbone feature stages rather than rerunning the same N_eff/coverage analysis.",
        "",
    ]
    (out / "a5_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (out / "a5_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out/'a5_report.md'}")
    print(f"Wrote {out/'a5_results.json'}")


if __name__ == "__main__":
    main()
