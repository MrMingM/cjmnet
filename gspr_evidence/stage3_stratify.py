"""Stage-2.5 denominator-aware stratification before Stage-3 intervention.

Uses existing Stage-1/Stage-2 JSON only. No model inference, no OPV2V-W test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gspr_evidence.stage3_common import ADVERSE, load_stage1_map, load_stage2_rows, scene_failure_table


def markdown(result):
    lines = [
        "# Stage-2.5 source-valid fusion-failure stratification",
        "",
        "> Development OPV2V validation + online weather only. Counts are target-frame occurrences, "
        "not independent vehicles or trajectories.",
        "",
        "The denominator is `any peer-alone detected`. The numerator is "
        "`any peer-alone detected + full fusion miss`.",
        "",
        "## Overall",
        "",
        "| Weather | Source-valid target occurrences | Full-miss occurrences | Failure rate | "
        "Source-valid unique frames | Failure unique frames |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for weather in ADVERSE:
        r = result[weather]["overall"]
        lines.append(
            f"| {weather} | {r['source_valid_target_occurrences']} | "
            f"{r['failure_target_occurrences']} | {r['failure_rate_given_source_valid']:.6f} | "
            f"{r['source_valid_unique_frames']} | {r['failure_unique_frames']} |"
        )

    for weather in ADVERSE:
        lines += [
            "",
            f"## {weather}: by scene",
            "",
            "| Scene | Source-valid occurrences | Full-miss occurrences | Failure rate | "
            "Source-valid frames | Failure frames |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name, r in result[weather]["by_scene"].items():
            rate = r["failure_rate_given_source_valid"]
            rate_text = "NA" if rate is None else f"{rate:.6f}"
            lines.append(
                f"| {name} | {r['source_valid_target_occurrences']} | "
                f"{r['failure_target_occurrences']} | {rate_text} | "
                f"{r['source_valid_unique_frames']} | {r['failure_unique_frames']} |"
            )
        lines += [
            "",
            f"## {weather}: by distance",
            "",
            "| Distance | Source-valid occurrences | Full-miss occurrences | Failure rate | "
            "Source-valid frames | Failure frames |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name, r in result[weather]["by_distance"].items():
            rate = r["failure_rate_given_source_valid"]
            rate_text = "NA" if rate is None else f"{rate:.6f}"
            lines.append(
                f"| {name} | {r['source_valid_target_occurrences']} | "
                f"{r['failure_target_occurrences']} | {rate_text} | "
                f"{r['source_valid_unique_frames']} | {r['failure_unique_frames']} |"
            )

    lines += [
        "",
        "## Interpretation boundary",
        "",
        "- A high raw failure count is not enough; inspect the denominator-aware failure rate.",
        "- These are target-frame occurrences, not unique physical vehicles.",
        "- This subset is conditioned on Stage-1 weather-induced ego misses and does not represent all detector errors.",
        "- No AP upper bound is inferred from these counts.",
    ]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1-root", required=True)
    p.add_argument("--stage2-root", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "schema": 1,
        "purpose": "Stage-2.5 denominator-aware source-valid/full-miss stratification",
        "development_only": True,
        "test_data_used": False,
        "stage1_root": str(Path(args.stage1_root).resolve()),
        "stage2_root": str(Path(args.stage2_root).resolve()),
    }
    for weather in ADVERSE:
        stage1 = load_stage1_map(args.stage1_root, weather)
        stage2 = load_stage2_rows(args.stage2_root, weather)
        result[weather] = scene_failure_table(stage1, stage2)

    (out / "stage2_5_stratification.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out / "stage2_5_stratification.md").write_text(markdown(result), encoding="utf-8")
    print(out / "stage2_5_stratification.md")


if __name__ == "__main__":
    main()
