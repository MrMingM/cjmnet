"""Merge four weather summaries into one compact Markdown table."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    args = p.parse_args()
    root = Path(args.root)

    data = {}
    for weather in ("clean", "fog", "rain", "snow"):
        path = root / weather / "summary.json"
        if not path.exists():
            raise FileNotFoundError(path)
        data[weather] = json.loads(path.read_text(encoding="utf-8"))

    lines = [
        "# Lossless communication summary",
        "",
        "| Weather | Mode | AP30 | AP50 | AP70 | Mean MiB/frame | Reduction vs raw | Enc ms | Dec ms | Recompute ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for weather, summary in data.items():
        for mode, row in summary["results"].items():
            lines.append(
                f"| {weather} | {mode} | {row['ap30']:.6f} | {row['ap50']:.6f} | "
                f"{row['ap70']:.6f} | {row['mean_bytes']/1048576:.4f} | "
                f"{100*row['wire_reduction_vs_raw_full']:.2f}% | "
                f"{row['mean_encode_ms']:.3f} | {row['mean_decode_ms']:.3f} | "
                f"{row['mean_recompute_ms']:.3f} |"
            )
    (root / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(root / "results.md")


if __name__ == "__main__":
    main()
