"""Summarize and optionally verify a generated GSPR point-label corpus."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--splits", nargs="+", default=("train", "val"))
    parser.add_argument("--verify-npz", action="store_true")
    parser.add_argument("--json-output", default="")
    return parser.parse_args()


def verify_npz(record):
    path = Path(record["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        required = {"points", "reliability", "noise_type", "source_index"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError("%s misses arrays %s" % (path, sorted(missing)))
        points = data["points"]
        reliability = data["reliability"]
        noise_type = data["noise_type"]
        source_index = data["source_index"]
        count = len(points)
        if points.ndim != 2 or points.shape[1] != 4:
            raise ValueError("invalid points shape %s in %s" %
                             (points.shape, path))
        for name, array in (("reliability", reliability),
                            ("noise_type", noise_type),
                            ("source_index", source_index)):
            if array.shape != (count,):
                raise ValueError("invalid %s shape %s in %s" %
                                 (name, array.shape, path))
        if not np.isfinite(points).all():
            raise ValueError("non-finite point in %s" % path)
        if not np.isin(reliability, (0.0, 1.0)).all():
            raise ValueError("non-binary reliability in %s" % path)
        noise = int((reliability < 0.5).sum())
        if count != int(record["output_points"]):
            raise ValueError("output point count mismatch in %s" % path)
        if noise != int(record["noise_points"]):
            raise ValueError("noise point count mismatch in %s" % path)
        if not np.all(source_index[reliability < 0.5] == -1):
            raise ValueError("noise source_index must be -1 in %s" % path)


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    groups = defaultdict(lambda: {
        "samples": 0, "sources": set(), "input_points": 0,
        "output_points": 0, "noise_points": 0, "lost_points": 0,
        "zero_noise_samples": 0, "noise_per_sample": [],
    })
    seen_paths = set()
    split_summary = {}
    for split in args.splits:
        manifest = root / (split + "_manifest.jsonl")
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        records = [json.loads(line) for line in
                   manifest.read_text(encoding="utf-8").splitlines()
                   if line.strip()]
        if not records:
            raise RuntimeError(
                "empty manifest: %s; if NPZ files already exist, rerun "
                "generate_gspr_opv2v.py with the original selection arguments "
                "and WITHOUT --overwrite to rebuild it" % manifest)
        duplicate_count = 0
        for record in records:
            path = str(Path(record["path"]).resolve())
            if path in seen_paths:
                duplicate_count += 1
            seen_paths.add(path)
            if args.verify_npz:
                verify_npz(record)
            key = (split, record["weather"], record["severity"])
            group = groups[key]
            noise = int(record["noise_points"])
            group["samples"] += 1
            group["sources"].add(record["source"])
            group["input_points"] += int(record["input_points"])
            group["output_points"] += int(record["output_points"])
            group["noise_points"] += noise
            group["lost_points"] += int(record["lost_points"])
            group["zero_noise_samples"] += int(noise == 0)
            group["noise_per_sample"].append(noise)
        split_summary[split] = {
            "manifest_records": len(records),
            "unique_sources": len({record["source"] for record in records}),
            "duplicate_paths": duplicate_count,
        }

    rows = []
    for (split, weather, severity), group in sorted(groups.items()):
        noise_array = np.asarray(group.pop("noise_per_sample"), dtype=np.int64)
        sources = group.pop("sources")
        output_points = group["output_points"]
        input_points = group["input_points"]
        row = {
            "split": split, "weather": weather, "severity": severity,
            **group,
            "unique_sources": len(sources),
            "noise_fraction": (group["noise_points"] / output_points
                               if output_points else 0.0),
            "lost_fraction": (group["lost_points"] / input_points
                              if input_points else 0.0),
            "zero_noise_fraction":
                group["zero_noise_samples"] / group["samples"],
            "noise_per_sample_q50": float(np.quantile(noise_array, 0.50)),
            "noise_per_sample_q95": float(np.quantile(noise_array, 0.95)),
            "noise_per_sample_max": int(noise_array.max()),
        }
        rows.append(row)

    report = {
        "root": str(root), "verified_npz": bool(args.verify_npz),
        "splits": split_summary, "groups": rows,
    }
    header = ("split\tweather\tseverity\tsamples\tnoise_points\tnoise_frac\t"
              "zero_noise\tzero_frac\tlost_points\tnoise_q50\tnoise_q95\t"
              "noise_max")
    print(header)
    for row in rows:
        print("{split}\t{weather}\t{severity}\t{samples}\t{noise_points}\t"
              "{noise_fraction:.8f}\t{zero_noise_samples}\t"
              "{zero_noise_fraction:.4f}\t{lost_points}\t"
              "{noise_per_sample_q50:.1f}\t{noise_per_sample_q95:.1f}\t"
              "{noise_per_sample_max}".format(**row))
    print("\nSplit summary:")
    print(json.dumps(split_summary, indent=2, ensure_ascii=False))
    if args.json_output:
        output = Path(args.json_output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                          encoding="utf-8")
        print("Saved JSON:", output)


if __name__ == "__main__":
    main()
