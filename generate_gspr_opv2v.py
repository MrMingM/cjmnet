"""Generate a compact point-supervision corpus from OPV2V with TripleMixer."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from gspr_supervision.triplemixer_adapter import TripleMixerWeatherAdapter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--triplemixer-root", required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--max-scenarios", type=int, default=4)
    parser.add_argument("--max-files-per-scenario", type=int, default=24)
    parser.add_argument("--weathers", nargs="+",
                        choices=("fog", "rain", "snow"),
                        default=("fog", "rain", "snow"))
    parser.add_argument("--severities", nargs="+",
                        choices=("light", "moderate", "heavy"),
                        default=("light", "moderate", "heavy"))
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def enforce_output_root(path):
    resolved = Path(path).resolve()
    allowed = Path("/data/cjm/datasets").resolve()
    if allowed not in (resolved, *resolved.parents):
        raise ValueError("--output-root must be inside /data/cjm/datasets")
    return resolved


def stable_seed(base_seed, *parts):
    text = "|".join((str(base_seed),) + tuple(str(part) for part in parts))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little")


def select_files(source_root, max_scenarios, max_files):
    scenarios = sorted(path for path in source_root.iterdir() if path.is_dir())
    selected = []
    for scenario in scenarios[:max_scenarios]:
        files = sorted(scenario.rglob("*.pcd"))
        if max_files > 0 and len(files) > max_files:
            indices = np.linspace(0, len(files) - 1, max_files, dtype=int)
            files = [files[index] for index in indices]
        selected.extend(files)
    return selected


def prepare_opv2v_points(points):
    lidar_range = np.asarray(
        [-140.8, -40.0, -3.0, 140.8, 40.0, 1.0], dtype=np.float32)
    in_range = np.all((points[:, :3] > lidar_range[:3]) &
                      (points[:, :3] < lidar_range[3:]), axis=1)
    ego = ((points[:, 0] >= -1.95) & (points[:, 0] <= 2.95) &
           (points[:, 1] >= -1.1) & (points[:, 1] <= 1.1))
    points = np.asarray(points[in_range & ~ego], dtype=np.float32)
    points[:, 3] = np.clip(points[:, 3], 0.0, 1.0)
    return points


def load_pcd_points(pcd_to_np, source_path):
    """Normalize pcd_to_np return values across OpenCOOD revisions."""
    loaded = pcd_to_np(str(source_path))
    if isinstance(loaded, np.ndarray):
        points = loaded
    elif isinstance(loaded, (tuple, list)):
        candidates = [
            item for item in loaded
            if isinstance(item, np.ndarray) and item.ndim == 2 and
            item.shape[1] >= 4
        ]
        if not candidates:
            raise ValueError(
                "pcd_to_np returned a tuple/list without an [N,4+] array "
                "for %s" % source_path)
        points = candidates[-1]
    else:
        raise TypeError(
            "Unsupported pcd_to_np return type %s for %s" %
            (type(loaded).__name__, source_path))

    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 4:
        raise ValueError(
            "Expected pcd_to_np output with shape [N,4+], got %s for %s" %
            (points.shape, source_path))
    return np.ascontiguousarray(points[:, :4])


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def existing_manifest_paths(manifest_path):
    if not manifest_path.is_file():
        return set()
    paths = set()
    with manifest_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                paths.add(str(Path(record["path"]).resolve()))
            except (json.JSONDecodeError, KeyError) as error:
                raise ValueError(
                    "Invalid manifest record at %s:%d" %
                    (manifest_path, line_number)) from error
    return paths


def recover_existing_record(destination, source_path, split, weather,
                            severity, seed, input_count):
    """Rebuild a missing manifest row without rerunning weather simulation."""
    with np.load(destination, allow_pickle=False) as data:
        required = {"points", "reliability", "noise_type", "source_index"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError("%s misses arrays %s" %
                             (destination, sorted(missing)))
        points = data["points"]
        reliability = data["reliability"]
        noise_type = data["noise_type"]
        source_index = data["source_index"]
        output_count = len(points)
        if points.ndim != 2 or points.shape[1] != 4:
            raise ValueError("invalid points shape %s in %s" %
                             (points.shape, destination))
        if reliability.shape != (output_count,) or \
                noise_type.shape != (output_count,) or \
                source_index.shape != (output_count,):
            raise ValueError("point-label arrays are not aligned in %s" %
                             destination)
        if not np.isfinite(points).all() or \
                not np.isin(reliability, (0.0, 1.0)).all():
            raise ValueError("invalid point values or labels in %s" %
                             destination)
        noise_count = int((reliability < 0.5).sum())
    if weather == "fog":
        lookup, alpha, beta = TripleMixerWeatherAdapter.FOG_LEVELS[severity]
        simulator_info = {
            "backend": "triplemixer_fog_simulation",
            "lookup": lookup, "alpha": alpha, "beta": beta,
            "manifest_recovered": True,
        }
    else:
        rates = (TripleMixerWeatherAdapter.RAIN_LEVELS if weather == "rain"
                 else TripleMixerWeatherAdapter.SNOW_LEVELS)
        simulator_info = {
            "backend": ("triplemixer_lisa_rain" if weather == "rain" else
                        "triplemixer_lisa_snow_branch"),
            "rate": float(rates[severity]),
            "manifest_recovered": True,
        }
    return {
        "path": str(destination),
        "source": str(source_path),
        "split": split,
        "weather": weather,
        "severity": severity,
        "seed": seed,
        "input_points": int(input_count),
        "output_points": int(output_count),
        "noise_points": noise_count,
        "lost_points": int(input_count - output_count),
        "simulator_info": simulator_info,
    }


def main():
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    output_root = enforce_output_root(args.output_root)
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Defer Open3D import until arguments and paths have been validated.
    from opencood.utils.pcd_utils import pcd_to_np
    adapter = TripleMixerWeatherAdapter(args.triplemixer_root)
    files = select_files(source_root, args.max_scenarios,
                         args.max_files_per_scenario)
    if not files:
        raise RuntimeError("No PCD files found under %s" % source_root)
    manifest_path = output_root / (args.split + "_manifest.jsonl")
    recorded_paths = (set() if args.overwrite else
                      existing_manifest_paths(manifest_path))
    mode = "w" if args.overwrite else "a"
    generated = recovered = skipped = 0
    with manifest_path.open(mode, encoding="utf-8") as manifest:
        for source_path in tqdm(files, desc="OPV2V source point clouds"):
            relative = source_path.relative_to(source_root)
            points = load_pcd_points(pcd_to_np, source_path)
            points = prepare_opv2v_points(points)
            for weather in args.weathers:
                for severity in args.severities:
                    destination = (output_root / args.split / weather /
                                   severity / relative).with_suffix(".npz")
                    if destination.exists() and not args.overwrite:
                        resolved_destination = str(destination.resolve())
                        if resolved_destination in recorded_paths:
                            skipped += 1
                            continue
                        seed = stable_seed(args.seed, args.split, relative,
                                           weather, severity)
                        record = recover_existing_record(
                            destination, source_path, args.split, weather,
                            severity, seed, len(points))
                        manifest.write(
                            json.dumps(record, ensure_ascii=False) + "\n")
                        manifest.flush()
                        recorded_paths.add(resolved_destination)
                        recovered += 1
                        continue
                    seed = stable_seed(args.seed, args.split, relative,
                                       weather, severity)
                    result = adapter.simulate(points, weather, severity, seed)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        destination,
                        points=result["points"],
                        reliability=result["reliability"],
                        noise_type=result["noise_type"],
                        source_index=result["source_index"])
                    record = {
                        "path": str(destination),
                        "source": str(source_path),
                        "split": args.split,
                        "weather": weather,
                        "severity": severity,
                        "seed": seed,
                        "input_points": int(len(points)),
                        "output_points": int(len(result["points"])),
                        "noise_points": int(
                            (result["reliability"] < 0.5).sum()),
                        "lost_points": int(result["lost_count"]),
                        "simulator_info": json_safe(result["simulator_info"]),
                    }
                    manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                    manifest.flush()
                    recorded_paths.add(str(destination.resolve()))
                    generated += 1
    summary = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "split": args.split,
        "source_files": len(files),
        "generated": generated,
        "recovered": recovered,
        "skipped": skipped,
        "weathers": list(args.weathers),
        "severities": list(args.severities),
        "manifest": str(manifest_path),
    }
    with (output_root / (args.split + "_generation_summary.json")).open(
            "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
