import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d
from tqdm import tqdm


DEFAULT_SPLITS = ("train", "validate", "test")


def generate_uncertainty_map(pcd_path: Path, config: dict) -> Optional[np.ndarray]:
    """Generate the density-derived uncertainty target for a single point cloud."""
    try:
        pcd = o3d.io.read_point_cloud(str(pcd_path))
        points = np.asarray(pcd.points)
    except Exception as exc:
        print(f"\n[Error] Failed to read {pcd_path}: {exc}")
        return None

    x_range = config["bev_range_x"]
    y_range = config["bev_range_y"]

    mask_x = (points[:, 0] >= x_range[0]) & (points[:, 0] < x_range[1])
    mask_y = (points[:, 1] >= y_range[0]) & (points[:, 1] < y_range[1])
    bev_points = points[mask_x & mask_y]

    if bev_points.shape[0] == 0:
        return np.ones((config["bev_h"], config["bev_w"]), dtype=np.float32)

    resolution = config["bev_resolution"]
    x_indices = ((bev_points[:, 0] - x_range[0]) / resolution).astype(np.int32)
    y_indices = ((bev_points[:, 1] - y_range[0]) / resolution).astype(np.int32)

    density_map = np.zeros((config["bev_h"], config["bev_w"]), dtype=np.float32)
    np.add.at(density_map, (y_indices, x_indices), 1)

    density_map = np.clip(density_map, 0, config["max_points_per_cell"])
    normalized_density = density_map / config["max_points_per_cell"]
    gt_umap = 1.0 - normalized_density

    return gt_umap.astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate UECP uncertainty maps for V2V4Real-style folders."
    )
    parser.add_argument(
        "--data-root",
        default="dataset/V2V4REAL",
        help="Dataset root that contains train/validate/test split folders.",
    )
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument("--bev-range-x", nargs=2, type=float, default=(-102.4, 102.4))
    parser.add_argument("--bev-range-y", nargs=2, type=float, default=(-51.2, 51.2))
    parser.add_argument("--bev-resolution", type=float, default=0.4)
    parser.add_argument("--max-points-per-cell", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    bev_range_x = list(args.bev_range_x)
    bev_range_y = list(args.bev_range_y)

    bev_h = int((bev_range_y[1] - bev_range_y[0]) / args.bev_resolution)
    bev_w = int((bev_range_x[1] - bev_range_x[0]) / args.bev_resolution)
    config = {
        "bev_range_x": bev_range_x,
        "bev_range_y": bev_range_y,
        "bev_resolution": args.bev_resolution,
        "bev_h": bev_h,
        "bev_w": bev_w,
        "max_points_per_cell": args.max_points_per_cell,
    }

    print("--- Uncertainty Map Generation Script (Local Save Mode) ---")
    print(f"BEV Grid Size for GT: {bev_h} (H) x {bev_w} (W)")
    print(f"Processing data within: {data_root}\n")

    for split in args.splits:
        split_path = data_root / split
        if not split_path.is_dir():
            continue

        print(f"Processing split: {split}")
        scenario_folders = sorted(
            d for d in os.listdir(split_path) if (split_path / d).is_dir()
        )
        for scenario in scenario_folders:
            scenario_path = split_path / scenario
            agent_folders = sorted(
                d for d in os.listdir(scenario_path) if (scenario_path / d).is_dir()
            )
            for agent_id in agent_folders:
                agent_path = scenario_path / agent_id
                pcd_files = sorted(f for f in os.listdir(agent_path) if f.endswith(".pcd"))
                if not pcd_files:
                    continue

                print(f"  -> Processing and overwriting {len(pcd_files)} files in: {agent_path}")
                for pcd_file in tqdm(
                    pcd_files, desc=f"    {split}/{scenario}/{agent_id}", leave=False
                ):
                    pcd_path = agent_path / pcd_file
                    gt_umap = generate_uncertainty_map(pcd_path, config)

                    if gt_umap is not None:
                        np.save(pcd_path.with_suffix(".npy"), gt_umap)

    print("\n--- All tasks completed! ---")


if __name__ == "__main__":
    main()
