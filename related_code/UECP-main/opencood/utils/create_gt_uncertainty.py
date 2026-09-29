import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d
from tqdm import tqdm


DEFAULT_DIRECTORIES = (
    "infrastructure-side/velodyne",
    "vehicle-side/velodyne",
)


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
        description="Generate UECP uncertainty maps for DAIR-V2X-C-style folders."
    )
    parser.add_argument(
        "--data-root",
        default="dataset/DAIR-V2X-C",
        help="Dataset root that contains infrastructure-side/ and vehicle-side/.",
    )
    parser.add_argument(
        "--directories",
        nargs="+",
        default=list(DEFAULT_DIRECTORIES),
        help="Point-cloud directories relative to --data-root.",
    )
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

    for dir_path in args.directories:
        input_dir = data_root / dir_path
        output_dir = input_dir.parent / "gt_uncertainty_maps"

        if not input_dir.is_dir():
            print(f"[Warning] Input directory not found, skipping: {input_dir}")
            continue

        print(f"Processing directory: {input_dir}")
        print(f"  -> Output will be saved to: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        pcd_files = [f for f in os.listdir(input_dir) if f.endswith(".pcd")]
        if not pcd_files:
            print("  No .pcd files found in this directory.")
            continue

        for pcd_file in tqdm(pcd_files, desc=f"  {dir_path}"):
            pcd_path = input_dir / pcd_file
            gt_umap = generate_uncertainty_map(pcd_path, config)

            if gt_umap is not None:
                output_path = output_dir / pcd_file.replace(".pcd", ".npy")
                np.save(output_path, gt_umap)

    print("\n--- All tasks completed! ---")


if __name__ == "__main__":
    main()
