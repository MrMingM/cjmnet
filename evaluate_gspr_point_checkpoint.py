"""Audit discrimination and calibration of a point-supervised GSPR checkpoint."""

import argparse
import json

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from attfuse_gspr.reliability import GeometryFirstPillarReliability
from gspr_supervision.dataset import GSPRPointDataset, collate_voxel_samples
from gspr_supervision.losses import BinaryHistogramMetrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", nargs="+", required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-voxels", type=int, default=32000)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    model_cfg = checkpoint["model_cfg"]
    voxel_size = checkpoint["voxel_size"]
    lidar_range = checkpoint["lidar_range"]

    dataset = GSPRPointDataset(
        args.manifest, voxel_size, lidar_range,
        max_voxels=args.max_voxels, seed=args.seed)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
        collate_fn=collate_voxel_samples, pin_memory=False)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device requests CUDA but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    model = GeometryFirstPillarReliability(
        model_cfg, voxel_size, lidar_range).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    metrics = {"overall": BinaryHistogramMetrics()}
    with torch.inference_mode():
        for batch in tqdm(loader, desc="GSPR calibration audit"):
            output = model(
                batch["voxel_features"].to(device),
                batch["voxel_num_points"].to(device),
                batch["voxel_coords"].to(device))
            targets = batch["point_targets"].to(device)
            valid = output["point_valid_mask"] & (targets >= 0)
            scores = output["point_reliability"][valid]
            labels = targets[valid]
            weather = batch["metadata"][0][0]
            if weather not in metrics:
                metrics[weather] = BinaryHistogramMetrics()
            metrics["overall"].update(scores, labels)
            metrics[weather].update(scores, labels)

    report = {
        "checkpoint": args.checkpoint,
        "samples": len(dataset),
        "interpretation": {
            "positive": "reliable point",
            "negative": "weather noise/replacement point",
            "decision": "score >= threshold means reliable",
        },
        "metrics": {name: metric.compute()
                    for name, metric in metrics.items()},
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
