"""Pretrain GSPR with TripleMixer-derived point provenance labels."""

import argparse
import json
import os
from pathlib import Path
import random
import statistics

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from attfuse_gspr.reliability import GeometryFirstPillarReliability
from gspr_supervision.dataset import (GSPRPointDataset,
                                      StratifiedGSPRSampler,
                                      collate_voxel_samples)
from gspr_supervision.losses import BinaryHistogramMetrics, evidential_point_loss


VOXEL_SIZE = [0.4, 0.4, 4.0]
LIDAR_RANGE = [-140.8, -40.0, -3.0, 140.8, 40.0, 1.0]
MODEL_CFG = {
    "hidden_dim": 32, "context_dim": 16,
    "max_points_per_voxel": 32, "intensity_index": 3,
    "ring_index": None, "spectral_downsample": 4,
    "reliability_floor": 0.05, "initial_reliability": 0.95,
    "evidence_cap": 30.0,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", nargs="+", required=True)
    parser.add_argument("--val-manifest", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--init-attfuse-checkpoint", default="")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--max-voxels", type=int, default=32000)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--sampling", choices=("shuffle", "stratified"),
                        default="shuffle")
    parser.add_argument("--noise-bearing-fraction", type=float, default=0.8)
    parser.add_argument("--samples-per-epoch", type=int, default=0)
    return parser.parse_args()


def enforce_output(path):
    output = Path(path).resolve()
    allowed = Path("/data/cjm/datasets").resolve()
    if allowed not in (output, *output.parents):
        raise ValueError("--output-dir must be inside /data/cjm/datasets")
    output.mkdir(parents=True, exist_ok=True)
    return output


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def initialize_from_attfuse(model, checkpoint_path):
    if not checkpoint_path:
        return
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    gspr_state = {key[len("gspr."):]: value for key, value in checkpoint.items()
                  if key.startswith("gspr.")}
    if not gspr_state:
        raise ValueError("checkpoint contains no gspr.* parameters")
    incompatible = model.load_state_dict(gspr_state, strict=False)
    if incompatible.unexpected_keys:
        raise ValueError("unexpected GSPR keys: %s" % incompatible.unexpected_keys)
    print("Initialized GSPR from", checkpoint_path,
          "missing keys:", len(incompatible.missing_keys))


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    metric = BinaryHistogramMetrics()
    group_metrics = {}
    total_loss = total_batches = 0.0
    iterator = tqdm(loader, desc="train" if training else "validate")
    for batch in iterator:
        features = batch["voxel_features"].to(device)
        counts = batch["voxel_num_points"].to(device)
        coords = batch["voxel_coords"].to(device)
        targets = batch["point_targets"].to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = model(features, counts, coords)
            loss, details = evidential_point_loss(output, targets)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        valid = output["point_valid_mask"] & (targets >= 0)
        metric.update(output["point_reliability"][valid], targets[valid])
        voxel_batch = coords[:, 0].long()
        for sample_index, (weather, severity) in enumerate(batch["metadata"]):
            key = "%s/%s" % (weather, severity)
            if key not in group_metrics:
                group_metrics[key] = BinaryHistogramMetrics()
            sample_valid = valid & (voxel_batch == sample_index)[:, None]
            group_metrics[key].update(
                output["point_reliability"][sample_valid],
                targets[sample_valid])
        total_loss += loss.item()
        total_batches += 1
        iterator.set_postfix(loss="%.4f" % loss.item(),
                             noise="%.3f" % details["noise_fraction"].item())
    result = metric.compute()
    result["loss"] = total_loss / max(total_batches, 1)
    computed_groups = {key: value.compute()
                       for key, value in sorted(group_metrics.items())}
    result["groups"] = computed_groups
    result["macro_auroc"] = statistics.mean(
        value["auroc"] for value in computed_groups.values())
    result["macro_best_balanced_accuracy"] = statistics.mean(
        value["best_balanced_accuracy"]
        for value in computed_groups.values())
    return result


def main():
    args = parse_args()
    output = enforce_output(args.output_dir)
    seed_everything(args.seed)
    train_data = GSPRPointDataset(
        args.train_manifest, VOXEL_SIZE, LIDAR_RANGE,
        max_voxels=args.max_voxels, seed=args.seed)
    val_data = GSPRPointDataset(
        args.val_manifest, VOXEL_SIZE, LIDAR_RANGE,
        max_voxels=args.max_voxels, seed=args.seed + 1)
    train_output_points = sum(record["output_points"] for record in
                              train_data.records)
    train_noise_points = sum(record["noise_points"] for record in
                             train_data.records)
    if train_noise_points <= 0 or train_noise_points >= train_output_points:
        raise ValueError(
            "training corpus must contain both reliable and noise points; "
            "got %d noise of %d" %
            (train_noise_points, train_output_points))
    print("Training corpus noise fraction:",
          train_noise_points / train_output_points)
    train_sampler = None
    if args.sampling == "stratified":
        train_sampler = StratifiedGSPRSampler(
            train_data,
            num_samples=(args.samples_per_epoch or len(train_data)),
            noise_bearing_fraction=args.noise_bearing_fraction,
            seed=args.seed)
        print("Stratified sampling groups:",
              json.dumps(train_sampler.group_counts, sort_keys=True))
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size,
        shuffle=train_sampler is None, sampler=train_sampler,
        num_workers=args.num_workers, collate_fn=collate_voxel_samples,
        pin_memory=False, drop_last=False)
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_voxel_samples,
        pin_memory=False, drop_last=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GeometryFirstPillarReliability(
        MODEL_CFG, VOXEL_SIZE, LIDAR_RANGE).to(device)
    initialize_from_attfuse(model, args.init_attfuse_checkpoint)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1.0e-4)
    history = []
    best_macro_auroc = -1.0
    for epoch in range(args.epochs):
        train_data.set_epoch(epoch)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_result = run_epoch(model, train_loader, device, optimizer)
        with torch.no_grad():
            val_result = run_epoch(model, val_loader, device)
        record = {"epoch": epoch + 1, "train": train_result,
                  "validation": val_result}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        checkpoint = {
            "epoch": epoch + 1, "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(), "metrics": record,
            "model_cfg": MODEL_CFG, "voxel_size": VOXEL_SIZE,
            "lidar_range": LIDAR_RANGE,
            "selection_metric": "validation_macro_auroc",
            "sampling": args.sampling,
            "noise_bearing_fraction": args.noise_bearing_fraction,
        }
        torch.save(checkpoint, output / ("gspr_epoch%d.pth" % (epoch + 1)))
        if val_result["macro_auroc"] > best_macro_auroc:
            best_macro_auroc = val_result["macro_auroc"]
            torch.save(checkpoint, output / "gspr_best.pth")
    with (output / "history.json").open("w", encoding="utf-8") as stream:
        json.dump(history, stream, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
