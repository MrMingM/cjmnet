"""Jointly fine-tune AttFuse detection and GSPR point reliability."""

import argparse
import json
import os
import random
import shutil
import statistics
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--gspr-checkpoint", required=True)
    parser.add_argument("--train-manifest", nargs="+", required=True)
    parser.add_argument("--val-manifest", nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--detection-batch-size", type=int, default=2)
    parser.add_argument("--point-batch-size", type=int, default=1)
    parser.add_argument("--max-train-steps", type=int, default=500,
                        help="Per epoch; 0 uses the complete detection loader")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2.0e-5)
    parser.add_argument("--lambda-point", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--point-sampling", choices=("shuffle", "stratified"),
                        default="shuffle")
    parser.add_argument("--noise-bearing-fraction", type=float, default=0.8)
    parser.add_argument("--point-samples-per-epoch", type=int, default=0)
    parser.add_argument("--freeze-detector-epochs", type=int, default=0)
    parser.add_argument("--min-point-macro-auroc", type=float, default=0.99)
    return parser.parse_args()


def configure_imports():
    here = os.path.abspath(os.path.dirname(__file__))
    root = here if os.path.isdir(os.path.join(here, "opencood")) else \
        os.path.abspath(os.path.join(here, "..", "OpenCOOD-main"))
    sys.path[:0] = [root, here]
    import attfuse_gspr.loss as local_loss
    import attfuse_gspr.point_pillar_gspr_attfuse as local_model
    sys.modules["opencood.models.point_pillar_gspr_attfuse"] = local_model
    sys.modules["opencood.loss.gspr_point_pillar_loss"] = local_loss


def load_state(path):
    state = torch.load(path, map_location="cpu")
    return state.get("state_dict", state) if isinstance(state, dict) else state


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def point_validation(model, loader, device, point_loss_fn, metric_class):
    model.eval()
    metric = metric_class()
    group_metrics = {}
    losses = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="point validation"):
            features = batch["voxel_features"].to(device)
            counts = batch["voxel_num_points"].to(device)
            coords = batch["voxel_coords"].to(device)
            targets = batch["point_targets"].to(device)
            output = model.gspr(features, counts, coords)
            loss, _ = point_loss_fn(output, targets)
            valid = output["point_valid_mask"] & (targets >= 0)
            metric.update(output["point_reliability"][valid], targets[valid])
            voxel_batch = coords[:, 0].long()
            for sample_index, (weather, severity) in enumerate(
                    batch["metadata"]):
                key = "%s/%s" % (weather, severity)
                if key not in group_metrics:
                    group_metrics[key] = metric_class()
                sample_valid = valid & \
                    (voxel_batch == sample_index)[:, None]
                group_metrics[key].update(
                    output["point_reliability"][sample_valid],
                    targets[sample_valid])
            losses.append(loss.item())
    result = metric.compute()
    result["loss"] = statistics.mean(losses)
    computed_groups = {key: value.compute()
                       for key, value in sorted(group_metrics.items())}
    result["groups"] = computed_groups
    result["macro_auroc"] = statistics.mean(
        value["auroc"] for value in computed_groups.values())
    result["macro_best_balanced_accuracy"] = statistics.mean(
        value["best_balanced_accuracy"]
        for value in computed_groups.values())
    return result


def configure_detector_stage(model, frozen):
    modules = (model.backbone, model.cls_head, model.reg_head)
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(not frozen)
        if frozen:
            module.eval()


def main():
    args = parse_args()
    configure_imports()
    from opencood.data_utils.datasets import build_dataset
    import opencood.hypes_yaml.yaml_utils as yaml_utils
    from opencood.tools import train_utils
    from gspr_supervision.dataset import (GSPRPointDataset,
                                          StratifiedGSPRSampler,
                                          collate_voxel_samples)
    from gspr_supervision.losses import BinaryHistogramMetrics, evidential_point_loss

    run_dir = os.path.abspath(args.run_dir)
    allowed = os.path.abspath("/data/cjm/datasets")
    if os.path.commonpath((run_dir, allowed)) != allowed:
        raise ValueError("--run-dir must be inside /data/cjm/datasets")
    os.makedirs(run_dir, exist_ok=True)
    shutil.copy2(args.config, os.path.join(run_dir, "config.yaml"))
    seed_everything(args.seed)
    hypes = yaml_utils.load_yaml(args.config)
    hypes["train_params"]["epoches"] = args.epochs
    train_set = build_dataset(hypes, visualize=False, train=True)
    val_set = build_dataset(hypes, visualize=False, train=False)
    train_loader = DataLoader(
        train_set, batch_size=args.detection_batch_size,
        shuffle=True, num_workers=args.num_workers,
        collate_fn=train_set.collate_batch_train, drop_last=True)
    val_loader = DataLoader(
        val_set, batch_size=args.detection_batch_size,
        shuffle=False, num_workers=args.num_workers,
        collate_fn=val_set.collate_batch_train, drop_last=False)
    point_train_set = GSPRPointDataset(
        args.train_manifest, hypes["model"]["args"]["voxel_size"],
        hypes["model"]["args"]["lidar_range"], seed=args.seed)
    point_val_set = GSPRPointDataset(
        args.val_manifest, hypes["model"]["args"]["voxel_size"],
        hypes["model"]["args"]["lidar_range"], seed=args.seed + 1)
    point_sampler = None
    if args.point_sampling == "stratified":
        point_sampler = StratifiedGSPRSampler(
            point_train_set,
            num_samples=(args.point_samples_per_epoch or
                         len(point_train_set)),
            noise_bearing_fraction=args.noise_bearing_fraction,
            seed=args.seed)
        print("Stratified point sampling groups:",
              json.dumps(point_sampler.group_counts, sort_keys=True))
    point_train_loader = DataLoader(
        point_train_set, batch_size=args.point_batch_size,
        shuffle=point_sampler is None, sampler=point_sampler,
        num_workers=args.num_workers, collate_fn=collate_voxel_samples)
    point_val_loader = DataLoader(
        point_val_set, batch_size=args.point_batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_voxel_samples)

    model = train_utils.create_model(hypes)
    model.load_state_dict(load_state(args.baseline_checkpoint), strict=False)
    point_state = load_state(args.gspr_checkpoint)
    point_state = {(key[5:] if key.startswith("gspr.") else key): value
                   for key, value in point_state.items()}
    incompatible = model.gspr.load_state_dict(point_state, strict=False)
    if incompatible.unexpected_keys:
        raise ValueError("unexpected GSPR keys: %s" % incompatible.unexpected_keys)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = train_utils.create_loss(hypes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1.0e-4)
    history = []
    best_detection_loss = float("inf")
    best_record = None

    for epoch in range(args.epochs):
        train_set.set_weather_augmentation_epoch(epoch) if hasattr(
            train_set, "set_weather_augmentation_epoch") else None
        point_train_set.set_epoch(epoch)
        if point_sampler is not None:
            point_sampler.set_epoch(epoch)
        detector_frozen = epoch < args.freeze_detector_epochs
        model.train()
        configure_detector_stage(model, detector_frozen)
        point_iterator = iter(point_train_loader)
        detection_losses, point_losses = [], []
        progress = tqdm(train_loader, desc="joint epoch %d" % (epoch + 1))
        for step, detection_batch in enumerate(progress):
            if args.max_train_steps > 0 and step >= args.max_train_steps:
                break
            try:
                point_batch = next(point_iterator)
            except StopIteration:
                point_iterator = iter(point_train_loader)
                point_batch = next(point_iterator)
            detection_batch = train_utils.to_device(detection_batch, device)
            point_features = point_batch["voxel_features"].to(device)
            point_counts = point_batch["voxel_num_points"].to(device)
            point_coords = point_batch["voxel_coords"].to(device)
            point_targets = point_batch["point_targets"].to(device)
            optimizer.zero_grad(set_to_none=True)
            detection_output = model(detection_batch["ego"])
            detection_loss = criterion(
                detection_output, detection_batch["ego"]["label_dict"])
            point_output = model.gspr(
                point_features, point_counts, point_coords)
            point_loss, _ = evidential_point_loss(
                point_output, point_targets)
            total = detection_loss + args.lambda_point * point_loss
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            detection_losses.append(detection_loss.item())
            point_losses.append(point_loss.item())
            progress.set_postfix(det="%.3f" % detection_loss.item(),
                                 point="%.3f" % point_loss.item())

        model.eval()
        val_detection_losses = []
        with torch.no_grad():
            for detection_batch in tqdm(val_loader, desc="detection validation"):
                detection_batch = train_utils.to_device(detection_batch, device)
                output = model(detection_batch["ego"])
                loss = criterion(output, detection_batch["ego"]["label_dict"])
                val_detection_losses.append(loss.item())
        point_result = point_validation(
            model, point_val_loader, device, evidential_point_loss,
            BinaryHistogramMetrics)
        record = {
            "epoch": epoch + 1,
            "detector_frozen": detector_frozen,
            "train_detection_loss": statistics.mean(detection_losses),
            "train_point_loss": statistics.mean(point_losses),
            "validation_detection_loss": statistics.mean(val_detection_losses),
            "validation_point": point_result,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        torch.save(model.state_dict(), os.path.join(
            run_dir, "net_epoch%d.pth" % (epoch + 1)))
        point_eligible = point_result["macro_auroc"] >= \
            args.min_point_macro_auroc
        if point_eligible and record["validation_detection_loss"] < \
                best_detection_loss:
            best_detection_loss = record["validation_detection_loss"]
            best_record = {
                "epoch": epoch + 1,
                "validation_detection_loss": best_detection_loss,
                "validation_point_macro_auroc":
                    point_result["macro_auroc"],
                "min_point_macro_auroc": args.min_point_macro_auroc,
            }
            torch.save(model.state_dict(), os.path.join(
                run_dir, "net_best_validation.pth"))
            with open(os.path.join(run_dir, "best_selection.json"), "w",
                      encoding="utf-8") as stream:
                json.dump(best_record, stream, indent=2,
                          ensure_ascii=False)
        with open(os.path.join(run_dir, "history.json"), "w",
                  encoding="utf-8") as stream:
            json.dump(history, stream, indent=2, ensure_ascii=False)
    if best_record is None:
        raise RuntimeError(
            "no checkpoint met --min-point-macro-auroc %.6f" %
            args.min_point_macro_auroc)


if __name__ == "__main__":
    main()
