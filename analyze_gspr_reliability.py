"""Streaming diagnostics for GSPR point reliability on an OpenCOOD split."""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True,
                        help="Directory containing config.yaml and checkpoint")
    parser.add_argument("--output", default="",
                        help="JSON output; defaults inside --model-dir")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=0,
                        help="0 scans the complete split")
    parser.add_argument("--hist-bins", type=int, default=1000)
    return parser.parse_args()


def configure_imports():
    here = os.path.abspath(os.path.dirname(__file__))
    root = here if os.path.isdir(os.path.join(here, "opencood")) else \
        os.path.abspath(os.path.join(here, "..", "OpenCOOD-main"))
    sys.path[:0] = [root, here]
    import attfuse_gspr.point_pillar_gspr_attfuse as local_model
    sys.modules["opencood.models.point_pillar_gspr_attfuse"] = local_model


class ScalarStats:
    def __init__(self, hist_bins=1000, low=0.0, high=1.0):
        self.count = 0
        self.total = 0.0
        self.total_sq = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf
        self.low = float(low)
        self.high = float(high)
        self.hist = np.zeros(hist_bins, dtype=np.int64)

    def update(self, value):
        value = value.detach().float().reshape(-1)
        if value.numel() == 0:
            return
        self.count += value.numel()
        self.total += value.double().sum().item()
        self.total_sq += value.double().square().sum().item()
        self.minimum = min(self.minimum, value.min().item())
        self.maximum = max(self.maximum, value.max().item())
        clipped = value.clamp(self.low, self.high)
        index = ((clipped - self.low) / max(self.high - self.low, 1e-12) *
                 (len(self.hist) - 1)).long()
        self.hist += torch.bincount(
            index.cpu(), minlength=len(self.hist)).numpy()

    def quantile(self, probability):
        if self.count == 0:
            return None
        target = max(1, int(math.ceil(probability * self.count)))
        index = int(np.searchsorted(np.cumsum(self.hist), target))
        return self.low + index / max(len(self.hist) - 1, 1) * \
            (self.high - self.low)

    def result(self):
        if self.count == 0:
            return {"count": 0}
        mean = self.total / self.count
        variance = max(self.total_sq / self.count - mean * mean, 0.0)
        return {
            "count": self.count,
            "mean": mean,
            "std": math.sqrt(variance),
            "min": self.minimum,
            "q01": self.quantile(0.01),
            "q05": self.quantile(0.05),
            "q25": self.quantile(0.25),
            "median": self.quantile(0.50),
            "q75": self.quantile(0.75),
            "q95": self.quantile(0.95),
            "q99": self.quantile(0.99),
            "max": self.maximum,
        }


class CorrelationStats:
    def __init__(self):
        self.n = 0
        self.sx = self.sy = 0.0
        self.sxx = self.syy = self.sxy = 0.0

    def update(self, x, y):
        x = x.detach().double().reshape(-1)
        y = y.detach().double().reshape(-1)
        if x.numel() != y.numel():
            raise ValueError("correlation inputs must have equal length")
        self.n += x.numel()
        self.sx += x.sum().item()
        self.sy += y.sum().item()
        self.sxx += x.square().sum().item()
        self.syy += y.square().sum().item()
        self.sxy += (x * y).sum().item()

    def result(self):
        if self.n < 2:
            return None
        covariance = self.sxy - self.sx * self.sy / self.n
        var_x = self.sxx - self.sx * self.sx / self.n
        var_y = self.syy - self.sy * self.sy / self.n
        denominator = math.sqrt(max(var_x * var_y, 0.0))
        return covariance / denominator if denominator > 0 else None


class BinnedStats:
    def __init__(self, edges):
        self.edges = list(edges)
        self.count = np.zeros(len(edges) - 1, dtype=np.int64)
        self.rel_sum = np.zeros(len(edges) - 1, dtype=np.float64)
        self.unc_sum = np.zeros(len(edges) - 1, dtype=np.float64)

    def update(self, value, reliability, uncertainty):
        value = value.detach().float().reshape(-1)
        reliability = reliability.detach().float().reshape(-1)
        uncertainty = uncertainty.detach().float().reshape(-1)
        boundaries = value.new_tensor(self.edges[1:-1])
        indices = torch.bucketize(value, boundaries)
        for index in range(len(self.count)):
            mask = indices == index
            if mask.any():
                self.count[index] += mask.sum().item()
                self.rel_sum[index] += reliability[mask].double().sum().item()
                self.unc_sum[index] += uncertainty[mask].double().sum().item()

    def result(self):
        output = []
        for index, count in enumerate(self.count):
            output.append({
                "lower": self.edges[index],
                "upper": self.edges[index + 1],
                "count": int(count),
                "reliability_mean": (self.rel_sum[index] / count
                                     if count else None),
                "uncertainty_mean": (self.unc_sum[index] / count
                                     if count else None),
            })
        return output


def safe_output_path(model_dir, requested):
    output = requested or os.path.join(
        model_dir, "gspr_reliability_stats.json")
    output = os.path.abspath(output)
    allowed = os.path.abspath("/data/cjm/datasets")
    if os.path.commonpath((output, allowed)) != allowed:
        raise ValueError("diagnostic output must be inside /data/cjm/datasets")
    return output


def mean_pair(evidence, valid):
    selected = evidence[valid]
    if selected.numel() == 0:
        return [None, None]
    return selected.double().mean(dim=0).cpu().tolist()


def main():
    args = parse_args()
    configure_imports()
    from opencood.data_utils.datasets import build_dataset
    import opencood.hypes_yaml.yaml_utils as yaml_utils
    from opencood.tools import train_utils

    output_path = safe_output_path(args.model_dir, args.output)
    option = argparse.Namespace(model_dir=args.model_dir)
    hypes = yaml_utils.load_yaml(None, option)
    dataset = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(
        dataset, batch_size=1, num_workers=args.num_workers,
        collate_fn=dataset.collate_batch_test, shuffle=False,
        pin_memory=False, drop_last=False)
    model = train_utils.create_model(hypes)
    _, model = train_utils.load_saved_model(args.model_dir, model)
    model.gspr.return_branch_support = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    reliability_stats = ScalarStats(args.hist_bins)
    uncertainty_stats = ScalarStats(args.hist_bins)
    intensity_scalar_stats = ScalarStats(
        args.hist_bins, low=0.0, high=256.0)
    range_stats = BinnedStats([0, 20, 40, 60, 80, 120, 200])
    intensity_stats = BinnedStats(
        [0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.7, 1.01,
         2, 5, 10, 25, 50, 100, 256, 1.0e6])
    density_stats = BinnedStats([0, 0.125, 0.25, 0.5, 0.75, 1.01])
    correlations = {
        "range": CorrelationStats(),
        "intensity": CorrelationStats(),
        "density": CorrelationStats(),
    }
    below = {threshold: 0 for threshold in (0.3, 0.5, 0.7, 0.9)}
    branch_sum = {name: np.zeros(2, dtype=np.float64) for name in
                  ("geometry", "radiometric", "physics", "fused")}
    branch_count = 0
    batches = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="GSPR reliability audit"):
            if batch is None:
                continue
            batch = train_utils.to_device(batch, device)
            ego = batch["ego"]
            prediction = model(ego)
            valid = prediction["point_valid_mask"]
            reliability = prediction["point_reliability"][valid]
            uncertainty = prediction["point_uncertainty"][valid]
            processed = ego["processed_lidar"]
            point_slots = processed["voxel_features"]
            ranges = torch.linalg.vector_norm(point_slots[..., :3], dim=-1)[valid]
            intensity = point_slots[..., 3][valid]
            density = (processed["voxel_num_points"].float() /
                       point_slots.shape[1])[:, None].expand(valid.shape)[valid]

            reliability_stats.update(reliability)
            uncertainty_stats.update(uncertainty)
            intensity_scalar_stats.update(intensity)
            range_stats.update(ranges, reliability, uncertainty)
            intensity_stats.update(intensity, reliability, uncertainty)
            density_stats.update(density, reliability, uncertainty)
            correlations["range"].update(ranges, reliability)
            correlations["intensity"].update(intensity, reliability)
            correlations["density"].update(density, reliability)
            for threshold in below:
                below[threshold] += (reliability < threshold).sum().item()

            pairs = {
                "geometry": mean_pair(prediction["geometry_support"], valid),
                "radiometric": mean_pair(
                    prediction["radiometric_support"], valid),
                "physics": mean_pair(prediction["physics_support"], valid),
                "fused": mean_pair(prediction["point_evidence"], valid),
            }
            current_count = reliability.numel()
            for name, pair in pairs.items():
                branch_sum[name] += np.asarray(pair) * current_count
            branch_count += current_count
            batches += 1
            if args.max_batches > 0 and batches >= args.max_batches:
                break

    point_count = reliability_stats.count
    result = {
        "model_dir": os.path.abspath(args.model_dir),
        "condition": os.path.basename(os.path.abspath(args.model_dir)),
        "validate_dir": hypes.get("validate_dir"),
        "dataset_samples": len(dataset),
        "processed_batches": batches,
        "complete_split": args.max_batches == 0 or batches >= len(loader),
        "scope": "valid points retained in voxel slots after range filtering and truncation",
        "reliability": reliability_stats.result(),
        "uncertainty": uncertainty_stats.result(),
        "raw_intensity": intensity_scalar_stats.result(),
        "fraction_below": {
            str(threshold): below[threshold] / max(point_count, 1)
            for threshold in below},
        "pearson_reliability_correlation": {
            name: stats.result() for name, stats in correlations.items()},
        "by_range_m": range_stats.result(),
        "by_raw_intensity": intensity_stats.result(),
        "by_pillar_density": density_stats.result(),
        "mean_positive_branch_support_reliable_noise": {
            name: (values / max(branch_count, 1)).tolist()
            for name, values in branch_sum.items() if name != "fused"},
        "mean_fused_evidence_reliable_noise": (
            branch_sum["fused"] / max(branch_count, 1)).tolist(),
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
    print("Wrote", output_path)


if __name__ == "__main__":
    main()
