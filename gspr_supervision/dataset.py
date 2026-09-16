"""NPZ point-label dataset and deterministic PointPillar slot alignment."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


def voxelize_with_labels(points, targets, voxel_size, lidar_range,
                         max_points=32, max_voxels=32000, seed=0):
    points = np.asarray(points, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32).reshape(-1)
    if len(points) != len(targets):
        raise ValueError("point and target counts differ")
    lower = np.asarray(lidar_range[:3], dtype=np.float32)
    upper = np.asarray(lidar_range[3:], dtype=np.float32)
    size = np.asarray(voxel_size, dtype=np.float32)
    keep = np.all((points[:, :3] > lower) & (points[:, :3] < upper), axis=1)
    points, targets = points[keep], targets[keep]
    if not len(points):
        raise ValueError("sample has no point inside the LiDAR range")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(points))
    points, targets = points[order], targets[order]
    xyz_coord = np.floor((points[:, :3] - lower) / size).astype(np.int32)
    grid = np.rint((upper - lower) / size).astype(np.int64)
    key = (xyz_coord[:, 0].astype(np.int64) +
           xyz_coord[:, 1].astype(np.int64) * grid[0] +
           xyz_coord[:, 2].astype(np.int64) * grid[0] * grid[1])
    sorted_index = np.argsort(key, kind="stable")
    sorted_key = key[sorted_index]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_key)) + 1]
    ends = np.r_[starts[1:], len(sorted_key)]
    if len(starts) > max_voxels:
        chosen = rng.choice(len(starts), max_voxels, replace=False)
        starts, ends = starts[chosen], ends[chosen]
    voxel_count = len(starts)
    features = np.zeros((voxel_count, max_points, 4), dtype=np.float32)
    labels = np.full((voxel_count, max_points), -1.0, dtype=np.float32)
    coords = np.zeros((voxel_count, 3), dtype=np.int32)
    counts = np.zeros(voxel_count, dtype=np.int32)
    for voxel_index, (start, end) in enumerate(zip(starts, ends)):
        selected = sorted_index[start:min(end, start + max_points)]
        count = len(selected)
        features[voxel_index, :count] = points[selected, :4]
        labels[voxel_index, :count] = targets[selected]
        xyz = xyz_coord[selected[0]]
        coords[voxel_index] = xyz[::-1]  # spconv convention: z, y, x
        counts[voxel_index] = count
    return features, labels, coords, counts


class GSPRPointDataset(Dataset):
    def __init__(self, manifests, voxel_size, lidar_range,
                 max_points=32, max_voxels=32000, seed=20260904):
        self.records = []
        for manifest in manifests:
            with Path(manifest).open("r", encoding="utf-8") as stream:
                self.records.extend(json.loads(line) for line in stream if line.strip())
        if not self.records:
            raise RuntimeError("point-supervision manifests are empty")
        self.voxel_size = voxel_size
        self.lidar_range = lidar_range
        self.max_points = max_points
        self.max_voxels = max_voxels
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with np.load(record["path"]) as data:
            features, labels, coords, counts = voxelize_with_labels(
                data["points"], data["reliability"], self.voxel_size,
                self.lidar_range, self.max_points, self.max_voxels,
                self.seed + self.epoch * len(self.records) + index)
        return {
            "voxel_features": features,
            "point_targets": labels,
            "voxel_coords": coords,
            "voxel_num_points": counts,
            "weather": record["weather"],
            "severity": record["severity"],
        }


class StratifiedGSPRSampler(Sampler):
    """Balance weather/severity while retaining some zero-noise examples."""

    def __init__(self, dataset, num_samples=None, noise_bearing_fraction=0.8,
                 seed=20260906):
        if not 0.0 < noise_bearing_fraction <= 1.0:
            raise ValueError("noise_bearing_fraction must be in (0, 1]")
        self.num_samples = int(num_samples or len(dataset))
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")
        self.seed = int(seed)
        self.epoch = 0
        groups = {}
        for index, record in enumerate(dataset.records):
            key = (record["weather"], record["severity"])
            groups.setdefault(key, {"noise": [], "zero": []})
            bucket = "noise" if int(record["noise_points"]) > 0 else "zero"
            groups[key][bucket].append(index)
        if not groups:
            raise RuntimeError("cannot stratify an empty dataset")

        weights = torch.zeros(len(dataset), dtype=torch.double)
        group_mass = 1.0 / len(groups)
        self.group_counts = {}
        for key, buckets in groups.items():
            noise_indices = buckets["noise"]
            zero_indices = buckets["zero"]
            self.group_counts["%s/%s" % key] = {
                "noise_bearing": len(noise_indices),
                "zero_noise": len(zero_indices),
            }
            if noise_indices and zero_indices:
                natural_zero_fraction = len(zero_indices) / \
                    (len(noise_indices) + len(zero_indices))
                zero_fraction = min(
                    1.0 - noise_bearing_fraction, natural_zero_fraction)
                zero_mass = group_mass * zero_fraction
                noise_mass = group_mass - zero_mass
                self.group_counts["%s/%s" % key][
                    "target_noise_bearing_fraction"] = 1.0 - zero_fraction
            elif noise_indices:
                noise_mass, zero_mass = group_mass, 0.0
                self.group_counts["%s/%s" % key][
                    "target_noise_bearing_fraction"] = 1.0
            else:
                noise_mass, zero_mass = 0.0, group_mass
                self.group_counts["%s/%s" % key][
                    "target_noise_bearing_fraction"] = 0.0
            if noise_indices:
                weights[noise_indices] = noise_mass / len(noise_indices)
            if zero_indices:
                weights[zero_indices] = zero_mass / len(zero_indices)
        if not torch.isfinite(weights).all() or weights.sum() <= 0:
            raise RuntimeError("invalid stratified sampling weights")
        self.weights = weights / weights.sum()

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights, self.num_samples, replacement=True,
            generator=generator)
        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples


def collate_voxel_samples(samples):
    features, targets, coords, counts = [], [], [], []
    metadata = []
    for batch_index, sample in enumerate(samples):
        features.append(sample["voxel_features"])
        targets.append(sample["point_targets"])
        coords.append(np.pad(sample["voxel_coords"], ((0, 0), (1, 0)),
                             constant_values=batch_index))
        counts.append(sample["voxel_num_points"])
        metadata.append((sample["weather"], sample["severity"]))
    return {
        "voxel_features": torch.from_numpy(np.concatenate(features)),
        "point_targets": torch.from_numpy(np.concatenate(targets)),
        "voxel_coords": torch.from_numpy(np.concatenate(coords)),
        "voxel_num_points": torch.from_numpy(np.concatenate(counts)),
        "metadata": metadata,
    }
