"""LMD-core feature construction for fixed cooperative top-256 candidates.

The implementation follows the feature families in the committed MetaDetect3D
source while intentionally omitting point-in-box/reflectance and 3D-IoU
features that do not have a semantics-preserving counterpart for intermediate
feature fusion.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


STAT_NAMES = ("min", "max", "mean", "std")
NEIGHBOR_ATTRS = (
    "x", "y", "z", "length", "width", "height", "yaw",
    "volume", "surface_area", "volume_surface_ratio", "score",
)
OWN_ATTRS = (
    "score", "x", "y", "z", "length", "width", "height", "yaw",
    "volume", "surface_area", "volume_surface_ratio",
)


@dataclass(frozen=True)
class FeatureSpec:
    proposal_iou_threshold: float = 0.2


def _decode_box(row: dict, box_order: str) -> dict:
    box = np.asarray(row["fused_decoded_box"], dtype=np.float64)
    if box.shape != (7,) or not np.isfinite(box).all():
        raise ValueError("Expected finite 7D decoded box")
    order = str(box_order).lower()
    if sorted(order) != ["h", "l", "w"] or len(order) != 3:
        raise ValueError(f"Unsupported OpenCOOD box order: {box_order!r}")
    dims = dict(zip(order, box[3:6]))
    length, width, height = float(dims["l"]), float(dims["w"]), float(dims["h"])
    if min(length, width, height) <= 0:
        raise ValueError("Non-positive decoded box size")
    volume = length * width * height
    surface = 2.0 * (length * width + length * height + width * height)
    return {
        "x": float(box[0]), "y": float(box[1]), "z": float(box[2]),
        "length": length, "width": width, "height": height,
        "yaw": float(box[6]), "volume": volume,
        "surface_area": surface,
        "volume_surface_ratio": volume / max(surface, 1e-12),
        "score": float(row["score"]),
    }


def _precise_bev_iou(corners: np.ndarray) -> np.ndarray:
    """Exact pairwise BEV polygon IoU using the repository geometry helper."""
    from opencood.utils import common_utils

    n = len(corners)
    out = np.eye(n, dtype=np.float32)
    if n <= 1:
        return out
    polygons = common_utils.convert_format(corners)
    lo = corners.min(axis=1)
    hi = corners.max(axis=1)
    for i in range(n):
        possible = np.flatnonzero(((hi[i] > lo) & (lo[i] < hi)).all(axis=1))
        possible = possible[possible > i]
        if not len(possible):
            continue
        values = np.asarray(common_utils.compute_iou(
            polygons[i], [polygons[int(j)] for j in possible]), dtype=np.float32)
        out[i, possible] = values
        out[possible, i] = values
    if not np.isfinite(out).all() or (out < -1e-6).any() or (out > 1 + 1e-6).any():
        raise ValueError("Invalid pairwise BEV IoU")
    return np.clip(out, 0.0, 1.0)


def feature_names() -> list[str]:
    names = [f"own_{name}" for name in OWN_ATTRS]
    names.append("neighbor_count")
    for attr in NEIGHBOR_ATTRS:
        names.extend(f"neighbor_{attr}_{stat}" for stat in STAT_NAMES)
    names.extend(f"neighbor_bev_iou_{stat}" for stat in STAT_NAMES)
    return names


def build_frame_features(rows: Sequence[dict], box_order: str,
                         proposal_iou_threshold: float = 0.2):
    """Build adapted LMD-core features for one frame.

    Each candidate is treated as the output box and all top-256 candidates whose
    BEV IoU exceeds the LMD proposal threshold are its proposal neighborhood.
    Geometry statistics include the candidate itself, matching MetaDetect3D's
    proposal-mask behavior; BEV-IoU dispersion excludes self overlap.
    """
    if not 0.0 <= proposal_iou_threshold < 1.0:
        raise ValueError("proposal_iou_threshold must lie in [0,1)")
    if not rows:
        return (np.empty((0, len(feature_names())), dtype=np.float32),
                np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64))
    corners = np.asarray([row["fused_bev_corners"] for row in rows], dtype=np.float32)
    if corners.shape != (len(rows), 4, 2) or not np.isfinite(corners).all():
        raise ValueError("Invalid candidate BEV corners")
    attrs = [_decode_box(row, box_order) for row in rows]
    values = {name: np.asarray([row[name] for row in attrs], dtype=np.float64)
              for name in NEIGHBOR_ATTRS}
    overlap = _precise_bev_iou(corners)
    features = np.empty((len(rows), len(feature_names())), dtype=np.float32)
    for i, _ in enumerate(rows):
        neighborhood = overlap[i] > proposal_iou_threshold
        neighborhood[i] = True
        indices = np.flatnonzero(neighborhood)
        vector = [attrs[i][name] for name in OWN_ATTRS]
        vector.append(float(len(indices)))
        for attr in NEIGHBOR_ATTRS:
            sample = values[attr][indices]
            vector.extend((float(np.min(sample)), float(np.max(sample)),
                           float(np.mean(sample)), float(np.std(sample))))
        nonself = indices[indices != i]
        ov = overlap[i, nonself]
        if len(ov):
            vector.extend((float(np.min(ov)), float(np.max(ov)),
                           float(np.mean(ov)), float(np.std(ov))))
        else:
            vector.extend((0.0, 0.0, 0.0, 0.0))
        features[i] = np.asarray(vector, dtype=np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Non-finite LMD features")
    quality = np.asarray([row["max_gt_iou"] for row in rows], dtype=np.float32)
    candidate_ids = np.asarray([row["candidate_id"] for row in rows], dtype=np.int64)
    if (quality < 0).any() or (quality > 1 + 1e-6).any():
        raise ValueError("Candidate GT quality outside [0,1]")
    return features, np.clip(quality, 0.0, 1.0), candidate_ids


def iter_frames(root, weather: str):
    """Yield (sample, scene, rows) without loading a whole split into memory."""
    import json
    from pathlib import Path

    root = Path(root)
    path = root / weather / "candidate_rows.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    current = None
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            raw = json.loads(line)
            if raw.get("weather") != weather:
                raise ValueError(f"{path}:{line_no}: weather mismatch")
            sample = int(raw["sample_index"])
            if current is None:
                current = sample
            if sample != current:
                yield current, rows[0].get("scene"), rows
                rows = []
                current = sample
            rows.append(raw)
    if rows:
        yield current, rows[0].get("scene"), rows
