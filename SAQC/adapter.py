"""PointPillar/Where2comm adapter for SAQC.

SAQC was introduced for a center-based BEV detector. This project is
anchor-based. We associate each decoded box with the nearest BEV cell center
from the existing anchor grid. Anchor-cell association is returned as a
diagnostic so the adaptation is auditable.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F


def detector_feature_map(arm, levels):
    """Return the exact fused feature map consumed by frozen cls/reg heads."""
    fused, info = arm.fusion(levels)
    joined = torch.cat([
        deblock(value)
        for deblock, value in zip(
            arm.detector.backbone.deblocks, fused)
    ], dim=1)
    prediction = {
        'psm': arm.detector.cls_head(joined),
        'rm': arm.detector.reg_head(joined),
    }
    return joined, prediction, info


def _anchor_tensor(anchor_box: torch.Tensor,
                   psm: torch.Tensor) -> torch.Tensor:
    anchor = anchor_box
    if anchor.ndim == 5:
        if anchor.shape[0] != 1:
            raise ValueError(
                'SAQC currently requires inference batch size 1')
        anchor = anchor[0]
    if anchor.ndim != 4 or anchor.shape[-1] != 7:
        raise ValueError(
            f'unsupported anchor_box shape {tuple(anchor.shape)}')
    _, anchors, h, w = psm.shape
    if tuple(anchor.shape[:3]) != (h, w, anchors):
        if anchor.numel() != h * w * anchors * 7:
            raise ValueError(
                'anchor_box does not match detector output grid')
        anchor = anchor.reshape(h, w, anchors, 7)
    return anchor


def decoded_center_cells(decoded_boxes, anchor_box, psm,
                         chunk: int = 512):
    """Map decoded xy centers to nearest feature-map cell."""
    anchor = _anchor_tensor(anchor_box, psm)
    _, w, _ = anchor.shape[:3]
    grid = anchor[:, :, 0, :2].reshape(-1, 2)
    decoded = torch.as_tensor(
        decoded_boxes,
        dtype=grid.dtype,
        device=grid.device,
    )
    if decoded.ndim != 2 or decoded.shape[1] != 7:
        raise ValueError(
            'decoded_boxes must be [N,7]')

    flat = []
    distances = []
    for start in range(0, len(decoded), chunk):
        d = torch.cdist(
            decoded[start:start + chunk, :2],
            grid,
        )
        value, index = d.min(dim=1)
        flat.append(index)
        distances.append(value)

    if not flat:
        empty = torch.empty(
            0, dtype=torch.long, device=grid.device)
        return (
            empty,
            empty,
            torch.empty(
                0,
                dtype=grid.dtype,
                device=grid.device,
            ),
        )

    flat = torch.cat(flat)
    return (
        flat // w,
        flat % w,
        torch.cat(distances),
    )


def anchor_cells(candidate_ids, psm):
    """Original anchor cell of flattened candidate IDs."""
    _, anchors, h, w = psm.shape
    ids = torch.as_tensor(
        candidate_ids,
        dtype=torch.long,
        device=psm.device,
    )
    cells = ids // anchors
    if (len(ids)
            and (ids.min() < 0
                 or ids.max() >= h * w * anchors)):
        raise ValueError(
            'candidate id outside detector grid')
    return cells // w, cells % w


def extract_patches(feature_map: torch.Tensor,
                    rows,
                    cols,
                    patch_size: int = 7):
    """Zero-padded local patches without a full unfold tensor."""
    if (feature_map.ndim != 4
            or feature_map.shape[0] != 1):
        raise ValueError(
            'feature_map must be [1,C,H,W]')
    if patch_size < 1 or patch_size % 2 != 1:
        raise ValueError(
            'patch_size must be positive and odd')

    rows = torch.as_tensor(
        rows,
        dtype=torch.long,
        device=feature_map.device,
    )
    cols = torch.as_tensor(
        cols,
        dtype=torch.long,
        device=feature_map.device,
    )
    if rows.shape != cols.shape:
        raise ValueError(
            'rows and cols differ')
    if len(rows) == 0:
        return feature_map.new_empty((
            0,
            feature_map.shape[1],
            patch_size,
            patch_size,
        ))

    h, w = feature_map.shape[-2:]
    if ((rows < 0).any()
            or (rows >= h).any()
            or (cols < 0).any()
            or (cols >= w).any()):
        raise ValueError(
            'patch center outside feature map')

    radius = patch_size // 2
    padded = F.pad(
        feature_map[0],
        (radius, radius, radius, radius),
    )
    offsets = torch.arange(
        patch_size,
        device=feature_map.device,
    )
    row_index = (
        rows[:, None, None]
        + offsets[None, :, None]
    )
    col_index = (
        cols[:, None, None]
        + offsets[None, None, :]
    )
    return (
        padded[:, row_index, col_index]
        .permute(1, 0, 2, 3)
        .contiguous()
    )


def positive_anchor_ids(label_dict, psm):
    labels = label_dict['pos_equal_one']
    if labels.ndim < 2:
        raise ValueError(
            'pos_equal_one has unexpected shape')
    flat = labels.reshape(
        psm.shape[0], -1)
    expected = (
        psm.shape[1]
        * psm.shape[2]
        * psm.shape[3]
    )
    if flat.shape != (
        psm.shape[0], expected):
        raise ValueError(
            'positive-anchor labels do not match detector output')
    if psm.shape[0] != 1:
        raise ValueError(
            'SAQC current training path requires batch size 1')
    return torch.nonzero(
        flat[0] > 0,
        as_tuple=False,
    ).reshape(-1)


def association_diagnostics(candidate_ids,
                            decoded_boxes,
                            anchor_box,
                            psm):
    drow, dcol, distance = (
        decoded_center_cells(
            decoded_boxes,
            anchor_box,
            psm,
        ))
    arow, acol = anchor_cells(
        candidate_ids, psm)
    return {
        'decoded_rows': drow,
        'decoded_cols': dcol,
        'anchor_rows': arow,
        'anchor_cols': acol,
        'decoded_to_anchor_cell_same': (
            (drow == arow)
            & (dcol == acol)),
        'decoded_to_nearest_grid_distance': (
            distance),
    }
