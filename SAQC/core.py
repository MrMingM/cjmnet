"""Pure SAQC score/calibration/metric helpers."""
from __future__ import annotations

import numpy as np
import torch


def paper_fused_score(score, quality, beta: float = 1.5):
    """SAQC quality-aware ranking score s' = s * q**beta."""
    if beta < 0:
        raise ValueError('beta must be nonnegative')
    if torch.is_tensor(score) or torch.is_tensor(quality):
        s = score if torch.is_tensor(score) else torch.as_tensor(score)
        q = quality if torch.is_tensor(quality) else torch.as_tensor(quality, device=s.device)
        return s.clamp(0, 1) * q.clamp(0, 1).pow(beta)
    s = np.asarray(score, dtype=np.float64)
    q = np.asarray(quality, dtype=np.float64)
    return np.clip(s, 0, 1) * np.power(np.clip(q, 0, 1), beta)


def platt_calibrate(score, a: float, b: float, eps: float = 1e-6):
    """Post-hoc SAQC sigmoid calibration sigmoid(a*logit(score)+b)."""
    if not 0 < eps < .5:
        raise ValueError('eps must be in (0,.5)')
    if torch.is_tensor(score):
        s = score.clamp(eps, 1 - eps)
        z = torch.log(s) - torch.log1p(-s)
        return torch.sigmoid(float(a) * z + float(b))
    s = np.clip(np.asarray(score, dtype=np.float64), eps, 1 - eps)
    z = np.log(s) - np.log1p(-s)
    return 1.0 / (1.0 + np.exp(-(float(a) * z + float(b))))


def _rank_average(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind='mergesort')
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = .5 * (start + stop - 1)
        start = stop
    return ranks


def spearman(score, quality) -> float:
    score = np.asarray(score, dtype=np.float64)
    quality = np.asarray(quality, dtype=np.float64)
    if score.shape != quality.shape or score.ndim != 1:
        raise ValueError('score and quality must be equal-length 1D arrays')
    if len(score) < 2:
        return float('nan')
    left, right = _rank_average(score), _rank_average(quality)
    if left.std() == 0 or right.std() == 0:
        return float('nan')
    return float(np.corrcoef(left, right)[0, 1])


def quality_ece(score, soft_quality, bins: int = 15) -> float:
    """Soft-target ECE used as a Q-ECE-style diagnostic."""
    score = np.asarray(score, dtype=np.float64)
    quality = np.asarray(soft_quality, dtype=np.float64)
    if score.shape != quality.shape or score.ndim != 1:
        raise ValueError('score and quality must be equal-length 1D arrays')
    if bins < 2:
        raise ValueError('bins must be >= 2')
    if len(score) == 0:
        return float('nan')
    score = np.clip(score, 0, 1)
    quality = np.clip(quality, 0, 1)
    edges = np.linspace(0., 1., bins + 1)
    total = 0.
    for i in range(bins):
        if i == bins - 1:
            mask = (score >= edges[i]) & (score <= edges[i + 1])
        else:
            mask = (score >= edges[i]) & (score < edges[i + 1])
        if mask.any():
            total += mask.mean() * abs(score[mask].mean() - quality[mask].mean())
    return float(total)
