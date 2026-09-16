"""Small NumPy-only statistics used by the Q-A observation diagnostic."""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


def finite_pair(x: Sequence[float], y: Sequence[float]):
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    a, b = finite_pair(x, y)
    if len(a) < 3 or np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def rankdata(values: Sequence[float]) -> np.ndarray:
    """Average ranks for ties; equivalent to scipy.stats.rankdata(method='average')."""
    x = np.asarray(values, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    a, b = finite_pair(x, y)
    if len(a) < 3:
        return float("nan")
    return pearson(rankdata(a), rankdata(b))


def residualize(y: Sequence[float], controls: np.ndarray) -> np.ndarray:
    target = np.asarray(y, dtype=np.float64)
    z = np.asarray(controls, dtype=np.float64)
    if z.ndim == 1:
        z = z[:, None]
    design = np.concatenate([np.ones((len(z), 1)), z], axis=1)
    beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    return target - design @ beta


def partial_corr(x: Sequence[float], y: Sequence[float], controls: Sequence[Sequence[float]]) -> float:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    z = np.asarray(controls, dtype=np.float64)
    if z.ndim == 1:
        z = z[:, None]
    mask = np.isfinite(a) & np.isfinite(b) & np.isfinite(z).all(axis=1)
    if int(mask.sum()) < max(8, z.shape[1] + 4):
        return float("nan")
    return pearson(residualize(a[mask], z[mask]), residualize(b[mask], z[mask]))


def binary_auc(score: Sequence[float], label: Sequence[int]) -> float:
    """Mann-Whitney AUROC. label=1 is the positive class."""
    s = np.asarray(score, dtype=np.float64)
    y = np.asarray(label, dtype=np.int64)
    mask = np.isfinite(s) & ((y == 0) | (y == 1))
    s, y = s[mask], y[mask]
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    if not pos or not neg:
        return float("nan")
    r = rankdata(s)
    rank_sum_pos = float(r[y == 1].sum())
    return (rank_sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)


def safe_mean(values: Iterable[float]) -> float:
    x = np.asarray(list(values), dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else float("nan")


def safe_quantile(values: Iterable[float], q: float, default: float = 0.0) -> float:
    x = np.asarray(list(values), dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.quantile(x, q)) if len(x) else float(default)


def distance_bin(distance: float) -> str:
    edges = (0.0, 20.0, 40.0, 60.0, 80.0, 100.0, float("inf"))
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo <= distance < hi:
            return f"{int(lo)}-{('inf' if math.isinf(hi) else int(hi))}"
    return "unknown"


def json_float(value: float):
    value = float(value)
    return value if math.isfinite(value) else None
