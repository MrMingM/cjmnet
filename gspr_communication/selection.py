"""Adapted selection primitives from CoSDH's comm_modules/where2comm.py.

Original author: Yue Hu <phyllis1sjtu@outlook.com>.
Original license: TDG-Attribution-NonCommercial-NoDistrib (see SOURCE.md).
Changes: fixed byte-budget K, request-conditioned ranking BEFORE selection,
deterministic ties, and a separate differentiable training surrogate.
This is an internal adapted baseline, NOT a complete CoSDH reproduction.
"""
import torch


def foreground(logits):
    return logits.sigmoid().max(dim=1, keepdim=True)[0]


def hard_topk(scores, k):
    if not torch.isfinite(scores).all():
        raise ValueError("Non-finite response scores")
    flat = scores.flatten()
    k = min(max(int(k), 0), flat.numel())
    # Stable ties also allow exact comparison across A/B variants.
    indices = torch.argsort(flat, descending=True, stable=True)[:k]
    mask = torch.zeros_like(flat).scatter(0, indices, 1)
    return mask.reshape_as(scores)


def soft_topk(scores, hard, temperature):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    chosen = scores.detach()[hard.bool()]
    if chosen.numel() == 0 or chosen.numel() == scores.numel():
        return hard
    cutoff = chosen.min()
    return torch.sigmoid((scores - cutoff) / temperature)
