"""Original scaled dot-product AttFuse with unavailable keys excluded.

Training uses exact hard attention forward and soft gate attention backward.
The backward surrogate uses cached full sender features; inference does not.
"""
import math
import torch
import torch.nn.functional as F


def fuse(features, hard_blocks, soft_blocks=None):
    n, c, h, w = features.shape
    hard = F.interpolate(hard_blocks, size=(h, w), mode="nearest").bool()
    if not hard[0].all():
        raise ValueError("Ego must always be available")
    received = torch.where(hard, features, torch.zeros_like(features))
    logits = (received[:1] * received).sum(1, keepdim=True) / math.sqrt(c)
    weights = logits.masked_fill(~hard, -torch.inf).softmax(0)
    exact = (weights * received).sum(0, keepdim=True)
    if soft_blocks is None:
        return exact
    soft = F.interpolate(soft_blocks, size=(h, w), mode="nearest")
    soft_logits = (features[:1] * features).sum(1, keepdim=True) / math.sqrt(c)
    soft_weights = (soft_logits + soft.clamp_min(1e-8).log()).softmax(0)
    surrogate = (soft_weights * features).sum(0, keepdim=True)
    return exact.detach() + (surrogate - surrogate.detach())
