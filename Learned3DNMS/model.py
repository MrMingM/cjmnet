"""D2D-Rescore adapted to single-class PointPillar/Where2comm top256.

Paper: Osterburg et al., Learned Non-Maximum Suppression for 3D Object
Detection (IV 2026), Sec. IV. No image/point/backbone features or GT
are consumed by this network at inference time.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class DetectionEmbedding(nn.Module):
    """7-D box + scalar score; velocity and class are unavailable/constant."""

    def __init__(self, width=64, frequencies=10, coordinate_scale=(140., 40., 10.)):
        super().__init__()
        self.frequencies = int(frequencies)
        self.register_buffer(
            'coordinate_scale',
            torch.as_tensor(coordinate_scale, dtype=torch.float32),
        )
        # xyz + log-dimensions + (sin yaw, cos yaw) + score +
        # one-hot vehicle class + xyz Fourier encoding.
        input_width = 10 + 6 * self.frequencies
        self.layers = nn.Sequential(
            nn.Linear(input_width, width),
            nn.ReLU(inplace=False),
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.ReLU(inplace=False),
        )

    def forward(self, boxes, scores):
        if boxes.ndim != 3 or boxes.shape[-1] != 7:
            raise ValueError('boxes must have shape [batch, candidates, 7]')
        if scores.shape != boxes.shape[:2]:
            raise ValueError('score and box candidate dimensions differ')
        xyz = boxes[..., :3] / self.coordinate_scale
        size = torch.log1p(boxes[..., 3:6].clamp_min(0.))
        yaw = boxes[..., 6]
        base = torch.cat((
            xyz, size, yaw.sin().unsqueeze(-1), yaw.cos().unsqueeze(-1),
            scores.unsqueeze(-1), torch.ones_like(scores).unsqueeze(-1),
        ), dim=-1)
        frequencies = (2. ** torch.arange(
            self.frequencies, dtype=boxes.dtype, device=boxes.device
        )) * math.pi
        phase = xyz.unsqueeze(-1) * frequencies
        fourier = torch.cat((phase.sin(), phase.cos()), dim=-1).flatten(-2)
        return self.layers(torch.cat((base, fourier), dim=-1))


class D2DBlock(nn.Module):
    """Four-head detection-to-detection attention with learned temperature."""

    def __init__(self, width=64, heads=4):
        super().__init__()
        if width % heads:
            raise ValueError('width must be divisible by attention heads')
        self.width = int(width)
        self.heads = int(heads)
        self.query_key_value = nn.Linear(width, 3 * width)
        self.projection = nn.Linear(width, width)
        self.log_temperature = nn.Parameter(torch.zeros(()))
        self.norm_attention = nn.LayerNorm(width)
        self.feed_forward = nn.Sequential(
            nn.Linear(width, 4 * width), nn.ReLU(inplace=False),
            nn.Linear(4 * width, width),
        )
        self.norm_ffn = nn.LayerNorm(width)

    def forward(self, x, mask):
        batch, count, width = x.shape
        head_width = width // self.heads
        qkv = self.query_key_value(x).reshape(
            batch, count, 3, self.heads, head_width
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        logits = (q @ k.transpose(-2, -1)) / math.sqrt(head_width)
        logits = logits / self.log_temperature.clamp(-4., 4.).exp()
        logits = logits.masked_fill(~mask[:, None, None, :], -1e4)
        attention = torch.softmax(logits, dim=-1)
        joined = (attention @ v).transpose(1, 2).contiguous()
        joined = joined.reshape(batch, count, width)
        x = self.norm_attention(x + self.projection(joined))
        x = self.norm_ffn(x + self.feed_forward(x))
        return x.masked_fill(~mask.unsqueeze(-1), 0.)


class GossipBlock(nn.Module):
    """Paper's local BEV-neighborhood message pooling adapted to 3D boxes."""

    def __init__(self, width=64, radius=5.):
        super().__init__()
        self.radius = float(radius)
        if self.radius <= 0.:
            raise ValueError('neighborhood radius must be positive')
        self.message = nn.Sequential(
            nn.Linear(2 * width + 8, width),
            nn.ReLU(inplace=False),
            nn.Linear(width, width),
        )

    def forward(self, features, boxes, mask):
        batch, count, width = features.shape
        centers = boxes[..., :3]
        dimensions = boxes[..., 3:6].clamp_min(1e-3)
        center_difference = centers[:, :, None, :] - centers[:, None, :, :]
        average_size = .5 * (
            dimensions[:, :, None, :] + dimensions[:, None, :, :])
        normalized_offset = center_difference / average_size
        normalized_size = (
            dimensions[:, :, None, :] - dimensions[:, None, :, :]
        ) / average_size
        distance = torch.linalg.vector_norm(
            center_difference, dim=-1)
        relative_heading = (
            boxes[:, :, None, 6] - boxes[:, None, :, 6]
        ).cos().unsqueeze(-1)
        geometry = torch.cat((
            normalized_offset, normalized_size, relative_heading,
            distance.unsqueeze(-1)), dim=-1)
        first = features[:, :, None, :].expand(
            batch, count, count, width)
        second = features[:, None, :, :].expand(
            batch, count, count, width)
        pair = torch.cat((first, second, geometry), dim=-1)
        local = (distance <= self.radius) & (
            mask[:, :, None] & mask[:, None, :])
        messages = self.message(pair).masked_fill(
            ~local.unsqueeze(-1), -1e4)
        pooled = messages.max(dim=2).values
        return (features + pooled).masked_fill(
            ~mask.unsqueeze(-1), 0.)


class D2DRescore(nn.Module):
    """Paper-scale 6-layer, 64-channel, 4-head residual score refiner."""

    def __init__(self, width=64, layers=6, heads=4, frequencies=10,
                 coordinate_scale=(140., 40., 10.), variant='d2d',
                 radius=5.):
        super().__init__()
        if variant not in ('d2d', 'gossip'):
            raise ValueError('variant must be d2d or gossip')
        self.variant = variant
        self.config = dict(
            width=int(width), layers=int(layers), heads=int(heads),
            frequencies=int(frequencies),
            coordinate_scale=list(map(float, coordinate_scale)),
            variant=variant, radius=float(radius),
        )
        self.embedding = DetectionEmbedding(
            width, frequencies, coordinate_scale)
        self.blocks = nn.ModuleList(
            (D2DBlock(width, heads) if variant == 'd2d'
             else GossipBlock(width, radius))
            for _ in range(layers))
        self.score_head = nn.Sequential(
            nn.Linear(width, width), nn.ReLU(inplace=False),
            nn.LayerNorm(width), nn.Linear(width, 1),
        )
        # A fresh/untrained head is an identity rescorer.
        nn.init.zeros_(self.score_head[-1].weight)
        nn.init.zeros_(self.score_head[-1].bias)

    def forward(self, boxes, scores, mask):
        if mask.shape != scores.shape or mask.dtype != torch.bool:
            raise ValueError('mask must be boolean with score dimensions')
        if boxes.shape[:2] != scores.shape or boxes.shape[-1] != 7:
            raise ValueError('invalid detection tensor dimensions')
        if not mask.any(dim=1).all():
            raise ValueError('empty candidate sets must be skipped before attention')
        if not torch.isfinite(boxes[mask]).all():
            raise ValueError('nonfinite box features')
        if not torch.isfinite(scores[mask]).all():
            raise ValueError('nonfinite original scores')
        scores = scores.clamp(1e-6, 1. - 1e-6)
        features = self.embedding(boxes, scores)
        features = features.masked_fill(~mask.unsqueeze(-1), 0.)
        for block in self.blocks:
            features = (block(features, mask) if self.variant == 'd2d'
                        else block(features, boxes, mask))
        residual = self.score_head(features).squeeze(-1)
        logits = torch.logit(scores) + residual
        return logits.masked_fill(~mask, -20.)

    @torch.no_grad()
    def rescore(self, boxes, scores, mask):
        return torch.sigmoid(self(boxes, scores, mask))
