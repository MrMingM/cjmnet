import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _regroup(x: torch.Tensor, record_len: torch.Tensor, lmax: int) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = int(record_len.numel())
    channels, height, width = x.shape[1:]
    out = x.new_zeros((batch_size, lmax, channels, height, width))
    mask = torch.zeros(batch_size, lmax, dtype=torch.bool, device=x.device)

    offset = 0
    for batch_idx in range(batch_size):
        cav_num = int(record_len[batch_idx].item())
        out[batch_idx, :cav_num] = x[offset : offset + cav_num]
        mask[batch_idx, :cav_num] = True
        offset += cav_num
    return out, mask


def _warp_affine(feats: torch.Tensor, affine: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
    cav_num, channels, _, _ = feats.shape
    height, width = out_hw
    affine = affine.to(device=feats.device, dtype=feats.dtype)
    grid = F.affine_grid(
        affine,
        size=(cav_num, channels, height, width),
        align_corners=False,
    )
    return F.grid_sample(feats, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


class UGPrecisionFusion(nn.Module):
    """Single-scale uncertainty-guided precision fusion."""

    def __init__(
        self,
        channels: int,
        mode: str = "sum",
        post_groups: int = 8,
        eps: float = 1e-3,
        smooth3x3: bool = True,
    ):
        super().__init__()
        if mode not in {"sum", "max", "mean"}:
            raise ValueError(f"Unsupported UGPF reduction mode: {mode}")

        self.channels = int(channels)
        self.mode = mode
        self.eps = float(eps)
        self.smooth3x3 = bool(smooth3x3)

        self.log_gamma = nn.Parameter(torch.zeros(1))
        self.logit_beta = nn.Parameter(torch.tensor(0.0))

        groups = math.gcd(self.channels, min(post_groups, self.channels)) or 1
        self.post = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, self.channels),
            nn.GELU(),
            nn.Conv2d(self.channels, self.channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, self.channels),
            nn.GELU(),
        )

        kernel = torch.tensor(
            [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        ) / 16.0
        self.register_buffer("gauss3x3", kernel.view(1, 1, 3, 3))

    def _blur3(self, x: torch.Tensor) -> torch.Tensor:
        if not self.smooth3x3:
            return x
        channels = x.shape[1]
        weight = self.gauss3x3.to(device=x.device, dtype=x.dtype).expand(channels, 1, 3, 3)
        return F.conv2d(x, weight=weight, stride=1, padding=1, groups=channels)

    def _reduce(self, weighted_features: torch.Tensor) -> torch.Tensor:
        if self.mode == "sum":
            return weighted_features.sum(dim=0)
        if self.mode == "max":
            return weighted_features.max(dim=0).values
        return weighted_features.mean(dim=0)

    def forward(
        self,
        feat: torch.Tensor,
        u_map: Optional[torch.Tensor],
        record_len: torch.Tensor,
        affine_matrix: torch.Tensor,
    ) -> torch.Tensor:
        _, _, height, width = feat.shape
        batch_size = int(record_len.numel())
        lmax = int(record_len.max().item())

        feats_b, mask = _regroup(feat, record_len, lmax)
        u_b = None
        if u_map is not None:
            u_b, _ = _regroup(u_map.detach(), record_len, lmax)

        out = []
        gamma = F.softplus(self.log_gamma)
        beta = torch.sigmoid(self.logit_beta)

        for batch_idx in range(batch_size):
            cav_num = int(mask[batch_idx].sum().item())
            affine = affine_matrix[batch_idx, 0, :cav_num]
            features = feats_b[batch_idx, :cav_num]
            features_warped = _warp_affine(features, affine, (height, width))

            if u_b is not None:
                uncertainty = u_b[batch_idx, :cav_num]
                uncertainty_warped = _warp_affine(uncertainty, affine, (height, width)).squeeze(1)
            else:
                uncertainty_warped = feat.new_full((cav_num, height, width), 0.5)

            valid_mask = _warp_affine(
                feat.new_ones((cav_num, 1, height, width)),
                affine,
                (height, width),
            ).squeeze(1) > 0.5
            uncertainty_warped = torch.where(
                valid_mask,
                uncertainty_warped,
                torch.ones_like(uncertainty_warped),
            )

            confidence = (1.0 - uncertainty_warped).clamp(0.0, 1.0)
            precision = (self.eps + gamma * confidence) * valid_mask.float()
            weights = precision / (precision.sum(dim=0, keepdim=True) + 1e-6)
            weights = self._blur3(weights.unsqueeze(0)).squeeze(0)
            weights = weights / (weights.sum(dim=0, keepdim=True) + 1e-6)

            fused_wls = self._reduce(weights.unsqueeze(1) * features_warped)
            ego_feature = features_warped[0]
            fused = ego_feature + beta * self.post((fused_wls - ego_feature).unsqueeze(0)).squeeze(0)
            out.append(fused)

        return torch.stack(out, dim=0)


class MSUGPrecisionFusionV1(nn.Module):
    """Uncertainty-aware pyramid fusion used by UECP."""

    def __init__(
        self,
        feature_dim: int,
        scales: Tuple[int, ...] = (4, 2, 1),
        fusion_operation: str = "sum",
    ):
        super().__init__()
        self.scales = tuple(scales)
        self.blocks = nn.ModuleList(
            [
                UGPrecisionFusion(
                    feature_dim,
                    mode=fusion_operation,
                    post_groups=8,
                    eps=1e-3,
                    smooth3x3=True,
                )
                for _ in self.scales
            ]
        )
        self.refine = nn.ModuleList(
            [nn.Conv2d(feature_dim, feature_dim, 1, bias=False) for _ in self.scales]
        )

    @staticmethod
    def _down_uweighted(
        feat: torch.Tensor,
        u_map: Optional[torch.Tensor],
        scale: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if u_map is None:
            u_map = feat.new_full((feat.shape[0], 1, feat.shape[2], feat.shape[3]), 0.5)
        if scale == 1:
            return feat, u_map

        confidence = (1.0 - u_map).clamp(0.0, 1.0)
        numerator = F.avg_pool2d(feat * confidence, kernel_size=scale, stride=scale)
        denominator = F.avg_pool2d(confidence, kernel_size=scale, stride=scale) + 1e-6
        feat_s = numerator / denominator
        u_s = 1.0 - F.avg_pool2d(confidence, kernel_size=scale, stride=scale)
        return feat_s, u_s

    @staticmethod
    def _up(x: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
        return F.interpolate(x, size=size_hw, mode="bilinear", align_corners=False)

    def forward(
        self,
        feature: torch.Tensor,
        u_map: Optional[torch.Tensor],
        record_len: torch.Tensor,
        affine_matrix: torch.Tensor,
    ) -> torch.Tensor:
        _, _, height, width = feature.shape
        lmax = int(record_len.max().item())

        prior = None
        out = None
        for scale, block, refine in zip(self.scales, self.blocks, self.refine):
            feature_s, u_s = self._down_uweighted(feature, u_map, scale)
            h_s, w_s = feature_s.shape[-2:]
            fused_s = block(feature_s, u_s, record_len, affine_matrix)

            if prior is not None:
                u_s_b, _ = _regroup(u_s, record_len, lmax)
                ego_confidence = (1.0 - u_s_b[:, 0]).clamp(0.0, 1.0)
                fused_s = fused_s + ego_confidence * self._up(prior, (h_s, w_s))

            fused_s = refine(fused_s)
            prior = fused_s
            out = fused_s

        if out is None:
            raise RuntimeError("MSUGPrecisionFusionV1 requires at least one pyramid scale.")
        if out.shape[-2:] != (height, width):
            out = self._up(out, (height, width))
        return out
