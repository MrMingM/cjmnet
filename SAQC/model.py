"""Local Spatial Quality Head (LSQH) for the SAQC benchmark.

The paper uses a 7x7 BEV patch augmented with two relative-coordinate channels,
then two 3x3 convolutions, ReLU, average pooling and a scalar quality output.
The original paper reports the block structure but does not expose an official
implementation in this repository, so hidden_channels is explicit and recorded
in every checkpoint instead of being treated as a paper constant.
"""
from __future__ import annotations

import torch
from torch import nn


def relative_coordinate_channels(
    batch: int,
    patch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return integer relative row/column offsets centered on the patch."""
    if patch_size < 1 or patch_size % 2 != 1:
        raise ValueError('patch_size must be a positive odd integer')
    radius = patch_size // 2
    axis = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    row = axis[:, None].expand(patch_size, patch_size)
    col = axis[None, :].expand(patch_size, patch_size)
    coords = torch.stack((row, col), dim=0)
    return coords.unsqueeze(0).expand(batch, -1, -1, -1)


class SpatialQualityHead(nn.Module):
    """Coordinate-augmented two-convolution SAQC quality head."""

    def __init__(self, feature_channels: int, hidden_channels: int = 16,
                 patch_size: int = 7, with_coordinates: bool = True):
        super().__init__()
        if feature_channels < 1 or hidden_channels < 1:
            raise ValueError('feature_channels and hidden_channels must be positive')
        if patch_size < 1 or patch_size % 2 != 1:
            raise ValueError('patch_size must be a positive odd integer')
        self.feature_channels = int(feature_channels)
        self.hidden_channels = int(hidden_channels)
        self.patch_size = int(patch_size)
        self.with_coordinates = bool(with_coordinates)
        inputs = self.feature_channels + (2 if self.with_coordinates else 0)
        self.conv1 = nn.Conv2d(inputs, self.hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(self.hidden_channels, self.hidden_channels,
                               kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.hidden_channels, 1)

    def forward_logits(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.ndim != 4 or patches.shape[1] != self.feature_channels:
            raise ValueError(
                f'expected [N,{self.feature_channels},K,K] patches, got {tuple(patches.shape)}')
        if patches.shape[-2:] != (self.patch_size, self.patch_size):
            raise ValueError('patch spatial shape differs from configured patch_size')
        x = patches
        if self.with_coordinates:
            coords = relative_coordinate_channels(
                len(x), self.patch_size, device=x.device, dtype=x.dtype)
            x = torch.cat((x, coords), dim=1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x).flatten(1)
        return self.fc(x).squeeze(1)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(patches))

    def extra_repr(self) -> str:
        return (f'feature_channels={self.feature_channels}, '
                f'hidden_channels={self.hidden_channels}, patch_size={self.patch_size}, '
                f'with_coordinates={self.with_coordinates}')
