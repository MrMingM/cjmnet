"""Lightweight pillar-level snow-noise suppression.

The module predicts whether an occupied weather pillar is likely to have been
created by snow particles.  It only scales the existing PointPillar feature;
the detector, communication module and detection heads can therefore remain
frozen during adaptation.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def snow_voxel_novelty_labels(clean_coords, weather_coords):
    """Mark weather occupied voxels absent from the paired clean input."""
    if clean_coords.ndim != 2 or clean_coords.shape[1] != 4 or \
            weather_coords.ndim != 2 or weather_coords.shape[1] != 4:
        raise ValueError('voxel coordinates must have shape [V, 4]')
    if weather_coords.shape[0] == 0:
        return weather_coords.new_zeros((0,), dtype=torch.float32)
    if clean_coords.shape[0] == 0:
        return weather_coords.new_ones(
            (weather_coords.shape[0],), dtype=torch.float32)

    clean = clean_coords.to(dtype=torch.int64)
    weather = weather_coords.to(device=clean.device, dtype=torch.int64)
    combined = torch.cat([clean, weather], dim=0)
    if torch.any(combined < 0):
        raise ValueError('voxel coordinates must be non-negative')
    limits = combined.amax(dim=0) + 1

    def encode(coords):
        key = coords[:, 0]
        for axis in range(1, 4):
            key = key * limits[axis] + coords[:, axis]
        return key

    clean_key = torch.sort(encode(clean)).values
    weather_key = encode(weather)
    positions = torch.searchsorted(clean_key, weather_key)
    safe_positions = positions.clamp(max=clean_key.numel() - 1)
    exists = (
        (positions < clean_key.numel()) &
        (clean_key[safe_positions] == weather_key))
    return (~exists).to(dtype=torch.float32)


class SnowVoxelDenoiser(nn.Module):
    """Predict and suppress snow-contaminated occupied pillars.

    The final classifier is initialized to zero.  A zero logit maps to an
    identity keep weight, so enabling the module does not perturb a pretrained
    checkpoint before optimization starts.
    """

    def __init__(self, channels=64, hidden_channels=64, max_points=32,
                 grid_size=None, gate_floor=0.05,
                 activation_threshold=0.5):
        super().__init__()
        if not 0.0 <= float(gate_floor) <= 1.0:
            raise ValueError('gate_floor must be in [0, 1]')
        if not 0.5 <= float(activation_threshold) < 1.0:
            raise ValueError(
                'activation_threshold must be in [0.5, 1.0)')
        self.max_points = max(int(max_points), 1)
        self.gate_floor = float(gate_floor)
        self.activation_threshold = float(activation_threshold)

        if grid_size is None:
            grid_size = [1, 1, 1]
        if len(grid_size) != 3:
            raise ValueError('grid_size must contain [x, y, z]')
        # voxel_coords use [batch, z, y, x].
        coord_scale = [
            max(float(grid_size[2] - 1), 1.0),
            max(float(grid_size[1] - 1), 1.0),
            max(float(grid_size[0] - 1), 1.0),
        ]
        self.register_buffer(
            'coord_scale',
            torch.tensor(coord_scale, dtype=torch.float32),
            persistent=False)

        hidden_channels = max(int(hidden_channels), 8)
        self.feature_norm = nn.LayerNorm(int(channels))
        self.classifier = nn.Sequential(
            nn.Linear(int(channels) + 4, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, 1))

        # Exact identity at initialization.  The raw zero logits still receive
        # a healthy BCE gradient, while suppression begins only for logits > 0.
        nn.init.zeros_(self.classifier[-1].weight)
        nn.init.zeros_(self.classifier[-1].bias)

    def forward(self, pillar_features, voxel_num_points, voxel_coords):
        if pillar_features.ndim != 2:
            raise ValueError('pillar_features must have shape [V, C]')
        if voxel_coords.ndim != 2 or voxel_coords.shape[1] != 4:
            raise ValueError('voxel_coords must have shape [V, 4]')
        if pillar_features.shape[0] != voxel_coords.shape[0]:
            raise ValueError('pillar feature/coordinate counts differ')
        if pillar_features.shape[0] != voxel_num_points.shape[0]:
            raise ValueError('pillar feature/point-count rows differ')

        dtype = pillar_features.dtype
        device = pillar_features.device
        point_feature = torch.log1p(
            voxel_num_points.to(device=device, dtype=dtype))
        point_feature = point_feature / math.log1p(self.max_points)
        point_feature = point_feature.clamp(0.0, 1.0).unsqueeze(1)

        coord_scale = self.coord_scale.to(device=device, dtype=dtype)
        coord_feature = voxel_coords[:, 1:].to(dtype=dtype)
        coord_feature = coord_feature / coord_scale.unsqueeze(0)
        coord_feature = coord_feature.mul(2.0).sub(1.0)

        predictor_input = torch.cat([
            self.feature_norm(pillar_features),
            point_feature,
            coord_feature,
        ], dim=1)
        noise_logit = self.classifier(predictor_input).squeeze(1)
        noise_probability = torch.sigmoid(noise_logit)

        # Only confidence above the calibrated threshold suppresses a pillar.
        # Since threshold >= 0.5, the zero-logit initialization stays identity.
        suppression = F.relu(
            (noise_probability - self.activation_threshold) /
            (1.0 - self.activation_threshold))
        keep_weight = 1.0 - (1.0 - self.gate_floor) * suppression
        gated_features = pillar_features * keep_weight.unsqueeze(1)
        return gated_features, {
            'noise_logit': noise_logit,
            'noise_probability': noise_probability,
            'keep_weight': keep_weight,
        }
