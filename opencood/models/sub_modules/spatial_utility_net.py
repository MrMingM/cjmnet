"""Lightweight spatial marginal-utility head for cooperative BEV features."""

import torch
import torch.nn as nn


INPUT_CHANNELS = 8


class SpatialUtilityNet(nn.Module):
    """Predict per-cell utility and harmful probability for one neighbor."""

    def __init__(self, hidden_channels=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, hidden_channels, 3, padding=1),
            nn.GroupNorm(4, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(4, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels // 2, 3, padding=1),
            nn.GroupNorm(4, hidden_channels // 2),
            nn.ReLU(inplace=True),
        )
        self.utility_head = nn.Conv2d(hidden_channels // 2, 1, 1)
        self.harm_head = nn.Conv2d(hidden_channels // 2, 1, 1)

    def forward(self, inputs):
        feature = self.encoder(inputs)
        return {
            'utility': self.utility_head(feature),
            'harm_logit': self.harm_head(feature)
        }


def build_neighbor_input(confidence, density, neighbor_index):
    """Create annotation-free spatial input for one neighbor.

    Parameters
    ----------
    confidence : torch.Tensor
        Shape ``[N, H, W]`` with sigmoid foreground confidence.
    density : torch.Tensor
        Shape ``[N, H, W]`` with normalized log point counts.
    neighbor_index : int
        Cooperative-agent index; ego is always index zero.
    """
    ego_conf = confidence[0]
    neighbor_conf = confidence[neighbor_index]
    ego_density = density[0]
    neighbor_density = density[neighbor_index]
    other_indices = [index for index in range(confidence.shape[0])
                     if index != neighbor_index]
    context_conf = confidence[other_indices].amax(dim=0)
    channels = [
        ego_conf,
        neighbor_conf,
        neighbor_conf - ego_conf,
        (neighbor_conf - ego_conf).abs(),
        ego_density,
        neighbor_density,
        torch.relu(neighbor_conf - ego_conf),
        context_conf,
    ]
    return torch.stack(channels, dim=0)


def utility_to_gate(utility, harm_logit, foreground_support,
                    utility_threshold=0.0, temperature=0.02,
                    gate_floor=0.05):
    """Convert predictions to a soft neighbor weight with safe background."""
    temperature = max(float(temperature), 1.0e-4)
    keep_from_utility = torch.sigmoid(
        (utility - float(utility_threshold)) / temperature)
    keep_from_harm = 1.0 - torch.sigmoid(harm_logit)
    keep = torch.sqrt(
        keep_from_utility.clamp_min(1.0e-6) *
        keep_from_harm.clamp_min(1.0e-6))
    keep = float(gate_floor) + (1.0 - float(gate_floor)) * keep
    support = foreground_support.to(dtype=keep.dtype)
    return 1.0 - support + support * keep
