"""Small shared heads used by every peer and every decision tile."""
import torch
from torch import nn
import torch.nn.functional as F


class TileLayerNorm(nn.Module):
    """Normalize descriptor channels independently at every peer/tile."""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(int(channels))

    def forward(self, value):
        return self.norm(value.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class UtilityPredictor(nn.Module):
    """Predict recovered GT, lost GT, new FP, and signed loss improvement.

    The first three outputs are constrained to non-negative values.  The fourth
    output exists only for the fair loss-gain ablation and is unconstrained.
    """

    def __init__(self, input_channels, hidden=64):
        super().__init__()
        input_channels, hidden = int(input_channels), int(hidden)
        if input_channels < 1 or hidden < 4:
            raise ValueError('Positive input size and hidden >= 4 required')
        self.input_channels = input_channels
        self.body = nn.Sequential(
            TileLayerNorm(input_channels),
            nn.Conv2d(input_channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 1),
            nn.SiLU(inplace=True),
        )
        self.outcome = nn.Conv2d(hidden, 3, 1)
        self.loss_gain = nn.Conv2d(hidden, 1, 1)

    def forward(self, descriptors):
        if descriptors.ndim != 4 or descriptors.shape[1] != self.input_channels:
            raise ValueError(f'Expected [peers,{self.input_channels},H,W] descriptors')
        if descriptors.shape[0] == 0:
            return {
                'outcome': descriptors.new_empty((0, 3, *descriptors.shape[-2:])),
                'loss_gain': descriptors.new_empty((0, 1, *descriptors.shape[-2:])),
            }
        hidden = self.body(descriptors)
        return {
            'outcome': F.softplus(self.outcome(hidden)),
            'loss_gain': self.loss_gain(hidden),
        }


def utility_score(outputs, lost_weight=1.0, fp_weight=1.0):
    values = outputs['outcome']
    if values.shape[1] != 3:
        raise ValueError('Outcome order must be recovered, lost, new_fp')
    return values[:, 0] - float(lost_weight)*values[:, 1] - float(fp_weight)*values[:, 2]
