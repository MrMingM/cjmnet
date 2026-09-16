"""Lightweight residual adapter for weather-corrupted BEV features."""

import torch.nn as nn


class WeatherFeatureAdapter(nn.Module):
    """Contextual residual correction with an exact identity initialization."""

    def __init__(self, channels=64, hidden_channels=64, groups=8):
        super(WeatherFeatureAdapter, self).__init__()
        groups = max(1, min(int(groups), int(hidden_channels)))
        while hidden_channels % groups != 0:
            groups -= 1
        self.correction = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1,
                      bias=False),
            nn.GroupNorm(groups, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=3, padding=1,
                      bias=True)
        )
        # A baseline checkpoint has no adapter weights.  Zero-initializing the
        # last layer makes both teacher and student exactly equal to the
        # original model before adapter training.
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def forward(self, feature):
        return feature + self.correction(feature)
