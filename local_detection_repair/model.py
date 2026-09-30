"""Small repair and keep/repair networks; no detector or fusion weights live here."""
import math

import torch
from torch import nn


class LocalRepairNet(nn.Module):
    """Predict bounded box residuals and a quality-aware score for one model candidate."""

    def __init__(self, feature_dim, config):
        super().__init__()
        hidden = int(config.get("hidden", 128))
        dropout = float(config.get("dropout", 0.1))
        self.center_scale = tuple(float(x) for x in config["center_scale_m"])
        self.max_log_size = float(config["max_log_size"])
        self.max_yaw_delta = float(config["max_yaw_delta"])
        if len(self.center_scale) != 3 or min(self.center_scale) <= 0:
            raise ValueError("repair.center_scale_m must contain three positive values")
        if self.max_log_size <= 0 or not 0 < self.max_yaw_delta <= math.pi:
            raise ValueError("invalid repair residual bounds")
        self.trunk = nn.Sequential(
            nn.Linear(int(feature_dim), hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.geometry = nn.Linear(hidden, 7)
        self.quality = nn.Linear(hidden, 1)

    def forward(self, features):
        hidden = self.trunk(features)
        return {
            "geometry_raw": self.geometry(hidden),
            "score_logit": self.quality(hidden).squeeze(-1),
        }

    def bounded_geometry(self, raw):
        """Map unconstrained network output to the same 7-D decoded-box residual convention."""
        scale = raw.new_tensor(
            [*self.center_scale, self.max_log_size, self.max_log_size,
             self.max_log_size, self.max_yaw_delta]
        )
        return torch.tanh(raw) * scale


class RepairSelector(nn.Module):
    """Conservative keep/repair gate trained on the fixed repairer's actual outcomes."""

    def __init__(self, feature_dim, config):
        super().__init__()
        hidden = int(config.get("hidden", 64))
        dropout = float(config.get("dropout", 0.1))
        self.net = nn.Sequential(
            nn.Linear(int(feature_dim), hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, features):
        return self.net(features).squeeze(-1)
