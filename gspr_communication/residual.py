"""A0B0 plus a bounded additive score correction, initialized exactly to zero."""
import math
import torch
from torch import nn


class PriorityCorrection(nn.Module):
    def __init__(self, hidden=16, max_adjustment=.1):
        super().__init__()
        if hidden < 1 or not math.isfinite(max_adjustment) or max_adjustment <= 0:
            raise ValueError('Invalid correction width/amplitude')
        self.max_adjustment = float(max_adjustment)
        # Six sender statistics + sender confidence + received request.
        self.net = nn.Sequential(nn.Conv2d(8, hidden, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(hidden, 1, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, sender_stats, sender_confidence, request):
        x = torch.cat([sender_stats, sender_confidence, request], 1)
        delta = self.max_adjustment * self.net(x).tanh()
        # Do not clamp: clamping would erase gradients and create ranking ties.
        # Additive correction allows a low base score to move upward.
        return request * sender_confidence + delta
