"""A has ego-only inputs; B has sender-only inputs and decoded request."""
import torch
from torch import nn


def predictor(inputs, hidden):
    return nn.Sequential(nn.Conv2d(inputs, hidden, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(hidden, 1, 1))


class RequestHead(nn.Module):
    def __init__(self, channels=64, hidden=32):
        super().__init__()
        self.net = predictor(channels + 7, hidden)

    def forward(self, ego_semantics, ego_stats, ego_confidence):
        return self.net(torch.cat([ego_semantics, ego_stats, ego_confidence], 1)).sigmoid()


class ResponseHead(nn.Module):
    def __init__(self, channels=64, hidden=32):
        super().__init__()
        self.net = predictor(channels + 8, hidden)

    def forward(self, sender_semantics, sender_stats, sender_confidence, request):
        # Already ego-aligned grid: relative pose is embodied in alignment.
        # No ego latent tensor / labels / other senders' features are accepted.
        return self.net(torch.cat([sender_semantics, sender_stats, sender_confidence, request], 1)).sigmoid()
