"""Single-frame evidence adaptation. See SOURCE_ADAPTATION.md for provenance.

CF-VQA factual/reference contrast + KalmanNet-inspired learned gain.
This is neither a causal-effect estimator nor a Kalman covariance filter.
"""
import torch
from torch import nn
from .evidence import PointReviewer


def neutral_reference(inputs):
    """Preserve transmitted evidence mass, u and support; remove class preference.

    Unlike zeroing a packet, this also preserves the quantized mass (which need
    not sum exactly to 1). It is a chosen comparison, not an unobserved truth.
    """
    reference = inputs.clone()
    mass = inputs[..., 7] + inputs[..., 8]
    reference[..., 7] = mass / 2
    reference[..., 8] = mass / 2
    return reference


class ContrastGain(nn.Module):
    def __init__(self, hidden, use_gain=True):
        super().__init__()
        self.use_gain = bool(use_gain)
        self.score = nn.Sequential(nn.Linear(11, hidden), nn.SiLU(),
                                   nn.Linear(hidden, hidden), nn.SiLU(),
                                   nn.Linear(hidden, 1, bias=False))
        # Zero initial update, while the hidden representation remains trainable.
        nn.init.zeros_(self.score[-1].weight)
        # Local p/u plus received reliable/noise/unknown/support. No remote features.
        self.gain = (nn.Sequential(nn.Linear(6, hidden), nn.SiLU(), nn.Linear(hidden, 1))
                     if self.use_gain else None)
        if self.gain is not None:
            nn.init.zeros_(self.gain[-1].weight)
            nn.init.zeros_(self.gain[-1].bias)  # initial gain .5, NOT zero

    def components(self, inputs):
        if inputs.shape[-1] != 11:
            raise ValueError('Expected seven local and four decoded peer fields')
        # Shared parameters cancel a common local offset. Never subtract the
        # reliability probabilities of two physically different points.
        contrast = self.score(inputs) - self.score(neutral_reference(inputs))
        gain = (torch.sigmoid(self.gain(torch.cat((inputs[..., :2], inputs[..., 7:]), -1)))
                if self.gain is not None else torch.ones_like(contrast))
        return contrast, gain

    def forward(self, inputs):
        contrast, gain = self.components(inputs)
        # PointReviewer applies max_adjustment*tanh(...) and preserves eligibility.
        return contrast * gain


class CounterfactualReviewer(PointReviewer):
    def __init__(self, hidden=32, max_adjustment=.25, floor=.05, use_gain=True):
        super().__init__(hidden, max_adjustment, floor)
        self.net = ContrastGain(hidden, use_gain)


def two_source_interactions(f00, f10, f01, f11):
    """Two-source specialization of InterSHAP's Shapley interaction matrix.

    f10: actual local/reference peer; f01: reference local/actual peer.
    Off-diagonal values each carry HALF the mixed difference. Diagonals are
    Shapley values minus the off-diagonal, so the matrix sums to f11-f00.
    The caller defines references; this is not dataset-level InterSHAP scoring.
    """
    off = (f11 + f00 - f10 - f01) / 2
    local_phi = ((f10-f00) + (f11-f01)) / 2
    peer_phi = ((f01-f00) + (f11-f10)) / 2
    return local_phi-off, peer_phi-off, off
