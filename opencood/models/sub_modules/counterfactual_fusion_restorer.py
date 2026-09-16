"""Post-fusion recovery guided by counterfactual collaboration utility."""

import torch
import torch.nn as nn


class CounterfactualFusionRestorer(nn.Module):
    """Correct fused BEV semantics using the corresponding Ego feature.

    Unlike the pre-fusion CURE adapter, this module operates at the detection
    feature resolution.  It observes the all-agent fused feature, the
    unchanged Ego feature, and their discrepancy.  Its residual branch is
    zero initialized, so adding the module to an existing checkpoint is an
    exact identity before training.
    """

    def __init__(self, channels=256, hidden_channels=64, groups=8):
        super().__init__()
        hidden_channels = int(hidden_channels)
        groups = max(1, min(int(groups), hidden_channels))
        while hidden_channels % groups != 0:
            groups -= 1

        self.context = nn.Sequential(
            nn.Conv2d(channels * 3, hidden_channels, 1, bias=False),
            nn.GroupNorm(groups, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                hidden_channels, hidden_channels, 3, padding=1,
                groups=hidden_channels, bias=False),
            nn.GroupNorm(groups, hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.correction_head = nn.Conv2d(
            hidden_channels, channels, 1, bias=True)
        self.value_head = nn.Conv2d(hidden_channels, 3, 1, bias=True)

        nn.init.zeros_(self.correction_head.weight)
        nn.init.zeros_(self.correction_head.bias)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)
        self.value_head.bias.data[0] = -2.0
        self.value_head.bias.data[1] = -1.0

    def forward(self, fused_feature, ego_feature):
        if fused_feature.shape != ego_feature.shape:
            raise ValueError(
                'Fused/Ego feature shapes differ: %s vs %s' %
                (tuple(fused_feature.shape), tuple(ego_feature.shape)))
        context = self.context(torch.cat([
            fused_feature,
            ego_feature,
            fused_feature - ego_feature,
        ], dim=1))
        correction = self.correction_head(context)
        value_logits = self.value_head(context)
        recoverability_logit = value_logits[:, 0:1]
        harm_logit = value_logits[:, 1:2]
        utility = torch.tanh(value_logits[:, 2:3])
        restoration_gate = torch.sigmoid(
            recoverability_logit) * (
            1.0 - torch.sigmoid(harm_logit))
        restored = fused_feature + restoration_gate * correction
        return restored, {
            'correction': correction,
            'recoverability_logit': recoverability_logit,
            'harm_logit': harm_logit,
            'utility': utility,
            'restoration_gate': restoration_gate,
        }
