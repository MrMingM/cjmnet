"""Counterfactual-utility-guided restoration for cooperative BEV features."""

import torch
import torch.nn as nn


class CounterfactualUtilityRestorer(nn.Module):
    """Restore per-agent BEV features and predict signed collaboration value.

    The restoration branch is initialized as an exact identity mapping.  A
    context head compares every agent with the corresponding Ego feature and
    predicts three spatial quantities:

    * recoverability: weather-lost clean marginal utility;
    * harm: probability that the agent has negative weather utility;
    * utility: normalized signed weather marginal utility.

    Ground truth and counterfactual passes are needed only to supervise these
    maps during training.  Inference uses the learned maps directly.
    """

    def __init__(self, channels=64, hidden_channels=64, groups=8,
                 fusion_gate_floor=0.5):
        super().__init__()
        hidden_channels = int(hidden_channels)
        groups = max(1, min(int(groups), hidden_channels))
        while hidden_channels % groups != 0:
            groups -= 1

        self.fusion_gate_floor = float(fusion_gate_floor)
        self.correction = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, 3, padding=1, bias=True),
        )
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
        self.value_head = nn.Conv2d(hidden_channels, 3, 1, bias=True)

        # A baseline checkpoint contains no restorer parameters.  The zero
        # correction guarantees exact baseline features before fine-tuning.
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)
        # Most BEV cells are background.  Start with sparse restoration and a
        # mildly conservative harm prior; the zero correction still makes the
        # complete module an exact identity before fine-tuning.
        self.value_head.bias.data[0] = -2.0
        self.value_head.bias.data[1] = -1.0

    @staticmethod
    def _ego_context(feature, record_len):
        """Repeat the Ego feature for every CAV in each batch item."""
        contexts = []
        ego_mask = torch.zeros(
            feature.shape[0], dtype=torch.bool, device=feature.device)
        offset = 0
        for length_value in record_len.detach().cpu().tolist():
            length = int(length_value)
            if length <= 0:
                raise ValueError('record_len entries must be positive')
            if offset + length > feature.shape[0]:
                raise ValueError(
                    'record_len describes more agents than feature rows')
            contexts.append(feature[offset:offset + 1].expand(
                length, -1, -1, -1))
            ego_mask[offset] = True
            offset += length
        if offset != feature.shape[0]:
            raise ValueError(
                'record_len describes %d agents but feature has %d rows' %
                (offset, feature.shape[0]))
        return torch.cat(contexts, dim=0), ego_mask

    def forward(self, feature, record_len):
        ego_feature, ego_mask = self._ego_context(feature, record_len)
        context_input = torch.cat(
            [feature, ego_feature, feature - ego_feature], dim=1)
        value_logits = self.value_head(self.context(context_input))
        recoverability_logit = value_logits[:, 0:1]
        harm_logit = value_logits[:, 1:2]
        utility_logit = value_logits[:, 2:3]

        recoverability = torch.sigmoid(recoverability_logit)
        harm_probability = torch.sigmoid(harm_logit)
        restoration_gate = recoverability * (1.0 - harm_probability)
        restoration_gate = restoration_gate.clone()
        # CURE learns the marginal value of cooperative agents.  Modifying Ego
        # here introduces an unsupervised shortcut and damages clean accuracy.
        restoration_gate[ego_mask] = 0.0

        correction = self.correction(feature)
        restored = feature + restoration_gate * correction

        floor = min(max(self.fusion_gate_floor, 0.0), 1.0)
        fusion_weight = floor + (1.0 - floor) * (
            1.0 - harm_probability)
        fusion_weight = fusion_weight.clone()
        fusion_weight[ego_mask] = 1.0
        return restored, {
            'correction': correction,
            'recoverability_logit': recoverability_logit,
            'harm_logit': harm_logit,
            'utility': torch.tanh(utility_logit),
            'restoration_gate': restoration_gate,
            'fusion_weight': fusion_weight,
            'ego_mask': ego_mask,
        }
