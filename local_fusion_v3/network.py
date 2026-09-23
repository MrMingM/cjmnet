import torch
from torch import nn
from local_fusion_utility_v2.fusion import attention_fusion, predict_from_levels


class SpatialSourceFusion(nn.Module):
    """Permutation-equivariant source router; all BEV cells include low-score objects.

    No batch statistics couple vehicles. A source sees its spatial embedding,
    the mean embedding of all sources, and the original fused embedding.
    """
    def __init__(self, channels, variant='residual', hidden=32, scales=(0, 1), max_gate=.5):
        super().__init__()
        if variant not in ('residual', 'attention') or not 0 < max_gate <= 1:
            raise ValueError('Invalid fusion variant or max_gate')
        self.channels = list(channels)
        self.variant, self.scales, self.max_gate = variant, tuple(scales), max_gate
        self.encoders, self.routers, self.gates = nn.ModuleDict(), nn.ModuleDict(), nn.ModuleDict()
        for scale in self.scales:
            key = str(scale)
            self.encoders[key] = nn.Sequential(nn.Conv2d(channels[scale], hidden, 1), nn.SiLU(),
                nn.Conv2d(hidden, hidden, 3, padding=1), nn.SiLU())
            self.routers[key] = nn.Sequential(nn.Conv2d(hidden*3+1, hidden, 1), nn.SiLU(),
                                             nn.Conv2d(hidden, 1, 1))
            self.gates[key] = nn.Conv2d(hidden*2, 1, 1)
            nn.init.zeros_(self.gates[key].weight)
            nn.init.constant_(self.gates[key].bias, -4.)

    def forward(self, levels):
        fused, changes, gates = [], [], []
        for scale, feature in enumerate(levels):
            original, weights = attention_fusion(feature)
            if scale not in self.scales or len(feature) == 1:
                fused.append(original)
                continue
            key = str(scale)
            z = self.encoders[key](feature)
            center = z.mean(0, keepdim=True)
            base = self.encoders[key](original)
            logits = self.routers[key](torch.cat((z, center.expand_as(z), base.expand_as(z), weights), 1))
            proposed = logits.softmax(0)
            gate = (self.max_gate * torch.sigmoid(self.gates[key](torch.cat((center, base), 1)))
                    if self.variant == 'residual' else torch.ones_like(weights[:1]))
            mixed = weights + gate*(proposed-weights)
            fused.append((feature*mixed).sum(0, keepdim=True))
            changes.append((mixed-weights).abs().sum(0).mean())
            gates.append(gate.mean())
        zero = levels[0].new_zeros(())
        return fused, {'change': torch.stack(changes).mean() if changes else zero,
                       'gate': torch.stack(gates).mean() if gates else zero}

    def predict(self, base, levels):
        fused, info = self(levels)
        return predict_from_levels(base, fused), info
