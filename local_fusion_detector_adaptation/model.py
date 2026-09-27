"""Keep source evidence fixed while giving fused features their own detector."""
import copy
import torch
from torch import nn

from local_fusion_utility_v2.fusion import predict_from_levels


class FusedDetector(nn.Module):
    def __init__(self, source_base):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.deblocks = copy.deepcopy(source_base.backbone.deblocks)
        self.cls_head = copy.deepcopy(source_base.cls_head)
        self.reg_head = copy.deepcopy(source_base.reg_head)

    def forward(self, levels):
        return predict_from_levels(self, levels)


class AdaptationArm(nn.Module):
    def __init__(self, fusion, source_base, adapt_detector):
        super().__init__()
        self.fusion = copy.deepcopy(fusion)
        self.detector = FusedDetector(source_base)
        self.adapt_detector = bool(adapt_detector)
        self.detector.requires_grad_(self.adapt_detector)

    def train(self, mode=True):
        super().train(mode)
        # Freeze BN running statistics in BOTH arms, including F+D.
        self.detector.eval()
        return self

    def predict(self, unused_base, levels):
        fused, info = self.fusion(levels)
        return self.detector(fused), info

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
