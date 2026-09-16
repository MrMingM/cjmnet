"""Detection loss plus optional point supervision and anti-collapse terms."""

import torch
import torch.nn.functional as F

from opencood.loss.point_pillar_loss import PointPillarLoss


class GsprPointPillarLoss(PointPillarLoss):
    def __init__(self, args):
        super().__init__(args)
        self.point_weight = float(args.get("point_weight", 1.0))
        self.keep_weight = float(args.get("keep_weight", 0.02))
        self.evidence_weight = float(args.get("evidence_weight", 1e-4))
        self.target_keep_rate = float(args.get("target_keep_rate", 0.90))

    def forward(self, output_dict, target_dict):
        detection_loss = super().forward(output_dict, target_dict)
        valid = output_dict["point_valid_mask"]
        reliability = output_dict["point_reliability"]
        point_loss = reliability.new_zeros(())
        if "point_reliability_target" in output_dict:
            target = output_dict["point_reliability_target"].to(
                reliability.device, reliability.dtype)
            if target.shape != reliability.shape:
                raise ValueError("point target must align with voxel point slots")
            point_loss = F.binary_cross_entropy(
                reliability[valid].clamp(1e-5, 1.0 - 1e-5), target[valid])
        keep_loss = (reliability[valid].mean() - self.target_keep_rate).pow(2)
        evidence_loss = output_dict["point_evidence"][valid].pow(2).mean()
        total = detection_loss + self.point_weight * point_loss + \
            self.keep_weight * keep_loss + self.evidence_weight * evidence_loss
        self.loss_dict["total_loss"] = total
        self.loss_dict["point_loss"] = point_loss
        self.loss_dict["keep_loss"] = keep_loss
        self.loss_dict["evidence_loss"] = evidence_loss
        return total

    def logging(self, epoch, batch_id, batch_len, writer, pbar=None):
        super().logging(epoch, batch_id, batch_len, writer, pbar)
        step = epoch * batch_len + batch_id
        writer.add_scalar("GSPR_point_loss", self.loss_dict["point_loss"].item(), step)
        writer.add_scalar("GSPR_keep_loss", self.loss_dict["keep_loss"].item(), step)
        writer.add_scalar("GSPR_evidence_loss",
                          self.loss_dict["evidence_loss"].item(), step)
