import torch
import torch.nn as nn
from opencood.loss.point_pillar_loss import PointPillarLoss
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from v2x_dgw_recoverability_audit.m8_ucq_route_train_protocol import ucq_route_loss

class M8UCQRouteLoss(nn.Module):
    """
    Combined loss: standard detection loss (PointPillarLoss) + UCQ-Route Student Distillation Loss.
    """
    def __init__(self, args):
        super(M8UCQRouteLoss, self).__init__()
        self.det_loss_func = PointPillarLoss(args)
        self.alpha_ucq = args.get('alpha_ucq', 1.0) # Weight for the UCQ distillation loss

    def forward(self, output_dict, target_dict):
        # 1. Standard Detection Loss (Classification + Regression)
        det_loss = self.det_loss_func(output_dict, target_dict)
        
        # 2. UCQ-Route Distillation Loss
        # In a full pipeline, teacher proxy provides these targets.
        # Here we mock them if not present (or user can populate target_dict with them via a modified Dataset/Dataloader)
        ucq_out = output_dict['ucq_out']
        
        # For a complete integration, the dataloader should yield 'teacher_clean_features' and 'utility_labels'
        # If they are not available (e.g. standard dataset), we might skip or dummy-mock them for now to make it runnable.
        if 'teacher_clean_features' in target_dict:
            clean_features = target_dict['teacher_clean_features']
            utility_labels = target_dict['utility_labels']
            ucq_loss_dict = ucq_route_loss(ucq_out, target_dict['teacher_proxy_output'])
            ucq_loss = ucq_loss_dict['loss']
        else:
            # Fallback for testing: no distillation supervision provided, return 0 loss
            ucq_loss = torch.tensor(0.0, device=det_loss.device, requires_grad=True)

        total_loss = det_loss + self.alpha_ucq * ucq_loss
        
        self.loss_dict = {
            'loss': total_loss,
            'det_loss': det_loss,
            'ucq_loss': ucq_loss
        }
        
        return total_loss

    def logging(self, epoch, batch_id, batch_len, writer, pbar=None):
        """
        Logging utility for tensorboard.
        """
        self.det_loss_func.logging(epoch, batch_id, batch_len, writer, pbar)
