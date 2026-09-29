import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridUncertaintyLoss(nn.Module):
    """
    一个混合损失函数，用于解决不确定性预测中的数据不均衡和细节模糊问题。
    结合了 Continuous Focal Loss 和 Gradient Difference Loss。
    """
    def __init__(self, alpha=0.5, gamma=2.0, grad_weight=1.0):
        """
        Args:
            alpha (float): Focal Loss的权重。 (1-alpha) 将是Gradient Loss的权重。
            gamma (float): Focal Loss中的调制因子gamma。
            grad_weight (float): 梯度损失的内部缩放系数。
        """
        super(HybridUncertaintyLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.grad_weight = grad_weight

        # 使用一个固定的Sobel滤波器来计算图像梯度
        # Sobel Gx
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        # Sobel Gy
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def _gradient_loss(self, pred, gt):
        """计算梯度差异损失 L_grad"""
        # 计算预测图的梯度
        pred_grad_x = F.conv2d(pred, self.sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred, self.sobel_y, padding=1)
        
        # 计算真值图的梯度
        gt_grad_x = F.conv2d(gt, self.sobel_x, padding=1)
        gt_grad_y = F.conv2d(gt, self.sobel_y, padding=1)

        # 计算梯度差异的L1损失
        grad_diff_loss = torch.mean(torch.abs(pred_grad_x - gt_grad_x)) + \
                         torch.mean(torch.abs(pred_grad_y - gt_grad_y))
        
        return grad_diff_loss

    def _focal_loss(self, pred, gt):
        """计算连续焦距损失 L_focal"""
        # 计算基础的L1误差
        l1_error = torch.abs(pred - gt)
        
        # 计算焦距调制因子
        focal_modulator = torch.pow(l1_error, self.gamma)
        
        # 计算最终的Focal L1 Loss
        focal_loss = torch.mean(focal_modulator * l1_error)
        
        return focal_loss

    def forward(self, pred, gt):
        """
        Args:
            pred (torch.Tensor): 模型预测的不确定性图, shape (N, 1, H, W).
            gt (torch.Tensor): 真值不确定性图, shape (N, 1, H, W).
        """
        # --- 计算 Focal Loss ---
        focal_loss = self._focal_loss(pred, gt)
        
        # --- 计算 Gradient Loss ---
        gradient_loss = self._gradient_loss(pred, gt)
        
        # --- 组合损失 ---
        hybrid_loss = self.alpha * focal_loss + \
                      (1 - self.alpha) * self.grad_weight * gradient_loss
        
        return hybrid_loss