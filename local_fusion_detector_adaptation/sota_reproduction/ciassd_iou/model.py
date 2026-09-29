"""The original-author 3x3, two-anchor IoU head and score equation."""
import numpy as np


def mapped_iou(raw):
    """Author-code map from the unconstrained [-1,1] regression target."""
    return (np.asarray(raw) + 1.) * .5


def rectified_score(class_score, raw):
    """Author-code confidence function: sigmoid(cls) * ((raw+1)/2)^4."""
    return np.asarray(class_score) * np.power(mapped_iou(raw), 4)


def weighted_smooth_l1(prediction, target):
    """Unreduced author-code sigma=3 WeightedSmoothL1Loss (weights applied by caller)."""
    import torch
    delta = torch.abs(prediction - target)
    return torch.where(delta < 1. / 9., 4.5 * delta.square(), delta - 1. / 18.)


def make_head(channels, anchors):
    """Equivalent to author Conv2d(C,A,3,padding=1,bias=False) on cached patches."""
    from torch import nn
    if channels < 1 or anchors < 1:
        raise ValueError('Positive channels and anchors required')
    return nn.Conv2d(channels, anchors, kernel_size=3, padding=1, bias=False)


def patch_logits(head, patches, anchor_indices):
    """Apply the same convolution weights to individual padded 3x3 neighborhoods."""
    import torch.nn.functional as F
    if patches.ndim != 4 or patches.shape[-2:] != (3, 3):
        raise ValueError('Expected [N,C,3,3] patches')
    # Conv over a 3x3 patch without padding yields its center map value.
    all_anchors = F.conv2d(patches, head.weight).flatten(1)
    return all_anchors.gather(1, anchor_indices[:, None]).flatten()
