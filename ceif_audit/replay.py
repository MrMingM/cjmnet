"""Receiver-local counterfactuals; original modules remain unchanged."""
import torch
import torch.nn.functional as F
from gspr_communication.masked_attfuse import fuse


def received_levels(encoded, masks, value_bytes):
    result = []
    for original in encoded['levels']:
        x = original.clone()
        if value_bytes == 2:
            x[1:] = x[1:].half().float()
        mask = F.interpolate(masks, size=x.shape[-2:], mode='nearest').bool()
        result.append(torch.where(mask, x, torch.zeros_like(x)))
    return result


def fused_levels(received, masks):
    return [fuse(x, masks) for x in received]


def apply_action(fused, received, masks, action):
    peer = action['peer']
    region = torch.as_tensor(action['region'], device=masks.device)[None, None]
    if (region & ~masks[peer:peer+1].bool()).any():
        raise ValueError('Attempted access to an unreceived feature block')
    result = []
    for z, sources in zip(fused, received):
        patch = F.interpolate(region.float(), size=z.shape[-2:], mode='nearest').bool()
        result.append(torch.where(patch, sources[peer:peer+1], z))
    return result


def detect(base, fused):
    joined = torch.cat([d(x) for d, x in zip(base.backbone.deblocks, fused)], 1)
    return dict(psm=base.cls_head(joined), rm=base.reg_head(joined))
