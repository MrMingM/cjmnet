"""Local interventions on immutable predictions/features; Torch is lazy-loaded."""
import math
import numpy as np
from .common import grid_xy, roi_mask


def masks_for_frame(anchors, corners, levels, lidar_range, expansion):
    head_xy = anchors.detach().cpu().numpy()[..., :2]
    if head_xy.ndim != 4:
        raise ValueError('Expected H/W/A/7 anchors')
    if not np.allclose(head_xy, head_xy[:, :, :1], rtol=0, atol=1e-6):
        raise ValueError('Anchor types do not share spatial centers')
    mask, fallback = roi_mask(head_xy[:, :, 0], corners, expansion)
    levels_out = []; stats = dict(head_cells=int(mask.sum()), head_fallback=fallback, levels=[])
    for level in levels:
        m, f = roi_mask(grid_xy(level.shape[-2:], lidar_range), corners, expansion)
        levels_out.append(m)
        stats['levels'].append(dict(shape=list(m.shape), cells=int(m.sum()), fallback=f))
    stats['definition'] = 'Output uses actual anchor centers; feature levels use metric BEV cell centers. If empty use nearest center, recorded explicitly; not an exact receptive-field boundary.'
    return mask, levels_out, stats


def replace_output(full, peer, mask, family):
    import torch
    key = {'score': 'psm', 'geometry': 'rm'}[family]
    m = torch.as_tensor(mask, device=full[key].device, dtype=torch.bool)[None, None]
    if tuple(m.shape[-2:]) != tuple(full[key].shape[-2:]):
        raise ValueError('Output ROI shape mismatch')
    out = dict(full)
    out[key] = torch.where(m, peer[key], full[key])
    # Every anchor channel at a location is replaced together.
    assert torch.equal(out[key].masked_select(~m), full[key].masked_select(~m))
    return out


class WeightCache:
    def __init__(self, model, encoded):
        self.model, self.levels = model, encoded['levels']
        self.weights = [((x[:1]*x).sum(1, keepdim=True)/math.sqrt(x.shape[1])).softmax(0) for x in self.levels]
        self.original = [(w*x).sum(0, keepdim=True) for x, w in zip(self.levels, self.weights)]
        self.peer_cache = {}

    def decode(self, features):
        import torch
        base = self.model.engine.base
        joined = torch.cat([d(x) for d, x in zip(base.backbone.deblocks, features)], 1)
        return dict(psm=base.cls_head(joined), rm=base.reg_head(joined))

    def predict(self, peer, scales, alpha, masks=None):
        import torch
        if not 0 <= alpha <= 1:
            raise ValueError('Invalid weight mixing strength')
        if peer not in self.peer_cache:
            self.peer_cache[peer] = [((x[peer:peer+1]*x).sum(1, keepdim=True)/math.sqrt(x.shape[1])).softmax(0)
                                     for x in self.levels]
        fused = []
        for i, (x, old, other, baseline) in enumerate(zip(self.levels, self.weights, self.peer_cache[peer], self.original)):
            if i not in scales or alpha == 0:
                fused.append(baseline); continue
            weights = other if alpha == 1 else (1-alpha)*old+alpha*other
            changed = (weights*x).sum(0, keepdim=True)
            if masks is not None:
                m = torch.as_tensor(masks[i], dtype=torch.bool, device=x.device)[None, None]
                changed = torch.where(m, changed, baseline)
                if not torch.equal(changed.masked_select(~m), baseline.masked_select(~m)):
                    raise AssertionError('Feature modification escaped local ROI')
            fused.append(changed)
        return self.decode(fused)
