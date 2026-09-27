"""Task-specific versus shared source fusion with a frozen detector.

Both arms contain two complete copies of the trained v3 router.  The shared
arm averages their source weights before BOTH tasks.  The split arm sends one
router's weights to classification and the other's to regression.

No detector parameter is copied into this module, so the only trainable
parameters are the two routers.
"""
import copy
import torch
from torch import nn

from local_fusion_utility_v2.fusion import attention_fusion


def _route(router, levels):
    fused, weights_out, changes, gates = [], [], [], []
    for scale, feature in enumerate(levels):
        original, base_weights = attention_fusion(feature)
        if scale not in router.scales or len(feature) == 1:
            fused.append(original)
            weights_out.append(base_weights)
            continue
        key = str(scale)
        z = router.encoders[key](feature)
        center = z.mean(0, keepdim=True)
        base = router.encoders[key](original)
        logits = router.routers[key](
            torch.cat((z, center.expand_as(z), base.expand_as(z), base_weights), 1)
        )
        proposed = logits.softmax(0)
        if router.variant == 'residual':
            gate = router.max_gate * torch.sigmoid(
                router.gates[key](torch.cat((center, base), 1))
            )
        else:
            gate = torch.ones_like(base_weights[:1])
        mixed = base_weights + gate * (proposed - base_weights)
        fused.append((feature * mixed).sum(0, keepdim=True))
        weights_out.append(mixed)
        changes.append((mixed - base_weights).abs().sum(0).mean())
        gates.append(gate.mean())
    zero = levels[0].new_zeros(())
    return fused, weights_out, {
        'change': torch.stack(changes).mean() if changes else zero,
        'gate': torch.stack(gates).mean() if gates else zero,
    }


def _fuse_from_weights(levels, weights):
    return [(feature * weight).sum(0, keepdim=True)
            for feature, weight in zip(levels, weights)]


def _mean_weights(left, right):
    return [(a + b) * 0.5 for a, b in zip(left, right)]


def _decode_task(base, levels, head):
    if len(levels) != len(base.backbone.deblocks):
        raise ValueError('Fused level count differs from detector backbone')
    joined = torch.cat(
        [deblock(value) for deblock, value in zip(base.backbone.deblocks, levels)], 1
    )
    return head(joined)


def _task_metrics(weights_a, weights_b, scales, levels):
    metrics = {}
    gaps, disagreements = [], []
    for scale in scales:
        key_gap = f'gap_s{scale}'
        key_disagree = f'disagree_s{scale}'
        if scale >= len(weights_a) or len(levels[scale]) <= 1:
            zero = levels[0].new_zeros(())
            metrics[key_gap] = zero
            metrics[key_disagree] = zero
            continue
        left, right = weights_a[scale], weights_b[scale]
        gap = (left - right).abs().mean()
        disagree = (left.argmax(0) != right.argmax(0)).float().mean()
        metrics[key_gap] = gap
        metrics[key_disagree] = disagree
        gaps.append(gap)
        disagreements.append(disagree)
    zero = levels[0].new_zeros(())
    metrics['task_gap'] = torch.stack(gaps).mean() if gaps else zero
    metrics['task_disagreement'] = (
        torch.stack(disagreements).mean() if disagreements else zero
    )
    return metrics


class TaskFusionArm(nn.Module):
    """Two-router arm with either shared or task-specific source weights."""

    def __init__(self, source_fusion, mode):
        super().__init__()
        if mode not in ('shared', 'split'):
            raise ValueError('mode must be shared or split')
        self.mode = mode
        self.router_a = copy.deepcopy(source_fusion)
        self.router_b = copy.deepcopy(source_fusion)

    def predict(self, base, levels, collapse=False, swap=False):
        if collapse and swap:
            raise ValueError('collapse and swap are mutually exclusive')
        _, weights_a, info_a = _route(self.router_a, levels)
        _, weights_b, info_b = _route(self.router_b, levels)

        if self.mode == 'shared' or collapse:
            common = _mean_weights(weights_a, weights_b)
            cls_levels = _fuse_from_weights(levels, common)
            reg_levels = _fuse_from_weights(levels, common)
        else:
            cls_weights, reg_weights = weights_a, weights_b
            if swap:
                cls_weights, reg_weights = reg_weights, cls_weights
            cls_levels = _fuse_from_weights(levels, cls_weights)
            reg_levels = _fuse_from_weights(levels, reg_weights)

        prediction = {
            'psm': _decode_task(base, cls_levels, base.cls_head),
            'rm': _decode_task(base, reg_levels, base.reg_head),
        }
        info = {
            'change': 0.5 * (info_a['change'] + info_b['change']),
            'gate': 0.5 * (info_a['gate'] + info_b['gate']),
            **_task_metrics(weights_a, weights_b, self.router_a.scales, levels),
        }
        return prediction, info

    def trainable_parameters(self):
        return list(self.parameters())

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())
