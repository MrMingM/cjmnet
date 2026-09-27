"""Small server-side invariants for the task-split pilot."""
import copy

import torch
from torch import nn

from local_fusion_v3.network import SpatialSourceFusion
from .model import TaskFusionArm


class DummyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.deblocks = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Identity()])


class DummyBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = DummyBackbone()
        self.cls_head = nn.Conv2d(12, 2, 1)
        self.reg_head = nn.Conv2d(12, 14, 1)


def _equal_state(left, right):
    a, b = left.state_dict(), right.state_dict()
    return a.keys() == b.keys() and all(
        torch.equal(a[key], b[key]) for key in a
    )


def _levels():
    torch.manual_seed(7)
    return [torch.randn(3, 4, 6, 8) for _ in range(3)]


def _has_gradient(module):
    values = [parameter.grad for parameter in module.parameters()]
    return values and all(value is not None for value in values)


def main():
    torch.manual_seed(3)
    source = SpatialSourceFusion(
        channels=[4, 4, 4],
        variant='residual',
        hidden=4,
        scales=(0, 1),
        max_gate=.5,
    )
    base = DummyBase().requires_grad_(False).eval()
    shared = TaskFusionArm(source, 'shared')
    split = TaskFusionArm(source, 'split')

    assert shared.parameter_count() == split.parameter_count()
    assert _equal_state(shared, split)
    assert _equal_state(shared.router_a, shared.router_b)
    assert _equal_state(split.router_a, split.router_b)
    assert all(parameter.requires_grad for parameter in shared.parameters())
    assert all(not parameter.requires_grad for parameter in base.parameters())

    levels = _levels()
    initial_shared, shared_info = shared.predict(base, levels)
    initial_split, split_info = split.predict(base, levels)
    assert torch.equal(initial_shared['psm'], initial_split['psm'])
    assert torch.equal(initial_shared['rm'], initial_split['rm'])
    assert float(shared_info['task_gap']) == 0.0
    assert float(split_info['task_gap']) == 0.0

    shared.zero_grad(set_to_none=True)
    prediction, _ = shared.predict(base, _levels())
    (prediction['psm'].sum() + prediction['rm'].sum()).backward()
    assert _has_gradient(shared.router_a)
    assert _has_gradient(shared.router_b)

    split.zero_grad(set_to_none=True)
    prediction, _ = split.predict(base, _levels())
    (prediction['psm'].sum() + prediction['rm'].sum()).backward()
    assert _has_gradient(split.router_a)
    assert _has_gradient(split.router_b)

    changed = copy.deepcopy(split)
    with torch.no_grad():
        first = next(changed.router_b.parameters())
        first.add_(0.05 * torch.randn_like(first))
    normal, info = changed.predict(base, _levels())
    collapsed, _ = changed.predict(base, _levels(), collapse=True)
    swapped, _ = changed.predict(base, _levels(), swap=True)
    assert float(info['task_gap']) > 0.0
    assert not torch.equal(normal['psm'], collapsed['psm'])
    assert not torch.equal(normal['rm'], collapsed['rm'])
    assert normal['psm'].shape == swapped['psm'].shape
    assert normal['rm'].shape == swapped['rm'].shape

    print('task-split core tests passed', flush=True)


if __name__ == '__main__':
    main()
