"""Server-side invariants for B1 local task-conflict gating."""
import copy

import torch
from torch import nn

from local_fusion_task_split_pilot.model import TaskFusionArm
from local_fusion_v3.network import SpatialSourceFusion

from .model import (
    TaskConflictGate,
    prepare_frozen_context,
    predict_with_gate,
)


class DummyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.deblocks = nn.ModuleList([
            nn.Identity(), nn.Identity(), nn.Identity()])


class DummyBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = DummyBackbone()
        self.cls_head = nn.Conv2d(12, 2, 1)
        self.reg_head = nn.Conv2d(12, 14, 1)


class ConstantGate(nn.Module):
    def __init__(self, scales, value):
        super().__init__()
        self.scales = tuple(scales)
        self.value = float(value)

    def forward(self, scale, features):
        return features.new_full(
            (features.shape[0], 1, features.shape[-2], features.shape[-1]),
            self.value,
        )


def _levels():
    torch.manual_seed(17)
    return [torch.randn(3, 4, 6, 8) for _ in range(3)]


def _equal_state(left, right):
    a, b = left.state_dict(), right.state_dict()
    return a.keys() == b.keys() and all(
        torch.equal(a[key], b[key]) for key in a)


def _assert_prediction_equal(left, right):
    torch.testing.assert_close(left['psm'], right['psm'], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(left['rm'], right['rm'], atol=1e-6, rtol=1e-6)


def main():
    torch.manual_seed(5)
    source = SpatialSourceFusion(
        channels=[4, 4, 4],
        variant='residual',
        hidden=4,
        scales=(0, 1),
        max_gate=.5,
    )
    split = TaskFusionArm(source, 'split')
    with torch.no_grad():
        # Make the two frozen B0 routers genuinely task-specific.
        first = next(split.router_b.parameters())
        first.add_(0.1 * torch.randn_like(first))
    split.requires_grad_(False).eval()

    base = DummyBase().requires_grad_(False).eval()
    levels = _levels()
    frozen = prepare_frozen_context(
        base, levels, split, scales=(0, 1))

    direct_split, _ = split.predict(base, levels)
    direct_collapse, _ = split.predict(base, levels, collapse=True)
    _assert_prediction_equal(
        frozen['split_prediction'], direct_split)
    _assert_prediction_equal(
        frozen['collapse_prediction'], direct_collapse)

    zero_prediction, _ = predict_with_gate(
        base, frozen, ConstantGate((0, 1), 0.0))
    one_prediction, _ = predict_with_gate(
        base, frozen, ConstantGate((0, 1), 1.0))
    _assert_prediction_equal(
        zero_prediction, frozen['collapse_prediction'])
    _assert_prediction_equal(
        one_prediction, frozen['split_prediction'])

    local = TaskConflictGate(
        scales=(0, 1), hidden=8, mode='local', initial_bias=-4.0)
    global_gate = TaskConflictGate(
        scales=(0, 1), hidden=8, mode='global', initial_bias=-4.0)
    global_gate.load_state_dict(local.state_dict(), strict=True)
    assert _equal_state(local, global_gate)
    assert local.parameter_count() == global_gate.parameter_count()

    initial_local, local_info = predict_with_gate(
        base, frozen, local)
    initial_global, global_info = predict_with_gate(
        base, frozen, global_gate)
    expected = float(torch.sigmoid(torch.tensor(-4.0)))
    assert abs(float(local_info['gate_mean']) - expected) < 1e-6
    assert abs(float(global_info['gate_mean']) - expected) < 1e-6
    _assert_prediction_equal(initial_local, initial_global)

    local.zero_grad(set_to_none=True)
    prediction, _ = predict_with_gate(base, frozen, local)
    (prediction['psm'].sum() + prediction['rm'].sum()).backward()
    assert all(parameter.grad is not None for parameter in local.parameters())
    assert all(parameter.grad is None for parameter in split.parameters())
    assert all(parameter.grad is None for parameter in base.parameters())

    # After a controlled nonzero final layer, Local may vary spatially while
    # Global must remain one broadcast value per scale.
    probe_local = copy.deepcopy(local)
    probe_global = copy.deepcopy(global_gate)
    with torch.no_grad():
        for gate in (probe_local, probe_global):
            for network in gate.networks.values():
                network[-1].weight.fill_(0.2)
                network[-1].bias.fill_(-1.0)
    _, local_probe = predict_with_gate(base, frozen, probe_local)
    _, global_probe = predict_with_gate(base, frozen, probe_global)
    for scale in (0, 1):
        local_map = local_probe['gate_maps'][scale]
        global_map = global_probe['gate_maps'][scale]
        assert float(global_map.std()) < 1e-7
        assert float(local_map.std()) > 0.0

    print('task-split B1 core tests passed', flush=True)


if __name__ == '__main__':
    main()
