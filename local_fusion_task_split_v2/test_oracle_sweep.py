"""Synthetic invariants for the B1 oracle/sweep audit."""
import numpy as np
import torch
from torch import nn

from local_fusion_task_split_pilot.model import TaskFusionArm
from local_fusion_v3.network import SpatialSourceFusion

from .model import prepare_frozen_context
from .oracle_sweep import (
    _constant_maps,
    _oracle_maps,
    _predict_with_maps,
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


def _levels():
    torch.manual_seed(31)
    return [torch.randn(3, 4, 6, 8) for _ in range(3)]


def _assert_prediction_equal(left, right):
    torch.testing.assert_close(
        left['psm'], right['psm'], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        left['rm'], right['rm'], atol=1e-6, rtol=1e-6)


def main():
    torch.manual_seed(29)
    source = SpatialSourceFusion(
        channels=[4, 4, 4],
        variant='residual',
        hidden=4,
        scales=(0, 1),
        max_gate=.5,
    )
    split = TaskFusionArm(source, 'split')
    with torch.no_grad():
        first = next(split.router_b.parameters())
        first.add_(0.1 * torch.randn_like(first))
    split.requires_grad_(False).eval()

    base = DummyBase().requires_grad_(False).eval()
    frozen = prepare_frozen_context(
        base, _levels(), split, scales=(0, 1))

    zero = _predict_with_maps(
        base, frozen, _constant_maps(frozen, 0.0))
    one = _predict_with_maps(
        base, frozen, _constant_maps(frozen, 1.0))
    _assert_prediction_equal(zero, frozen['collapse_prediction'])
    _assert_prediction_equal(one, frozen['split_prediction'])

    # One centered oriented target should open only a strict spatial subset.
    corners = np.array([[
        [-0.5, -0.5, 0.0],
        [ 0.5, -0.5, 0.0],
        [ 0.5,  0.5, 0.0],
        [-0.5,  0.5, 0.0],
        [-0.5, -0.5, 1.0],
        [ 0.5, -0.5, 1.0],
        [ 0.5,  0.5, 1.0],
        [-0.5,  0.5, 1.0],
    ]], dtype=np.float32)
    maps, fallback, active, total = _oracle_maps(
        frozen,
        corners,
        {0},
        [-4.0, -3.0, -1.0, 4.0, 3.0, 1.0],
        1.0,
    )
    for scale in (0, 1):
        key = str(scale)
        assert fallback[key] == 0
        assert 0 < active[key] < total[key]
        values = maps[scale]
        assert float(values.min()) == 0.0
        assert float(values.max()) == 1.0

    empty_maps, _, empty_active, empty_total = _oracle_maps(
        frozen,
        corners,
        set(),
        [-4.0, -3.0, -1.0, 4.0, 3.0, 1.0],
        1.0,
    )
    for scale in (0, 1):
        key = str(scale)
        assert empty_active[key] == 0
        assert empty_total[key] > 0
        assert float(empty_maps[scale].max()) == 0.0

    print('B1 oracle/sweep tests passed', flush=True)


if __name__ == '__main__':
    main()
