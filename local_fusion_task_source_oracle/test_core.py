"""Synthetic checks for the direction-B task/source Oracle."""
import torch
from torch import nn

from local_fusion_task_source_oracle.oracle import (
    KEEP,
    build_candidate_pool,
    compose_prediction,
    decide,
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


def main():
    torch.manual_seed(41)
    base = DummyBase().eval()
    levels = [torch.randn(3, 4, 5, 7) for _ in range(3)]
    shared = {
        'psm': torch.randn(1, 2, 5, 7),
        'rm': torch.randn(1, 14, 5, 7),
    }
    pool = build_candidate_pool(
        base, levels, shared, ('single', 'query'))
    assert KEEP in pool
    assert len(pool) == 1 + 2 * 3
    for name, pred in pool.items():
        assert pred['psm'].shape == shared['psm'].shape, name
        assert pred['rm'].shape == shared['rm'].shape, name

    mask = torch.zeros(5, 7, dtype=torch.bool)
    mask[1:3, 2:5] = True
    mixed = compose_prediction(
        shared, pool, mask.numpy(), 'single:1', 'query:2')
    m = mask[None, None]
    torch.testing.assert_close(
        mixed['psm'].masked_select(m),
        pool['single:1']['psm'].masked_select(m))
    torch.testing.assert_close(
        mixed['rm'].masked_select(m),
        pool['query:2']['rm'].masked_select(m))
    torch.testing.assert_close(
        mixed['psm'].masked_select(~m),
        shared['psm'].masked_select(~m))
    torch.testing.assert_close(
        mixed['rm'].masked_select(~m),
        shared['rm'].masked_select(~m))

    keep = compose_prediction(shared, pool, mask.numpy(), KEEP, KEEP)
    torch.testing.assert_close(keep['psm'], shared['psm'])
    torch.testing.assert_close(keep['rm'], shared['rm'])

    spec = {
        'minimum_mean_task_vs_shared': 0.015,
        'minimum_weathers_ge_1pp': 2,
        'per_weather_task_vs_shared': 0.010,
        'minimum_mean_task_vs_same': 0.005,
    }
    good = {
        w: {'results': {
            'Shared': {'ap70': .70},
            'Oracle-Same': {'ap70': .71},
            'Oracle-Task': {'ap70': .72},
        }}
        for w in ('fog', 'rain', 'snow')
    }
    assert decide(good, spec)['direction_b_survives']
    weak = {
        w: {'results': {
            'Shared': {'ap70': .70},
            'Oracle-Same': {'ap70': .704},
            'Oracle-Task': {'ap70': .708},
        }}
        for w in ('fog', 'rain', 'snow')
    }
    assert decide(weak, spec)['direction_b_kill']

    print('direction-B task/source Oracle tests passed', flush=True)


if __name__ == '__main__':
    main()
