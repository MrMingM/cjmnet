"""Server acceptance for the transport objective and the real LR0 integration."""
import math

import torch

from common import atomic_json, read_json, sha256
from checkpoint import restore, rng, save, seed, state
from fusion import CollaborativeFusion, support_maps
from runtime import SEED, build, create_model, loader
from transport import Grid, dustbin, unbalanced
from v2xr_weather_l012.data import tensor_digest


def check_close(actual, expected, **tolerances):
    torch.testing.assert_close(actual, expected, **tolerances)


def numerical(cfg):
    import numpy as np
    import ot
    from opencood.models.common_modules.self_attn import AttFusion

    device = torch.device('cuda:0')
    grid = Grid(1, 3, 5, device)
    matrix = torch.tensor([[.15, 1.0, 2.4], [.9, .3, 1.9], [2.2, 1.6, .5]], device=device)
    cost = matrix[torch.arange(3, device=device)[:, None], grid.neighbor][None].requires_grad_()
    a = torch.tensor([[.5, .2, .1]], device=device)
    b = torch.tensor([[.3, .6, .5]], device=device)
    eps, r1, r2, steps = .24, .45, .62, 55
    plan = unbalanced(cost, a, b, grid, eps, r1, r2, steps)
    dense = plan.weights[0, :, 12-2:12+3].masked_select(grid.valid[:, 10:15]).reshape(3, 3)
    oracle = ot.unbalanced.sinkhorn_unbalanced(a[0].cpu().numpy().astype('float64'),
        b[0].cpu().numpy().astype('float64'), matrix.cpu().numpy().astype('float64'),
        eps, (r1, r2), reg_type='entropy', numItermax=4000, stopThr=1e-12)
    reference = torch.as_tensor(oracle, device=device, dtype=dense.dtype)
    check_close(dense, reference, rtol=3e-5, atol=2e-6)
    stationarity = (matrix + eps * dense.log() + r1 * (dense.sum(1)/a[0]).log()[:, None]
                    + r2 * (dense.sum(0)/b[0]).log()[None])
    assert float(stationarity.detach().abs().max()) < 1e-5
    check_close(plan.row, dense.sum(1)[None])
    check_close(plan.col, dense.sum(0)[None])
    assert abs(float(dense.detach().sum()) - float(a.sum())) > .01
    assert abs(float(dense.detach().sum()) - float(b.sum())) > .01

    def probe(c):
        return unbalanced(c, a, b, grid, eps, r1, r2, steps).weights.square().sum()

    grad, = torch.autograd.grad(probe(cost), cost)
    direction = torch.zeros_like(cost)
    direction[0, 1, 12] = 1.
    h = .002
    with torch.no_grad():
        fd = (probe(cost + h*direction) - probe(cost - h*direction)) / (2*h)
    check_close((grad*direction).sum(), fd, rtol=.003, atol=3e-5)
    assert torch.isfinite(grad).all() and torch.count_nonzero(grad)

    # With a hard fixed mass cap, cheap correspondences are used up to that cap.
    partial = ot.partial.partial_wasserstein(np.array([.6, .4]), np.array([.4, .6]),
        np.array([[0., 4.], [4., 2.]]), m=.4)
    np.testing.assert_allclose(partial, [[.4, 0.], [0., 0.]], atol=1e-8)

    bin_logit = -1.4
    aug = np.full((4, 4), -bin_logit * eps)
    aug[:3, :3] = matrix.cpu().numpy()
    av, bv = a[0].cpu().numpy().astype('float64'), b[0].cpu().numpy().astype('float64')
    expected_bin = ot.sinkhorn(np.r_[av, bv.sum()], np.r_[bv, av.sum()], aug,
                              eps, numItermax=2000, stopThr=1e-12)
    reject = dustbin(cost, a, b, grid, eps, bin_logit, 150)
    actual_bin = reject.weights[0, :, 10:15].masked_select(grid.valid[:, 10:15]).reshape(3, 3)
    check_close(actual_bin, torch.as_tensor(expected_bin[:3, :3], device=device).float(),
                rtol=1e-4, atol=2e-6)
    assert float(reject.residual) < 1e-5

    # A one-cell coupling has an analytic optimum even when the two priors differ.
    one = Grid(1, 1, 3, device)
    c = torch.full((1, 1, 9), 4., device=device)
    a1, b1 = a[:, :1], b[:, :1]
    test = unbalanced(c, a1, b1, one, eps, r1, r2, steps)
    expected = ((r1*a1.log() + r2*b1.log() - 4.) / (eps+r1+r2)).exp()
    check_close(test.row, expected)
    check_close(test.col_remaining, b1 - expected)
    assert float(test.col_remaining) > .25
    for no_a, no_b in ((a*0, b), (a, b*0), (a*0, b*0)):
        empty = unbalanced(cost, no_a, no_b, grid, eps, r1, r2, steps)
        assert not torch.count_nonzero(empty.weights)
        check_close(empty.row_remaining, no_a)
        check_close(empty.col_remaining, no_b)
        empty_grad, = torch.autograd.grad(empty.weights.sum(), cost)
        assert torch.isfinite(empty_grad).all() and not torch.count_nonzero(empty_grad)

    torch.manual_seed(20261017)
    native = AttFusion(8).to(device)
    layer = CollaborativeFusion(native, 8, cfg, 'unbalanced', 'matched_plus_unmatched').to(device)
    x = torch.randn(5, 8, 3, 4, device=device)
    counts = torch.tensor([2, 1, 2], device=device)
    evidence = torch.zeros(5, 12, 2, dtype=torch.long, device=device)
    evidence[0, 0, 0], evidence[1, 1, 0], evidence[1, 11, 1] = 3, 3, 3
    evidence[3, 5, 0], evidence[4, 5, 0] = 3, 3
    layer.sources = evidence
    with torch.no_grad():
        expected_out = native(x, counts)
        assert torch.equal(layer(x, counts), expected_out)
        layer.mode = 'module_off'
        assert torch.equal(layer(x, counts), expected_out)
        layer.mode = 'matched_plus_unmatched'
        layer.matched.weight.fill_(.2)
        layer.unmatched.weight.fill_(.2)
        layer.observe = True
        both = layer(x, counts)
        report = layer.report
        assert float(report['row'][0, 0]) > .1
        assert float(report['col'][0, 11]) == 0 and float(report['unmatched_weight'][0, 11]) > 0
        assert torch.equal(both[1], expected_out[1]) and not torch.equal(both[0], expected_out[0])
        x_other = x.clone()
        x_other[3:] *= -2
        assert torch.equal(layer(x_other, counts)[:2], both[:2])
        for mode in ('matched_only', 'unmatched_only'):
            layer.mode = mode
            assert not torch.equal(layer(x, counts), both)
        layer.sources = torch.zeros_like(evidence)
        assert torch.equal(layer(x, counts), expected_out)
    return dict(pot=ot.__version__, max_plan_error=float((dense-reference).abs().max().detach()),
                kkt_residual=float(stationarity.abs().max().detach()),
                fixed_point_residual=float(plan.residual), dustbin_error=float((actual_bin-torch.as_tensor(
                    expected_bin[:3, :3], device=device)).abs().max().detach()),
                transported_mass=float(dense.detach().sum()), input_mass=[float(a.sum()), float(b.sum())],
                finite_difference=float(fd), autograd=float((grad*direction).sum()),
                cases=['unequal mass', 'high cost rejection', 'zero support', 'gradient',
                       'dustbin', 'fixed partial mass reference', 'scene isolation',
                       'single vehicle', 'match/unmatched switches'])


def assert_layout(ego):
    counts = ego['record_len'].tolist()
    for flags, count in zip(ego['ego_flag'], counts, strict=True):
        if list(map(bool, flags)) != [True] + [False]*(count - 1):
            raise ValueError('Native vehicle order does not put ego first')
    for sensor in ('lidar', 'radar'):
        pillar = ego[f'processed_{sensor}']
        coordinates = pillar['voxel_coords']
        n = pillar['voxel_num_points']
        if (len(n) != len(coordinates) or not bool(((n > 0) & (n <= 32)).all())
                or not bool(((coordinates >= 0) & (coordinates < coordinates.new_tensor(
                    [sum(counts), 1, 200, 704]))).all())):
            raise ValueError(f'Invalid nonempty {sensor} pillar contract')


def integration(config, settings, groups, factory, weight):
    from opencood.tools import train_utils

    dataset, cfg = build(config, condition='clean')
    # Ensure the real equality test exercises cooperation, not just the single-car shortcut.
    ego = None
    for batch in loader(dataset, evaluation=True, limit=12):
        if int(batch['ego']['record_len'][0]) > 1:
            ego = train_utils.to_device(batch['ego'], torch.device('cuda:0'))
            break
    if ego is None:
        raise ValueError('No multicar input in the fixed smoke frames')
    assert_layout(ego)
    seed(SEED)
    native = create_model(cfg).eval()
    shared = tensor_digest(native.state_dict())
    init_rng = tensor_digest(rng())
    native.load_state_dict(torch.load(weight, map_location='cpu', weights_only=True), strict=True)
    evidence = {}
    shapes = []
    hooks = [layer.register_forward_pre_hook(lambda _, args: shapes.append(list(args[0].shape)))
             for layer in native.backbone.fuse_modules]
    with torch.no_grad():
        expected = native(ego)
    for hook in hooks:
        hook.remove()
    assert [row[1:] for row in shapes] == [[128, 100, 352], [256, 50, 176], [512, 25, 88]]
    for group in groups:
        seed(SEED)
        model = factory(cfg, group).eval()
        current = model.state_dict()
        if tensor_digest({k: current[k] for k in native.state_dict()}) != shared or tensor_digest(rng()) != init_rng:
            raise ValueError(f'{group} changed the shared initialization or RNG')
        extra = {k: v for k, v in current.items() if k not in native.state_dict()}
        evidence[group] = dict(new_parameters=sum(v.numel() for v in extra.values()),
                               initialization_sha256=tensor_digest(extra))
        missing, unexpected = model.load_state_dict(native.state_dict(), strict=False)
        assert set(missing) == set(extra) and not unexpected
        default = model.fusion.mode
        with torch.no_grad():
            for mode in (default, 'module_off'):
                model.fusion.mode = mode
                got = model(ego)
                assert all(torch.equal(got[k], expected[k]) for k in got), (group, mode)
            model.fusion.mode = default
            if group == 'P3':
                for sensors in (('lidar',), ('radar',), ('lidar', 'radar')):
                    subset = dict(ego)
                    for sensor in sensors:
                        key = f'processed_{sensor}'
                        subset[key] = {k: v[:0] for k, v in ego[key].items()}
                    sources = support_maps(subset, sum(ego['record_len'].tolist()), model.coarse_shape, model.stride)
                    for sensor in sensors:
                        assert not torch.count_nonzero(sources[..., ('lidar', 'radar').index(sensor)])
                    assert all(torch.isfinite(v).all() for v in model(subset).values())
                model.single_vehicle = True
                assert all(torch.isfinite(v).all() for v in model(ego).values())
        del model
    return dict(shared_initialization_sha256=shared, rng_sha256=init_rng, variants=evidence,
                attention_inputs=shapes, head_shapes={k: list(v.shape) for k, v in expected.items()},
                exact_native_and_zero_init=True, real_empty_modalities=True, real_single_vehicle=True)


def roundtrip(path, model, optimizer, scheduler, criterion, contract):
    save(path, state(model, optimizer, scheduler, criterion, 1, contract))
    components = [model, optimizer, scheduler, criterion]
    expected = tensor_digest([part.state_dict() for part in components] + [rng()])
    seed(SEED+1)
    restore(path, model, optimizer, scheduler, criterion, contract, 1)
    if expected != tensor_digest([part.state_dict() for part in components] + [rng()]):
        raise ValueError('Checkpoint did not restore full training state')
    return sha256(path)
