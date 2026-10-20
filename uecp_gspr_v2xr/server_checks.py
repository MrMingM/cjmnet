"""Must run on Linux/HIP before real smoke. Never imported by local checks."""
import copy
import gc
from pathlib import Path

import numpy as np
import torch

from .common import ROOT, commit, read
from .runtime import host_config
from .model_adapter import make, original, criterion, losses
from .dual_quality_fusion import native, DualFusion, enable_precision_gradient
from .quality_adapter import QualityAdapter, aggregate, regional, shuffled
from .uncertainty_targets import pairwise


def run(output):
    root = Path(output); c = read(root / 'config.json'); checks = {}
    free, total = torch.cuda.mem_get_info()
    if free/total < .9:
        raise RuntimeError('Selected physical GPU has less than90% free VRAM; inspect rocm-smi and choose an idle card')
    checks['startup_capacity'] = dict(free_bytes=free, total_bytes=total,
        note='VRAM occupancy check; user must also inspect rocm-smi utilization before selecting GPU')
    # Reuse native AP None/empty/GT contract tests in the server environment.
    import importlib.util
    spec = importlib.util.spec_from_file_location('_uecp_native_ap_checks', ROOT / 'v2xr_protocol_audit/test_contract.py')
    tests = importlib.util.module_from_spec(spec); spec.loader.exec_module(tests); tests.run_tests()
    checks['native_ap_contract'] = True
    a = host_config(root)['model']['args']
    torch.manual_seed(23)
    base = original.BEVtextv1(a).cuda().eval()
    adapted = make(root, 'E0').cuda().eval()
    adapted.load_state_dict(base.state_dict(), strict=True)
    coords = torch.tensor([[b, 0, 30+i//20, 200+i%20] for b in range(2) for i in range(60)], device='cuda', dtype=torch.int32)
    vf = torch.zeros((len(coords), 32, 4), device='cuda')
    vf[:, :6] = torch.randn((len(coords), 6, 4), device='cuda')*.1
    vf[:, :6, 0] += coords[:, 3:4]*.4-140.6
    vf[:, :6, 1] += coords[:, 2:3]*.4-39.8
    vf[:, :6, 3] = vf[:, :6, 3].abs()
    transforms = [np.eye(4), np.eye(4)]; transforms[1][0, 3] = 1.6
    data = dict(processed_lidar=dict(voxel_features=vf, voxel_coords=coords,
        voxel_num_points=torch.full((len(coords),), 6, device='cuda', dtype=torch.int32)),
        record_len=torch.tensor([2], device='cuda'),
        pairwise_t_matrix=torch.tensor(pairwise(transforms)[None], device='cuda', dtype=torch.float32),
        empty_voxels=torch.zeros(len(coords), device='cuda', dtype=torch.bool),
        point_counts=torch.ones((2, 1, 200, 704), device='cuda'),
        un_maps=torch.rand((2, 200, 704), device='cuda'), quality_seeds=[1, 2])
    with torch.no_grad():
        ref = base(data); actual = adapted(copy.deepcopy(data))
    for key in ref:
        torch.testing.assert_close(actual[key], ref[key], atol=0, rtol=0)
    checks['E0_exact_original_forward_including_auxiliary'] = True
    h, w = actual['cls_preds'].shape[-2:]
    def labels(n):
        pos = torch.zeros((n, h, w, 2), device='cuda'); pos[:, h//2, w//2, 0] = 1
        return dict(pos_equal_one=pos, neg_equal_one=1-pos, targets=torch.zeros((n, h, w, 14), device='cuda'))
    cr = criterion(root).cuda()
    l, parts = losses(cr, actual, labels(1), labels(2))
    reference_loss = cr(ref, labels(1)) + cr(ref, labels(2), suffix='_single')
    torch.testing.assert_close(l, reference_loss, atol=0, rtol=0)
    checks['original_detection_direction_single_aux_loss'] = parts
    del base, adapted, ref, actual; gc.collect(); torch.cuda.empty_cache()
    # Test translation through actual normalization + actual grid_sample helper.
    transform = torch.eye(4, device='cuda').reshape(1, 1, 1, 4, 4).repeat(1, 2, 2, 1, 1)
    transform[0, 0, 1, 0, 3] = -1
    affine = original.normalize_pairwise_tfm(transform, 8, 12, 1)
    marker = torch.zeros((1, 1, 8, 12), device='cuda'); marker[0, 0, 4, 2] = 1
    warped = native._warp_affine(marker, affine[0, 0, 1:2], (8, 12))
    assert abs(float(warped[0, 0, 4, 3])-1) < 1e-5
    checks['source_to_ego_translation'] = True
    feature = torch.randn((2, 8, 8, 12), device='cuda')
    record = torch.tensor([2], device='cuda')
    identity_affine = torch.eye(2, 3, device='cuda').reshape(1, 1, 1, 2, 3).repeat(1, 2, 2, 1, 1)
    u = torch.rand((2, 1, 8, 12), device='cuda', requires_grad=True)
    ug = native.UGPrecisionFusion(8).cuda()
    result = ug(feature, u, record, identity_affine)
    assert torch.autograd.grad(result.square().mean(), u, allow_unused=True)[0] is None
    fusion = native.MSUGPrecisionFusionV1(8).cuda()
    enable_precision_gradient(fusion)
    result = fusion.blocks[0](feature, u, record, identity_affine)
    grad = torch.autograd.grad(result.square().mean(), u)[0]
    assert grad.isfinite().all() and grad.abs().sum() > 0
    checks['single_scale_detach_original_vs_controlled_removal'] = True
    dual = DualFusion.from_original(native.MSUGPrecisionFusionV1(8)).cuda()
    valid = torch.zeros_like(u, dtype=torch.bool); valid[:, :, 2:6, 3:9] = True
    coverage = torch.rand_like(u, requires_grad=True)
    quality = torch.rand_like(u, requires_grad=True)
    result = dual(feature, coverage, quality, valid, record, identity_affine)
    grads = torch.autograd.grad(result.square().mean(), (coverage, quality))
    assert all(g.isfinite().all() and g.abs().sum() > 0 for g in grads)
    assert all(b['empty_weight_mass'] == 0 for s in dual.last_stats for b in s['batches'])
    assert torch.count_nonzero(dual(feature, coverage, quality, valid & False, record, identity_affine)) == 0
    checks['E3_C_Q_gradients_empty_mask_and_all_missing'] = True
    adapter = QualityAdapter().cuda()
    q = torch.rand((2, 4, 8, 12), device='cuda')
    correction = adapter(u, 1-u, q, valid)
    torch.testing.assert_close(correction, u, atol=1e-6, rtol=1e-5)
    correction.mean().backward()
    assert adapter.net[-1].weight.grad.abs().sum() > 0
    shuffled_q = shuffled(q, valid, torch.ones_like(u)*8, [1, 2])
    assert torch.equal(shuffled_q[~valid.expand_as(q)], q[~valid.expand_as(q)])
    checks['identity_adapter_gradient_and_shuffling_mask'] = True
    packed = dict(voxel_coords=torch.tensor([[0, 0, 0, 0], [1, 0, 0, 0]], device='cuda'))
    points = dict(point_valid_mask=torch.tensor([[1, 0], [1, 0]], dtype=torch.bool, device='cuda'),
        point_reliability=torch.tensor([[.7, 99], [.9, 99]], device='cuda'),
        point_uncertainty=torch.tensor([[.4, 99], [.2, 99]], device='cuda'),
        point_evidence=torch.ones((2, 2, 2), device='cuda'))
    grid = aggregate(points, packed, 2, torch.tensor([False, True], device='cuda'), height=4, width=8)
    values, observed = regional(grid, (2, 4))
    assert abs(float(values[0, 0, 0, 0])-.7) < 1e-6
    assert abs(float(values[0, 2, 0, 0])-.4) < 1e-6
    assert not observed[1].any() and values[1].abs().sum() == 0
    checks['valid_point_aggregation_padding_and_independent_uncertainty'] = True
    for variant in c['variants'][1:]:
        model = make(root, variant).cuda().train()
        inputs = copy.deepcopy(data)
        result = model(inputs)
        loss, _ = losses(criterion(root).cuda(), result, labels(1), labels(2))
        loss.backward()
        assert torch.isfinite(loss)
        assert not any(p.grad is not None for p in model.gspr.parameters())
        if variant != 'E3_coverage':
            assert model.quality_adapter.net[-1].weight.grad.abs().sum() > 0
        checks[variant+'_full_model_loss_frozen_GSPR_and_adapter_gradient'] = True
        del model, result, loss; gc.collect(); torch.cuda.empty_cache()
    path = root / 'self_test/complete.json'
    return commit(path, [], checks=checks, synthetic_tensors_only=True, executed_on='server HIP')


def resume_check(output, variant, epoch=2):
    from .common import complete, variant_path
    root = Path(output)
    expected = variant_path(root, variant, True) / 'epochs' / f'{epoch:02d}'
    actual = root / 'smoke_replay' / variant / 'epochs' / f'{epoch:02d}'
    a, b = complete(expected / 'train_complete.json'), complete(actual / 'train_complete.json')
    state_a = torch.load(expected / 'resume.pth', map_location='cpu', weights_only=False)
    state_b = torch.load(actual / 'resume.pth', map_location='cpu', weights_only=False)
    tolerance = read(root / 'config.json')['resume_replay']
    maximum = 0.
    def same(x, y):
        nonlocal maximum
        if isinstance(x, torch.Tensor):
            if x.is_floating_point():
                maximum = max(maximum, float((x-y).abs().max()) if x.numel() else 0.)
                torch.testing.assert_close(x, y, **tolerance)
            else:
                assert torch.equal(x, y)
        elif isinstance(x, dict):
            assert x.keys() == y.keys()
            for k in x:
                same(x[k], y[k])
        elif isinstance(x, (list, tuple)):
            assert len(x) == len(y)
            for p, q in zip(x, y):
                same(p, q)
        elif isinstance(x, np.ndarray):
            np.testing.assert_equal(x, y)
        elif isinstance(x, float):
            np.testing.assert_allclose(x, y, **tolerance)
        else:
            assert x == y
    for key in ('model', 'optimizer', 'scheduler', 'rng', 'global_step', 'totals', 'actual'):
        same(state_a[key], state_b[key])
    for key in ('position', 'order_identity', 'microbatch', 'accumulation_position'):
        if state_a[key] != state_b[key]:
            raise ValueError('Smoke recovery metadata differs: '+key)
    return commit(root / 'smoke' / (variant+f'_epoch{epoch}_resume_complete.json'), [],
        exact_model_tensors=a['final_tensor_sha256'] == b['final_tensor_sha256'],
        maximum_float_tensor_delta=maximum, tolerance=tolerance,
        optimizer_rng_sampler_restored=True, global_step=state_a['global_step'])
