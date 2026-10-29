"""Server-only structural, numerical and real-loader checks; no A0 optimization."""
import copy
from pathlib import Path
import itertools
import torch
from auditlib import METHOD, ROOT, atomic_json, complete, load_frozen
from checkpoint import seed, save


def bn_state(model):
    return {k: v.clone() for k, v in model.state_dict().items()
            if any(s in k for s in ('running_mean', 'running_var', 'num_batches_tracked'))}


def same(a, b):
    if set(a) != set(b) or any(not torch.equal(a[k], b[k]) for k in a):
        raise ValueError('Exact tensor/state equality failed')


def finite_gradients(model):
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads or any(not torch.isfinite(g).all() for g in grads):
        raise ValueError('Nonfinite/missing full-model gradients')
    if any(not torch.isfinite(v).all() for v in model.state_dict().values() if v.is_floating_point()):
        raise ValueError('Nonfinite model state')


def module_tests():
    from modules import GGCA
    seed()
    rows = {}
    for counts in ([0, 0], [1, 1], [2, 2], [3, 6], [4, 8]):
        for level in (1, 2, 3):
            n = torch.tensor(counts, device='cuda:0')
            valid = torch.arange(8, device='cuda:0')[None] < n[:, None]
            f = torch.randn(2, 8, 64, device='cuda:0', requires_grad=True)
            base = torch.randn(2, 64, device='cuda:0', requires_grad=True)
            xyz = torch.zeros(2, 8, 3, device='cuda:0', requires_grad=True)
            # Two separate local surfaces; no single-center outlier decision.
            with torch.no_grad():
                xyz[:, 4:, 2] = 2
            r = torch.full((2, 8), .9 if sum(counts) < 10 else .06, device='cuda:0', requires_grad=True)
            u = torch.full((2, 8), .2, device='cuda:0', requires_grad=True)
            net = GGCA(METHOD).cuda().train()
            with torch.no_grad():
                net.project.weight.normal_(0, .02)
            ranges = torch.ones_like(r) * 70
            result = net(f, base, xyz, valid, r, u, ranges, level)
            dirty_f = f.detach().masked_fill(~valid[..., None], float('nan'))
            dirty_xyz = xyz.detach().masked_fill(~valid[..., None], float('nan'))
            other = net(dirty_f, base, dirty_xyz, valid, r, u, ranges, level)
            torch.testing.assert_close(result, other, rtol=0, atol=0)
            if level == 3 and max(counts) <= 2:
                torch.testing.assert_close(result, base, rtol=0, atol=0)
            result.square().mean().backward()
            if not torch.isfinite(result).all() or any(v.grad is not None and not torch.isfinite(v.grad).all() for v in (f, base, xyz, r, u)):
                raise ValueError('Degenerate pillar numerical failure')
            rows[str(counts) + ':' + str(level)] = net.last
    empty = GGCA(METHOD).cuda()(torch.empty(0, 8, 64, device='cuda:0'), torch.empty(0, 64, device='cuda:0'),
        torch.empty(0, 8, 3, device='cuda:0'), torch.empty(0, 8, dtype=torch.bool, device='cuda:0'),
        *(torch.empty(0, 8, device='cuda:0') for _ in range(3)), 3)
    assert empty.shape == (0, 64)
    return rows


def b1_factorial(pfn, decorated, valid, r):
    """Eight independent same-weight PFN probes, not extra training experiments."""
    from torch.nn import functional as F
    outputs, states = {}, {}
    decorated, valid, r = decorated[:2048].detach(), valid[:2048], r[:2048].detach()
    for weighted, valid_bn, masked_max in itertools.product((False, True), repeat=3):
        # Use clones so diagnostic BN updates never touch the audited model.
        linear, norm = copy.deepcopy(pfn.linear), copy.deepcopy(pfn.norm)
        norm.train()
        x = linear(decorated * r[..., None] if weighted else decorated)
        with torch.backends.cudnn.flags(enabled=False):
            if valid_bn:
                if int(valid.sum()) < 2:
                    raise ValueError('Factorial probe needs at least two valid slots')
                selected = norm(x[valid])
                x = x.new_zeros(x.shape).index_put((valid,), selected)
            else:
                x = norm(x.transpose(1, 2)).transpose(1, 2)
        x = F.relu(x)
        if masked_max:
            x = x.masked_fill(~valid[..., None], -1e4)
        key = f'gspr_weighted={weighted},valid_only_bn={valid_bn},masked_max={masked_max}'
        outputs[key] = x.amax(1).detach()
        states[key] = {'running_mean': norm.running_mean.detach(), 'running_var': norm.running_var.detach()}
    baseline = 'gspr_weighted=True,valid_only_bn=False,masked_max=False'
    return dict(scope='first up to 2048 observed pillars; same tensors, weights, initial BN state; no detector optimization',
        comparisons={k: dict(pillar_rms=float((v-outputs[baseline]).square().mean().sqrt()),
            bn_mean_rms=float((states[k]['running_mean']-states[baseline]['running_mean']).square().mean().sqrt()),
            bn_var_rms=float((states[k]['running_var']-states[baseline]['running_var']).square().mean().sqrt()))
            for k, v in outputs.items()})


def run(args):
    from opencood.hypes_yaml import yaml_utils
    from models import Model
    from bridge import load
    root = Path(args.output_root); root.mkdir(parents=True, exist_ok=True)
    from lineage import plan
    from test_geometry_server import run as geometry_tests
    from test_a1_restore_server import run as restore_A1_test
    migration = plan(args.experiment_root)
    a1_restore = restore_A1_test(args)
    geometry_mechanism = geometry_tests(root, migration['old_source'])
    legacy = load_frozen('ggca_fixtures', 'server_tests.py', ROOT / 'v2xr_weather_l012')
    legacy.test_full_state(root)
    degenerate = module_tests()
    h = yaml_utils.load_yaml(str(Path(args.runtime_root) / 'opencood/hypes_yaml/V2X-R/LiDAR/V2XR_attfuse.yaml'))
    cfg = h['model']['args']
    batch = legacy.batch()
    batch.update(lidar_pose=[[[0., 0., 0., 0., 0., 0.]]], ego_flag=[[True]])
    original_class = load('old_repe_models', 'gspr_repe/models.py').Model
    seed(); reference = original_class(cfg, 'B0').cuda()
    ref_state = copy.deepcopy(reference.state_dict())
    # Train-mode equality checks both outputs and actual running BN buffers.
    reference.train()
    original_pfn = []
    pfn_handle = reference.pillar_vfe.register_forward_hook(
        lambda module, inputs, result: original_pfn.append(result['pillar_features'].detach().clone()))
    with torch.no_grad():
        expected = reference(copy.deepcopy(batch))
    pfn_handle.remove()
    expected_bn = bn_state(reference)
    rows = {}
    for variant in ('A0', 'A1', 'A2', 'A3'):
        seed(); model = Model(cfg, variant).cuda().train()
        if variant != 'A0':
            model.capture_pfn = True
        shared = model.state_dict()
        shared.update(ref_state)
        model.load_state_dict(shared, strict=True)
        with torch.no_grad():
            actual = model(copy.deepcopy(batch))
        for key in ('psm', 'rm'):
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        same(bn_state(model), expected_bn)
        if variant != 'A0':
            torch.testing.assert_close(model.last_pfn, original_pfn[0], rtol=0, atol=0)
        if variant == 'A0':
            rows[variant] = dict(exact=True, bn_exact=True)
            del model, actual
            continue
        model.load_state_dict(shared, strict=True)
        model.enabled = False
        with torch.no_grad():
            disabled = model(copy.deepcopy(batch))
        for key in ('psm', 'rm'):
            torch.testing.assert_close(disabled[key], expected[key], rtol=0, atol=0)
        same(bn_state(model), expected_bn)
        model.enabled = True
        model.ggca.force_zero = True
        model.load_state_dict(shared, strict=True)
        with torch.no_grad():
            zero = model(copy.deepcopy(batch))
        for key in ('psm', 'rm'):
            torch.testing.assert_close(zero[key], expected[key], rtol=0, atol=0)
        same(bn_state(model), expected_bn)
        model.ggca.force_zero = False
        # Synthetic fixture uses eight points so A3 has enough evidence to update.
        train_batch = copy.deepcopy(batch)
        train_batch['processed_lidar']['voxel_num_points'].fill_(8)
        train_batch['processed_lidar']['voxel_features'][:, 4:8] = train_batch['processed_lidar']['voxel_features'][:, :4]
        optimizer = torch.optim.Adam(model.parameters(), lr=.002)
        before = {k: v.clone() for k, v in model.ggca.state_dict().items()}
        seen = set()
        for step in range(3):
            optimizer.zero_grad()
            prediction = model(copy.deepcopy(train_batch))
            loss = prediction['psm'].square().mean() + prediction['rm'].square().mean()
            loss.backward(); finite_gradients(model); model.repe_check_gradients()
            seen.update(k for k, p in model.ggca.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0)
            optimizer.step(); finite_gradients(model)
        if not any(not torch.equal(v, model.ggca.state_dict()[k]) for k, v in before.items()):
            raise ValueError('GGCA parameters did not update')
        if not any(k.startswith('query.') or k.startswith('geometry.') for k in seen):
            raise ValueError('Relation branch never received gradients after residual warm start')
        model.eval()
        with torch.no_grad():
            expected_reload = model(copy.deepcopy(train_batch))
            save(root / (variant + '.pth'), model.state_dict())
            restored = Model(cfg, variant).cuda().eval()
            restored.load_state_dict(torch.load(root / (variant + '.pth'), map_location='cpu', weights_only=False), strict=True)
            actual_reload = restored(copy.deepcopy(train_batch))
            for key in ('psm', 'rm'):
                torch.testing.assert_close(expected_reload[key], actual_reload[key], rtol=0, atol=0)
        rows[variant] = dict(zero_residual_exact=True, disabled_exact=True, original_bn_exact=True,
                             original_pfn_output_exact=True, updated=True, gradient_parameters=sorted(seen), strict_reload=True)
        del model, restored, actual, zero, disabled, optimizer, prediction, loss
        torch.cuda.empty_cache()
    if getattr(args, 'synthetic_only', False):
        atomic_json(root / 'SYNTHETIC_ONLY.json', dict(status='complete', models=rows,
            degenerate=degenerate, full_state_resume=True, real_data_test_executed=False))
        return
    # Use a real official dataset item, with the A0 selected weight only for diagnosis.
    from training import config
    from data import build_train
    from opencood.tools import train_utils
    from torch.utils.data import DataLoader, Subset
    actual_h = config(args, root)
    seed()
    dataset = build_train(actual_h, args.prepared[0])
    loader = DataLoader(Subset(dataset, [0]), batch_size=1, num_workers=0, collate_fn=dataset.collate_batch_train)
    sample = next(iter(loader)); audit = sample.pop('_audit')
    sample = train_utils.to_device(sample, torch.device('cuda:0'))['ego']
    ref = complete(Path(args.experiment_root) / 'baseline_reference.json')
    reference.load_state_dict(torch.load(ref['checkpoint'], map_location='cpu', weights_only=False), strict=True)
    reference.eval()
    captured = []
    pfn = reference.pillar_vfe.pfn_layers[0]
    handle = pfn.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0].detach()), prepend=True)
    with torch.no_grad():
        output = reference(sample)
        factorial = b1_factorial(pfn, captured[0], output['point_valid_mask'], output['point_reliability'])
    handle.remove()
    atomic_json(root / 'B1_FACTORIAL_DIAGNOSTIC.json', factorial)
    atomic_json(root / 'summary.json', dict(status='complete', models=rows, degenerate=degenerate,
        geometry_revision=2, geometry_mechanism=geometry_mechanism, A1_preserved_state=a1_restore,
        actual_loader_sample=audit, b1_factorial=factorial, full_state_resume=True,
        A0_optimization_steps=0, actual_training_and_inference_gate='subsequent A1/A2/A3 smoke stages'))
