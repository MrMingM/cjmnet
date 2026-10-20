"""Server synthetic Torch/OpenCOOD tests; no real-data optimization or AP selection."""
import copy
from pathlib import Path
import random
import subprocess
import sys
from auditlib import HERE, ROOT, atomic_json, identity, load_frozen


def batch():
    import torch
    coords = torch.tensor([[0, 0, 80, 250], [0, 0, 81, 251], [0, 0, 82, 252], [0, 0, 83, 253]],
                          dtype=torch.int32, device='cuda:0')
    voxels = torch.zeros((4, 32, 4), device='cuda:0')
    for i in range(4):
        voxels[i, :4, 0] = coords[i, 3] * .4 - 140.8 + .2 + torch.linspace(-.1, .1, 4, device='cuda:0')
        voxels[i, :4, 1] = coords[i, 2] * .4 - 40 + .2
        voxels[i, :4, 2] = -1
        voxels[i, :4, 3] = .5
    return dict(processed_lidar=dict(voxel_features=voxels, voxel_coords=coords,
        voxel_num_points=torch.full((4,), 4, device='cuda:0', dtype=torch.int32)),
        record_len=torch.tensor([1], device='cuda:0'))


def target(prediction):
    import torch
    n, a, h, w = prediction['psm'].shape
    positive = torch.zeros((n, h, w, a), device='cuda:0')
    positive[:, h//2, w//2, 0] = 1
    return dict(pos_equal_one=positive, neg_equal_one=1-positive,
        targets=torch.zeros((n, h, w, 7*a), device='cuda:0'))


def test_full_state(output):
    import numpy as np
    import torch
    from checkpoint import restore, save, seed, state
    seed()
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout(.2), torch.nn.Linear(4, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=.002)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 15], gamma=.1)
    class StatefulCriterion(torch.nn.MSELoss):
        def __init__(self):
            super().__init__()
            self.register_buffer('calls', torch.zeros((), dtype=torch.int64))
            self.loss_cnt = 0
            self.loss_dict = {}
        def forward(self, prediction, target):
            self.calls.add_(1)
            self.loss_cnt += 1
            result = super().forward(prediction, target)
            self.loss_dict = dict(last=float(result.detach()))
            return result
    criterion = StatefulCriterion()
    def step():
        optimizer.zero_grad()
        x = torch.randn(3, 4)
        value = criterion(model(x), torch.randn(3, 1))
        value.backward()
        optimizer.step()
        return float(value), random.random(), float(np.random.rand()), torch.rand(3, device='cuda:0').cpu()
    step()
    scheduler.step(10)
    file = output / 'full_state_fixture.pth'
    save(file, state(model, optimizer, scheduler, criterion, 1, 'synthetic'))
    a = step()
    expected = copy.deepcopy(model.state_dict())
    expected_criterion = (int(criterion.calls), criterion.loss_cnt, criterion.loss_dict.copy())
    optimizer.param_groups[0]['lr'] = .55
    scheduler.step(20)
    restore(file, model, optimizer, scheduler, criterion, 'synthetic', 1)
    if scheduler.last_epoch != 10 or optimizer.param_groups[0]['lr'] != .0002:
        raise ValueError('Scheduler milestones/optimizer lr failed restoration')
    b = step()
    if a[:3] != b[:3] or not torch.equal(a[3], b[3]) or any(not torch.equal(v, model.state_dict()[k]) for k, v in expected.items()):
        raise ValueError('Model/Adam/scheduler/Python/NumPy/Torch CPU+CUDA resume next-step differs')
    if expected_criterion != (int(criterion.calls), criterion.loss_cnt, criterion.loss_dict):
        raise ValueError('Criterion buffer/log accumulators failed restoration')
    try:
        restore(file, model, optimizer, scheduler, criterion, 'wrong', 1)
    except ValueError:
        pass
    else:
        raise ValueError('Wrong checkpoint contract was accepted')


def test_models(args, output):
    import torch
    from opencood.hypes_yaml import yaml_utils
    from opencood.models.point_pillar_intermediate import PointPillarIntermediate
    from opencood.loss.point_pillar_loss_opencood import PointPillarLossOpenCOOD
    from checkpoint import save, seed
    from models import (add_film, bn_difference, create, criterion,
                        frozen_backbone, start_bn, strict_backbone)
    h = yaml_utils.load_yaml(str(Path(args.runtime_root) / 'opencood/hypes_yaml/V2X-R/LiDAR/V2XR_attfuse.yaml'))
    seed()
    official = PointPillarIntermediate(h['model']['args'])
    expected_rng = torch.get_rng_state().clone()
    reference = {k: v.clone() for k, v in official.state_dict().items()}
    seed()
    c = create(h, 'C')
    if any(not torch.equal(v, c.state_dict()[k]) for k, v in reference.items()):
        raise ValueError('L0 initialized weights differ from official')
    official, c = official.cuda().eval(), c.cuda().eval()
    inputs = batch()
    with torch.no_grad():
        a, b = official(inputs), c(inputs)
        for key in ('psm', 'rm'):
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    del official, a, b
    seed()
    d = create(h, 'D')
    if not torch.equal(torch.get_rng_state(), expected_rng):
        raise ValueError('GSPR extra initialization perturbed shared CPU RNG')
    if any(not torch.equal(v, d.state_dict()[k]) for k, v in reference.items()):
        raise ValueError('C/D shared initialization differs')
    d = d.cuda().eval()
    d.force_unit_weights = True
    with torch.no_grad():
        a, b = c(inputs), d(inputs)
        for key in ('psm', 'rm'):
            torch.testing.assert_close(a[key], b[key], rtol=1e-5, atol=1e-5)
    d.force_unit_weights = False
    del a, b, reference
    d.train()
    prediction = d(inputs)
    labels = target(prediction)
    valid = prediction['point_valid_mask']
    if int(valid.sum()) != 16 or prediction['point_reliability'][~valid].abs().sum() != 0:
        raise ValueError('GSPR valid mask/padding differs')
    detection = PointPillarLossOpenCOOD(h['loss']['args'])(prediction, labels)
    grads = torch.autograd.grad(detection, tuple(d.gspr.parameters()), retain_graph=True, allow_unused=True)
    active = [g for g in grads if g is not None]
    if not active or not all(torch.isfinite(g).all() for g in active) or not any(g.abs().sum() > 0 for g in active):
        raise ValueError('Detection gradient does not reach GSPR')
    loss_fn = criterion(h, 'D')
    total = loss_fn(prediction, labels)
    expected = detection + .02 * (prediction['point_reliability'][valid].mean()-.9).square() + .0001 * prediction['point_evidence'][valid].square().mean()
    torch.testing.assert_close(total, expected)
    total.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in d.parameters()):
        raise ValueError('Nonfinite GSPR synthetic gradient')
    loss_state = copy.deepcopy(loss_fn.state_dict())
    loss_fn._gspr_regularizer_sum.add_(1)
    loss_fn.load_state_dict(loss_state, strict=True)
    if any(not torch.equal(v, loss_fn.state_dict()[k]) for k, v in loss_state.items()):
        raise ValueError('GSPR regularizer buffer reload differs')
    del prediction, total, expected, detection, grads, active
    d.zero_grad(set_to_none=True)
    checks = {}
    for name, model in (('C', c), ('D', d)):
        before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.eval()
        with torch.no_grad():
            a = model(inputs)['psm'].detach().clone()
            layers = start_bn(model)
            for _ in range(3):
                model(inputs)
            audit = bn_difference(before, model.state_dict(), layers)
            model.eval()
            b = model(inputs)['psm']
        if torch.equal(a, b):
            raise ValueError('Synthetic BN model outputs did not change')
        model.load_state_dict(before, strict=True)
        model.eval()
        with torch.no_grad():
            identity_output = {k: v.detach().clone() for k, v in model(inputs).items() if k in ('psm', 'rm')}
        add_film(model)
        model.film.cuda()
        frozen_backbone(model)
        with torch.no_grad():
            a = model(inputs)
            for key in identity_output:
                torch.testing.assert_close(a[key], identity_output[key], rtol=0, atol=0)
        if model.film.last['shape'][1] != 64 or model.film.last['delta_rms'] != 0 or model.film.last['beta_rms'] != 0:
            raise ValueError('FiLM insertion/identity initialization invalid')
        # Compare to the unchanged model with the hook returning its input exactly.
        model.zero_grad()
        prediction = model(inputs)
        value = criterion(h, name, film=True)(prediction, target(prediction))
        value.backward()
        film_grads = [p.grad for p in model.film.parameters() if p.grad is not None]
        if not film_grads or not any(g.abs().sum() > 0 for g in film_grads) or not all(torch.isfinite(g).all() for g in film_grads):
            raise ValueError('FiLM received no finite nonzero gradients')
        if any(p.grad is not None for k, p in model.named_parameters() if not k.startswith('film.')):
            raise ValueError('Frozen backbone received gradients')
        optimizer = torch.optim.Adam(model.film.parameters(), lr=.001)
        optimizer.step()
        strict_backbone(model, before)
        model.eval()
        with torch.no_grad():
            expected_output = model(inputs)['psm'].detach().clone()
        file = output / (name + '_film_synthetic.pth')
        save(file, model.state_dict())
        state = torch.load(file, map_location='cpu', weights_only=False)
        with torch.no_grad():
            model.film.net[-1].bias.add_(1)
        model.load_state_dict(state, strict=True)
        with torch.no_grad():
            torch.testing.assert_close(model(inputs)['psm'], expected_output, rtol=0, atol=0)
        checks[name] = dict(bn=audit, film_shape=model.film.last['shape'],
            film_gradients=True, backbone_exact=True, film_strict_reload=True,
            film_parameters=sum(p.numel() for p in model.film.parameters()))
        del prediction, value, state, before, a, b
        model.film.observations.clear()
    return checks


def test_dataset_observer(args):
    """Identical worker RNG: observed mixed dataset must equal unobserved official dataset."""
    import torch
    from checkpoint import rng, restore_rng, seed
    from data import build_train, tensor_digest
    from training import config
    from opencood.data_utils.datasets import build_dataset
    manifest, evidence = args.prepared
    h = config(args, Path(args.output_root))
    observed = build_train(h, manifest)
    plain = build_dataset(h, visualize=False, train=True)
    seed()
    state = rng()
    a = observed[0]
    restore_rng(state)
    b = plain[0]
    audit = a.pop('_audit')
    if tensor_digest(a) != tensor_digest(b):
        raise ValueError('Read/GT observer changed actual official sample, augmentation or GT')
    if audit['final_gt_ids'] != list(map(str, b['ego']['object_ids'])):
        raise ValueError('GT observer changed IDs')
    if audit['configured_world_augmentation_calls'] != 0:
        raise ValueError('Reviewed official inactive world-augmentation path changed')
    # Exercise the real multi-worker boundary and capture worker-seeded actual sampling.
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(observed, range(16)),
        batch_size=8, num_workers=16, shuffle=True, pin_memory=True, drop_last=True,
        prefetch_factor=4, collate_fn=observed.collate_batch_train)
    traces = []
    for item in loader:
        traces += item['_audit']
    if len(traces) != 16 or {r['condition'] for t in traces for r in t['reads']} != {'clean', 'fog'}:
        raise ValueError('Multi-worker mixed dataset did not produce both weather sources')
    return dict(observed_equals_original=True, original_gt_equal=True,
        worker_trace_identity=identity(traces), first_sample=audit)


def run(args):
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, '-B', str(HERE / 'tests_local.py')], check=True)
    test_full_state(output)
    models = test_models(args, output)
    # These are pipeline checks on training data, no validation-driven calibration or fitting.
    dataset = test_dataset_observer(args)
    module = load_frozen('protocol_tests', 'test_contract.py', ROOT / 'v2xr_protocol_audit')
    module.run_tests()
    atomic_json(output / 'summary.json', dict(status='complete', synthetic_models=True,
        original_dataset_observer_smoke=dataset, model_checks=models,
        full_model_adam_scheduler_criterion_rng_next_step_equal=True,
        official_AP_contract_tests_passed=True, no_real_data_optimization=True))
