"""Server-only equivalence and exact gradient-repeat checks; no optimizer update."""
import copy
from pathlib import Path

import numpy as np

from .common import read, write


def equivalence(bev, owners, centers):
    import torch
    from .stable_crop import configure, check_backend, crop_condition, reference_crop
    upstream = torch.randn(len(owners), bev.shape[1], 32, 32, device=bev.device)
    leaf = bev.detach().requires_grad_(True)
    # This is the declared old-operation reference only, not a training fallback.
    try:
        torch.use_deterministic_algorithms(False)
        old = reference_crop(leaf, owners, centers)
        old_gradient = torch.autograd.grad(old, leaf, upstream)[0]
    finally:
        configure(torch)
    new = crop_condition(leaf, owners, centers)
    new_gradient = torch.autograd.grad(new, leaf, upstream)[0]
    forward = torch.allclose(old, new, rtol=1e-5, atol=1e-6)
    backward = torch.allclose(old_gradient, new_gradient, rtol=1e-5, atol=1e-6)
    if not forward or not backward or not torch.isfinite(new_gradient).all() or not new_gradient.abs().sum():
        raise ValueError('Stable crop does not match the original bilinear forward/feature gradient')
    repeats = []
    for _ in range(2):
        out = crop_condition(leaf, owners, centers)
        grad = torch.autograd.grad(out, leaf, upstream)[0]
        repeats.append(bool(torch.equal(out, new) and torch.equal(grad, new_gradient)))
    if not all(repeats):
        raise ValueError('Fixed-input stable crop is not exactly repeatable')
    check_backend(torch)
    return dict(forward_original_tolerance=True, backward_original_tolerance=True,
                maximum_forward_delta=float((old-new).abs().max()),
                maximum_backward_delta=float((old_gradient-new_gradient).abs().max()),
                exact_crop_repeats=repeats, tolerance=dict(rtol=1e-5, atol=1e-6),
                scope='Reference grid_sample backward temporarily permits nondeterminism in this check only')


def run(output):
    import torch
    from opencood.tools import train_utils
    from .cache import load_windows
    from .data import Dataset
    from .geometry import loss_weights
    from .losses import auxiliary_loss
    from .model import FeatureTap
    from .runtime import hypes
    from .training import make_models, load_resume_state, rng_state, tensor_hash
    from .resume_compare import compare_states
    from .stable_crop import POLICY, POLICY_ID, check_backend
    root = Path(output)
    check_backend(torch)
    torch.manual_seed(20261019)
    # Overlapping owners and partial/outside boundaries, independent from a real scene.
    synthetic = equivalence(torch.randn(2, 3, 100, 352, device='cuda'), [0, 0, 1, 1],
                            np.array([[0., 0., -1.], [.4, .4, -1.], [140.6, 39.8, -1.], [-140.9, -40.1, -1.]]))
    print('Stable synthetic crop equivalence and exact repeats passed', flush=True)
    config = read(root / 'config.json')
    indices = read(root / 'preflight/smoke_indices.json')['indices']
    index = next((i for i in indices if any(loss_weights('E2', row['target']).any()
                  for row in load_windows(root, i, 1))), None)
    if index is None:
        raise ValueError('No supervised smoke sample for stable calculation check')
    dataset = Dataset(root, epoch=1, auxiliary=True)
    batch = dataset.collate([dataset[index]])
    detector, aux = make_models(root, 'E2', config)
    detector, aux = detector.cuda().train(), aux.cuda().train()
    criterion = train_utils.create_loss(hypes(root)).cuda()
    named = [('detector.' + n, p) for n, p in detector.named_parameters()] + [('aux.' + n, p) for n, p in aux.named_parameters()]
    optimizer = torch.optim.Adam([p for n, p in named], **config['optimizer'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **config['scheduler'])
    def host(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {k: host(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return type(value)(host(v) for v in value)
        return copy.deepcopy(value)
    state = host(dict(model=detector.state_dict(), aux=aux.state_dict(), optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(), criterion=criterion.state_dict(),
        criterion_accumulators={k: getattr(criterion, k) for k in ('total_loss', 'reg_loss', 'conf_loss', 'diff_loss', 'loss_cnt', 'loss_dict')
                                if hasattr(criterion, k)}, rng=rng_state(), global_step=0))
    before = tensor_hash(dict(named))
    tap, baseline, comparisons = FeatureTap(detector), None, []
    try:
        for repeat in range(3):
            optimizer.zero_grad(set_to_none=True)
            load_resume_state(detector, aux, criterion, optimizer, scheduler, copy.deepcopy(state))
            inputs = train_utils.to_device(batch['inputs'], torch.device('cuda:0'))
            labels = train_utils.to_device(batch['labels'], torch.device('cuda:0'))
            out = detector(inputs)
            det = criterion(out, labels)
            gen, counts = auxiliary_loss('E2', aux, tap.value, batch, 1, config)
            if not counts['valid_windows']:
                raise ValueError('Stable joint test has no auxiliary gradient')
            joint = det + config['lambda_gen'] * gen
            joint.backward()
            if not torch.isfinite(joint) or any(p.grad is not None and not torch.isfinite(p.grad).all() for n, p in named):
                raise FloatingPointError('Nonfinite stable joint result')
            current = host(dict(outputs=out, loss=joint, gradients={n: p.grad for n, p in named}, rng=rng_state()))
            if baseline is not None:
                checked = compare_states(current, baseline,
                    lambda x: x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x if isinstance(x, np.ndarray) else None,
                    exact=True)
                comparisons.append(checked)
                if checked['status'] != 'complete':
                    raise ValueError('Stable joint gradients differ: ' + repr(checked['failures'][:8]))
            baseline = current if baseline is None else baseline
            print('Stable real E2 forward/backward repeat ' + str(repeat+1) + '/3 passed; no optimizer update', flush=True)
            bev = tap.value.detach()
            tap.clear()
        owners = [i for i, rows in enumerate(batch['auxiliary']) for row in rows]
        centers = np.array([row['center'] for rows in batch['auxiliary'] for row in rows])
        real = equivalence(bev, owners, centers)
        if tensor_hash(dict(named)) != before:
            raise ValueError('Stable diagnostic unexpectedly updated parameters')
    finally:
        tap.handle.remove(); tap.clear()
    report = dict(status='complete', policy=POLICY, policy_identity=POLICY_ID, synthetic=synthetic,
                  real_crop=real, real_E2_exact_joint_repeats=comparisons, sample_index=index,
                  optimizer_updates=0, backend=check_backend(torch), server_validation=True)
    write(root / 'smoke/stable_calculation_checks.json', report)
