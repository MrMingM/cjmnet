"""One training epoch per process. Evaluation starts only after this process releases its GPU."""
import hashlib
import os
from pathlib import Path
import random
import time

import numpy as np

from .common import atomic_bytes, bound, identity, progress, read, sha, stream_seed, write


def tensor_hash(state):
    h = hashlib.sha256()
    for k, v in sorted(state.items()):
        a = v.detach().cpu().contiguous().numpy()
        h.update(k.encode()); h.update(str((a.shape, a.dtype)).encode()); h.update(a.tobytes())
    return h.hexdigest()


def atomic_torch(path, value):
    import torch
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    torch.save(value, tmp)
    with tmp.open('rb') as f:
        os.fsync(f.fileno())
    os.replace(tmp, p)


def rng_state():
    import torch
    return dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(),
                hip=torch.cuda.get_rng_state_all())


def restore_rng(rng):
    import torch
    random.setstate(rng['python']); np.random.set_state(rng['numpy'])
    torch.set_rng_state(rng['cpu']); torch.cuda.set_rng_state_all(rng['hip'])


def make_models(root, group, config):
    import torch
    from opencood.tools import train_utils
    from .runtime import hypes
    from .model import ConditionalUNet, attach_empty_voxel_padding
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    detector = train_utils.create_model(hypes(root))
    attach_empty_voxel_padding(detector)
    aux = None
    if group != 'E0':
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(stream_seed(config['seed'], 0, 'aux', 'initialization'))
            aux = ConditionalUNet()
    return detector, aux


def completion(root, epoch_root):
    record = read(epoch_root / 'train_complete.json')
    if record['contract_identity'] != read(Path(root) / 'contract.json')['identity']:
        raise ValueError('Training commit contract differs')
    for name, digest in record['output_hashes'].items():
        if sha(epoch_root / name) != digest:
            raise ValueError('Committed training artifact changed: ' + name)
    return record


def run(output, group, epoch, smoke=False, probe_micro=None, replay=False):
    import torch
    from opencood.tools import train_utils
    from .data import Dataset
    from .losses import auxiliary_loss
    from .model import FeatureTap
    from .runtime import hypes
    root = Path(output)
    config = read(root / 'config.json')
    branch = root / ('probe/micro' + str(probe_micro) if probe_micro else 'smoke_replay' if replay else 'smoke' if smoke else 'groups') / group
    epoch_root = branch / 'epochs' / f'{epoch:02d}'
    epoch_root.mkdir(parents=True, exist_ok=True)
    if (epoch_root / 'train_complete.json').exists():
        return completion(root, epoch_root)
    micro = probe_micro or read(root / 'execution.json')['microbatch']
    accumulation = config['effective_batch'] // micro
    if config['effective_batch'] % micro:
        raise ValueError('Invalid accumulation boundary')
    detector, aux = make_models(root, group, config)
    init = dict(detector=tensor_hash(detector.state_dict()), aux=None if aux is None else tensor_hash(aux.state_dict()))
    bound(branch / 'initialization.json', init)
    bound(branch / 'parameters.json', dict(detector=sum(p.numel() for p in detector.parameters()),
        auxiliary=0 if aux is None else sum(p.numel() for p in aux.parameters()),
        detector_class=type(detector).__module__ + '.' + type(detector).__name__,
        auxiliary_class=None if aux is None else type(aux).__name__))
    reference = root / ('smoke' if smoke else 'groups') / 'E0/initialization.json'
    if group != 'E0' and reference.exists() and init['detector'] != read(reference)['detector']:
        raise ValueError('Host initialization differs between groups')
    aux_reference = root / ('smoke' if smoke else 'groups') / 'E1/initialization.json'
    if group in ('E2', 'E3') and aux_reference.exists() and init['aux'] != read(aux_reference)['aux']:
        raise ValueError('Auxiliary initialization differs')
    bound(branch / 'config.json', dict(group=group, shared=config, microbatch=micro,
        accumulation=accumulation, lambda_gen=0 if group == 'E0' else config['lambda_gen'],
        parsed_host_config=hypes(root),
        smoke_only=bool(smoke or probe_micro)))
    detector = detector.cuda()
    aux = aux.cuda() if aux is not None else None
    criterion = train_utils.create_loss(hypes(root)).cuda()
    params = list(detector.parameters()) + ([] if aux is None else list(aux.parameters()))
    opt = torch.optim.Adam(params, **config['optimizer'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, **config['scheduler'])
    previous = branch / 'epochs' / f'{epoch-1:02d}'
    global_step = 0
    if epoch > 1:
        prior = completion(root, previous)
        state = torch.load(previous / 'resume.pth', map_location='cpu', weights_only=False)
        if state['epoch'] != epoch - 1 or state['accumulation_position'] != 0 or state['group'] != group:
            raise ValueError('Invalid epoch resume boundary')
        detector.load_state_dict(state['model'], strict=True)
        if aux is not None:
            aux.load_state_dict(state['aux'], strict=True)
        opt.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        criterion.load_state_dict(state['criterion'])
        for k, v in state['criterion_accumulators'].items():
            setattr(criterion, k, v)
        global_step = state['global_step']
        restore_rng(state['rng'])
        del state
    else:
        atomic_torch(branch / 'initial_detector.pth', {k: v.detach().cpu() for k, v in detector.state_dict().items()})
    detector.train()
    if aux is not None:
        aux.train()
    tap = FeatureTap(detector) if aux is not None else None
    dataset = Dataset(root, epoch=epoch, auxiliary=aux is not None)
    from .runtime import audit_imports
    imports = audit_imports(root)
    order = dataset.order()
    if smoke or probe_micro:
        order = read(root / 'preflight/smoke_indices.json')['indices']
        order = order[:config['effective_batch'] * (1 if probe_micro else config['smoke_updates'])]
    usable = len(order) // config['effective_batch'] * config['effective_batch']
    if usable == 0:
        raise ValueError('No full effective batch')
    write(epoch_root / 'sampling.json', dict(epoch=epoch, order=order, used=order[:usable], dropped=order[usable:],
        weather=[dataset.schedule(i) for i in order], microbatch=micro, accumulation=accumulation,
        order_identity=identity(order)))
    start = time.monotonic()
    totals = dict(det_loss=0., gen_loss=0., microsteps=0, frames=0, updates=0)
    counters = {}
    diagnostics = []
    actual_samples, input_digests = [], []
    used_lr = [g['lr'] for g in opt.param_groups]
    opt.zero_grad(set_to_none=True)
    for step, offset in enumerate(range(0, usable, micro)):
        batch_indices = order[offset:offset + micro]
        progress(epoch_root / 'progress.json', status='running', group=group, epoch=epoch,
                 sample_uid=dataset.keys[batch_indices[0]], completed=offset, total=usable)
        batch = dataset.collate([dataset[i] for i in batch_indices])
        actual_samples.extend(batch['audits'])
        if smoke:
            from .smoke import tensor_digest
            input_digests.append(tensor_digest(dict(inputs=batch['inputs'], labels=batch['labels'])))
        inputs = train_utils.to_device(batch['inputs'], torch.device('cuda:0'))
        targets = train_utils.to_device(batch['labels'], torch.device('cuda:0'))
        out = detector(inputs)
        det_loss = criterion(out, targets)
        if not torch.isfinite(det_loss):
            raise FloatingPointError('Nonfinite detection loss UID=' + repr(batch['audits']))
        gen_loss = det_loss * 0
        if aux is not None:
            gen_loss, counts = auxiliary_loss(group, aux, tap.value, batch, epoch, config)
            for k, value in counts.items():
                counters[k] = counters.get(k, 0) + value
            if step == 0:
                # Separate loss gradients on shared encoder/fusion tensors, before the joint update.
                named = [(k, p) for k, p in detector.named_parameters() if p.requires_grad and
                         (k.startswith('pillar_vfe') or 'fuse_modules' in k)]
                gd = torch.autograd.grad(det_loss, [p for k, p in named], retain_graph=True, allow_unused=True)
                gg = torch.autograd.grad(gen_loss, [p for k, p in named], retain_graph=True, allow_unused=True)
                feature_gradient = torch.autograd.grad(gen_loss, tap.value, retain_graph=True, allow_unused=True)[0]
                diagnostics.append(dict(parameters=[k for k, p in named],
                    detection_norm=float(torch.sqrt(sum((g.square().sum() for g in gd if g is not None), det_loss.new_zeros(()))).item()),
                    generation_norm=float(torch.sqrt(sum((g.square().sum() for g in gg if g is not None), det_loss.new_zeros(()))).item()),
                    fused_feature_gradient_norm=0. if feature_gradient is None else float(feature_gradient.norm().item()),
                    lambda_gen=config['lambda_gen'], valid_windows=counts['valid_windows']))
                diagnostics[-1]['weighted_generation_norm'] = diagnostics[-1]['generation_norm'] * config['lambda_gen']
                if counts['valid_windows'] and (not diagnostics[-1]['generation_norm'] or not diagnostics[-1]['fused_feature_gradient_norm']):
                    raise ValueError('Nonempty generation supervision failed to reach shared encoder/fusion')
                del gd, gg, feature_gradient
        loss = (det_loss + (0 if group == 'E0' else config['lambda_gen']) * gen_loss) / accumulation
        loss.backward()
        if (step + 1) % accumulation == 0:
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
                raise FloatingPointError('Nonfinite parameter gradient')
            opt.step(); opt.zero_grad(set_to_none=True)
            global_step += 1
            totals['updates'] += 1
        totals['microsteps'] += 1; totals['frames'] += micro
        totals['det_loss'] += float(det_loss.detach()); totals['gen_loss'] += float(gen_loss.detach())
        if tap:
            tap.clear()
        if step % 20 == 0 or offset + micro == usable:
            print(f'{group} epoch={epoch} frames={offset+micro}/{usable} det={float(det_loss):.6f} gen={float(gen_loss):.6f} '
                  f'updates={totals["updates"]} seconds={time.monotonic()-start:.1f}', flush=True)
    # Explicit completed-epoch schedule: epochs 1..10 use .002; 11..15 .0002; 16..20 .00002.
    scheduler.step()
    torch.cuda.synchronize()
    duration = time.monotonic() - start
    actual = dict(samples=actual_samples, smoke_input_and_target_digests=input_digests)
    reference_actual = root / ('smoke' if smoke else 'groups') / 'E0/epochs' / f'{epoch:02d}' / 'actual_samples.json'
    if group != 'E0' and not probe_micro and reference_actual.exists() and read(reference_actual) != actual:
        raise ValueError('Actual sample/weather/CAV/geometry/voxel schedule differs from E0')
    write(epoch_root / 'actual_samples.json', actual)
    contract_identity = read(root / 'contract.json')['identity']
    model_state = {k: v.detach().cpu() for k, v in detector.state_dict().items()}
    atomic_torch(epoch_root / 'detector.pth', model_state)
    # Strict key/shape load into a fresh official class, with no auxiliary prefix/keys.
    saved_rng = rng_state()
    fresh = train_utils.create_model(hypes(root))
    fresh.load_state_dict(model_state, strict=True)
    restore_rng(saved_rng)
    state = dict(contract_identity=contract_identity, group=group, epoch=epoch, model=model_state,
        aux=None if aux is None else aux.state_dict(), optimizer=opt.state_dict(), scheduler=scheduler.state_dict(),
        criterion=criterion.state_dict(), criterion_accumulators={k: getattr(criterion, k) for k in
            ('total_loss', 'reg_loss', 'conf_loss', 'diff_loss', 'loss_cnt', 'loss_dict') if hasattr(criterion, k)},
        rng=saved_rng, amp_scaler=None, global_step=global_step, sampler=dict(epoch=epoch, order=order),
        accumulation_position=0, config=config, input_identity=read(root / 'contract.json')['inputs'],
        microbatch=micro, completed_training=True)
    atomic_torch(epoch_root / 'resume.pth', state)
    record = dict(status='complete', contract_identity=contract_identity, group=group, epoch=epoch,
        smoke_only=bool(smoke or probe_micro), initialization=init, totals=totals, label_counts=counters,
        gradients=diagnostics, seconds=duration, frames_per_second=usable / duration,
        used_lr=used_lr, next_lr=[g['lr'] for g in opt.param_groups], global_step=global_step,
        max_memory_allocated=torch.cuda.max_memory_allocated(), max_memory_reserved=torch.cuda.max_memory_reserved(),
        gpu_name=torch.cuda.get_device_name(0), physical_gpu=os.environ.get('GPU'),
        imported_modules=imports,
        detector_tensor_sha256=tensor_hash(model_state), strict_official_reload=True,
        output_hashes={n: sha(epoch_root / n) for n in ('detector.pth', 'resume.pth', 'sampling.json', 'actual_samples.json')})
    write(epoch_root / 'train_complete.json', record)
    return record
