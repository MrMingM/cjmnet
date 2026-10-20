"""FP32, common accumulation boundaries, resumable optimizer-boundary checkpoints."""
import os
from pathlib import Path
import time

import numpy as np
import torch

from v2xr_raygen_e0e3.training import atomic_torch, tensor_hash, rng_state, restore_rng
from .common import bound, commit, complete, identity, progress, read, sha, variant_path, write
from .data_adapter import Dataset
from .model_adapter import make, criterion, losses, shared_state
from .runtime import host_config, audit_imports
from .upstream import inventory
from .state_logic import verify_resume


def tensor_digest(value):
    fields = {}
    def walk(x, key):
        if isinstance(x, dict):
            for k in sorted(x):
                walk(x[k], key+'/'+str(k))
        elif isinstance(x, (tuple, list)):
            for i, v in enumerate(x):
                walk(v, key+'/'+str(i))
        elif isinstance(x, np.ndarray):
            fields[key] = torch.from_numpy(x.copy())
        elif isinstance(x, torch.Tensor):
            fields[key] = x
    walk(value, '')
    return tensor_hash(fields)


def recovery(branch, state):
    folder = branch / 'recovery'
    folder.mkdir(parents=True, exist_ok=True)
    pointer = folder / 'latest.json'
    old = read(pointer) if pointer.exists() else dict(slot=1)
    slot = 1-old['slot']
    path = folder / f'slot{slot}.pth'
    # Pointer still references the other complete slot until this atomic commit.
    atomic_torch(path, state)
    write(pointer, dict(slot=slot, sha256=sha(path), epoch=state['epoch'], global_step=state['global_step']))


def load_recovery(branch, epoch):
    pointer = branch / 'recovery/latest.json'
    if not pointer.exists():
        return None
    r = read(pointer)
    if r['epoch'] > epoch:
        raise ValueError('Recovery is ahead of an uncommitted earlier epoch')
    if r['epoch'] != epoch:
        return None
    path = pointer.parent / f'slot{r["slot"]}.pth'
    if sha(path) != r['sha256']:
        raise ValueError('Committed recovery checkpoint damaged')
    return torch.load(path, map_location='cpu', weights_only=False)


def run(output, variant, epoch, smoke=False, replay=False, interrupt_once=False):
    from opencood.tools import train_utils
    root = Path(output); c = read(root / 'config.json')
    branch = variant_path(root, variant, smoke)
    if replay:
        branch = root / 'smoke_replay' / variant
    ep = branch / 'epochs' / f'{epoch:02d}'
    ep.mkdir(parents=True, exist_ok=True)
    if (ep / 'train_complete.json').exists():
        return complete(ep / 'train_complete.json')
    micro = c['microbatch']; accumulation = 8//micro
    model = make(root, variant)
    initial = dict(shared=tensor_hash(shared_state(model)), full=tensor_hash(model.state_dict()))
    bound(branch / 'initialization.json', initial)
    reference = variant_path(root, 'E0', smoke) / 'initialization.json'
    if variant != 'E0' and read(reference)['shared'] != initial['shared']:
        raise ValueError('Shared UECP initialization differs from E0')
    bound(branch / 'config.json', dict(shared=c, variant=variant, model=host_config(root),
        frozen_quality=variant != 'E0', microbatch=micro, accumulation=accumulation,
        precision_gradient_removed=variant in ('E2', 'E3_no_separation'),
        original_auxiliary_loss_calls=2, smoke_only=smoke))
    bound(branch / 'parameters.json', dict(total=sum(p.numel() for p in model.parameters()),
        trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
        adapter=sum(p.numel() for n, p in model.named_parameters() if n.startswith('quality_adapter.')),
        frozen=sum(p.numel() for p in model.parameters() if not p.requires_grad),
        pretrained_cost='none' if variant == 'E0' else c['gspr']['domain'],
        note='Coverage-only keeps adapter parameters but does not use its output; constant-Q control keeps all adapter parameters active'))
    model = model.cuda(); crit = criterion(root).cuda()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, **c['optimizer'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, **c['scheduler'])
    dataset = Dataset(root, epoch=epoch)
    order = dataset.order(epoch)
    if smoke:
        # Repeat a fixed real subset to exercise full effective batches and recovery boundaries.
        ids = list(range(min(len(dataset), c['smoke_frames'])))
        order = (ids*(8*c['smoke_updates']//len(ids)+1))[:8*c['smoke_updates']]
    usable = len(order)//8*8
    if not usable:
        raise ValueError('No full effective batch')
    sampling = dict(epoch=epoch, order=order, usable=usable, dropped=order[usable:],
        uids=[dataset.keys[dataset.indices[i]] for i in order],
        weather=[dataset.base.schedule(dataset.indices[i]) for i in order],
        microbatch=micro, accumulation=accumulation)
    bound(ep / 'sampling.json', sampling)
    e0ep = variant_path(root, 'E0', smoke) / 'epochs' / f'{epoch:02d}'
    if variant != 'E0' and read(e0ep / 'sampling.json') != sampling:
        raise ValueError('Frame order, Mix3 weather or optimization budget differs')
    contract = read(root / 'contract.json')['identity']
    totals = dict(frames=0, updates=0, microsteps=0, loss=0., fused=0., single=0., auxiliary_each=0.)
    position, global_step, elapsed, actual, gradients, distributions = 0, 0, 0., [], [], []
    restored = load_recovery(branch, epoch)
    if restored is not None:
        verify_resume(restored, contract, variant, epoch, micro, order)
    elif epoch > 1:
        prev = branch / 'epochs' / f'{epoch-1:02d}'
        complete(prev / 'train_complete.json')
        restored = torch.load(prev / 'resume.pth', map_location='cpu', weights_only=False)
        if restored['contract_identity'] != contract or restored['epoch'] != epoch-1 or restored['variant'] != variant:
            raise ValueError('Previous completed epoch identity differs')
    if restored is not None:
        model.load_state_dict(restored['model'], strict=True)
        opt.load_state_dict(restored['optimizer']); scheduler.load_state_dict(restored['scheduler'])
        crit.load_state_dict(restored['criterion'], strict=True)
        global_step = restored['global_step']
        if restored['epoch'] == epoch:
            position, totals, elapsed = restored['position'], restored['totals'], restored['elapsed']
            actual, gradients, distributions = restored['actual'], restored['gradients'], restored['distributions']
        restore_rng(restored['rng'])
        del restored
    model.train(); used_lr = [g['lr'] for g in opt.param_groups]
    started = time.monotonic()
    def state():
        return dict(contract_identity=contract, variant=variant, epoch=epoch, position=position,
            model={k: v.detach().cpu() for k, v in model.state_dict().items()}, optimizer=opt.state_dict(),
            scheduler=scheduler.state_dict(), criterion=crit.state_dict(), rng=rng_state(),
            global_step=global_step, order_identity=identity(order), microbatch=micro, accumulation_position=0,
            totals=totals, elapsed=elapsed+time.monotonic()-started, actual=actual,
            gradients=gradients, distributions=distributions, amp_scaler=None)
    opt.zero_grad(set_to_none=True)
    for offset in range(position, usable, micro):
        ids = order[offset:offset+micro]
        progress(ep / 'progress.json', status='RUNNING', variant=variant, epoch=epoch,
            completed=offset, total=usable, uid=dataset.keys[dataset.indices[ids[0]]])
        batch = dataset.collate_batch_train([dataset[i] for i in ids])
        audits = batch.pop('_audit')
        actual.append(dict(samples=audits, input_digest=tensor_digest(dict(inputs=batch['ego']['uecp_input'],
            fused_labels=batch['ego']['label_dict'], single_labels=batch['_single'])) if smoke else None))
        inputs = train_utils.to_device(batch['ego']['uecp_input'], torch.device('cuda:0'))
        targets = train_utils.to_device(batch['ego']['label_dict'], torch.device('cuda:0'))
        single = train_utils.to_device(batch['_single'], torch.device('cuda:0'))
        output_dict = model(inputs)
        loss, parts = losses(crit, output_dict, targets, single)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite loss UID=' + repr(audits))
        (loss/accumulation).backward()
        totals['frames'] += micro; totals['microsteps'] += 1; totals['loss'] += float(loss.detach())
        for key, val in parts.items():
            totals[key] += val
        if offset == 0 or offset % (20*micro) == 0:
            distributions.append(dict(offset=offset, stats=model.last_stats))
        if (offset+micro) % 8 == 0:
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
                raise FloatingPointError('Nonfinite gradient UID=' + repr(audits))
            if hasattr(model, 'gspr') and any(p.grad is not None for p in model.gspr.parameters()):
                raise ValueError('Frozen GSPR received optimizer gradients')
            if totals['updates'] == 0:
                norms = {n: float(p.grad.norm()) for n, p in model.named_parameters()
                         if p.grad is not None and ('quality_adapter.net.2' in n or n.startswith('uncertainty_head.head.3'))}
                gradients.append(norms)
                if variant not in ('E0', 'E3_coverage') and not any(v > 0 for k, v in norms.items() if k.startswith('quality_adapter')):
                    raise ValueError('Adapter has no detection gradient on first effective batch')
            opt.step(); opt.zero_grad(set_to_none=True)
            if any(not torch.isfinite(p).all() for p in params):
                raise FloatingPointError('Nonfinite optimized parameter')
            totals['updates'] += 1; global_step += 1; position = offset+micro
            if totals['updates'] % 100 == 0 or smoke:
                recovery(branch, state())
            if interrupt_once and totals['updates'] == 1 and not (ep / 'expected_train_interruption.json').exists():
                if not smoke or replay:
                    raise ValueError('Training fault injection is restricted to independent smoke')
                write(ep / 'expected_train_interruption.json', dict(position=position, global_step=global_step,
                    recovery_pointer_sha256=sha(branch / 'recovery/latest.json')))
                raise SystemExit(44)
        if offset % (20*micro) == 0:
            print(f'{variant} epoch={epoch} frames={offset+micro}/{usable} loss={float(loss):.6f} updates={totals["updates"]}', flush=True)
    scheduler.step()
    torch.cuda.synchronize()
    seconds = elapsed+time.monotonic()-started
    saved = state()
    atomic_torch(ep / 'detector.pth', saved['model'])
    atomic_torch(ep / 'resume.pth', saved)
    # A fresh instance must load every tensor; this is not a relaxed/partial checkpoint.
    fresh = make(root, variant); fresh.load_state_dict(saved['model'], strict=True)
    write(ep / 'actual_samples.json', actual)
    if variant != 'E0' and read(e0ep / 'actual_samples.json') != actual:
        raise ValueError('Actual CAV/weather/geometry differs from E0')
    write(ep / 'statistics.json', dict(gradients=gradients, distributions=distributions))
    artifacts = [ep / n for n in ('detector.pth', 'resume.pth', 'sampling.json', 'actual_samples.json', 'statistics.json')]
    if (ep / 'expected_train_interruption.json').exists():
        artifacts.append(ep / 'expected_train_interruption.json')
    return commit(ep / 'train_complete.json', artifacts,
        contract_identity=contract, variant=variant, epoch=epoch, smoke_only=smoke,
        totals=totals, seconds=seconds, used_lr=used_lr, global_step=global_step,
        next_lr=[g['lr'] for g in opt.param_groups], frames_per_second=usable/seconds,
        max_memory_allocated=torch.cuda.max_memory_allocated(), max_memory_reserved=torch.cuda.max_memory_reserved(),
        gpu_name=torch.cuda.get_device_name(0), physical_gpu=os.environ['GPU'],
        final_tensor_sha256=tensor_hash(saved['model']), native_imports=audit_imports(root), uecp_imports=inventory())
