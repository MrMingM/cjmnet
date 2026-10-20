"""train_dev chooses models; frozen829 O/O*/F is run only after selection is locked."""
import copy
import gc
import os
from pathlib import Path
import time

import torch

from .common import atomic_bytes, bound, commit, complete, identity, read, sha, variant_path, write
from .data_adapter import Dataset
from .model_adapter import make
from .runtime import arguments, host_config
from .state_logic import choose


def dev(output, variant, epoch, smoke=False):
    from opencood.tools import train_utils, inference_utils
    from opencood.utils import eval_utils_opv2v
    from evaluate import statistics
    root = Path(output); branch = variant_path(root, variant, smoke)
    ep = branch / 'epochs' / f'{epoch:02d}'
    complete(ep / 'train_complete.json')
    marker = ep / 'dev_complete.json'
    if marker.exists():
        return complete(marker)
    model = make(root, variant).cuda().eval()
    model.load_state_dict(torch.load(ep / 'detector.pth', map_location='cpu', weights_only=False), strict=True)
    dataset = Dataset(root, split='train_dev', epoch=0)
    n = min(len(dataset), read(root / 'config.json')['smoke_frames']) if smoke else len(dataset)
    stat = statistics(); stats, trace = [], []
    torch.cuda.synchronize(); started = time.monotonic()
    for i in range(n):
        item = dataset.get_global(dataset.indices[i], 'clean')
        batch = dataset.collate_batch_test([item])
        batch = train_utils.to_device(batch, torch.device('cuda:0'))
        with torch.no_grad():
            boxes, scores, gt = inference_utils.inference_intermediate_fusion(batch, model, dataset)
        if any(t is not None and not torch.isfinite(t).all() for t in (boxes, scores, gt)):
            raise FloatingPointError('Nonfinite train_dev prediction/GT')
        # Exact native O policy skips None frames; no replacement IoU/AP implementation.
        if boxes is not None:
            for t in (.3, .5, .7):
                eval_utils_opv2v.caluclate_tp_fp(boxes, scores, gt, stat, t)
        trace.append(dict(uid=dataset.keys[dataset.indices[i]], none=boxes is None, gt_count=len(gt)))
        if i % 50 == 0:
            print(f'train_dev Clean {variant} epoch={epoch} {i+1}/{n}', flush=True)
            stats.append(dict(index=i, stats=model.last_stats))
    aps = {f'ap{int(t*100)}': float(eval_utils_opv2v.calculate_ap(copy.deepcopy(stat), t, False)[0]) for t in (.3, .5, .7)}
    if any(not 0 <= x <= 1 for x in aps.values()):
        raise ValueError('Invalid native train_dev AP')
    write(ep / 'dev_statistics.json', stat)
    write(ep / 'dev_trace.json', dict(frames=trace, distributions=stats))
    return commit(marker, [ep / 'dev_statistics.json', ep / 'dev_trace.json'],
        contract_identity=read(root / 'contract.json')['identity'], variant=variant, epoch=epoch,
        checkpoint_sha256=sha(ep / 'detector.pth'), smoke_only=smoke, split='train_dev',
        condition='clean', protocol='O', AP=aps, frames=n, seconds=time.monotonic()-started)


def select(output, variant, smoke=False):
    root = Path(output); branch = variant_path(root, variant, smoke)
    n = 2 if smoke else read(root / 'config.json')['epochs']
    rows = []
    for e in range(1, n+1):
        ep = branch / 'epochs' / f'{e:02d}'
        complete(ep / 'train_complete.json')
        rows.append(complete(ep / 'dev_complete.json'))
        if rows[-1]['checkpoint_sha256'] != sha(ep / 'detector.pth'):
            raise ValueError('Selected candidate checkpoint differs')
    best = choose(rows)
    selected = dict(status='COMPLETE', variant=variant, epoch=best['epoch'], smoke_only=smoke,
        checkpoint=str(branch / 'epochs' / f'{best["epoch"]:02d}' / 'detector.pth'),
        checkpoint_sha256=best['checkpoint_sha256'], train_dev_clean_O_ap70=best['AP']['ap70'],
        candidate_count=n, policy='maximum unrounded train_dev Clean O AP70; exact tie earliest',
        candidates={str(e): sha(branch / 'epochs' / f'{e:02d}' / 'dev_complete.json') for e in range(1, n+1)})
    bound(branch / 'selected.json', selected)
    return selected


def final(output, variant, smoke=False, interrupt_once=False):
    from predict import generate
    from evaluate import replay
    from support import DEFINITIONS, patched
    from common import link
    from opencood.tools import train_utils
    import opencood.data_utils.datasets as native_datasets
    from v2xr_raygen_e0e3.runtime import evidence, frozen_gt
    root = Path(output); branch = variant_path(root, variant, smoke)
    selected = read(branch / 'selected.json')
    if selected['status'] != 'COMPLETE' or selected['smoke_only'] != smoke:
        raise ValueError('Final evaluation requires a locked selected checkpoint')
    checkpoint = Path(selected['checkpoint'])
    if sha(checkpoint) != selected['checkpoint_sha256']:
        raise ValueError('Locked weights changed')
    dest = branch / 'final_evaluation'
    dest.mkdir(parents=True, exist_ok=True)
    count = read(root / 'config.json')['smoke_frames'] if smoke else 829
    ev, args = evidence(root), arguments(root)
    if os.environ.get('PYTHONHASHSEED') != '20261016':
        raise ValueError('Native final evaluation requires separate process hashseed20261016')
    contract = dict(parent_contract=read(root / 'contract.json')['identity'], variant=variant,
        selected_sha256=sha(branch / 'selected.json'), checkpoint_sha256=sha(checkpoint),
        frame_keys=ev['b']['selected_frame_keys'][:count], smoke_only=smoke, protocols=DEFINITIONS,
        seed=20261016, frozen_F_identity=read(root / 'contract.json')['inputs']['frozen_F'],
        actual_uecp_config=host_config(root),
        compatibility='Native D1A YAML controls only dataset/GT/postprocess; explicit private UECP factory controls model; direction uses native dm branch')
    contract['identity'] = identity(contract)
    bound(dest / 'contract.json', contract)
    if (dest / 'complete.json').exists():
        result = complete(dest / 'complete.json')
        if result['contract_identity'] != contract['identity']:
            raise ValueError('Final evaluation identity changed')
        for row in read(dest / 'prediction_manifest.json')['files']:
            if sha(row['path']) != row['sha256']:
                raise ValueError('Final immutable prediction changed')
        frozen_gt(root, read(dest / 'gt_manifest.json'), count)
        return result
    alias = dest / 'native_loader_config'
    alias.mkdir(exist_ok=True)
    original_config = Path(args.model_dir) / 'config.yaml'
    if not (alias / 'config.yaml').exists():
        atomic_bytes(alias / 'config.yaml', original_config.read_bytes())
    if sha(alias / 'config.yaml') != sha(original_config):
        raise ValueError('Native loader YAML bytes changed')
    for name in ('0.pth', 'net_epoch0.pth'):
        link(checkpoint, alias / name)
    args.model_dir = str(alias)
    native_build = native_datasets.build_dataset
    native_create = train_utils.create_model
    timings, distributions = [], []
    def build(hypes, visualize=False, train=True):
        return Dataset(root, native=native_build(hypes, visualize=visualize, train=train))
    def create(hypes):
        from v2xr_raygen_e0e3.training import rng_state, restore_rng
        # Native workers are seeded after model construction. Reproduce the original
        # construction's RNG advancement, so GT fallback behavior cannot drift merely
        # because UECP has a different number of parameters. Reference is never inferred.
        reference = native_create(hypes)
        native_rng = rng_state()
        del reference
        model = make(root, variant)
        restore_rng(native_rng)
        def before(layer, inputs):
            torch.cuda.synchronize(); layer._timing = time.monotonic()
        def after(layer, inputs, out):
            torch.cuda.synchronize(); timings.append(time.monotonic()-layer._timing)
            if len(timings) % 25 == 1:
                distributions.append(dict(forward=len(timings), statistics=layer.last_stats))
        model.register_forward_pre_hook(before); model.register_forward_hook(after)
        return model
    manifests = {}
    # The native generator still verifies strict tensor loading, real CAVs and every live GT.
    with patched(native_datasets, 'build_dataset', build), patched(train_utils, 'create_model', create):
        for condition in ('clean', 'fog', 'snow'):
            manifests[condition] = generate(condition, dest, args, ev, contract)
            if interrupt_once and condition == 'clean' and not (dest / 'expected_interruption.json').exists():
                if not smoke:
                    raise ValueError('Fault injection is smoke-only')
                write(dest / 'expected_interruption.json', dict(checkpoint_sha256=sha(checkpoint),
                    clean_manifest_sha256=sha(dest / 'predictions/clean/manifest.json')))
                raise SystemExit(43)
            gc.collect(); torch.cuda.empty_cache()
    results, predictions, gt = replay(dest, ev, contract, manifests)
    frozen_gt(root, gt, count)
    if (dest / 'expected_interruption.json').exists():
        old = read(dest / 'expected_interruption.json')
        if old['checkpoint_sha256'] != sha(checkpoint) or old['clean_manifest_sha256'] != sha(dest / 'predictions/clean/manifest.json'):
            raise ValueError('Recovery changed trained weights or cached Clean predictions')
    write(dest / 'model_statistics.json', dict(distributions=distributions,
        measured_forwards=len(timings), forward_seconds=timings,
        timing_note='synchronized GPU forward incl frozen GSPR, excludes data loading/NMS; cached frames excluded'))
    artifacts = [dest / n for n in
        ('contract.json', 'prediction_manifest.json', 'gt_manifest.json', 'per_frame_metrics.jsonl',
         'removed_gt_metrics.jsonl', 'model_statistics.json')]
    if (dest / 'expected_interruption.json').exists():
        artifacts.append(dest / 'expected_interruption.json')
    return commit(dest / 'complete.json', artifacts,
        contract_identity=contract['identity'], variant=variant, epoch=selected['epoch'],
        smoke_only=smoke, full_validation=count == 829, results=results,
        mean_forward_seconds=sum(timings)/len(timings) if timings else None,
        timing_complete=len(timings) == 3*count, checkpoint_sha256=sha(checkpoint))
