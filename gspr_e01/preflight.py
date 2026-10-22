"""Server resource and frozen-input checks. No extraction, copying or training."""
import argparse
from pathlib import Path
import os
import shutil
import sys
from e01_common import PROJECT, SETTINGS, atomic, reference_results, safe, sha, sources


def probe(destination):
    import torch
    if (not torch.version.hip or not torch.cuda.is_available() or torch.cuda.device_count() != 1
            or not os.environ.get('ROCR_VISIBLE_DEVICES')
            or any(k in os.environ for k in ('HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES'))):
        raise ValueError('Require one physical ROCm GPU; only ROCR_VISIBLE_DEVICES')
    free, total = torch.cuda.mem_get_info(0)
    atomic(destination, dict(status='complete', physical_gpu=os.environ['ROCR_VISIBLE_DEVICES'],
        name=torch.cuda.get_device_name(0), hip=torch.version.hip,
        free_GiB=free/2**30, total_GiB=total/2**30, free_fraction=free/total))


def resources(root, mode, probes):
    requirements = SETTINGS['resources']
    available = {line.split(':', 1)[0]: int(line.split()[1])*1024
        for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')}
    ram = available['MemAvailable']/2**30
    minimum = requirements[mode+'_ram_available_GiB']
    if ram < minimum or shutil.disk_usage(root).free/2**30 < requirements['disk_free_GiB']:
        raise ValueError('Insufficient available CPU RAM or output disk; see settings.json; no automatic downsizing')
    if mode == 'parallel' and probes['E0']['physical_gpu'] == probes['E1']['physical_gpu']:
        raise ValueError('Parallel E0/E1 must use different physical GPUs')
    for name, row in probes.items():
        if row['free_fraction'] < .95 or row['free_GiB'] < requirements[name+'_gpu_free_GiB']:
            raise ValueError('GPU is occupied or lacks the registered free-memory reserve: '+name)
    io_file = root / 'logs/io_probe.json'
    atomic(io_file, dict(status='complete', io='create/fsync/atomic replace on output filesystem'))
    return dict(status='complete', mode=mode, available_ram_GiB=ram,
        available_disk_GiB=shutil.disk_usage(root).free/2**30, cpu_count=os.cpu_count(),
        gpu_probes=probes, requirements=requirements,
        limitation='Static reserves plus real smoke; runtime resource changes still cause technical failure')


def inputs(cli):
    sys.path.insert(0, str(PROJECT / 'gspr_e0_diagnosis'))
    from e0_runtime import setup, model_inventory, cache_audit
    from e0_common import sources as e0_sources
    e0_sources()
    e0 = argparse.Namespace(output=Path(cli.model_root), original=Path(cli.original_root))
    args, evidence, frozen, audit, _, _ = setup(e0)
    models = model_inventory(e0, audit)
    reference, reference_source = reference_results()
    if any(models[v]['summary'] != reference['results'][v] for v in ('C', 'D')):
        raise ValueError('Actual T0 summaries do not equal the full-precision reference')
    caches = {v: cache_audit(m, evidence, frozen) for v, m in models.items()}
    args.train_root = str(cli.train_root)
    manifest = audit.verify_train(args)
    from e1_data import inventory
    pairs = inventory(manifest, args.train_root)
    from auditlib import load_frozen
    load_frozen('training', 'training.py').verify_clean_loss_view(args, evidence)
    keys = {(s['scene'], t) for s in manifest['scenes'] for t in s['timestamps']}
    if keys & {tuple(k) for k in evidence['b']['selected_frame_keys']}:
        raise ValueError('Train/validation overlap')
    from adapters import retrieve_tree, training_tree
    import ast
    retrieve_tree((Path(args.runtime_root) / 'opencood/data_utils/datasets/opv2v/basedataset.py').read_text(encoding='utf-8'))
    _, edits = training_tree((PROJECT / 'v2xr_weather_l012/training.py').read_text(encoding='utf-8'))
    result = dict(status='complete', sources=sources(), T0=models, caches=caches, T0_reference=reference_source,
        train_manifest_sha256=sha(Path(args.train_root) / 'manifest.json'),
        train_manifest_identity=manifest['identity'], train_data_acceptance=manifest['train_data_acceptance'],
        three_weather_training_identity=pairs['identity'], train_frames=8083,
        validation_frames=829, input_identity=evidence['input_manifest']['identity'],
        frozen_GT_identity=frozen['gt']['identity'], training_AST_edits=edits,
        source_grammar='Python '+str(ast.__name__),
        no_training_performed=True, no_test=True)
    atomic(Path(cli.model_root) / 'preflight.json', result)


def validate_paths(root, original, train, existing=None):
    root, original, train = safe(root).resolve(), safe(original).resolve(strict=True), safe(train).resolve(strict=True)
    if train.name == 'train' and (train.parent / 'manifest.json').exists():
        train = train.parent
    # Check output against all fixed data/runtime/evaluation roots without importing OpenCOOD.
    sys.path.insert(0, str(PROJECT / 'v2xr_weather_l012'))
    import auditlib
    inputs_ = [PROJECT, original, train]+[safe(v).resolve() for k, v in auditlib.DEFAULTS.items()
        if k not in ('output_root', 'seven_zip', 'train_root')]
    if Path('/data') not in root.parents or any(root == p or root in p.parents or p in root.parents for p in inputs_):
        raise ValueError('Output overlaps a read-only source or is outside /data')
    if existing:
        other = safe(existing).resolve(strict=True)
        if other == root or root in other.parents or other in root.parents or any(
                other == p or other in p.parents or p in other.parents for p in inputs_):
            raise ValueError('Existing E0 output must be independent of current output and all frozen inputs')
    return root, original, train
