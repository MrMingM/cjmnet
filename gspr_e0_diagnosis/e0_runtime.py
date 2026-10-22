"""Server-only bridges. Historical directories are always opened read-only."""
import argparse
from pathlib import Path
import os
import sys
from e0_common import HERE, WEATHERS, historical, identity, read, safe, sha


def setup(cli, gpu=True):
    if sys.platform != 'linux':
        raise RuntimeError('Research execution is server-only')
    if os.environ.get('PYTHONHASHSEED') != '20261016':
        raise ValueError('Frozen inference requires PYTHONHASHSEED=20261016')
    if not os.environ.get('ROCR_VISIBLE_DEVICES') or any(os.environ.get(k) is not None for k in ('HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES')):
        raise ValueError('Set ROCR_VISIBLE_DEVICES only')
    audit, evaluation, models = historical()
    audit.gpu()
    args = argparse.Namespace(**audit.DEFAULTS)
    args.output_root = str(cli.output)
    args.variant = 'C'
    args.resume = True
    args.smoke = False
    from provenance import verify_inputs
    evidence = verify_inputs(args)
    frozen = audit.frozen_c(args)
    audit.runtime(args, evidence['a'])
    import torch
    if gpu and (not torch.version.hip or not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise ValueError('Exactly one visible ROCm GPU required')
    return args, evidence, frozen, audit, evaluation, models


def selected_model(root, variant, audit):
    """Read selection without calling the historical selector (which writes selected.json)."""
    from support import check_identity
    root = safe(root)
    selected = read(root / 'selected.json')
    contract = read(root / 'contract.json')
    check_identity(selected)
    check_identity(contract)
    if (contract['smoke_only'] or contract['variant'] != variant or contract['epochs'] != 20
            or contract['config_sha256'] != sha(root / 'config.yaml')
            or contract['sources'] != audit.sources() or selected['weather_used']
            or selected['candidate_count'] != 20 or selected['policy'] != audit.SELECTION
            or selected['training_contract_sha256'] != sha(root / 'contract.json')):
        raise ValueError('Original training/selection identity differs: ' + variant)
    candidates = []
    for epoch in range(1, 21):
        row = audit.complete(root / 'epochs' / f'{epoch:02d}' / 'complete.json')
        ap = audit.complete(safe(row['clean_summary']))
        if (row['epoch'] != epoch or row['contract'] != contract['identity'] or row['smoke_only']
                or sha(row['clean_summary']) != row['clean_summary_sha256']
                or ap['conditions'] != ['clean'] or ap['frame_count'] != 829 or not ap['full_validation']
                or ap['checkpoint_sha256'] != row['checkpoint_sha256']
                or row['AP70'] != ap['results']['clean']['AP']['O']['ap70']):
            raise ValueError('Original Clean-only candidate evidence differs')
        candidates.append(row)
    best = min(candidates, key=lambda r: (-r['AP70'], r['epoch']))
    if (selected['epoch'] != best['epoch'] or selected['checkpoint'] != best['checkpoint']
            or selected['checkpoint_sha256'] != best['checkpoint_sha256']
            or selected['clean_O_AP70'] != best['AP70']
            or sha(safe(selected['checkpoint'])) != selected['checkpoint_sha256']):
        raise ValueError('Selected checkpoint is not the committed Clean O optimum')
    summary = audit.complete(root / 'final_evaluation/summary.json')
    if (summary['checkpoint_sha256'] != selected['checkpoint_sha256']
            or summary['conditions'] != list(WEATHERS) or summary['frame_count'] != 829
            or not summary['full_validation'] or summary['model_variant'] != variant):
        raise ValueError('Original full evaluation/selection mismatch')
    return dict(root=str(root), selected=selected, contract=contract, summary=summary)


def cache_audit(model, evidence, frozen):
    from support import check_identity, compare_gt, load_prediction, prediction_tensors
    # prediction_tensors additionally rejects nonfinite values and malformed None/empty states.
    import numpy as np
    root = Path(model['root']) / 'final_evaluation'
    contract = read(root / 'contract.json')
    check_identity(contract)
    if (contract['frame_keys'] != evidence['b']['selected_frame_keys']
            or contract['checkpoint_sha256'] != model['selected']['checkpoint_sha256']
            or contract['conditions'] != list(WEATHERS)):
        raise ValueError('Historical cache contract differs')
    gt = read(root / 'gt_manifest.json')
    from evaluation import inherited
    inherited.assert_frozen_gt(gt, frozen, WEATHERS, 829)
    inventory = read(root / 'prediction_manifest.json')
    check_identity(inventory)
    if len(inventory['files']) != 3*829 or inventory['identity'] != model['summary']['prediction_cache_identity']:
        raise ValueError('Historical prediction inventory differs')
    catalog = {(r['condition'], r['index']): r for r in inventory['files']}
    if len(catalog) != 3*829:
        raise ValueError('Duplicate cache records')
    for weather in WEATHERS:
        manifest = read(root / 'predictions' / weather / 'manifest.json')
        if manifest['status'] != 'complete' or len(manifest['entries']) != 829 or manifest['contract_identity'] != contract['identity']:
            raise ValueError('Incomplete historical predictions')
        for index, key in enumerate(contract['frame_keys']):
            path = root / 'predictions' / weather / f'{index:06d}.json'
            digest = sha(path)
            if digest != manifest['entries'][str(index)]['sha256'] or digest != catalog[(weather, index)]['sha256']:
                raise ValueError('Historical prediction hash mismatch')
            if safe(catalog[(weather, index)]['path']).resolve() != path.resolve():
                raise ValueError('Historical prediction path mismatch')
            record = load_prediction(path, contract, weather, index, key)
            boxes, scores = prediction_tensors(record)
            if boxes is not None and not np.isfinite(boxes.numpy()).all():
                raise ValueError('Nonfinite historical boxes')
            compare_gt(record['official_gt'], evidence['frames'][index]['conditions'][weather]['final_gt'], 'E0 historical GT')
        for name, digest in manifest['official_output_sha256'].items():
            if sha(root / name) != digest:
                raise ValueError('Historical official inference evidence changed')
    return dict(contract_sha256=sha(root / 'contract.json'),
        checkpoint_sha256=model['selected']['checkpoint_sha256'],
        prediction_identity=inventory['identity'], gt_identity=gt['identity'], frames=829,
        cached_records_verified=2487)


def runtime_info():
    import torch
    from importlib import metadata
    return dict(python=sys.version, executable=sys.executable,
        packages={k: metadata.version(k) for k in ('torch', 'numpy', 'scipy', 'shapely', 'matplotlib', 'Pillow', 'PyYAML')},
        hip=torch.version.hip, physical_gpu=os.environ['ROCR_VISIBLE_DEVICES'],
        gpu=torch.cuda.get_device_name(0), torch_threads=torch.get_num_threads(),
        env={k: os.environ.get(k) for k in ('PYTHONHASHSEED', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
        backend=dict(benchmark=torch.backends.cudnn.benchmark, deterministic=torch.backends.cudnn.deterministic,
            enabled=torch.backends.cudnn.enabled, algorithms_deterministic=torch.are_deterministic_algorithms_enabled(),
            cudnn_tf32=torch.backends.cudnn.allow_tf32, matmul_tf32=torch.backends.cuda.matmul.allow_tf32))


def model_inventory(cli, audit):
    reference = read(HERE / 'original_FINAL_RESULTS.json')
    if reference['status'] != 'complete':
        raise ValueError('Archived original final result incomplete')
    result = {}
    for variant, directory in (('C', 'L0_C'), ('D', 'L1_D')):
        model = selected_model(cli.original / directory, variant, audit)
        if model['summary'] != reference['results'][variant]:
            raise ValueError('Server final summary differs from archived FINAL_RESULTS.json: ' + variant)
        result[variant] = model
    return result


def state_hash(model):
    from e0_numeric import array_hash
    return identity({k: array_hash(v.detach().cpu().numpy()) for k, v in model.state_dict().items()})


def seed():
    import random
    import numpy as np
    import torch
    random.seed(20261016)
    np.random.seed(20261016)
    torch.manual_seed(20261016)


def tensor_numpy(value):
    import numpy as np
    if hasattr(value, 'detach'):
        return value.detach().cpu().numpy()
    if isinstance(value, (tuple, list)):
        return np.asarray([tensor_numpy(v) for v in value])
    return np.asarray(value)


def points_in_targets(xyz, targets, expansion=1.):
    import numpy as np
    import torch
    from opencood.pcdet_utils.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu
    if not targets:
        return np.zeros((0, len(xyz)), dtype=np.bool_)
    boxes = np.asarray([r['box'] for r in targets], dtype=np.float32)[:, [0, 1, 2, 5, 4, 3, 6]].copy()
    boxes[:, 3:6] *= expansion
    return tensor_numpy(points_in_boxes_cpu(torch.from_numpy(np.asarray(xyz, dtype=np.float32).copy()), torch.from_numpy(boxes))) > 0


def smoke_indices(evidence):
    keys = evidence['b']['selected_frame_keys']
    counts, indices = {}, []
    for index, key in enumerate(keys):
        n = counts.get(key[0], 0)
        if n < 2:
            indices.append(index)
            counts[key[0]] = n+1
    return indices


def npz_write(path, **arrays):
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(str(path)+'.tmp')
    with temp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
