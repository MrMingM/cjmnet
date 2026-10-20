"""Isolate frozen runtime and reuse audited GT/prediction/AP functions, never old selection wrappers."""
import argparse
import importlib.metadata
import os
from pathlib import Path
import sys

from .common import ROOT, bound, read, safe, sha, stamp, write


def legacy_paths():
    for name in ('v2xr_official_repro', 'v2xr_train_attfuse', 'v2xr_protocol_audit'):
        path = str(ROOT / name)
        if path not in sys.path:
            sys.path.insert(1, path)


def arguments(output):
    from .common import settings
    legacy_paths()
    from auditlib import DEFAULTS
    values = dict(DEFAULTS)
    for k in values:
        values[k] = os.environ.get(k.upper(), values[k])
    train = safe(os.environ.get('TRAIN_ROOT', values['train_root'])).resolve()
    if train.name == 'train' and (train.parent / 'manifest.json').exists():
        train = train.parent
    validate = safe(os.environ.get('VALIDATE_ROOT', values['view_root'] + '/validate')).resolve()
    if validate.name != 'validate':
        raise ValueError('VALIDATE_ROOT must be the existing official validate directory')
    values.update(train_root=str(train), view_root=str(validate.parent), output_root=str(output),
                  resume=True, stage='all', seed=settings()['seed'])
    return argparse.Namespace(**values)


def evidence(output):
    return read(Path(output) / 'preflight/evidence.json')


def bootstrap(output, gpu=True):
    legacy_paths()
    from observe import bootstrap as official_bootstrap, module_inventory
    args, ev = arguments(output), evidence(output)
    official_bootstrap(args.runtime_root)
    import torch
    from opencood.tools import train_utils
    if ({k: importlib.metadata.version(k) for k in ev['a']['versions']} != ev['a']['versions']
            or sys.version != read(Path(output) / 'contract.json')['environment']['python']):
        raise ValueError('Python/package versions changed since preflight')
    if torch.version.hip != ev['a']['hip_version']:
        raise ValueError('HIP version differs from accepted runtime')
    if gpu and (not torch.version.hip or not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise RuntimeError('One explicitly selected HIP GPU is required')
    execution = Path(output) / 'execution.json'
    if gpu and execution.exists():
        selected = read(execution)
        if (selected['physical_gpu'] != os.environ.get('GPU')
                or selected['gpu_name'] != torch.cuda.get_device_name(0)):
            raise ValueError('Physical GPU/hardware differs from frozen capacity contract')
    actual = module_inventory(Path(args.runtime_root))
    for name, row in actual.items():
        if row['path'].endswith('.so') and ev['a']['imported_modules'].get(name) != row:
            raise ValueError('Native module differs from completed D-1A: ' + name)
    return args, ev


def audit_imports(output):
    from observe import module_inventory
    ev, args = evidence(output), arguments(output)
    actual = module_inventory(Path(args.runtime_root))
    for name, row in actual.items():
        if row['path'].endswith('.so') and ev['a']['imported_modules'].get(name) != row:
            raise ValueError('Native import differs from successful D-1A: ' + name)
    return actual


def hypes(output):
    from opencood.hypes_yaml import yaml_utils
    args = arguments(output)
    c = read(Path(output) / 'config.json')
    h = yaml_utils.load_yaml(str(Path(args.runtime_root) / 'opencood/hypes_yaml/V2X-R/LiDAR/V2XR_attfuse.yaml'))
    h.update(root_dir=str(Path(args.train_root) / 'train'), validate_dir=str(Path(args.view_root) / 'validate'),
             train_sim=False, eval_sim=False)
    if (h['fusion']['core_method'] != 'IntermediateFusionDatasetV2XR'
            or h['model']['core_method'] != 'point_pillar_intermediate'
            or h['model']['args']['use_modality'] != 'processed_lidar'
            or not h['fov'] or not h['fusion']['args']['proj_first']
            or h['preprocess']['args']['voxel_size'] != [.4, .4, 4]
            or h['postprocess']['order'] != 'hwl'
            or h['postprocess']['target_args']['score_threshold'] != .2
            or h['postprocess']['nms_thresh'] != .15
            or 'max_cav' in h['train_params']):
        raise ValueError('Unexpected official host configuration')
    h['train_params'].update(batch_size=c['effective_batch'], epoches=c['epochs'])
    return h


def frozen_gt(output, actual, limit):
    """IDs/order/geometry must equal already frozen D-1C, including the smoke prefix."""
    args = arguments(output)
    from auditlib import frozen_c
    frozen = frozen_c(args)
    reference = {(r['condition'], r['index']): r for r in frozen['gt']['frames']}
    fields = ('frame_key', 'official_gt_identity', 'fixed_gt_identity', 'official_ids',
              'fixed_ids', 'official_count', 'fixed_count', 'removed_ids')
    if len(actual['frames']) != 3 * limit:
        raise ValueError('GT frame coverage differs')
    for row in actual['frames']:
        if any(row[k] != reference[(row['condition'], row['index'])][k] for k in fields):
            raise ValueError('Frozen D-1C GT changed: ' + str(row['frame_key']))


def input_stamps(output):
    """Cheap metadata comparison on every process; changed files are never silently accepted."""
    root = Path(output)
    for row in read(root / 'preflight/input_stamps.json'):
        if stamp(row['path']) != row:
            raise ValueError('Input changed since shared verification; use a new output: ' + row['path'])
    ev = evidence(output)
    for name, digest in ev['a']['upstream_source_hashes'].items():
        if sha(Path(arguments(output).runtime_root) / name) != digest:
            raise ValueError('Runtime source changed: ' + name)
