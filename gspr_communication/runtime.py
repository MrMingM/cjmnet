"""Shared standalone entrypoint utilities. No monkey patching or model registry edits."""
import copy
import hashlib
import json
import os
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen():
    rows = json.loads((ROOT/'gspr_communication/frozen_sources.json').read_text(encoding='utf-8-sig'))
    mismatches = [r['path'] for r in rows if not (ROOT/r['path']).is_file() or sha256(ROOT/r['path']) != r['sha256']]
    if mismatches:
        raise RuntimeError('Frozen source mismatch; reconcile server version before running: ' + ', '.join(mismatches))
    return rows


def new_output(path):
    target = Path(path).resolve()
    allowed = Path('/data/cjm/datasets/logs').resolve()
    if target == allowed or allowed not in target.parents:
        raise ValueError('Outputs must be in a new subdirectory of /data/cjm/datasets/logs')
    target.mkdir(parents=True, exist_ok=False)
    return target


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    import numpy as np
    import torch
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def device():
    import torch
    physical = os.environ.get('ROCR_VISIBLE_DEVICES', '')
    if physical not in tuple(str(i) for i in range(7)):
        raise ValueError('Set exactly one available HCU 0-6 via ROCR_VISIBLE_DEVICES')
    if 'HIP_VISIBLE_DEVICES' in os.environ or 'CUDA_VISIBLE_DEVICES' in os.environ:
        raise ValueError('Unset HIP_VISIBLE_DEVICES and CUDA_VISIBLE_DEVICES')
    if not torch.cuda.is_available():
        raise RuntimeError('No GPU/HCU visible')
    return torch.device('cuda:0')


def load_config(experiment, frontend_config):
    import yaml
    from opencood.hypes_yaml.yaml_utils import load_yaml
    options = yaml.safe_load(Path(experiment).read_text(encoding='utf-8'))
    hypes = load_yaml(str(frontend_config))
    if hypes['fusion']['core_method'] != 'IntermediateFusionDataset':
        raise ValueError('Expected classic IntermediateFusionDataset frontend config')
    hypes['fusion']['args'] = {'proj_first': True}
    # Existing GSPR checkpoint architecture is authoritative.
    for key in ('root_dir', 'validate_dir'):
        if options.get(key):
            hypes[key] = options[key]
    if Path(hypes['root_dir']).resolve() == Path(hypes['validate_dir']).resolve():
        raise ValueError('Training and validation roots must differ')
    # Changes only this private configuration, not the base training config.
    hypes['data_augment'] = []
    if 'weather_augmentation' in options:
        hypes['weather_augmentation'] = copy.deepcopy(options['weather_augmentation'])
    return options, hypes


def load_model(hypes, options, checkpoint, target_device):
    import torch
    from .model import CommunicationModel
    before = sha256(checkpoint)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    model = CommunicationModel(hypes['model']['args'], options['communication'])
    model.load_frontend(state)
    if sha256(checkpoint) != before:
        raise RuntimeError('Frontend checkpoint changed during load; use an immutable completed snapshot')
    return model.to(target_device), before


def make_loader(hypes, options, train=False, test=False):
    import torch
    from torch.utils.data import DataLoader, Subset
    from .dataset_adapter import CommunicationDataset
    ds = CommunicationDataset(copy.deepcopy(hypes), train=train)
    split = 'train' if train else 'validation'
    indices = list(range(len(ds)))
    # Scene IDs refer to sorted scenario directories, not random frame splits.
    selected = options.get(split + '_scene_indices')
    if selected is not None:
        if len(set(selected)) != len(selected) or not selected:
            raise ValueError('Scene indices must be nonempty and unique')
        indices = []
        for scene in selected:
            if scene < 0 or scene >= len(ds.len_record):
                raise ValueError('Scene index out of range')
            indices.extend(range(ds.len_record[scene-1] if scene else 0, ds.len_record[scene]))
    stride = int(options.get('frame_stride', 1))
    if stride < 1:
        raise ValueError('frame_stride must be positive')
    indices = indices[::stride]
    if not indices:
        raise ValueError('Empty dataset selection')
    generator = torch.Generator().manual_seed(int(options['seed']))
    loader = DataLoader(Subset(ds, indices), batch_size=1 if test else int(options['batch_size']),
                        shuffle=train, drop_last=False, num_workers=int(options['workers']),
                        collate_fn=ds.collate_batch_test if test else ds.collate_batch_train,
                        worker_init_fn=seed_worker, generator=generator)
    return ds, loader, indices


def model_input(ego, options):
    # Whitelist: supervision/labels never enter A/B, even in a centralized batch.
    key = options.get('lidar_key', 'processed_lidar')
    if key not in ('processed_lidar', 'processed_lidar_weather') or key not in ego:
        raise ValueError('Configured LiDAR branch unavailable: ' + key)
    return {'processed_lidar': ego[key], 'record_len': ego['record_len']}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
