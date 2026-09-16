"""Full-split loaders and immutable experiment contracts."""
import copy
from pathlib import Path
from gspr_communication.runtime import (ROOT, sha256, verify_frozen, new_output, seed_all,
                                        seed_worker, device, write_json)
from gspr_review.runtime import load_config as base_config


def load_config(path, frontend):
    options, hypes = base_config(path, frontend)
    for key, minimum in (('queries', 1), ('teacher_candidates', 2), ('risk_epochs', 1), ('gain_epochs', 1), ('workers', 0)):
        if not isinstance(options.get(key), int) or options[key] < minimum:
            raise ValueError(f'{key} must be an integer >= {minimum}')
    if options.get('gain_scale', 0) <= 0 or options.get('learning_rate', 0) <= 0:
        raise ValueError('Positive gain scale and learning rate required')
    if (options.get('frame_stride', 1) != 1 or options.get('train_scene_indices') is not None
            or options.get('validation_scene_indices') is not None or options.get('steps_per_epoch')):
        raise ValueError('This experiment requires full splits, stride 1, no step cap')
    train, val = Path(hypes['root_dir']).resolve(), Path(hypes['validate_dir']).resolve()
    if train == val or train in val.parents or val in train.parents:
        raise ValueError('Train/validation roots overlap')
    if not train.is_dir() or not val.is_dir():
        raise FileNotFoundError(f'Dataset roots unavailable: {train}, {val}')
    overlap = {p.name for p in train.iterdir() if p.is_dir()} & {p.name for p in val.iterdir() if p.is_dir()}
    if overlap:
        raise ValueError('Scenario names overlap; reconcile scene split: '+str(sorted(overlap)))
    return options, hypes


def make_loader(hypes, options, train=False, weather='mixed', smoke=0):
    import torch
    import numpy as np
    from torch.utils.data import DataLoader, Subset
    from gspr_communication.dataset_adapter import CommunicationDataset

    class EvidenceDataset(CommunicationDataset):
        def __getitem__(self, index):
            self._clouds = []
            self._weather_clouds = []
            result = super().__getitem__(index)
            result['ego']['evidence_clouds'] = self._clouds
            result['ego']['evidence_weather_clouds'] = self._weather_clouds
            return result

        def get_item_single_car(self, selected_cav_base, ego_pose, **kwargs):
            result = super().get_item_single_car(selected_cav_base, ego_pose, **kwargs)
            self._clouds.append(np.array(result['projected_lidar'], copy=True))
            self._weather_clouds.append(np.array(result.get('weather_projected_lidar', result['projected_lidar']), copy=True))
            return result

        def collate_batch_train(self, batch):
            result = super().collate_batch_train(batch)
            for key in ('evidence_clouds', 'evidence_weather_clouds'):
                result['ego'][key] = [torch.from_numpy(c) for b in batch for c in b['ego'][key]]
            return result

    local = copy.deepcopy(hypes)
    if weather == 'clean':
        local.pop('weather_augmentation', None)
    elif weather != 'mixed':
        local['weather_augmentation']['mode'] = 'physics_'+weather
    ds = EvidenceDataset(local, train=train)
    indices = list(range(len(ds)))
    if smoke:
        indices = indices[:smoke]
    generator = torch.Generator().manual_seed(int(options['seed']))
    loader = DataLoader(Subset(ds, indices), batch_size=1, shuffle=False,
                        num_workers=int(options['workers']), collate_fn=ds.collate_batch_test,
                        worker_init_fn=seed_worker, generator=generator)
    return ds, loader, indices


def input_branch(ego, branch):
    key = 'processed_lidar' if branch == 'clean' else 'processed_lidar_weather'
    if key not in ego:
        raise ValueError('Requested weather branch missing')
    return dict(processed_lidar=ego[key], record_len=ego['record_len'],
                transforms=ego['communication_transforms'],
                clouds=ego['evidence_clouds' if branch == 'clean' else 'evidence_weather_clouds'])


def load_model(hypes, options, checkpoint, target):
    import torch
    from .model import EvidenceModel
    digest = sha256(checkpoint)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    model = EvidenceModel(hypes['model']['args'], options)
    model.engine.load_frontend(state)
    if sha256(checkpoint) != digest:
        raise RuntimeError('Frontend changed during load')
    return model.to(target).eval(), digest


def contract(options, frontend, digest):
    files = ['gspr_evidence/'+x+'.py' for x in ('geometry', 'heads', 'model', 'protocol', 'supervision', 'runtime', 'prepare')]
    files += ['gspr_communication/'+x+'.py' for x in ('model', 'dataset_adapter', 'codec', 'masked_attfuse')]
    files += ['opencood/data_utils/datasets/intermediate_fusion_dataset.py', 'opencood/loss/point_pillar_loss.py']
    return dict(schema=1, frontend_sha256=digest, frontend_config_sha256=sha256(frontend), options=options,
                pipeline_sources={f: sha256(ROOT/f) for f in files})


def validate_contract(specification):
    if not specification.get('pipeline_sources'):
        raise ValueError('Cache has no source fingerprint')
    for path, expected in specification['pipeline_sources'].items():
        if sha256(ROOT/path) != expected:
            raise ValueError('Cached teacher/source mismatch: '+path)


def load_heads(model, path, expected):
    import torch
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state['contract'] != expected:
        raise ValueError('Checkpoint/data/protocol/frontend contract mismatch')
    model.heads.load_state_dict(state['heads'], strict=True)
    model.heads['gain'].matching = state['variant'] != 'concat'
    model.disable_u = state['variant'] == 'no_u'
    return state
