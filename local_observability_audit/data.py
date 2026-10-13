"""Read-only fixed validation data, independent sampling and provenance adapter."""
import copy
from pathlib import Path
import random
import numpy as np
from .common import safe_path, object_hash, digest, select_frames

def inventory(root, max_cav):
    root = safe_path(root)
    if root.name not in ('validate', 'validation') or not root.is_dir():
        raise ValueError('An existing validation directory is required')
    records = []
    for scene in sorted(p for p in root.iterdir() if p.is_dir()):
        safe_path(scene)
        cavs = sorted(p.name for p in scene.iterdir() if p.is_dir())
        if not cavs or any(not c.lstrip('-').isdigit() for c in cavs):
            raise ValueError('Unexpected source directory')
        if int(cavs[0]) < 0:
            cavs = cavs[1:] + cavs[:1]
        cavs = cavs[:max_cav]
        for c in cavs:
            safe_path(scene/c)
        stamps = sorted(p.stem for p in (scene/cavs[0]).glob('*.yaml')
                        if 'additional' not in p.name and p.name!='data_protocol.yaml')
        for stamp in stamps:
            records.append({'index': len(records), 'scene_id': scene.name,
                            'frame_id': stamp, 'cav_ids': cavs})
    if not records:
        raise ValueError('Empty validation inventory')
    return records

def selection_and_manifest(cfg, max_cav, mode):
    inventories = {w: inventory(cfg['data'][w], max_cav) for w in cfg['weathers']}
    reference = inventories['clean']
    for w, inv in inventories.items():
        if inv != reference:
            raise ValueError('Fixed-weather frame/source inventory differs from clean: '+w)
    count = None if mode == 'full-validation' else cfg['smoke_frames_per_weather'] if mode == 'smoke' else cfg['frames_per_weather']
    selected = select_frames(reference, count, cfg['seed'])
    files = []
    for w in cfg['weathers']:
        root = safe_path(cfg['data'][w])
        for r in selected:
            for source in r['cav_ids']:
                for suffix in ('.pcd', '.yaml'):
                    p = root/r['scene_id']/source/(r['frame_id']+suffix)
                    resolved = safe_path(p)
                    if not resolved.is_file():
                        raise FileNotFoundError(p)
                    # Metadata may symlink to official validation; test links forbidden.
                    if not any(part in ('validate', 'validation') for part in resolved.parts):
                        raise ValueError('Selected data resolves outside validation: '+str(p))
                    files.append({'path': str(p), 'resolved': str(resolved),
                                  'size': resolved.stat().st_size, 'sha256': digest(resolved)})
    return selected, {'inventory_hash': object_hash(inventories), 'files': files,
                      'selection': selected, 'selection_hash': object_hash(selected)}

def verify_files(manifest):
    for row in manifest['files']:
        p = safe_path(row['path'])
        if str(p) != row['resolved'] or not p.is_file() or p.stat().st_size != row['size'] or digest(p) != row['sha256']:
            raise ValueError('Dataset changed: '+row['path'])

def build_dataset(hypes, root):
    from opencood.data_utils.datasets.intermediate_fusion_dataset import IntermediateFusionDataset
    class AuditDataset(IntermediateFusionDataset):
        def __getitem__(self, index):
            self.audit_sources = []
            sample = super().__getitem__(index)
            ego = sample['ego']
            if len(self.audit_sources) != ego['cav_num'] or len(ego['cav_ids']) != ego['cav_num']:
                raise ValueError('Source provenance cardinality mismatch')
            # OpenCOOD deduplicates through set(); canonicalize GT rows by stable object ID.
            n = int(ego['object_bbx_mask'].sum())
            if n != len(ego['object_ids']):
                raise ValueError('GT ID alignment mismatch')
            order = sorted(range(n), key=lambda k: str(ego['object_ids'][k]))
            ego['object_bbx_center'][:n] = ego['object_bbx_center'][:n][order]
            ego['object_ids'] = [str(ego['object_ids'][k]) for k in order]
            return sample

        def get_item_single_car(self, base, ego_pose, **kwargs):
            result = super().get_item_single_car(base, ego_pose, **kwargs)
            t = np.asarray(base['params']['transformation_matrix'], float)
            if (not np.isfinite(t).all() or not np.allclose(t[3], [0,0,0,1], atol=1e-6)
                    or not np.allclose(t[:3,:3].T@t[:3,:3], np.eye(3), atol=2e-5)):
                raise ValueError('Invalid source-to-ego rigid transform')
            self.audit_sources.append({'transform': t.copy(),
                'pose': np.asarray(base['params']['lidar_pose'], float),
                'points': np.asarray(result['projected_lidar']).copy(),
                'is_ego': bool(base['ego'])})
            return result
    h = copy.deepcopy(hypes)
    h['root_dir'] = h['validate_dir'] = str(safe_path(root))
    h['weather_augmentation'] = {'mode': 'none'}
    h['prediction_distillation'] = {}
    h['data_augment'] = []
    args = h['fusion'].get('args') or {}
    if not isinstance(args, dict) or args.get('proj_first', True) is not True:
        raise ValueError('Only proj_first=True supported')
    if args.get('cur_ego_pose_flag', True) is not True:
        raise ValueError('Only current ego coordinate system supported')
    if h.get('wild_setting', {}).get('async', False) or h.get('wild_setting', {}).get('loc_err', False):
        raise ValueError('Async/localization error unsupported')
    h['fusion']['args'] = dict(args, proj_first=True, cur_ego_pose_flag=True)
    return AuditDataset(h, visualize=False, train=False)

def get_frame(dataset, record, seed):
    value = int(object_hash([seed, record['scene_id'], record['frame_id']])[:8], 16)
    random.seed(value); np.random.seed(value)
    import torch
    torch.manual_seed(value)
    sample = dataset[record['index']]
    metadata = dataset.audit_sources
    ids = sample['ego']['cav_ids']
    if not ids or ids[0] != record['cav_ids'][0] or not metadata[0]['is_ego']:
        raise ValueError('Ego/source order mismatch')
    if not np.allclose(metadata[0]['transform'], np.eye(4), atol=1e-6):
        raise ValueError('Nonidentity ego frame')
    for row, source in zip(metadata, ids):
        row['source_id'] = source
    batch = dataset.collate_batch_test([sample])
    return batch, metadata
