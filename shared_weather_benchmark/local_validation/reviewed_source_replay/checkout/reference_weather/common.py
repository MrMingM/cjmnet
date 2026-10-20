"""Protocol, identity and IO helpers; never require a Git checkout."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parent


def server_only():
    if not sys.platform.startswith('linux'):
        raise RuntimeError('Research execution is server-only. Sync this directory to the Linux server.')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def read_json(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    os.replace(temporary, path)


def package_hashes():
    return {p.relative_to(REPO).as_posix(): digest(p) for p in sorted(PACKAGE.rglob('*'))
            if p.is_file() and (p.suffix in ('.py', '.yaml', '.json', '.sh', '.txt', '.md')
                                or p.name.startswith('LICENSE'))
            and p.name != 'source_snapshot.json' and '__pycache__' not in p.parts}


def verify_sources():
    snapshot = read_json(PACKAGE / 'source_snapshot.json')
    current = package_hashes()
    if current != snapshot['sources']:
        changed = sorted(k for k in set(current) | set(snapshot['sources'])
                         if current.get(k) != snapshot['sources'].get(k))
        raise RuntimeError('Source hash mismatch. Sync the entire reference_weather directory: '
                           + ', '.join(changed))
    return current


def project_hashes():
    # Record the actual model/data/evaluation implementation on the server.
    names = ['attfuse_code/point_pillar_intermediate.py', 'attfuse_code/att_bev_backbone.py',
             'attfuse_code/self_attn.py', 'opencood/data_utils/datasets/basedataset.py',
             'opencood/data_utils/datasets/intermediate_fusion_dataset.py',
             'opencood/data_utils/pre_processor/sp_voxel_preprocessor.py',
             'opencood/data_utils/post_processor/voxel_postprocessor.py',
             'opencood/models/sub_modules/pillar_vfe.py',
             'opencood/models/sub_modules/point_pillar_scatter.py',
             'opencood/models/sub_modules/auto_encoder.py',
             'opencood/data_utils/datasets/__init__.py',
             'opencood/data_utils/augmentor/data_augmentor.py',
             'opencood/data_utils/pre_processor/base_preprocessor.py',
             'opencood/data_utils/post_processor/base_postprocessor.py',
             'opencood/hypes_yaml/yaml_utils.py', 'opencood/utils/pcd_utils.py',
             'opencood/utils/eval_utils.py', 'opencood/utils/box_utils.py',
             'opencood/utils/common_utils.py', 'opencood/utils/transformation_utils.py',
             'opencood/tools/inference_utils.py', 'opencood/tools/train_utils.py']
    return {name: digest(REPO / name) for name in names}


def development_path(path, split):
    if split not in ('train', 'validate'):
        raise ValueError('Only original train/validate are allowed.')
    path = Path(path).expanduser().resolve(strict=True)
    if path.name != split or any(x.lower() == 'test' or x.lower().startswith('test_')
                                 or 'opv2v-w' in x.lower() for x in path.parts):
        raise ValueError(f'Expected original clean {split}, refusing {path}')
    return path


def external_output(path):
    path = Path(path).expanduser().resolve()
    if path == REPO or REPO in path.parents:
        raise ValueError('Large generated data and logs must be outside the source checkout.')
    if any(x.lower() == 'test' or 'opv2v-w' in x.lower() for x in path.parts):
        raise ValueError(f'Refusing test/official-weather output path: {path}')
    return path


def load_config(path):
    import yaml
    with Path(path).open(encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    if config.get('schema') != 1:
        raise ValueError('Unsupported protocol schema')
    if not config['splits'] or set(config['splits']) - {'train', 'validate'}:
        raise ValueError('Only train/validate are supported')
    if not config['conditions'] or set(config['conditions']) - {'fog', 'rain', 'snow'}:
        raise ValueError('Unknown weather condition')
    if len(config['conditions']) != len(set(config['conditions'])):
        raise ValueError('Duplicate weather conditions')
    validate_paths(config)
    if int(config['seed']) < 0 or int(config['loader_workers']) < 0:
        raise ValueError('Seed and loader worker count must be nonnegative')
    if int(config['generation_workers']) < 1:
        raise ValueError('Generation workers must be positive')
    if float(config['fog']['alpha']) <= 0 or float(config['fog']['gamma']) <= 0:
        raise ValueError('Fog alpha/gamma must be positive')
    if int(config['fog']['noise']) < 0:
        raise ValueError('Fog noise must be nonnegative')
    rain = config['rain']
    if float(rain['m']) != 1.328 or float(rain['lam']) != 905:
        raise ValueError('Pinned LISA Mie table requires m=1.328 and lam=905nm')
    if (float(rain['rain_rate_mm_h']) <= 0 or not 0 < rain['rmin'] < rain['rmax']
            or rain['bdiv'] <= 0 or rain['dst'] <= 0 or rain['mode'] not in ('strongest', 'last')):
        raise ValueError('Invalid LISA Rain configuration')
    if 'snow' in config['conditions']:
        snow = config['snow']
        if (snow['adaptation'] != 'opv2v_elevation_no_focal_v1'
                or snow['ground_plane'] != 'opv2v_lower_envelope_ransac'):
            raise ValueError('Snow adaptation/ground plane must be explicitly recorded')
        if not 4 <= int(snow['max_channels']) <= 64 or int(snow['channel_workers']) < 1:
            raise ValueError('Snow requires 4..64 channels and positive channel workers')
        if any(float(snow[k]) <= 0 for k in ('snowfall_rate_mm_h', 'terminal_velocity_m_s',
                'snow_density_g_cm3', 'snowflake_diameter_m', 'beam_divergence_rad',
                'noise_floor', 'elevation_cluster_tolerance_deg')):
            raise ValueError('Snow physical parameters must be positive')
        if not 0 <= float(snow['max_unmatched_fraction']) <= .01:
            raise ValueError('Snow unmatched-ray tolerance must be at most 1%')
    return config


def validate_paths(config):
    config['output_root'] = str(external_output(config['output_root']))
    config['assets_root'] = str(external_output(config['assets_root']))
    clean = Path(config['clean_root']).expanduser().resolve(strict=True)
    for split in config['splits']:
        development_path(clean / split, split)
    config['clean_root'] = str(clean)
    roots = [clean, Path(config['output_root']), Path(config['assets_root'])]
    for i, a in enumerate(roots):
        for b in roots[i + 1:]:
            if a == b or a in b.parents or b in a.parents:
                raise ValueError('Clean source, generated dataset and assets must have disjoint directory trees')


def stable_seed(seed, *keys):
    value = object_hash([int(seed), *map(str, keys)])
    return int(value[:8], 16)


def iter_frames(source):
    """Match BaseDataset scene/CAV/timestamp ordering, retaining all CAVs."""
    sample = 0
    for scene in sorted(p for p in Path(source).iterdir() if p.is_dir()):
        cavs = sorted(p.name for p in scene.iterdir() if p.is_dir())
        if not cavs or any(not x.lstrip('-').isdigit() for x in cavs):
            raise ValueError(f'Invalid CAV directories in {scene}')
        if int(cavs[0]) < 0:
            cavs = cavs[1:] + cavs[:1]
        stamps = sorted(p.stem for p in (scene / cavs[0]).glob('*.yaml')
                        if 'additional' not in p.name)
        if not stamps:
            raise ValueError(f'No frame YAML in {scene}')
        for cav in cavs:
            other = sorted(p.stem for p in (scene / cav).glob('*.yaml')
                           if 'additional' not in p.name)
            if other != stamps:
                raise ValueError(f'CAV timestamp mismatch: {scene / cav}')
        for stamp in stamps:
            entries = []
            for cav in cavs:
                pcd = scene / cav / (stamp + '.pcd')
                if not pcd.is_file():
                    raise FileNotFoundError(pcd)
                entries.append((cav, str(pcd)))
            yield dict(index=sample, scene=scene.name, timestamp=stamp, entries=entries)
            sample += 1


def read_points(path):
    import numpy as np
    from opencood.utils.pcd_utils import pcd_to_np
    points = pcd_to_np(str(path))
    if points.ndim != 2 or points.shape[1] != 4 or not np.isfinite(points).all():
        raise ValueError(f'Invalid XYZI: {path}')
    if len(points) and (points[:, 3].min() < 0 or points[:, 3].max() > 1):
        raise ValueError(f'Expected normalized RGB intensity: {path}')
    return np.ascontiguousarray(points, dtype=np.float32)


def write_pcd(path, points):
    import numpy as np
    import open3d as o3d
    path = Path(path)
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 4 or not np.isfinite(points).all():
        raise ValueError(f'Invalid generated cloud: {path}')
    if len(points) and (points[:, 3].min() < -1e-6 or points[:, 3].max() > 1 + 1e-6):
        raise ValueError(f'Generated intensity outside [0,1]: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + '.partial.pcd')
    if len(points):
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points[:, :3].astype(float))
        cloud.colors = o3d.utility.Vector3dVector(np.repeat(points[:, 3:4], 3, axis=1).astype(float))
        if not o3d.io.write_point_cloud(str(tmp), cloud, write_ascii=False, compressed=False):
            raise OSError(f'Failed to write {tmp}')
    else:
        tmp.write_bytes(b'# .PCD v0.7\nVERSION 0.7\nFIELDS x y z rgb\n'
                        b'SIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n'
                        b'WIDTH 0\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS 0\nDATA binary\n')
    loaded = read_points(tmp)
    if loaded.shape != points.shape or (len(points) and (
            not np.allclose(loaded[:, :3], points[:, :3], atol=2e-5, rtol=0)
            or not np.allclose(loaded[:, 3], points[:, 3], atol=1 / 255 + 1e-6, rtol=0))):
        raise ValueError(f'PCD round-trip mismatch: {path}')
    if path.exists():
        if digest(path) != digest(tmp):
            raise FileExistsError(f'Existing PCD differs from deterministic regeneration: {path}')
        tmp.unlink()
    else:
        os.replace(tmp, path)


def link_metadata(source, target):
    source, target = Path(source), Path(target)
    target.mkdir(parents=True, exist_ok=True)
    for entry in source.iterdir():
        if entry.suffix.lower() == '.pcd':
            continue
        dst = target / entry.name
        if dst.is_symlink() and dst.resolve(strict=True) == entry.resolve(strict=True):
            continue
        if dst.exists() or dst.is_symlink():
            raise FileExistsError(f'Refusing unrelated metadata: {dst}')
        dst.symlink_to(entry.resolve(strict=True), target_is_directory=entry.is_dir())
