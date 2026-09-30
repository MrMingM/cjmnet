"""Materialize the existing local-sensor physics weather stream as OPV2V PCDs.

Run this on the research server. This is a fixed epoch-0 realization of the
existing online augmentors, suitable for reuse by many detector experiments.
The source OPV2V tree is never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import yaml

from opencood.data_utils.augmentor.physics_fog import PhysicsFogAugmentor
from opencood.data_utils.augmentor.physics_rain import PhysicsRainAugmentor
from opencood.data_utils.augmentor.physics_snow import PhysicsSnowAugmentor
from opencood.data_utils.datasets.intermediate_fusion_dataset import (
    select_mixed_physics_weather,
)
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils.pcd_utils import mask_ego_points, pcd_to_np
from opencood.utils.transformation_utils import x1_to_x2
from gspr_review.resources import resolve_weather


REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path('/data/cjm/datasets/opv2v-physics-fixed-v1')
MODES = ('fog', 'rain', 'snow')
LEGACY_SAFE_YAML_GENERATOR_SHA256 = (
    '5e5323f2445463f1a76b65f57de1b230162673f4af7462fc57f89aa51b960734')
CLASSES = dict(fog=PhysicsFogAugmentor, rain=PhysicsRainAugmentor,
               snow=PhysicsSnowAugmentor)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
    os.replace(temporary, path)


def cav_order(scene):
    names = sorted(p.name for p in scene.iterdir() if p.is_dir())
    if not names or any(not name.lstrip('-').isdigit() for name in names):
        raise ValueError(f'Unexpected CAV directories in {scene}')
    # Match BaseDataset: one negative roadside ID, if first, is moved last.
    if int(names[0]) < 0:
        names = names[1:] + names[:1]
    return names


def frames(source):
    """Yield the exact (sample_index, agent_index) keys used by BaseDataset."""
    sample = 0
    for scene in sorted(p for p in source.iterdir() if p.is_dir()):
        cavs = cav_order(scene)
        ego = scene / cavs[0]
        stamps = sorted(p.stem for p in ego.glob('*.yaml')
                        if 'additional' not in p.name)
        if not stamps:
            raise ValueError(f'No ego timestamps in {ego}')
        for timestamp in stamps:
            for agent, cav in enumerate(cavs):
                source_pcd = scene / cav / (timestamp + '.pcd')
                if not source_pcd.is_file():
                    raise FileNotFoundError(source_pcd)
                yield sample, agent, scene.name, cav, timestamp, source_pcd
            sample += 1


def load_pose(path):
    """Use OpenCOOD's loader for trusted OPV2V NumPy-tagged frame YAML."""
    pose = np.asarray(load_yaml(str(path))['lidar_pose'], dtype=np.float64)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError(f'Invalid lidar_pose in {path}')
    return pose.tolist()


def ensure_link(source, target):
    source = source.resolve(strict=True)
    if target.is_symlink():
        if target.resolve(strict=True) != source:
            raise FileExistsError(f'Wrong existing link: {target}')
        return
    if target.exists():
        raise FileExistsError(f'Refusing to overwrite: {target}')
    target.symlink_to(source, target_is_directory=source.is_dir())


def link_metadata(source_cav, target_cav):
    target_cav.mkdir(parents=True, exist_ok=True)
    for entry in source_cav.iterdir():
        if entry.suffix.lower() != '.pcd':
            ensure_link(entry, target_cav / entry.name)


def make_augmentors(weather_config, lidar_range):
    seed = int(weather_config['seed'])
    result = {}
    for mode in MODES:
        config = dict(weather_config.get('physics_' + mode, {}))
        if mode == 'fog':
            lookup = Path(config['lookup_dir']).expanduser()
            config['lookup_dir'] = str((lookup if lookup.is_absolute()
                                        else REPO / lookup).resolve())
        result[mode] = CLASSES[mode](config, lidar_range, seed=seed)
    return result


def severity(augmentor, mode, sample):
    if mode == 'fog':
        return augmentor.sample_alpha(0, sample)
    if mode == 'rain':
        return augmentor.sample_rain_rate(0, sample)
    return augmentor.sample_snowfall_rate(0, sample)


def write_pcd(path, points, verify=False):
    """Use the same Open3D RGB convention consumed by pcd_to_np."""
    import open3d as o3d

    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 4 or not np.isfinite(points).all():
        raise ValueError(f'Invalid generated XYZI for {path}: {points.shape}')
    if len(points) and (points[:, 3].min() < -1e-5 or
                        points[:, 3].max() > 1 + 1e-5):
        raise ValueError(f'Intensity outside [0,1] for {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + '.tmp.pcd')
    if len(points):
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points[:, :3].astype(np.float64))
        colors = np.repeat(np.clip(points[:, 3:4], 0, 1), 3, axis=1)
        cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
        if not o3d.io.write_point_cloud(str(temporary), cloud,
                                        write_ascii=False, compressed=False):
            raise OSError(f'Open3D could not write {temporary}')
    else:
        # Open3D does not reliably write an empty cloud across versions.
        temporary.write_bytes(
            b'# .PCD v0.7\nVERSION 0.7\nFIELDS x y z rgb\n'
            b'SIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n'
            b'WIDTH 0\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n'
            b'POINTS 0\nDATA binary\n')
    try:
        if verify:
            loaded = pcd_to_np(str(temporary))
            if loaded.shape != points.shape or not np.isfinite(loaded).all():
                raise ValueError(f'PCD round trip failed for {path}')
            if len(points) and (not np.allclose(loaded[:, :3], points[:, :3],
                                                atol=2e-5, rtol=0) or
                                not np.allclose(loaded[:, 3], points[:, 3],
                                                atol=1 / 255 + 1e-5, rtol=0)):
                raise ValueError(f'PCD coordinates/intensity changed for {path}')
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def in_range(points, transform, lidar_range):
    if not len(points):
        return np.zeros(0, dtype=bool), np.empty((0, 3))
    xyz = points[:, :3].astype(np.float64) @ transform[:3, :3].T
    xyz += transform[:3, 3]
    low, high = np.asarray(lidar_range[:3]), np.asarray(lidar_range[3:])
    return ((xyz > low) & (xyz < high)).all(axis=1), xyz


def make_weather_file(raw, target_pcd, augmentor, mode, sample, agent,
                      transform, lidar_range,
                      verify=False):
    if target_pcd.is_file() and target_pcd.stat().st_size > 0:
        return False
    if target_pcd.exists():
        raise FileExistsError(f'Invalid existing PCD: {target_pcd}')
    if raw.ndim != 2 or raw.shape[1] != 4 or not np.isfinite(raw).all():
        raise ValueError(f'Invalid source XYZI for {target_pcd}')
    rng = augmentor.make_rng(0, sample, agent, stream=1)
    if len(raw):
        raw = raw[rng.permutation(len(raw))]
    raw = mask_ego_points(raw)
    output, _ = augmentor.augment(raw, rng, severity(augmentor, mode, sample))
    clean_inside, clean_xyz = in_range(raw, transform, lidar_range)
    weather_inside, _ = in_range(output, transform, lidar_range)
    if not clean_inside.any():
        # Match the online branch's clean-empty source alignment rule.
        output = np.empty((0, 4), dtype=np.float32)
    elif not weather_inside.any():
        # Match its one-clean-point fallback when weather has no ego-range voxel.
        possible = np.flatnonzero(clean_inside)
        nearest = possible[np.argmin(np.linalg.norm(clean_xyz[possible], axis=1))]
        output = raw[[nearest]].copy()
    write_pcd(target_pcd, output, verify=verify)
    return True


def config_manifest(args, experiment, frontend, augmentors):
    config = dict(experiment['weather_augmentation'])
    config['physics_fog'] = dict(config.get('physics_fog', {}))
    config['physics_fog']['lookup_dir'] = augmentors['fog'].lookup_dir
    fog_tables = {p.name: digest(p) for p in sorted(
        Path(augmentors['fog'].lookup_dir).glob('alpha_*.pickle'))}
    return dict(schema=1, status='in_progress', epoch=0,
                source_roots={key: str(Path(experiment[
                    'root_dir' if key == 'train' else 'validate_dir']).resolve())
                    for key in ('train', 'validate')},
                experiment_config_sha256=digest(args.experiment_config),
                frontend_config_sha256=digest(args.frontend_config),
                augmentor_source_sha256={mode: digest(REPO / 'opencood' /
                    'data_utils' / 'augmentor' / ('physics_' + mode + '.py'))
                    for mode in MODES},
                generator_source_sha256=digest(__file__),
                fog_lookup_sha256=fog_tables,
                weather_config=config,
                lidar_range=list(frontend['preprocess']['cav_lidar_range']),
                metadata='symlinks to official clean OPV2V',
                pcd_encoding='Open3D binary PCD, RGB intensity quantized to 8 bit',
                conditions={})


def validate_output_location(output, sources):
    output = output.resolve()
    for source in sources.values():
        source = Path(source).resolve()
        if output == source or output in source.parents or source in output.parents:
            raise ValueError('Output must be outside official source roots')
    data_root = Path('/data/cjm/datasets')
    if data_root not in output.parents:
        raise ValueError('Output must be under /data/cjm/datasets/')


def run(args):
    experiment = yaml.safe_load(Path(args.experiment_config).read_text('utf-8'))
    frontend = yaml.safe_load(Path(args.frontend_config).read_text('utf-8'))
    weather_cfg = resolve_weather(experiment['weather_augmentation'], REPO)
    if weather_cfg['mode'] != 'mixed_physics_weather':
        raise ValueError('Expected the existing mixed_physics_weather config')
    experiment['weather_augmentation'] = weather_cfg
    augmentors = make_augmentors(
        weather_cfg, frontend['preprocess']['cav_lidar_range'])
    expected = config_manifest(args, experiment, frontend, augmentors)
    output = Path(args.output).resolve()
    validate_output_location(output, expected['source_roots'])
    for source in expected['source_roots'].values():
        if not Path(source).is_dir():
            raise FileNotFoundError(source)
    manifest_path = output / 'manifest.json'
    if manifest_path.exists():
        actual = json.loads(manifest_path.read_text('utf-8'))
        migrated = False
        for key in expected:
            if (key == 'generator_source_sha256' and args.resume
                    and actual.get('status') == 'in_progress'
                    and actual.get(key) == LEGACY_SAFE_YAML_GENERATOR_SHA256):
                migrated = True
                continue
            if key not in ('status', 'conditions') and actual.get(key) != expected[key]:
                raise ValueError(f'Existing dataset has different {key}')
        if actual['status'] == 'complete':
            print(f'Already complete: {output}', flush=True)
            return
        if not args.resume:
            raise RuntimeError('Incomplete dataset: rerun with --resume')
        expected['conditions'] = actual.get('conditions', {})
        expected['generator_migrations'] = actual.get('generator_migrations', [])
        if migrated:
            expected['generator_migrations'].append(dict(
                from_sha256=LEGACY_SAFE_YAML_GENERATOR_SHA256,
                to_sha256=expected['generator_source_sha256'],
                reason='Use OpenCOOD loader for NumPy-tagged frame poses; '
                       'reuse atomically written PCDs'))
            atomic_json(manifest_path, expected)
            print('Resuming prior generator after frame-YAML loader fix',
                  flush=True)
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f'Nonempty unrecognized output: {output}')
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(manifest_path, expected)

    probabilities = {name: float(weather_cfg.get(
        'mixed_physics_weather', {}).get(name + '_probability', 1 / 3))
        for name in ('physics_fog', 'physics_rain', 'physics_snow')}
    for split in ('train', 'validate'):
        source = Path(expected['source_roots'][split])
        counts = {mode: 0 for mode in MODES}
        mixed_counts = {mode: 0 for mode in MODES}
        frame_count = 0
        previous_sample = -1
        prepared = set()
        ego_pose = None
        for sample, agent, scene, cav, stamp, source_pcd in frames(source):
            if sample != previous_sample:
                frame_count += 1
                previous_sample = sample
                ego_pose = load_pose(source_pcd.with_suffix('.yaml'))
            selected = select_mixed_physics_weather(
                probabilities, weather_cfg['seed'], 0, sample)
            selected = selected.removeprefix('physics_')
            targets = {}
            for mode in MODES:
                target_cav = output / mode / split / scene / cav
                key = (mode, scene, cav)
                if key not in prepared:
                    link_metadata(source_pcd.parent, target_cav)
                    prepared.add(key)
                targets[mode] = target_cav / source_pcd.name
            raw = (pcd_to_np(str(source_pcd)) if any(
                not target.is_file() or target.stat().st_size == 0
                for target in targets.values())
                   else None)
            if raw is not None:
                cav_pose = load_pose(source_pcd.with_suffix('.yaml'))
                transform = x1_to_x2(cav_pose, ego_pose)
            for mode in MODES:
                if raw is not None:
                    make_weather_file(raw, targets[mode], augmentors[mode],
                                      mode, sample, agent, transform,
                                      expected['lidar_range'],
                                      verify=(counts[mode] % args.verify_every == 0))
                counts[mode] += 1
            mixed_cav = output / 'mixed' / split / scene / cav
            key = ('mixed', scene, cav)
            if key not in prepared:
                link_metadata(source_pcd.parent, mixed_cav)
                prepared.add(key)
            ensure_link(output / selected / split / scene / cav / source_pcd.name,
                        mixed_cav / source_pcd.name)
            mixed_counts[selected] += 1
            if args.progress_every and frame_count % args.progress_every == 0 and agent == 0:
                print(f'{split}: {frame_count} frames', flush=True)
        if not frame_count:
            raise ValueError(f'No frames in {source}')
        expected['conditions'][split] = dict(frames=frame_count,
                                             pcd_files=counts,
                                             mixed_pcd_files=mixed_counts)
        atomic_json(manifest_path, expected)
        print(f'{split}: {frame_count} frames, {counts}', flush=True)
    expected['status'] = 'complete'
    atomic_json(manifest_path, expected)
    print(f'Complete: {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-config', default=str(
        REPO / 'local_fusion_v3' / 'experiment.yaml'))
    parser.add_argument('--frontend-config', required=True)
    parser.add_argument('--output', default=str(DEFAULT_OUTPUT))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--progress-every', type=int, default=100)
    parser.add_argument('--verify-every', type=int, default=100,
                        help='Read back every Nth generated PCD per weather')
    args = parser.parse_args()
    if args.progress_every < 0 or args.verify_every < 1:
        parser.error('progress-every must be >=0; verify-every must be >=1')
    try:
        run(args)
    except Exception as error:
        print(f'Weather generation stopped: {error}', file=sys.stderr)
        raise


if __name__ == '__main__':
    main()
