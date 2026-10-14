"""Persist reference weather in each transmitting CAV's local coordinates."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path

import numpy as np

from reference_weather.common import (atomic_json, digest, iter_frames, object_hash,
    project_hashes, read_json, read_points, stable_seed, verify_sources, write_pcd)
from reference_weather.prepare_assets import verify_assets

_SIMULATORS = {}
_CONFIG = None


def worker_init(config):
    global _CONFIG, _SIMULATORS
    _CONFIG, _SIMULATORS = config, {}


def distance_stats(points):
    edges = [0, 20, 40, 60, 80, 120, 160, np.inf]
    return dict(count=int(len(points)), mean_intensity=float(points[:, 3].mean()) if len(points) else 0.0,
                range_counts=np.histogram(np.linalg.norm(points[:, :3], axis=1), bins=edges)[0].tolist())


def metadata(source, target, timestamp):
    target.mkdir(parents=True, exist_ok=True)
    for entry in source.iterdir():
        # Keep smoke frame YAML selection exact; do not link all frame YAMLs.
        if entry.suffix.lower() == '.pcd':
            continue
        if entry.is_file() and not (entry.name.startswith(timestamp + '.')
                                    or entry.name.startswith(timestamp + '_')
                                    or entry.name == 'additional.yaml'):
            continue
        link(entry, target / entry.name)


def link(source, target):
    source = Path(source).resolve(strict=True)
    if target.is_symlink() and target.resolve(strict=True) == source:
        return
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.symlink_to(source, target_is_directory=source.is_dir())


def generate_one(task):
    mode, split, frame, cav, source_path, identity = task
    config = _CONFIG
    root = Path(config['output_root'])
    relative = Path(frame['scene']) / cav / (frame['timestamp'] + '.pcd')
    target = root / mode / split / relative
    record_path = root / 'records' / mode / split / relative.with_suffix('.json')
    source = Path(source_path)
    source_hash, yaml_hash = digest(source), digest(source.with_suffix('.yaml'))
    if record_path.exists():
        row = read_json(record_path)
        if (row['identity'] != identity or row['source_sha256'] != source_hash
                or row['yaml_sha256'] != yaml_hash or not target.is_file()
                or row['output_sha256'] != digest(target)):
            raise ValueError(f'Resume identity/data mismatch: {record_path}')
        return row
    if mode not in _SIMULATORS:
        from reference_weather.adapters import ReferenceSimulator
        _SIMULATORS[mode] = ReferenceSimulator(mode, config, config['assets_root'])
    raw = read_points(source)
    seed = stable_seed(config['seed'], mode, split, frame['scene'], cav, frame['timestamp'])
    try:
        output, statistics = _SIMULATORS[mode].augment(raw, seed)
        write_pcd(target, output)
    except Exception as error:
        raise RuntimeError(f'{mode} failed at {source}. No clean fallback was inserted. {error}') from error
    row = dict(identity=identity, frame_index=frame['index'], scene=frame['scene'], cav=cav,
               timestamp=frame['timestamp'], seed=seed, source=str(source),
               source_sha256=source_hash, yaml_sha256=yaml_hash, output_sha256=digest(target),
               input=distance_stats(raw), output=distance_stats(read_points(target)), simulation=statistics)
    atomic_json(record_path, row)
    return row


def protocol(config, limit_frames):
    assets = verify_assets(config)
    return dict(schema=1, config=config, limit_frames=int(limit_frames),
                sources=verify_sources(), project_sources=project_hashes(),
                assets=assets, coordinate_frame='each transmitting CAV local sensor; before ego projection',
                intensity='Fog/Snow: normalized -> x255 -> reference -> /255; Rain: normalized',
                pcd_encoding='Open3D binary RGB; intensity quantized to 8 bit; read back every output',
                fallback='none: failure stops generation; empty clouds stay empty',
                snow_adaptation='reconstructed elevation channels; no real HDL focal/channel corrections; '
                    'OPV2V lower-envelope road fit; 80m published particle field; waveform extended beyond 120m',
                official_weather_recipe_known=False)


def generate(config, limit_frames=0, resume=False):
    root = Path(config['output_root'])
    contract = protocol(config, limit_frames)
    identity = object_hash(contract)
    manifest_path = root / 'manifest.json'
    if manifest_path.exists():
        previous = read_json(manifest_path)
        if previous['identity'] != identity:
            raise ValueError('Dataset protocol changed; use a new output_root.')
        if not resume:
            raise FileExistsError('Dataset exists. Set RESUME=1 to verify/reuse it.')
    elif root.exists() and any(root.iterdir()):
        # Launcher logs live beside the dataset, never inside a new dataset root.
        raise FileExistsError(f'Refusing a nonempty unrelated output directory: {root}')
    root.mkdir(parents=True, exist_ok=True)
    manifest = dict(identity=identity, protocol=contract, status='in_progress', splits={})
    atomic_json(manifest_path, manifest)
    for split in config['splits']:
        source = Path(config['clean_root']) / split
        selected = list(iter_frames(source))
        total_frames = len(selected)
        expected = config.get('expected_frames', {}).get(split)
        if expected is not None and total_frames != int(expected):
            raise ValueError(f'{split} contains {total_frames} frames, expected {expected}')
        if limit_frames:
            selected = selected[:limit_frames]
        if not selected:
            raise ValueError(f'No selected source frames in {source}')
        split_info = dict(source=str(source), full_source_frames=total_frames,
                          selected_frames=len(selected), complete_split=len(selected) == total_frames,
                          frame_keys=[[f['scene'], f['timestamp']] for f in selected], conditions={})
        manifest['splits'][split] = split_info
        for frame in selected:
            for cav, pcd in frame['entries']:
                src_cav = Path(pcd).parent
                for mode in ['clean', *config['conditions']]:
                    dst_cav = root / mode / split / frame['scene'] / cav
                    metadata(src_cav, dst_cav, frame['timestamp'])
                    if mode == 'clean':
                        link(pcd, dst_cav / Path(pcd).name)
        # One condition at a time keeps memory bounded and gives completed-stage progress.
        for mode in config['conditions']:
            tasks = [(mode, split, frame, cav, pcd, identity)
                     for frame in selected for cav, pcd in frame['entries']]
            rows = []
            with ProcessPoolExecutor(max_workers=int(config['generation_workers']),
                    mp_context=mp.get_context('spawn'), initializer=worker_init,
                    initargs=(config,)) as pool:
                for row in pool.map(generate_one, tasks, chunksize=1):
                    rows.append(row)
                    if len(rows) % 25 == 0 or len(rows) == len(tasks):
                        print(f'{split}/{mode}: {len(rows)}/{len(tasks)} CAV clouds', flush=True)
            source_index = object_hash([[r['scene'], r['cav'], r['timestamp'], r['source_sha256'],
                                       r['yaml_sha256']] for r in rows])
            previous_indices = [v['source_index'] for v in split_info['conditions'].values()]
            if previous_indices and any(v != source_index for v in previous_indices):
                raise ValueError('Clean source changed between weather generation stages')
            aggregate = dict(pcd_count=len(rows), source_index=source_index,
                input_points=sum(r['input']['count'] for r in rows),
                output_points=sum(r['output']['count'] for r in rows),
                empty_clouds=sum(r['output']['count'] == 0 for r in rows),
                input_range_counts=np.sum([r['input']['range_counts'] for r in rows], axis=0).tolist(),
                output_range_counts=np.sum([r['output']['range_counts'] for r in rows], axis=0).tolist(),
                replaced_points=sum(r['simulation'].get(mode + '_replaced', 0) for r in rows),
                lost_points=sum(r['simulation'].get('lost', 0) for r in rows))
            split_info['conditions'][mode] = aggregate
            atomic_json(manifest_path, manifest)
        atomic_json(root / f'{split}_frames.json', selected)
    manifest['status'] = 'complete'
    atomic_json(manifest_path, manifest)
    return manifest


def verify_dataset(config, limit_frames):
    root = Path(config['output_root'])
    manifest = read_json(root / 'manifest.json')
    if manifest['status'] != 'complete' or manifest['identity'] != object_hash(protocol(config, limit_frames)):
        raise ValueError('Dataset incomplete or source/config/assets changed')
    for split, info in manifest['splits'].items():
        frames = read_json(root / f'{split}_frames.json')
        if [[f['scene'], f['timestamp']] for f in frames] != info['frame_keys']:
            raise ValueError('Frame selection changed')
        for frame in frames:
            for cav, source in frame['entries']:
                relative = Path(frame['scene']) / cav / (frame['timestamp'] + '.pcd')
                source_hash, yaml_hash = digest(source), digest(Path(source).with_suffix('.yaml'))
                clean_link = root / 'clean' / split / relative
                if not clean_link.is_symlink() or clean_link.resolve(strict=True) != Path(source).resolve():
                    raise ValueError(f'Clean link mismatch: {clean_link}')
                clean_annotation = root / 'clean' / split / relative.with_suffix('.yaml')
                if (not clean_annotation.is_symlink() or clean_annotation.resolve(strict=True)
                        != Path(source).with_suffix('.yaml').resolve(strict=True)):
                    raise ValueError(f'Clean annotation link mismatch: {clean_annotation}')
                for mode in config['conditions']:
                    row = read_json(root / 'records' / mode / split / relative.with_suffix('.json'))
                    annotation = root / mode / split / relative.with_suffix('.yaml')
                    if (not annotation.is_symlink() or annotation.resolve(strict=True)
                            != Path(source).with_suffix('.yaml').resolve(strict=True)):
                        raise ValueError(f'Annotation link mismatch: {annotation}')
                    if (row['identity'] != manifest['identity'] or row['source_sha256'] != source_hash
                            or row['yaml_sha256'] != yaml_hash or row['output_sha256'] != digest(root / mode / split / relative)):
                        raise ValueError(f'Dataset integrity mismatch: {mode}/{relative}')
    return manifest
