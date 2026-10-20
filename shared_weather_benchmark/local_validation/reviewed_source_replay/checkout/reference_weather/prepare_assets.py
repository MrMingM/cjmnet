"""Download pinned tables and original snow particles on the server only."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import urllib.request
import zipfile

from reference_weather.common import PACKAGE, atomic_json, digest, read_json, server_only


def download(url, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + '.partial')
    print(f'Download {url} -> {target}', flush=True)
    # Retry reruns reuse a finished file. A partial transfer restarts cleanly.
    with urllib.request.urlopen(url, timeout=120) as response, partial.open('wb') as stream:
        size = 0
        next_log = 256 * 1024 * 1024
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            stream.write(block)
            size += len(block)
            if size >= next_log:
                print(f'  downloaded {size / 1024**3:.2f} GiB', flush=True)
                next_log += 256 * 1024 * 1024
    os.replace(partial, target)


def git_blob_hash(path):
    path = Path(path)
    h = hashlib.sha1(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def particle_parameters(config):
    s = config['snow']
    rate = (s['snowfall_rate_mm_h'] / (487 * s['snow_density_g_cm3']
             * s['snowflake_diameter_m'] * s['terminal_velocity_m_s'])) ** 1.5
    occupancy = s['snowfall_rate_mm_h'] / (
        3.6e6 * s['snow_density_g_cm3'] * s['terminal_velocity_m_s'])
    return float(rate), float(occupancy)


def select_particle_members(names, config):
    rate, occupancy = particle_parameters(config)
    dist = config['snow']['distribution']
    pattern = re.compile(r'^(gunn|sekhon)_([^_]+)_([^_]+)_(\d+)\.npy$')
    found = {}
    for member in names:
        match = pattern.match(Path(member).name)
        if not match or match[1] != dist:
            continue
        r, o, channel = float(match[2]), float(match[3]), int(match[4])
        if abs(r - rate) <= max(1e-10, rate * 1e-10) and abs(o - occupancy) < 1e-14:
            if channel in found:
                raise ValueError('Duplicate particle member in archive')
            found[channel] = member
    if set(found) != set(range(1, 65)):
        raise ValueError(f'Archive lacks the requested 64 patterns: rate={rate}, occupancy={occupancy}. '
                         'Use a rate/velocity pair from upstream sampling.py grid.')
    return found


def verify_assets(config):
    root = Path(config['assets_root'])
    manifest = read_json(root / 'manifest.json')
    requested = {k: config[k] for k in ('conditions', 'snow')}
    if manifest.get('request') != requested or manifest.get('status') != 'complete':
        raise ValueError('Assets do not match this protocol. Run the assets stage.')
    for name, sha in manifest['files'].items():
        if digest(root / name) != sha:
            raise ValueError(f'Asset hash mismatch: {root / name}')
    return manifest


def prepare(config):
    import fcntl
    server_only()
    root = Path(config['assets_root'])
    lock_path = root.with_name(root.name + '.lock')
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another run is preparing the same asset directory') from error
        return _prepare(config)


def _prepare(config):
    server_only()
    root = Path(config['assets_root'])
    root.mkdir(parents=True, exist_ok=True)
    lock = read_json(PACKAGE / 'assets_lock.json')
    requested = {k: config[k] for k in ('conditions', 'snow')}
    manifest_path = root / 'manifest.json'
    if manifest_path.exists():
        old = read_json(manifest_path)
        if old.get('request') == requested and old.get('status') == 'complete':
            return verify_assets(config)
        if old.get('request') != requested:
            raise ValueError('Use a new assets_root for a different particle/configuration request.')
    record = dict(status='in_progress', request=requested, files={},
                  table_origin='pinned upstream Git blob hashes; no Git executable required')
    atomic_json(manifest_path, record)
    for item in lock['assets']:
        asset_family = item['relative_path'].split('/')[0]
        condition = 'rain' if asset_family == 'lisa' else asset_family
        if condition not in config['conditions']:
            continue
        target = root / item['relative_path']
        url = f"https://raw.githubusercontent.com/{item['repo']}/{item['ref']}/{item['upstream_path']}"
        if not target.exists():
            download(url, target)
        if target.stat().st_size != item['size'] or git_blob_hash(target) != item['git_blob_sha1']:
            raise ValueError(f'Pinned binary asset does not match upstream: {target}')
        record['files'][target.relative_to(root).as_posix()] = digest(target)
    if 'snow' in config['conditions']:
        import numpy as np
        import yaml
        s = config['snow']
        if s['particle_source'] != 'upstream_archive' or s['particle_radius_m'] != 80:
            raise ValueError('Published archive contains radius-80m particles only.')
        archive = root / 'snowflakes.zip'
        old_archive_record = root / 'snow_archive.json'
        if not archive.exists():
            download(lock['snow_archive_url'], archive)
        archive_hash = digest(archive)
        if old_archive_record.exists() and read_json(old_archive_record)['sha256'] != archive_hash:
            raise ValueError('Snow archive changed since the previous preparation')
        archive_info = dict(url=lock['snow_archive_url'], sha256=archive_hash,
                            published_expected_sha256=lock['snow_archive_expected_sha256'],
                            radius_m=80, note=lock['snow_archive_note'])
        atomic_json(old_archive_record, archive_info)
        with zipfile.ZipFile(archive) as z:
            selected = select_particle_members(z.namelist(), config)
            particle_root = root / 'snow' / 'training' / 'snowflakes' / 'npy'
            particle_root.mkdir(parents=True, exist_ok=True)
            for channel, member in selected.items():
                target = particle_root / f'particles_{channel}.npy'
                # Fixed destination names prevent zip path traversal.
                h = hashlib.sha256()
                temporary = target.with_name(target.name + '.partial')
                with z.open(member) as source, temporary.open('wb') as destination:
                    for block in iter(lambda: source.read(1024 * 1024), b''):
                        h.update(block)
                        destination.write(block)
                if target.exists():
                    if digest(target) != h.hexdigest():
                        raise ValueError(f'Existing particle file differs: {target}')
                    temporary.unlink()
                else:
                    os.replace(temporary, target)
                array = np.load(target, mmap_mode='r', allow_pickle=False)
                if array.ndim != 2 or array.shape[1] != 3 or not np.isfinite(array).all() or (array[:, 2] <= 0).any():
                    raise ValueError(f'Invalid particle array: {target}')
                record['files'][target.relative_to(root).as_posix()] = h.hexdigest()
        record['snow_archive'] = archive_info
        record['snow_equivalent_rain_rate'], record['snow_occupancy'] = particle_parameters(config)
        sensor = dict(num_lasers=64, lasers=[dict(laser_id=i, focal_distance=0.0,
                      focal_slope=0.0, min_intensity=0.0) for i in range(64)])
        sensor_path = root / 'snow' / 'opv2v_virtual_sensor.yaml'
        sensor_path.write_text(yaml.safe_dump(sensor, sort_keys=True), encoding='utf-8')
        record['files'][sensor_path.relative_to(root).as_posix()] = digest(sensor_path)
    record['status'] = 'complete'
    atomic_json(manifest_path, record)
    return record
