"""Reuse accepted extraction and frozen evaluation evidence; one shared decode verification."""
import importlib.metadata
import os
from pathlib import Path
import platform
import shutil
import sys
import time

from .common import HERE, ROOT, bound, identity, progress, read, safe, sha, stamp, verify_sources, write
from .runtime import arguments, legacy_paths


def run(output):
    root = Path(output)
    legacy_paths()
    args = arguments(root)
    from auditlib import check_scene_partition, frozen_c
    from train_acceptance import verify_manifest_acceptance
    from provenance import verify_inputs
    from support import verify_sources as legacy_verify
    from common import verify_package
    own = verify_sources()
    verify_package(); legacy_verify()
    train = read(Path(args.train_root) / 'manifest.json')
    if (train['status'] != 'complete' or not train['full_train'] or not train['integrity_complete']
            or train['counts']['cooperative_frames'] != 8083
            or train['counts']['scene_count'] != 32 or train['counts']['scene_agent_count'] != 92
            or any(train['counts'][k] != 24088 for k in ('clean_pcd', 'fog', 'snow'))
            or identity({k: v for k, v in train.items() if k != 'identity'}) != train['identity']):
        raise ValueError('Train extraction does not match the accepted complete 8083-frame identity')
    check_scene_partition(train)
    verify_manifest_acceptance(train, Path(args.train_root))
    for name, digest in train['output_hashes'].items():
        if sha(safe(Path(args.train_root) / name)) != digest:
            raise ValueError('Accepted extraction artifact changed: ' + name)
    records = {r['relative']: r for r in train['files']}
    inputs, counts = [], {w: 0 for w in ('clean', 'fog', 'snow')}
    c = read(root / 'config.json')
    for s in train['scenes']:
        for t in s['timestamps']:
            for cav in s['agents']:
                prefix = s['scene'] + '/' + cav + '/' + t
                for weather, suffix in c['weather'].items():
                    rel = prefix + suffix + '.pcd'
                    if rel not in records:
                        raise ValueError('Missing same-frame three-weather pair: ' + rel)
                    counts[weather] += 1
    for i, r in enumerate(train['files']):
        p = safe(Path(args.train_root) / 'train' / r['relative'])
        if p.stat().st_size != r['size']:
            raise ValueError('Accepted extracted file missing/size changed: ' + str(p))
        inputs.append(stamp(p))
        if (i + 1) % 10000 == 0:
            print(f'Accepted train metadata {i+1}/{len(train["files"])}', flush=True)
    # This frozen verifier hashes the validation view ONCE, never once per epoch/group.
    ev = verify_inputs(args)
    frozen = frozen_c(args)
    if ev['b']['selected_frame_keys'] != [r['frame_key'] for r in
            frozen['gt']['frames'] if r['condition'] == 'clean']:
        raise ValueError('Frozen D-1C validation UID order differs')
    paired = safe(os.environ.get('PAIRED_VALIDATE_ROOT', '/data/cjm/datasets/v2xr-lidar-weather-validate-v1')).resolve()
    if not (paired / 'manifest.json').is_file():
        raise ValueError('PAIRED_VALIDATE_ROOT has no accepted manifest')
    if (sha(paired / 'manifest.json') != ev['input_manifest']['paired_manifest_sha256']
            or sha(paired / 'plan.json') != ev['input_manifest']['paired_plan_sha256']):
        raise ValueError('Requested paired validation is not the frozen evaluated extraction')
    for r in ev['input_manifest']['files']:
        inputs.append(stamp(r['source']))
        inputs.append(stamp(Path(ev['input_manifest']['overlay']) / r['relative']))
    for p in (Path(args.train_root) / 'manifest.json', Path(args.train_root) / 'data_acceptance.json',
              Path(args.view_root) / 'manifest.json', paired / 'manifest.json',
              Path(args.d1c_root) / 'gt_manifest.json', Path(args.gt_audit_root) / 'input_manifest.json'):
        inputs.append(stamp(p))
    for name in ('config.yaml', 'provenance.json', '0.pth', 'net_epoch0.pth'):
        inputs.append(stamp(Path(args.model_dir) / name))
    for folder, names in ((Path(args.d1a_root), ev['d1a_artifact_sha256']),
                          (Path(args.gt_audit_root), ev['d1b_artifact_sha256']),
                          (Path(args.d1c_root), frozen['hashes'])):
        for name in names:
            inputs.append(stamp(folder / name))
    train_keys = {(s['scene'], t) for s in train['scenes'] for t in s['timestamps']}
    if train_keys & set(map(tuple, ev['b']['selected_frame_keys'])):
        raise ValueError('Physical train/validate frames overlap')
    versions = {k: importlib.metadata.version(k) for k in ev['a']['versions']}
    if versions != ev['a']['versions'] or sys.version_info[:2] != (3, 10):
        raise ValueError('Environment differs from accepted Python3.10 D-1A runtime')
    roots = {k: str(safe(getattr(args, k)).resolve()) for k in
             ('train_root', 'runtime_root', 'model_dir', 'd1a_root', 'gt_audit_root', 'd1c_root', 'view_root')}
    roots['paired_validate_root'] = str(paired)
    contract = dict(revision=1, config=c, sources=own, inputs=dict(train=train['identity'],
                    validation=ev['input_manifest']['identity'], frozen_F=frozen['gt']['identity'],
                    paired_manifest_sha256=sha(paired / 'manifest.json')), roots=roots,
                    environment=dict(python=sys.version, versions=versions, platform=platform.platform(),
                                     physical_gpu=os.environ.get('GPU'), pythonhashseed=os.environ.get('PYTHONHASHSEED')),
                    initialization='random official create_model; independent aux RNG', no_test=True,
                    resume_boundary='epoch training commit, then evaluation commit; partial epoch replay')
    contract['identity'] = identity(contract)
    bound(root / 'contract.json', contract)
    bound(root / 'preflight/train_manifest.json', train)
    bound(root / 'preflight/evidence.json', ev)
    write(root / 'preflight/input_stamps.json', list({r['path']: r for r in inputs}.values()))
    write(root / 'source_snapshot.json', own)
    free = shutil.disk_usage(root).free
    write(root / 'data_check.json', dict(status='complete', counts=train['counts'], weather_pairs=counts,
        validation_frames=829, frozen_F_identity=frozen['gt']['identity'],
        extraction_reused=True, repeated_archive_CRC=False, free_disk_bytes=free,
        validation_roots_verified=True))
    print('PLAN groups=E0,E1,E2,E3 epochs=' + str(c['epochs']) + ' train=8083 validate=829 per weather', flush=True)
    print('Disk estimate before measured cache: rays roughly 10-80 GiB; states/checkpoints/predictions roughly 30-100 GiB; '
          'actual sample estimates follow smoke. Free GiB=' + str(round(free / 2**30, 2)), flush=True)
    if free < 20 * 2**30:
        raise ValueError('Less than 20 GiB free before smoke; no training started')
    write(root / 'preflight/complete.json', dict(status='complete', contract_identity=contract['identity']))


def verify_training_content(output):
    """PCD weather content that extraction only header-checked is decoded once with the native reader."""
    import numpy as np
    from opencood.utils import pcd_utils
    from .runtime import arguments
    root, args = Path(output), arguments(output)
    manifest = read(root / 'preflight/train_manifest.json')
    rows = [r for r in manifest['files'] if r['relative'].endswith('.pcd') and not r['relative'].endswith('_radar.pcd')]
    start = time.monotonic()
    for i, row in enumerate(rows):
        p = safe(Path(args.train_root) / 'train' / row['relative'])
        journal = root / 'preflight/content' / (row['relative'] + '.json')
        state = stamp(p)
        if journal.exists():
            if read(journal)['source'] != state:
                raise ValueError('Previously decoded weather content changed: ' + str(p))
        else:
            if sha(p) != row['sha256']:
                raise ValueError('Extracted PCD content differs: ' + str(p))
            # Open3D native reader supports ASCII/binary and unpacks RGB red into intensity.
            points, weather = pcd_utils.pcd_to_np(str(p), get_weather=True)
            if points.ndim != 2 or points.shape[1] != 4 or not np.isfinite(points).all():
                raise ValueError('Decoded PCD is invalid: ' + str(p))
            if len(points) and (points[:, 3].min() < 0 or points[:, 3].max() > 1):
                raise ValueError('Native red-channel intensity outside [0,1]')
            write(journal, dict(status='complete', source=state, sha256=row['sha256'], points=len(points)))
        if i % 100 == 0 or i + 1 == len(rows):
            progress(root / 'preflight/content_progress.json', status='running', completed=i + 1,
                     total=len(rows), file=row['relative'])
            print(f'Shared weather decode {i+1}/{len(rows)} elapsed={time.monotonic()-start:.1f}s', flush=True)
    write(root / 'preflight/content_complete.json', dict(status='complete', files=len(rows), seconds=time.monotonic()-start))
