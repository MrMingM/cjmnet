"""Accepted8083 extraction, frozen829 evaluation, independent scene split, immutable sources."""
import importlib.metadata
import os
from pathlib import Path
import shutil
import sys

from .common import bound, identity, read, safe, sha, sources, stamp, write
from .runtime import arguments, legacy_paths
from .uncertainty_targets import split_scenes


def run(output):
    root = Path(output)
    legacy_paths()
    from auditlib import check_scene_partition, frozen_c
    from train_acceptance import verify_manifest_acceptance
    from provenance import verify_inputs
    from support import verify_sources
    from common import verify_package
    args = arguments(root)
    c = read(root / 'config.json')
    verify_package(); verify_sources()
    train = read(Path(args.train_root) / 'manifest.json')
    if (train['status'] != 'complete' or not train['integrity_complete'] or not train['full_train']
            or train['counts']['cooperative_frames'] != 8083
            or any(train['counts'][k] != 24088 for k in ('clean_pcd', 'fog', 'snow'))):
        raise ValueError('Require completed accepted8083/24088 extraction')
    if train['identity'] != identity({k: v for k, v in train.items() if k != 'identity'}):
        raise ValueError('Accepted train manifest identity differs')
    check_scene_partition(train); verify_manifest_acceptance(train, Path(args.train_root))
    if len(train['scenes']) != 32 or sum(len(s['agents']) for s in train['scenes']) != 92:
        raise ValueError('Accepted32-scene/92-CAV-combination identity differs')
    for name, digest in train['output_hashes'].items():
        if sha(safe(Path(args.train_root) / name)) != digest:
            raise ValueError('Accepted extraction record changed: ' + name)
    rows = {r['relative']: r for r in train['files']}
    for s in train['scenes']:
        for t in s['timestamps']:
            for agent in s['agents']:
                for suffix in c['weather'].values():
                    if s['scene'] + '/' + agent + '/' + t + suffix + '.pcd' not in rows:
                        raise ValueError('Missing same-frame Mix3 pair')
    inputs = []
    for i, row in enumerate(train['files']):
        p = safe(Path(args.train_root) / 'train' / row['relative'])
        if p.stat().st_size != row['size']:
            raise ValueError('Accepted extracted file size changed: ' + str(p))
        inputs.append(stamp(p))
        if (i + 1) % 10000 == 0:
            print('Accepted train metadata', i + 1, '/', len(train['files']), flush=True)
    ev = verify_inputs(args)  # Shared validation byte verification once, never per epoch.
    fixed = frozen_c(args)
    if [r['frame_key'] for r in fixed['gt']['frames'] if r['condition'] == 'clean'] != ev['b']['selected_frame_keys']:
        raise ValueError('D-1C fixed target frame identities differ')
    split = split_scenes(train, c['seed'], c['train_dev_scenes'])
    if {tuple(u[:2]) for u in split['uids']} & set(map(tuple, ev['b']['selected_frame_keys'])):
        raise ValueError('Train/official validation overlap')
    paired = safe(os.environ['PAIRED_VALIDATE_ROOT'])
    if sha(paired / 'manifest.json') != ev['input_manifest']['paired_manifest_sha256']:
        raise ValueError('Requested paired validation identity differs')
    for row in ev['input_manifest']['files']:
        inputs += [stamp(row['source']), stamp(Path(ev['input_manifest']['overlay']) / row['relative'])]
    quality = safe(c['gspr']['checkpoint'])
    if sha(quality) != c['gspr']['checkpoint_sha256']:
        raise ValueError('Missing/incompatible frozen supervised OPV2V quality checkpoint; random replacement forbidden')
    import yaml
    qconfig = quality.parent / 'config.yaml'
    qh = yaml.safe_load(qconfig.read_text())
    qa = qh['model']['args']
    if qa['lidar_range'] != c['range'] or qa['voxel_size'] != c['voxel']:
        raise ValueError('Frozen GSPR normalization/range/voxel domain mismatch')
    for key, value in c['gspr']['model'].items():
        if qa['gspr'].get(key, value) != value:
            raise ValueError('GSPR architecture mismatch: ' + key)
    inputs += [stamp(quality), stamp(qconfig), stamp(Path(args.train_root) / 'manifest.json'),
               stamp(Path(args.train_root) / 'data_acceptance.json'), stamp(paired / 'manifest.json')]
    for folder, names in ((Path(args.d1c_root), fixed['hashes']), (Path(args.gt_audit_root), ev['d1b_artifact_sha256']),
                          (Path(args.d1a_root), ev['d1a_artifact_sha256'])):
        inputs += [stamp(folder / name) for name in names]
    label_manifest = os.environ.get('QUALITY_LABEL_MANIFEST')
    quality_labels = None
    if label_manifest:
        quality_labels = read(safe(label_manifest))
        if (quality_labels.get('kind') != 'simulator_recorded_point_origin'
                or quality_labels.get('split') != 'train' or not quality_labels.get('provenance')):
            raise ValueError('Quality labels require explicit simulator point-origin provenance, train only')
        inputs.append(stamp(label_manifest))
        for rel, entry in quality_labels['files'].items():
            if rel not in rows or entry['pcd_sha256'] != rows[rel]['sha256'] or sha(safe(entry['path'])) != entry['sha256']:
                raise ValueError('Quality point-label/PCD identity differs: ' + rel)
            inputs.append(stamp(entry['path']))
        bound(root / 'preflight/quality_labels.json', quality_labels)
    versions = {k: importlib.metadata.version(k) for k in ev['a']['versions']}
    if versions != ev['a']['versions'] or sys.version_info[:2] != (3, 10):
        raise ValueError('Require accepted Python3.10/HIP package versions')
    contract = dict(family='uecp_gspr_v2xr', sources=sources(), config=c,
        inputs=dict(train=train['identity'], validation=ev['input_manifest']['identity'], frozen_F=fixed['gt']['identity'],
                    gspr_sha256=sha(quality), quality_labels=None if quality_labels is None else identity(quality_labels)),
        environment=dict(python=sys.version, versions=versions, physical_gpu=os.environ.get('GPU')),
        split_identity=identity(split), selection=c['selection'],
        validation_history='829 validation has already been repeatedly inspected; final report is not an independent blind test')
    contract['identity'] = identity(contract)
    bound(root / 'contract.json', contract)
    bound(root / 'data_manifest.json', split)
    bound(root / 'preflight/train_manifest.json', train)
    bound(root / 'preflight/evidence.json', ev)
    bound(root / 'preflight/gspr_source_config.json', qh)
    write(root / 'preflight/input_stamps.json', list({r['path']: r for r in inputs}.values()))
    write(root / 'source_snapshot.json', contract['sources'])
    write(root / 'preflight/data_check.json', dict(status='COMPLETE', counts=train['counts'],
        core_frames=len(split['train_core']), dev_frames=len(split['train_dev']),
        quality_labels='unavailable; E1 noise-identification tests INCONCLUSIVE' if quality_labels is None else 'provenance-bound',
        external_gspr_training_cost='additional historical supervised OPV2V training; not equal total computation'))
    if shutil.disk_usage(root).free < 30 * 2**30:
        raise ValueError('Less than30GiB free before smoke')
    print('Plan: 9 detector runs x', c['epochs'], 'epochs; train_core', len(split['train_core']),
          'train_dev', len(split['train_dev']), 'final829 per weather; FP32 effectivebatch8; estimate50-150GiB.', flush=True)


def content(output):
    # The shared verifier decodes native RGB red-channel intensity, ASCII/binary, and hashes once per source.
    from v2xr_raygen_e0e3.preflight import verify_training_content
    verify_training_content(output)
    from .common import commit
    root = Path(output)
    commit(root / 'preflight/training_content.json', [root / 'preflight/content_complete.json'],
           contract_identity=read(root / 'contract.json')['identity'],
           note='All accepted weather PCD decoded and sha256 checked; per-source journal retained')
