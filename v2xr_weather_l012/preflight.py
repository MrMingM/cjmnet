"""Validate accepted data and all frozen dependencies, never extract or visit test."""
import os
from pathlib import Path
import sys
from auditlib import (CONFIG, ROOT, atomic_json, baseline, no_test, read_json, sha256, sources)


def run(args):
    from data import paired_inventory
    manifest, evidence = args.prepared
    original = baseline(args)
    paired = paired_inventory(manifest, args.train_root)
    if len(paired['frames']) != 8083 or len(paired['pairs']) != 24088:
        raise ValueError('Expected 8083 cooperative frames / 24088 single-CAV pairs')
    if manifest['counts']['cooperative_frames'] != 8083:
        raise ValueError('Accepted train count differs')
    refs = {}
    def add(path, expected=None):
        path = no_test(path)
        value = sha256(path)
        if expected is not None and value != expected:
            raise ValueError('Preflight byte identity mismatch: ' + str(path))
        refs[str(path)] = value
    train = Path(args.train_root)
    for row in manifest['files']:
        add(train / 'train' / row['relative'], row['sha256'])
    for row in manifest['archives']:
        add(row['path'], row['sha256'])
        add(train / 'records' / row['scene'] / (row['cav'] + '.json'))
    for row in read_json(train / 'plan.json').get('scene_metadata', []):
        add(row['path'], row['sha256'])
    for name in ('manifest.json', 'plan.json', 'data_acceptance.json'):
        add(train / name)
    for row in evidence['input_manifest']['files']:
        add(row['source'], row['sha256'])
        add(Path(evidence['input_manifest']['overlay']) / row['relative'], row['sha256'])
    for name, digest in evidence['a']['upstream_source_hashes'].items():
        add(Path(args.runtime_root) / name, digest)
    for row in evidence['a']['imported_modules'].values():
        add(row['path'], row['sha256'])
    add(Path(args.runtime_root) / 'native_build_selection.json')
    for key in ('d1a_root', 'gt_audit_root', 'd1c_root', 'model_dir', 'view_root'):
        directory = Path(getattr(args, key))
        for name in ('summary.json', 'protocol.json', 'manifest.json', 'input_manifest.json', 'contract.json', 'provenance.json', 'config.yaml'):
            if (directory / name).is_file():
                add(directory / name)
    pins = read_json(ROOT / 'v2xr_gspr_clean/PINS.json')['baseline']
    for name, digest in pins['files'].items():
        add(Path(args.baseline_root) / name, digest)
    add(read_json(Path(args.baseline_root) / 'selected.json')['checkpoint'], pins['checkpoint_sha256'])
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / 'external_inputs.json', refs)
    trees = {}
    for directory in (train / 'train', Path(evidence['input_manifest']['overlay']), Path(args.view_root) / 'validate'):
        trees[str(directory)] = dict(files=sorted(p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()),
            directories=sorted(p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_dir()))
    atomic_json(root / 'input_trees.json', trees)
    atomic_json(root / 'paired_inventory.json', paired)
    atomic_json(root / 'summary.json', dict(status='complete', sources=sources(), settings=CONFIG,
        cooperative_frames=8083, single_cav_pairs=24088, validation_frames=829,
        data_identity=manifest['identity'], pair_identity=paired['identity'],
        baseline_checkpoint_sha256=original['checkpoint_sha256'],
        upstream_commit=evidence['a']['upstream_commit'],
        device='cuda:0', physical_gpu=os.environ['ROCR_VISIBLE_DEVICES'], python=sys.version,
        sampling='per-CAV official randint(0,6)<=2, P(Fog)=3/7; validation train_sim inactive',
        train_GT='native point/FOV filter follows actually selected points; empty-GT resampling preserved',
        augmentation='official IntermediateFusionDataset never invokes configured world flip/rotation/scaling; original path retained; actual calls counted',
        no_data_extraction=True, no_test=True))
