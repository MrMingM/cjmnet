"""Bind D-1C to the supplied, successful D-1A/D-1B evidence and real inputs."""
from pathlib import Path

from support import (CONDITIONS, HERE, REPRO, check_identity, compare_gt, identity,
                     jsonl, no_test, read_json, sha256)


def verify_inputs(args):
    pins = read_json(HERE / 'EVIDENCE.json')
    b_root = Path(args.gt_audit_root)
    for name, digest in pins['d1b_artifacts'].items():
        if sha256(no_test(b_root / name)) != digest:
            raise ValueError(f'D-1B evidence differs from logs/D/D1: {name}')
    b, bc, inputs = (read_json(b_root / n) for n in ('protocol.json', 'contract.json', 'input_manifest.json'))
    check_identity(bc)
    check_identity(inputs)
    if (bc['identity'] != pins['d1b_contract_identity'] or b['identity'] != bc['identity']
            or b['input_identity'] != inputs['identity']
            or b['input_manifest_sha256'] != sha256(b_root / 'input_manifest.json')):
        raise ValueError('D-1B protocol/contract/input chain differs')
    if (sha256(REPRO / 'source_snapshot.json') != pins['historical_snapshot_sha256']
            or read_json(REPRO / 'source_snapshot.json')['files'] != bc['source_hashes']):
        raise ValueError('Historical D-1A/B source snapshot differs; do not rewrite it for D-1C')
    for name, digest in bc['source_hashes'].items():
        if sha256(REPRO / name) != digest:
            raise ValueError(f'Historical reproduction source changed: {name}')
    a_root = Path(args.d1a_root)
    for name, digest in b['d1a_evidence'].items():
        if sha256(no_test(a_root / name)) != digest:
            raise ValueError(f'D-1A evidence changed: {name}')
    a = read_json(a_root / 'protocol.json')
    if a['model'] != b['d1a_model_provenance'] or a['upstream_commit'] != pins['upstream_commit']:
        raise ValueError('D-1A model/source provenance changed')
    model = Path(args.model_dir)
    if (read_json(model / 'provenance.json') != a['model']
            or sha256(model / 'config.yaml') != a['model']['runtime_yaml_sha256']):
        raise ValueError('D-1A model/config changed')
    for name in ('0.pth', 'net_epoch0.pth'):
        if sha256(no_test(model / name)) != a['model']['checkpoint_sha256']:
            raise ValueError('Official checkpoint alias changed')
    runtime = Path(args.runtime_root)
    for name, digest in a['upstream_source_hashes'].items():
        if sha256(runtime / name) != digest:
            raise ValueError(f'Official runtime source changed: {name}')
    for name, row in a['imported_modules'].items():
        path = no_test(row['path']).resolve(strict=True)
        if runtime not in path.parents or sha256(path) != row['sha256']:
            raise ValueError(f'Successful D-1A imported artifact changed: {name}')
    if read_json(runtime / 'native_build_selection.json') != a['native_build_selection']:
        raise ValueError('Native build selection changed')
    view_manifest = Path(args.view_root) / 'manifest.json'
    if sha256(view_manifest) != inputs['d1a_view_manifest_sha256']:
        raise ValueError('Original D-1A view manifest changed')
    view = read_json(view_manifest)
    check_identity(view)
    if view['identity'] != inputs['d1a_data_identity'] or view['frame_keys'] != bc['selected_frame_keys']:
        raise ValueError('D-1A data/frame identity changed')
    for path, digest in inputs['paired_extraction_record_sha256'].items():
        if sha256(no_test(path)) != digest:
            raise ValueError(f'Paired extraction record changed: {path}')
    overlay = no_test(inputs['overlay'])
    expected_files = set()
    for i, row in enumerate(inputs['files']):
        source, target = no_test(row['source']), overlay / row['relative']
        if (target.resolve(strict=True) != source.resolve(strict=True)
                or source.stat().st_size != row['size'] or sha256(source) != row['sha256']):
            raise ValueError(f'D-1B/D-1A input bytes or view link changed: {target}')
        expected_files.add(row['relative'])
        if (i + 1) % 3000 == 0:
            print(f'Input SHA256 {i+1}/{len(inputs["files"])}', flush=True)
    if {p.relative_to(overlay).as_posix() for p in overlay.rglob('*') if p.is_file()} != expected_files:
        raise ValueError('Unexpected/missing files in official weather view')
    summary = read_json(b_root / 'summary.json')
    if (summary['status'] != 'complete' or not summary['full_validation']
            or summary['validation_frame_count'] != 829 or summary['contract_identity'] != bc['identity']
            or summary['skipped_frames'] != 0 or not summary['clean_per_frame_counts_match_d1a']):
        raise ValueError('D-1B GT audit is incomplete')
    for name, digest in summary['output_hashes'].items():
        if sha256(b_root / name) != digest:
            raise ValueError(f'D-1B completed output hash mismatch: {name}')
    frames = jsonl(b_root / 'per_frame.jsonl')
    if len(frames) != 829 or [r['frame_key'] for r in frames] != bc['selected_frame_keys']:
        raise ValueError('D-1B frame keys/count differ')
    totals = {c: 0 for c in CONDITIONS}
    changed = dict(fog=0, snow=0)
    losses = dict(fog=0, snow=0)
    for index, frame in enumerate(frames):
        if frame['index'] != index:
            raise ValueError('D-1B frame index/order differs')
        clean = frame['conditions']['clean']
        for condition in CONDITIONS:
            row = frame['conditions'][condition]
            compare_gt(row['final_gt'], row['final_gt'], f'D-1B {index}/{condition}')
            if row['final_gt_count'] != len(row['final_gt']):
                raise ValueError('D-1B final GT count/ID mismatch')
            totals[condition] += len(row['final_gt'])
            if condition == 'clean':
                continue
            selected = {r['object_id'] for r in row['final_gt']}
            fixed = {r['object_id'] for r in clean['final_gt']}
            if not selected <= fixed:
                raise ValueError('D-1B weather GT is not a subset of official Clean')
            compare_gt(row['final_gt'], [r for r in clean['final_gt'] if r['object_id'] in selected], f'D-1B paired {index}/{condition}')
            removed = fixed - selected
            losses[condition] += len(removed)
            changed[condition] += bool(removed)
            for object_id in removed:
                evidence = []
                for cav in row['cavs']:
                    if object_id in cav['before_point_filter_ids']:
                        evidence.append(cav['official_points_in_box'][cav['before_point_filter_ids'].index(object_id)])
                if not evidence or any(evidence):
                    raise ValueError('D-1B removed object lacks all-CAV zero-point evidence')
    if totals != dict(clean=6224, fog=3673, snow=5854) or losses != dict(fog=2551, snow=370) or changed != dict(fog=637, snow=179):
        raise ValueError('D-1B statistics differ from accepted full-validation facts')
    artifacts = {n: sha256(b_root / n) for n in ('protocol.json', 'contract.json', 'input_manifest.json', 'summary.json', 'per_frame.jsonl', 'removed_gt.jsonl')}
    return dict(a=a, b=b, input_manifest=inputs, frames=frames, totals=totals,
        d1b_artifact_sha256=artifacts, d1a_artifact_sha256=b['d1a_evidence'],
        d1a_view_manifest_sha256=sha256(view_manifest), d1a_model=a['model'])
