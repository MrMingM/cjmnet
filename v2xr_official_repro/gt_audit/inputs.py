"""D-1B provenance and additive view; never extract or alter historical inputs."""
from pathlib import Path

from common import COMMIT, PACKAGE, atomic_json, identity, link, no_test, read_json, sha256

WEATHER = {'clean': None, 'fog': '_fog_0.060', 'snow': '_snow_2.5_2.0'}


def verify_d1a(args):
    root = Path(args.d1a_root)
    names = ('summary.json', 'protocol.json', 'contract.json', 'frames.jsonl')
    evidence = {n: sha256(no_test(root / n)) for n in names}
    summary, protocol = read_json(root / 'summary.json'), read_json(root / 'protocol.json')
    previous_contract = read_json(root / 'contract.json')
    if (previous_contract['identity'] != identity(previous_contract['contract'])
            or previous_contract['contract']['stage'] != 'D-1A'
            or previous_contract['contract']['upstream_commit'] != COMMIT):
        raise ValueError('D-1A contract identity invalid')
    expected_ap = {'ap30': .85993605, 'ap50': .82279434, 'ap70': .66625452}
    if (summary['status'] != 'complete' or not summary['full_validation']
            or summary['validation_frame_count'] != 829 or summary['GT_total'] != 6224
            or summary['upstream_commit'] != COMMIT
            or any(abs(summary['AP'][k] - v) > 5e-8 for k, v in expected_ap.items())
            or summary['eval_utils'] != 'opencood.utils.eval_utils_opv2v'):
        raise ValueError('D-1A evidence differs from the accepted 829-frame reproduction; stop D-1B')
    upstream = read_json(PACKAGE / 'UPSTREAM.json')['files']
    if protocol['upstream_source_hashes'] != upstream:
        raise ValueError('D-1A upstream source identity differs')
    runtime = Path(args.runtime_root)
    for name, digest in upstream.items():
        if sha256(runtime / name) != digest:
            raise ValueError(f'D-1A runtime source changed: {name}')
    # Includes the actual native .so files loaded by the successful D-1A run.
    native = {}
    for name, row in protocol['imported_modules'].items():
        path = Path(row['path'])
        if runtime not in path.resolve().parents or sha256(path) != row['sha256']:
            raise ValueError(f'D-1A imported runtime artifact changed: {name}')
        if path.suffix != '.py':
            native[name] = row
    if not any('roiaware_pool3d_cuda' in n for n in native):
        raise ValueError('Missing successful D-1A native ROI evidence')
    selection = read_json(runtime / 'native_build_selection.json')
    if selection != protocol['native_build_selection']:
        raise ValueError('D-1A native build selection changed')
    model = Path(args.model_dir)
    info = read_json(model / 'provenance.json')
    if (sha256(model / 'config.yaml') != info['runtime_yaml_sha256']
            or info['runtime_yaml_sha256'] != summary['runtime_yaml_sha256']
            or info != protocol['model'] or info != previous_contract['contract']['metadata']):
        raise ValueError('D-1A model/config provenance changed')
    import json
    frames = [json.loads(line) for line in (root / 'frames.jsonl').read_text().splitlines()]
    if (len(frames) != 829 or [r['index'] for r in frames] != list(range(829))
            or sum(r['gt_count'] for r in frames) != 6224):
        raise ValueError('D-1A frame evidence incomplete')
    return dict(artifacts=evidence, summary=summary, protocol=protocol,
                native=native, model=info), frames


def prepare_view(args, output, d1a, frames):
    """Reuse each D-1A file plus exact weather members verified by extraction records."""
    view_root, paired = Path(args.view_root), Path(args.paired_root)
    manifest_path = view_root / 'manifest.json'
    manifest = read_json(manifest_path)
    unsigned = {k: v for k, v in manifest.items() if k != 'identity'}
    if (manifest['status'] != 'complete' or manifest['validation_frame_count'] != 829
            or manifest['identity'] != identity(unsigned)
            or d1a['protocol']['data_identity'] != manifest['identity']
            or manifest['frame_keys'] != [r['frame_key'] for r in frames]):
        raise ValueError('D-1A data view/order identity mismatch')
    plan_path, paired_manifest = paired / 'plan.json', paired / 'manifest.json'
    plan, completed = read_json(plan_path), read_json(paired_manifest)
    import hashlib
    import json
    plan_hash = hashlib.sha256(json.dumps(plan['protocol'], sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    p = plan['protocol']
    if (plan_hash != plan['identity'] or completed['identity'] != plan['identity']
            or completed['status'] != 'complete' or p['split'] != 'validate'
            or p['frame_keys'] != manifest['frame_keys']
            or p['selected_variants'] != {'fog': 'fog_0.060', 'snow': 'snow_2.5_2.0'}):
        raise ValueError('Paired weather plan/manifest differs from required official suffixes')
    overlay = output / 'data_view' / 'validate'
    rows = []
    extraction_record_hashes = {}
    # Rehash every actual input on every launch/resume; do not trust old size/mtime.
    for row in manifest['files']:
        original = no_test(row['view'])
        relative = original.relative_to(view_root / 'validate')
        if sha256(original) != row['sha256']:
            raise ValueError(f'D-1A data input changed: {original}')
        target = overlay / relative
        link(original, target)
        rows.append(dict(relative=relative.as_posix(), source=str(original.resolve()),
                         sha256=row['sha256'], size=original.stat().st_size))
    for scene in manifest['scenes']:
        for cav in scene['agents']:
            record_path = paired / 'records' / 'validate' / scene['scene'] / (cav + '.json')
            record = read_json(record_path)
            extraction_record_hashes[str(record_path)] = sha256(record_path)
            if record['identity'] != plan['identity']:
                raise ValueError('Paired per-CAV extraction record differs')
            archive = next(r for r in p['archives'] if r['scene'] == scene['scene'] and r['cav'] == cav)
            members = {r['name']: r for r in archive['members']}
            for stamp in scene['timestamps']:
                for condition, suffix in WEATHER.items():
                    if suffix is None:
                        continue
                    name = stamp + suffix + '.pcd'
                    member = members[name]
                    if member['timestamp'] != stamp or member['condition'] != condition:
                        raise ValueError('Weather member/frame/condition mismatch')
                    source = no_test(paired / 'payload' / 'validate' / scene['scene'] / cav / name)
                    digest = sha256(source)
                    if digest != record['files'][name] or source.stat().st_size != member['size']:
                        raise ValueError(f'Weather input hash/size changed: {source}')
                    relative = Path(scene['scene']) / cav / name
                    link(source, overlay / relative)
                    rows.append(dict(relative=relative.as_posix(), source=str(source.resolve()),
                                     sha256=digest, size=source.stat().st_size))
            print(f'Input SHA256 verified: {scene["scene"]}/{cav}', flush=True)
    if len(rows) != 3041 * 9 or len({r['relative'] for r in rows}) != len(rows):
        raise ValueError('Expected 3041 agent frames, each with 9 official input files')
    if {p.relative_to(overlay).as_posix() for p in overlay.rglob('*') if p.is_file()} != {r['relative'] for r in rows}:
        raise ValueError('Unexpected files in D-1B overlay')
    result = dict(files=rows, scenes=manifest['scenes'], frame_keys=manifest['frame_keys'],
        d1a_view_manifest_sha256=sha256(manifest_path), d1a_data_identity=manifest['identity'],
        paired_plan_sha256=sha256(plan_path), paired_manifest_sha256=sha256(paired_manifest),
        paired_identity=plan['identity'], paired_extraction_record_sha256=extraction_record_hashes,
        overlay=str(overlay))
    result['identity'] = identity(result)
    atomic_json(output / 'input_manifest.json', result)
    return result
