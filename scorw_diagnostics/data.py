"""Freeze only validation inputs, actual D6 cache and original cloud receipts."""
from pathlib import Path
import numpy as np
from .core import digest, identity, read, write


def checked_commit(folder):
    folder = Path(folder)
    c = read(folder / 'COMMIT.json'); contract = read(folder / 'CONTRACT.json')
    if c['status'] != 'passed' or c['contract'] != contract['identity']:
        raise ValueError('Invalid historical commit: ' + str(folder))
    if identity(contract['request']) != contract['identity']:
        raise ValueError('Historical contract encoding mismatch')
    for rel, sha in c['hashes'].items():
        p = folder / rel
        if Path(rel).is_absolute() or '..' in Path(rel).parts or digest(p) != sha:
            raise ValueError('Historical artifact changed: ' + str(p))
    return c


def catalogue(c):
    root = Path(c['dataset']); d = read(root / 'manifests/VALIDATE_SELECTION.json')
    if d['split'] != 'validate' or d['frames'] != c['expected_frames'] or len(d['rows']) != c['expected_pcds']:
        raise ValueError('Validation scope changed')
    frames = [(s['scene'], t) for s in d['layout'] for t in s['timestamps']]
    rows = {(r['scene'], r['timestamp'], r['cav']): r for r in d['rows']}
    if len(rows) != len(d['rows']) or len(set(frames)) != len(frames):
        raise ValueError('Duplicate scene/frame/CAV')
    return root, d, frames, rows


def freeze(c, out):
    root, d, frames, rows = catalogue(c)
    p = read(root / 'DATASET_PROTOCOL.json'); report = read(root / 'VALIDATION_ATTFUSE_RESULTS.json')
    from .core import REPO
    historical_sources = {**p['sources']['pinned'], **p['sources']['native_runtime']}
    changed = [name for name, sha in historical_sources.items() if not (REPO/name).is_file() or digest(REPO/name) != sha]
    if changed:
        raise ValueError('D6 historical runtime source changed: '+str(changed))
    if digest(root / 'VALIDATION_ATTFUSE_RESULTS.json') != c['d6_report_sha256']:
        raise ValueError('D6 final report differs from reviewed publication')
    if read(root / 'VALIDATION_GATES.json')['status'] != 'DATASET_COMPLETE':
        raise ValueError('Dataset is not complete')
    for key, expected in [('checkpoint', c['checkpoint_sha256']), ('config', c['model_config_sha256'])]:
        if digest(p['config']['model'][key]) != expected:
            raise ValueError('Frozen model identity mismatch: ' + key)
    cache = []; hashes = {}
    for i, (s, t) in enumerate(frames):
        folder = root / 'stages/D6/frames' / s / t
        checked_commit(folder)
        record = read(folder / 'RESULT.json')
        if (record['scene'], record['timestamp']) != (s, t):
            raise ValueError('Cache frame identity mismatch')
        for mode in ('clean', 'fog'):
            pred = record['predictions'][mode]
            if len(pred['boxes']) != len(pred['scores']):
                raise ValueError('Cache prediction count mismatch')
            for m in pred['matches'].values():
                if len(m['tp']) != len(pred['boxes']) or m['gt'] != len(record['GT_ids']):
                    raise ValueError('Cached matches incomplete')
        cache.append(record); hashes[str(folder / 'RESULT.json')] = digest(folder / 'RESULT.json')
        if (i + 1) % 200 == 0:
            print('Freeze D6 caches %d/%d' % (i + 1, len(frames)), flush=True)
    if identity(cache) != report['frame_results_identity']:
        raise ValueError('D6 frame cache aggregate identity mismatch')
    for i, r in enumerate(d['rows'], 1):
        rec = root / 'records/validate' / r['scene'] / 'clouds' / r['cav'] / r['timestamp']
        # All original per-cloud completion artifacts (including sidecar) are hashed.
        checked_commit(rec)
        receipt = read(rec / 'RESULT.json')
        if receipt['relative'] != r['relative'] or Path(r['relative']).parts[0] != 'validate':
            raise ValueError('Unexpected split/identity')
        expected = [(Path(r['pcd']), r['pcd_sha256']), (Path(r['yaml']), r['yaml_sha256']),
                    (root / 'fog' / r['relative'], receipt['output_sha256']),
                    (rec / 'provenance.npz', receipt['provenance_sha256'])]
        if receipt['input_sha256'] != r['pcd_sha256'] or receipt['yaml_sha256'] != r['yaml_sha256']:
            raise ValueError('Receipt/source identity mismatch')
        if (root / 'clean' / r['relative']).resolve() != Path(r['pcd']).resolve():
            raise ValueError('Clean link target changed')
        for path, sha in expected:
            if digest(path) != sha:
                raise ValueError('Frozen input changed: ' + str(path))
            hashes[str(path)] = sha
        hashes[str(rec / 'RESULT.json')] = digest(rec / 'RESULT.json')
        if i % 200 == 0:
            print('Freeze original/Fog/sidecar %d/%d' % (i, len(d['rows'])), flush=True)
    for name in ('DATASET_PROTOCOL.json', 'VALIDATION_GATES.json', 'VALIDATION_ATTFUSE_RESULTS.json',
                 'manifests/VALIDATE_SELECTION.json', 'stages/D6/MODEL_AUDIT.json'):
        hashes[str(root / name)] = digest(root / name)
    ref = Path(p['config']['model']['reference_protocol']).parent
    for path in (ref / 'protocol.json', ref / 'clean/result.json'):
        if path.exists():
            hashes[str(path)] = digest(path)
    result = dict(status='passed', frames=len(frames), pcds=len(rows),
                  GT=sum(len(x['GT_ids']) for x in cache), inputs=hashes,
                  inputs_identity=identity(hashes), model=p['config']['model'],
                  inference=False, GPU_used=False, split='validate',
                  D6_historical_source_hashes=historical_sources, D6_historical_sources_exact=True)
    if result['GT'] != c['expected_gt']:
        raise ValueError('D6 GT total mismatch')
    write(out / 'INPUT_FREEZE.json', result)
    return result


def fixed_gt(c, layout, stamp, rows):
    import yaml
    from shared_weather_benchmark.audit import canonical_gt
    from shared_weather_benchmark.geometry import object_matrix, box_corners, pose_matrix, inverse
    root = Path(c['dataset']); config = read(root / 'DATASET_PROTOCOL.json')['config']
    metadata = {}
    for cav in layout['cavs']:
        with Path(rows[(layout['scene'], stamp, cav)]['yaml']).open(encoding='utf-8') as f:
            metadata[cav] = yaml.load(f, Loader=yaml.Loader)
    objects, _ = canonical_gt(metadata, config['evaluation']['gt_pair_tolerance_m'])
    ego = layout['cavs'][0]; pose = rows[(layout['scene'], stamp, ego)]['pose']
    from opencood.hypes_yaml.yaml_utils import load_yaml
    h = load_yaml(config['model']['config']); limit = h.get('train_params', {}).get('max_cav', 7)
    active = [v for v in layout['cavs'] if np.linalg.norm(np.asarray(rows[(layout['scene'], stamp, v)]['pose'][:2])
              - pose[:2]) <= config['evaluation']['communication_xy_m']][:limit]
    selected = {k: o for k, o in objects.items() if k != ego and min(np.linalg.norm(
        np.asarray(object_matrix(o))[:3, 3] - rows[(layout['scene'], stamp, v)]['pose'][:3])
        for v in active) <= config['evaluation']['max_object_range_m']}
    ids = sorted(selected); m = np.asarray(inverse(pose_matrix(pose)))
    gt = np.asarray([np.asarray(box_corners(selected[i])) @ m[:3, :3].T + m[:3, 3]
                     for i in ids], dtype=np.float32).reshape(-1, 8, 3)
    b = config['evaluation']['GT_RANGE']
    mask = ((gt[:, :, :2] >= b[:2]) & (gt[:, :, :2] <= b[3:5])).all(axis=(1, 2))
    ids = [i for i, yes in zip(ids, mask) if yes]; gt = gt[mask]
    return gt, ids, active, pose


def cache_frame(root, s, t):
    return read(root / 'stages/D6/frames' / s / t / 'RESULT.json')
