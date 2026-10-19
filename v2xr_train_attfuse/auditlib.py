"""D-1D utilities. Historical packages are read-only, hash-pinned dependencies."""
from contextlib import contextmanager
from importlib import metadata
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPRO = ROOT / 'v2xr_official_repro'
PROTOCOL = ROOT / 'v2xr_protocol_audit'
sys.path.insert(1, str(PROTOCOL))
sys.path.insert(2, str(REPRO))
from common import atomic_json, identity, jsonable, lock, no_test, read_json, server_only, sha256
from support import DEFINITIONS, patched
# support adds the historical package to sys.path; keep D-1C's test_contract ahead
# of D-1A's equally named module while retaining this independent script directory.
sys.path.insert(1, str(PROTOCOL))

DEFAULTS = dict(
    data_root='/data/cjm/datasets/V2X-R/train',
    train_root='/data/cjm/datasets/v2xr-official-train-v2',
    output_root='/data/cjm/datasets/logs/v2xr-clean-attfuse-train-v1',
    runtime_root='/data/cjm/datasets/v2xr-official-runtime-v1',
    model_dir='/data/cjm/datasets/logs/v2xr-official-attfuse-model-v1',
    d1a_root='/data/cjm/datasets/logs/v2xr-official-attfuse-clean-repro-v1',
    gt_audit_root='/data/cjm/datasets/logs/v2xr-official-gt-audit-v1',
    d1c_root='/data/cjm/datasets/logs/v2xr-dminus1c-protocol-v1',
    view_root='/data/cjm/datasets/v2xr-official-validate-v1',
    seven_zip='/data/cjm/tools/7zip-26.04/bin/7zz')
SELECTION = dict(metric='Clean validation F AP70', direction='maximum',
    tie='earliest epoch; compare full-precision AP, no rounding', epochs=20,
    validation_frames=829, weather_used=False, random_seed=20261017,
    official_selection_audit='train.py saves every epoch; best-validation-loss block is commented; no active best/AP rule',
    final_protocols=DEFINITIONS, initialization='random; no pretrained weights')


def sources():
    pins = read_json(HERE / 'PINS.json')
    for name, digest in pins['dependencies'].items():
        if sha256(ROOT / name) != digest:
            raise ValueError('Frozen dependency changed: ' + name)
    expected = read_json(HERE / 'source_snapshot.json')['files']
    actual = {p.relative_to(HERE).as_posix(): sha256(p) for p in HERE.rglob('*')
              if p.is_file() and '__pycache__' not in p.parts and p.name != 'source_snapshot.json'}
    if actual != expected:
        raise ValueError('Synchronize the whole v2xr_train_attfuse directory, including its own snapshot')
    from common import verify_package
    from support import verify_sources
    verify_package()
    verify_sources()
    return actual


def paths(args):
    for key in DEFAULTS:
        setattr(args, key, str(no_test(getattr(args, key)).resolve()))
    for key in ('output_root', 'train_root'):
        path = Path(getattr(args, key))
        if Path('/data') not in path.parents:
            raise ValueError(key + ' must be on /data')
        if path == ROOT or ROOT in path.parents or path in ROOT.parents:
            raise ValueError('Data/log output must be outside the repository')
    a, b = Path(args.train_root), Path(args.output_root)
    if a == b or a in b.parents or b in a.parents:
        raise ValueError('Extracted data and training logs must have independent roots')
    for key in ('runtime_root', 'model_dir', 'd1a_root', 'gt_audit_root', 'd1c_root', 'view_root', 'data_root'):
        source = Path(getattr(args, key))
        for dest in (Path(args.train_root), Path(args.output_root)):
            if dest == source or dest in source.parents or source in dest.parents:
                raise ValueError('New output must be independent of historical inputs: ' + key)
    temporary = no_test(os.environ.get('TMPDIR', '')).resolve()
    if Path('/data') not in temporary.parents:
        raise ValueError('TMPDIR must be on /data')


def evaluation_path(args, destination):
    path = no_test(destination).resolve()
    if Path('/data') not in path.parents:
        raise ValueError('Evaluation output must be on /data')
    for source in [ROOT, Path(args.train_root)] + [Path(getattr(args, key)) for key in
            ('runtime_root', 'model_dir', 'd1a_root', 'gt_audit_root', 'd1c_root', 'view_root', 'data_root')]:
        if path == source or path in source.parents or source in path.parents:
            raise ValueError('Evaluation output overlaps frozen input: ' + str(source))
    return path


def request(output, record, resume):
    marker = output / 'request.json'
    if output.exists() and any(output.iterdir()):
        if not resume or not marker.exists() or read_json(marker) != jsonable(record):
            raise ValueError('Existing output needs --resume and identical source/config/input arguments')
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(marker, record)


def bound_json(path, record):
    record = jsonable(record)
    if path.exists() and read_json(path) != record:
        raise ValueError('Immutable contract changed: ' + str(path))
    atomic_json(path, record)


def complete(path):
    record = read_json(path)
    if record['status'] != 'complete':
        raise ValueError('Incomplete result: ' + str(path))
    for name, digest in record.get('output_hashes', {}).items():
        if sha256(path.parent / name) != digest:
            raise ValueError('Completed artifact changed: ' + name)
    return record


def frozen_c(args):
    """Verify the exact completed D-1C evidence, without loading its predictions."""
    pins = read_json(HERE / 'PINS.json')['d1c']
    root = Path(args.d1c_root)
    for name, digest in pins['files'].items():
        if sha256(no_test(root / name)) != digest:
            raise ValueError('Frozen D-1C artifact differs: ' + name)
    summary = read_json(root / 'summary.json')
    protocol = read_json(root / 'protocol.json')
    gt = read_json(root / 'gt_manifest.json')
    if (summary['status'] != 'complete' or not summary['full_validation']
            or summary['scientific_status'] != 'READY_FOR_INTERPRETATION'
            or protocol['protocols'] != DEFINITIONS or gt['definition'] != DEFINITIONS
            or gt['identity'] != pins['gt_identity'] or gt['fixed_total_per_condition'] != 6224):
        raise ValueError('D-1C prerequisite/GT definition changed')
    return dict(protocol=protocol, gt=gt, hashes=pins['files'])


def runtime(args, a):
    from observe import bootstrap, module_inventory
    root = bootstrap(args.runtime_root)
    import torch
    import opencood.models.point_pillar_intermediate
    from opencood.utils import eval_utils_opv2v
    if ({k: metadata.version(k) for k in a['versions']} != a['versions']
            or torch.version.hip != a['hip_version']):
        raise ValueError('Versions/HIP differ from successful D-1A')
    for name, row in module_inventory(root).items():
        if row['path'].endswith('.so') and a['imported_modules'].get(name) != row:
            raise ValueError('Native ROI/overlap extension differs from D-1A')
    return root


def gpu():
    if (not os.environ.get('ROCR_VISIBLE_DEVICES')
            or os.environ.get('HIP_VISIBLE_DEVICES') is not None
            or os.environ.get('CUDA_VISIBLE_DEVICES') is not None):
        raise ValueError('Set only ROCR_VISIBLE_DEVICES; unset HIP/CUDA_VISIBLE_DEVICES')
    if os.environ.get('PYTHONHASHSEED') != '20261016':
        raise ValueError('Use run_server.sh: frozen inference requires PYTHONHASHSEED=20261016')
    if any(os.environ.get(k) for k in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'SLURM_PROCID')):
        raise ValueError('This preregistration uses the official single-GPU path; distributed environment refused')


def verify_train(args, require_full=True):
    from train_acceptance import required_train_count, verify_manifest_acceptance
    root = Path(args.train_root)
    manifest = complete(root / 'manifest.json')
    if require_full and (not manifest['full_train'] or manifest['counts']['cooperative_frames'] != required_train_count()):
        raise ValueError('Training requires the completed, explicitly accepted full 8083-frame Clean train extraction')
    if identity({k: v for k, v in manifest.items() if k != 'identity'}) != manifest['identity']:
        raise ValueError('Train manifest identity differs')
    check_scene_partition(manifest)
    if manifest['full_train']:
        verify_manifest_acceptance(manifest, root)
    for i, row in enumerate(manifest['files']):
        file = no_test(root / 'train' / row['relative'])
        if file.stat().st_size != row['size'] or sha256(file) != row['sha256']:
            raise ValueError('Train input changed: ' + str(file))
        if (i + 1) % 5000 == 0:
            print('Train input SHA256', i + 1, '/', len(manifest['files']), flush=True)
    actual = {p.relative_to(root / 'train').as_posix() for p in (root / 'train').rglob('*') if p.is_file()}
    if actual != {r['relative'] for r in manifest['files']}:
        raise ValueError('Unexpected/missing train files')
    expected_directories = {s['scene'] for s in manifest['scenes']} | {
        s['scene'] + '/' + cav for s in manifest['scenes'] for cav in s['agents']}
    actual_directories = {p.relative_to(root / 'train').as_posix()
                          for p in (root / 'train').rglob('*') if p.is_dir()}
    if actual_directories != expected_directories:
        raise ValueError('Unexpected/missing train directories; empty CAV placeholders must not enter the loader view')
    for archive in manifest['archives']:
        if sha256(no_test(archive['path'])) != archive['sha256']:
            raise ValueError('Original train archive changed: ' + archive['path'])
        record = read_json(root / 'records' / archive['scene'] / (archive['cav'] + '.json'))
        if (record['status'] != 'complete' or not record['crc_test_passed']
                or record['archive_sha256'] != archive['sha256']
                or record['kind'] != archive['kind']
                or record['excluded_from_loader'] != archive['excluded_from_loader']):
            raise ValueError('Archive lacks matching completed CRC/exclusion evidence: ' + archive['path'])
    for row in read_json(root / 'plan.json').get('scene_metadata', []):
        if sha256(no_test(row['path'])) != row['sha256']:
            raise ValueError('Original train scene metadata changed: ' + row['path'])
    return manifest


def check_scene_partition(manifest):
    """Every source scene/archive must be either real data or an evidenced zero-frame placeholder."""
    from train_acceptance import required_train_count
    active, empty = manifest['scenes'], manifest.get('empty_scenes', [])
    names = [s['scene'] for s in active + empty]
    if len(names) != len(set(names)) or manifest['counts']['source_scene_count'] != len(names):
        raise ValueError('Source/data/zero-frame scene partition differs')
    if (manifest['counts']['scene_count'] != len(active)
            or manifest['counts']['cooperative_frames'] != sum(len(s['timestamps']) for s in active)):
        raise ValueError('Data scene/frame count differs from the actual active timestamp inventory')
    if manifest.get('full_train') and (
            manifest['counts']['source_scene_count'] != manifest['counts']['raw_source_scene_count']
            or manifest['counts']['cooperative_frames'] != required_train_count()):
        raise ValueError('Full train requires every raw scene classified and exactly 8083 user-approved real cooperative frames')
    archives = manifest['archives']
    keys = [(a['scene'], a['cav']) for a in archives]
    if len(keys) != len(set(keys)) or {a['scene'] for a in archives} != set(names):
        raise ValueError('Original archives are missing, duplicated or absent from scene inventory')
    if manifest['counts']['empty_scene_count'] != len(empty):
        raise ValueError('Zero-frame scene count differs')
    for scene in active + empty:
        rows = [a for a in archives if a['scene'] == scene['scene']]
        data = {a['cav'] for a in rows if a['kind'] == 'data_cav'}
        placeholders = {a['cav'] for a in rows if a['kind'] == 'empty_directory_placeholder'}
        if len(data) + len(placeholders) != len(rows):
            raise ValueError('Unknown source archive classification')
        if set(scene['empty_agents']) != placeholders:
            raise ValueError('Empty CAV identity partition differs: ' + scene['scene'])
        if scene in active:
            if not data or data != set(scene['agents']) or not scene['timestamps']:
                raise ValueError('Data-bearing scene lacks complete active CAV/frame inventory')
        else:
            if (data or not placeholders or scene['timestamps'] or scene['cooperative_frames'] != 0
                    or not scene['excluded_from_loader'] or scene['kind'] != 'zero_frame_scene_placeholder'):
                raise ValueError('Nonempty/incomplete scene cannot be classified as a zero-frame placeholder')
            if {(a['cav'], a['path'], a['sha256']) for a in scene['archives']} != {
                    (a['cav'], a['path'], a['sha256']) for a in rows}:
                raise ValueError('Zero-frame scene archive evidence differs')
        for row in rows:
            if row['kind'] == 'empty_directory_placeholder':
                proof = row['placeholder_evidence']
                if (not row['excluded_from_loader'] or proof['regular_file_count'] != 0
                        or proof['data_blocks'] != 0 or proof['directory'] != row['cav']
                        or proof['archive_sha256'] != row['sha256']):
                    raise ValueError('Empty archive lacks conclusive matching content/hash evidence')
            elif row['excluded_from_loader']:
                raise ValueError('Data-bearing archive cannot be omitted from the loader view')
