"""Stdlib IO, provenance, test-data guards and server-only preflight.

Workers write separate atomic status files; only the driver writes manifest.
No Git invocation, model inference or label creation occurs in preflight.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = Path(__file__).resolve().parent
KEEP = 'KEEP_SHARED'
WEATHERS = ('clean', 'fog', 'rain', 'snow')
ADVERSE = WEATHERS[1:]
FORBIDDEN = (
    '/data/scd/datasets/opv2v_official_data_dumping/test',
    '/data/cjm/datasets/opv2v-w/fog/test',
    '/data/cjm/datasets/opv2v-w/rain/test',
    '/data/cjm/datasets/opv2v-w/snow/test',
)


def now():
    return datetime.now(timezone.utc).isoformat()


def safe_path(value, dataset=False):
    path = Path(value).resolve()
    normalized = path.as_posix().lower()
    if any(part.lower() == 'test' for part in path.parts) or any(
            normalized == item or normalized.startswith(item + '/') for item in FORBIDDEN):
        raise ValueError('Formal test access prohibited: ' + str(path))
    if dataset and path.name not in ('train', 'validate', 'validation'):
        raise ValueError('Explicit development split required: ' + str(path))
    return path


def external_run(value, create=False):
    path = safe_path(value)
    if Path('/data/cjm/datasets/logs').resolve() not in path.parents:
        raise ValueError('RUN must be below /data/cjm/datasets/logs')
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path):
    return json.loads(safe_path(path).read_text(encoding='utf-8'))


def digest(path):
    result = hashlib.sha256()
    with safe_path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_text(path, value):
    path = safe_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def settings():
    import yaml
    return yaml.safe_load((PACKAGE / 'experiment.yaml').read_text(encoding='utf-8'))


def source_hashes():
    files = list(PACKAGE.glob('*.py')) + list(PACKAGE.glob('*.sh'))
    files += [PACKAGE / 'experiment.yaml', PACKAGE / 'README.md']
    for directory in ('local_fusion_task_source_oracle', 'local_fusion_task_split_pilot',
                      'local_fusion_utility_v2', 'local_fusion_v3', 'gspr_communication',
                      'gspr_evidence', 'gspr_review'):
        files += list((ROOT / directory).glob('*.py'))
    files += [ROOT / p for p in (
        'local_fusion_task_source_oracle/experiment.yaml',
        'local_fusion_v3/experiment.yaml', 'local_fusion_action_utility_audit/counterfactual.py',
        'local_fusion_action_utility_audit/common.py', 'qa_local_intervention/operators.py',
        'qa_local_intervention/common.py', 'ceif_audit/scoring.py',
        'opencood/utils/eval_utils.py', 'opencood/utils/common_utils.py',
        'opencood/utils/box_utils.py', 'opencood/tools/train_utils.py',
        'opencood/data_utils/datasets/intermediate_fusion_dataset.py')]
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(set(files))}


def static_preflight():
    import ast
    import yaml
    for path in PACKAGE.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    for path in PACKAGE.glob('*.sh'):
        data = path.read_bytes()
        if b'\r' in data or not data.endswith(b'\n'):
            raise ValueError('Shell must use LF: ' + path.name)
        subprocess.run(['sh', '-n', path.as_posix()], check=True)
    spec = yaml.safe_load(safe_path(ROOT / settings()['oracle_config']).read_text(encoding='utf-8'))
    fixed = dict(seed=20260927, target_iou=.7, roi_expansions=[1., 1.5],
                 candidate_families=['single', 'query'], shortlist_k=6)
    if any(spec[k] != v for k, v in fixed.items()):
        raise ValueError('Original section 33 settings changed')
    context = (ROOT / 'AI_CONTEXT.md').read_text(encoding='utf-8')
    if 'Source Utility Kill Protocol：Stage 1 预注册' not in context:
        raise ValueError('Stage 1 preregistration missing')
    from gspr_communication.runtime import verify_frozen
    verify_frozen()
    snapshot = PACKAGE / 'source_snapshot.json'
    if read_json(snapshot)['source_hashes'] != source_hashes():
        raise ValueError('Delivered source_snapshot differs; sync the entire Stage 1 directory')
    return {'status': 'complete', 'scope': 'static only; no server cache/data/checkpoint access',
            'source_hashes': source_hashes()}


def discover_runs():
    """Evidence-derived defaults; never guess a cache by globbing latest runs."""
    context = (ROOT / 'AI_CONTEXT.md').read_text(encoding='utf-8')
    def section(number):
        found = re.search(r'^## ' + str(number) + r'\. .*?(?=^## |\Z)', context, re.M | re.S)
        if found is None:
            raise ValueError('Missing provenance section ' + str(number))
        return found.group()
    s0 = re.findall(r'/data/cjm/datasets/logs/action_utility_audit_[0-9_]+', section(38))
    oracle = re.findall(r'/data/cjm/datasets/logs/task_source_oracle_v2_[0-9_]+', section(33))
    if len(set(s0)) != 1 or len(set(oracle)) != 1:
        raise ValueError('Ambiguous historical run evidence; set S0_RUN / ORACLE_RUN explicitly')
    return os.environ.get('S0_RUN', s0[0]), os.environ.get('ORACLE_RUN', oracle[0])


def guard_tree(root):
    """Resolve every nested directory/file BEFORE traversing it or loading data."""
    root = safe_path(root, dataset=True)
    seen = set()
    def walk(path):
        path = safe_path(path)
        if path in seen:
            return
        seen.add(path)
        for child in path.iterdir():
            resolved = safe_path(child)
            if resolved.is_dir():
                walk(resolved)
    walk(root)


def check_hashes(recorded):
    if not recorded:
        raise ValueError('Missing source hashes')
    for name, expected in recorded.items():
        path = safe_path(ROOT / name)
        if ROOT not in path.parents or digest(path) != expected:
            raise ValueError('Source mismatch: ' + name)


def verify_s0(run):
    from local_fusion_action_utility_audit.common import verify_protocol_snapshot, Manifest
    protocol = read_json(run / 'protocol.json')
    for value in protocol['inputs'].values():
        safe_path(value)
    for value in protocol['dataset_paths'].values():
        safe_path(value, dataset=True)
    # The legacy helper opens artifacts through its own hash routine. Guard every
    # recorded path first, including aliases, before handing it the old manifest.
    recorded_manifest = read_json(run / 'manifest.json')
    for entry in recorded_manifest['entries'].values():
        for name in entry.get('artifacts', {}):
            path = safe_path(run / name)
            if run not in path.parents:
                raise ValueError('S0 artifact escapes its run: ' + name)
    for name in protocol['source_hashes']:
        path = safe_path(ROOT / name)
        if ROOT not in path.parents:
            raise ValueError('S0 source escapes repository')
    for name in ('protocol.json', 'decision_results.json', 'Shared.pth'):
        safe_path(Path(protocol['inputs']['b0_run']) / name)
    protocol = verify_protocol_snapshot(run)
    manifest = Manifest(run)
    legacy = None
    if protocol.get('repairs'):
        repair = protocol['repairs']
        if len(repair) != 1 or repair[0]['repair_id'] != 'shared_snapshot_v2':
            raise ValueError('Unknown historical S0 repair; cannot infer label provenance')
        record = repair[0]
        folder = run / 'repair_shared_drift'
        original = read_json(folder / 'original_protocol.json')
        original_manifest = read_json(folder / 'original_manifest.json')
        journal = read_json(folder / 'transaction.json')
        if journal['status'] != 'complete' or journal['repair_id'] != record['repair_id']:
            raise ValueError('Incomplete original S0 repair transaction')
        for name, expected in journal['proposed_hashes'].items():
            if digest(folder / name) != expected:
                raise ValueError('Original repair proposal changed')
        if (original['identity'] != record['original_identity'] or
                original_manifest['identity'] != original['identity'] or
                object_hash(original['source_hashes']) != original['identity']['source_hash']):
            raise ValueError('Original S0 source identity differs')
        compatibility = read_json(ROOT / 'local_fusion_action_utility_audit/repair_compatibility.json')
        for name, expected in compatibility['legacy_audit_sources'].items():
            if original['source_hashes'].get(name) != expected:
                raise ValueError('Unsupported original label producer source: ' + name)
        for name in ('config', 'inputs', 'train_indices', 'validation_indices', 'train_scenes',
                     'validation_scenes', 'probe_fit_scenes', 'probe_calibration_scenes'):
            if original[name] != protocol[name]:
                raise ValueError('Historical repair changed scientific protocol: ' + name)
        if original['identity']['checkpoint_hashes'] != protocol['identity']['checkpoint_hashes']:
            raise ValueError('Historical repair changed model checkpoints')
        if digest(run / record['evidence']) != record['evidence_sha256']:
            raise ValueError('Historical repair diagnostic evidence changed')
        legacy = (record, original_manifest)
    for stage in ('baseline', 'S0-build', 'S0', 'validation-reference'):
        if not manifest.complete(stage):
            raise ValueError('S0 incomplete: ' + stage)
    for split in ('train', 'validation'):
        indices = protocol[split + '_indices']
        if not indices or len(indices) != len(set(indices)):
            raise ValueError('Invalid S0 indices')
        for weather in WEATHERS:
            if not manifest.complete('S0-weather', split, weather):
                raise ValueError('S0 weather incomplete')
            for index in indices:
                if not manifest.complete('S0-frame', split, weather, index):
                    raise ValueError('S0 frame incomplete')
                entry = manifest.value['entries'][manifest.key('S0-frame', split, weather, index)]
                name = f'cache/{split}/{weather}/{index:08d}.pt'
                if name not in entry.get('artifacts', {}):
                    raise ValueError('S0 frame lacks complete action cache hash')
                if entry['source_hash'] != protocol['identity']['source_hash']:
                    if legacy is None or split != 'train':
                        raise ValueError('Unrecognized S0 label producer source')
                    record, original_manifest = legacy
                    old_entry = original_manifest['entries'][manifest.key('S0-frame', split, weather, index)]
                    if (entry.get('reuse_approval') != record['repair_id'] or
                            entry['source_hash'] != record['original_identity']['source_hash'] or
                            old_entry['artifacts'] != entry['artifacts']):
                        raise ValueError('Retained original S0 cache provenance changed')
    fit, cal = set(protocol['probe_fit_scenes']), set(protocol['probe_calibration_scenes'])
    if fit & cal or fit | cal != set(protocol['train_scenes']):
        raise ValueError('S0 fit/calibration split changed')
    fixed = dict(shared_top_k=256, roi_expansion=1.5, candidate_families=['single', 'query'],
                 target_iou=.7, fp_identity_iou=.7)
    if any(protocol['config'].get(k) != v for k, v in fixed.items()):
        raise ValueError('S0 label protocol changed')
    return protocol


def verify_inputs(s0_run, oracle_run):
    """Read-only: no loader, CUDA or detector is constructed here."""
    import yaml
    s0 = verify_s0(s0_run)
    old = read_json(oracle_run / 'protocol.json')
    results = read_json(oracle_run / 'death_test_results.json')
    b0_run = external_run(s0['inputs']['b0_run'])
    b0 = read_json(b0_run / 'protocol.json')
    if safe_path(old['b0_run']) != b0_run:
        raise ValueError('S0 and original Oracle used different B0 runs')
    for split in ('train', 'validation'):
        for field in ('indices', 'scenes'):
            key = split + '_' + field
            if s0[key] != b0[key]:
                raise ValueError('S0/B0 split mismatch: ' + key)
    if old['validation_indices'] != b0['validation_indices']:
        raise ValueError('Oracle validation frames changed')
    check_hashes(old['source_hashes'])
    check_hashes(b0['source_hashes'])
    spec = yaml.safe_load((ROOT / settings()['oracle_config']).read_text(encoding='utf-8'))
    if old['config'] != spec:
        raise ValueError('Original Oracle config differs')
    checks = dict(b0_protocol_sha256=digest(b0_run / 'protocol.json'),
                  b0_shared_checkpoint_sha256=digest(b0_run / 'Shared.pth'),
                  frontend_sha256=digest(s0['inputs']['frontend_checkpoint']),
                  v3_checkpoint_sha256=digest(s0['inputs']['v3_checkpoint']))
    if any(old.get(k) != v for k, v in checks.items()):
        raise ValueError('Original Oracle checkpoint/protocol differs')
    ck = s0['identity']['checkpoint_hashes']
    if (ck['b0_shared'] != checks['b0_shared_checkpoint_sha256'] or
            ck['frontend_checkpoint'] != checks['frontend_sha256'] or
            ck['v3_checkpoint'] != checks['v3_checkpoint_sha256']):
        raise ValueError('S0 checkpoint identity differs')
    for weather in WEATHERS:
        row = results['conditions'][weather]
        if row['frames'] != len(old['validation_indices']) or row['global_sort'] is not False:
            raise ValueError('Original Oracle result scope differs')
        for metric in ('ap30', 'ap50', 'ap70'):
            expected = s0['b0_saved_results']['conditions'][weather]['results']['Shared'][metric]
            if abs(row['results']['Shared'][metric] - expected) > 1e-6:
                raise ValueError('Original Shared reference differs')
    # Scan the effective raw YAML roots before any dataset enumeration/import.
    front = yaml.safe_load(safe_path(s0['inputs']['frontend_config']).read_text(encoding='utf-8'))
    options = yaml.safe_load(safe_path(s0['inputs']['v3_config']).read_text(encoding='utf-8'))
    for key in ('root_dir', 'validate_dir'):
        path = safe_path(options.get(key) or front[key], dataset=True)
        if str(path) != s0['dataset_paths'][key]:
            raise ValueError('Recorded dataset root differs: ' + key)
        guard_tree(path)
    return s0, old


def preregister_clock():
    """One project clock across RUNs; resource stop has no hypothesis verdict."""
    import fcntl
    path = Path('/data/cjm/datasets/logs/source_kill_suite_clock.json')
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            value = read_json(path)
        else:
            started = datetime.now(timezone.utc)
            value = {'started_at': started.isoformat(),
                     'resource_stop_at': (started + timedelta(days=14)).isoformat(),
                     'meaning': 'resource stop; not hypothesis rejection'}
            atomic_json(path, value)
    if datetime.now(timezone.utc) >= datetime.fromisoformat(value['resource_stop_at']):
        raise RuntimeError('14-day RESOURCE_STOP; no scientific hypothesis rejection')
    return value


def preflight(run):
    static_preflight()
    s0_name, oracle_name = discover_runs()
    s0_run, oracle_run = external_run(s0_name), external_run(oracle_name)
    s0, old = verify_inputs(s0_run, oracle_run)
    refs = {'s0_run': str(s0_run), 's0_manifest_hash': digest(s0_run / 'manifest.json'),
            's0_protocol_hash': digest(s0_run / 'protocol.json'),
            'oracle_run': str(oracle_run), 'oracle_protocol_hash': digest(oracle_run / 'protocol.json'),
            'oracle_results_hash': digest(oracle_run / 'death_test_results.json'),
            'oracle_config_hash': digest(ROOT / settings()['oracle_config']),
            'b0_checkpoint_hash': digest(Path(s0['inputs']['b0_run']) / 'Shared.pth'),
            'label_audit_hash': digest(ROOT / 'logs/sectionB/all/LABEL_AUDIT.md'),
            'frontend_config_hash': digest(s0['inputs']['frontend_config']),
            'v3_config_hash': digest(s0['inputs']['v3_config'])}
    if s0.get('repairs'):
        refs['s0_repair_hashes'] = {p.relative_to(s0_run).as_posix(): digest(p)
            for p in sorted((s0_run / 'repair_shared_drift').glob('*.json'))}
    protocol = dict(stage=settings()['stage'], source_commit=None, source_hashes=source_hashes(),
                    references=refs, config=settings(), test_data_used=False,
                    inputs=s0['inputs'], validation_indices=old['validation_indices'])
    protocol['identity'] = object_hash(protocol)
    path = run / 'protocol.json'
    if path.exists():
        saved = read_json(path)
        if {k: v for k, v in saved.items() if k != 'project_clock'} != protocol:
            raise ValueError('Resume inputs/source drift; no cache reuse allowed')
        protocol = saved
    else:
        protocol['project_clock'] = preregister_clock()
        atomic_json(path, protocol)
    preregister_clock()
    refresh_manifest(run, 'running')
    print('PREFLIGHT COMPLETE: S0 hashes/splits, original Oracle, checkpoints, test guards', flush=True)


def contract(run):
    protocol = read_json(run / 'protocol.json')
    if protocol['source_hashes'] != source_hashes():
        raise ValueError('Stage 1 source drift')
    refs = protocol['references']
    for root, field, filename in (
            ('s0_run', 's0_manifest_hash', 'manifest.json'),
            ('s0_run', 's0_protocol_hash', 'protocol.json'),
            ('oracle_run', 'oracle_protocol_hash', 'protocol.json'),
            ('oracle_run', 'oracle_results_hash', 'death_test_results.json')):
        if digest(Path(refs[root]) / filename) != refs[field]:
            raise ValueError('Historical run changed: ' + field)
    for field in ('frontend_config', 'v3_config'):
        if digest(protocol['inputs'][field]) != refs[field + '_hash']:
            raise ValueError('Input config changed')
    for name, expected in refs.get('s0_repair_hashes', {}).items():
        if digest(Path(refs['s0_run']) / name) != expected:
            raise ValueError('Historical S0 repair provenance changed')
    return protocol


def state_file(run, component):
    return Path(run) / (component.lower() + '_status.json')


def component_status(run, component):
    path = state_file(run, component)
    value = read_json(path) if path.exists() else {'status': 'pending'}
    if value['status'] == 'complete':
        protocol = read_json(Path(run) / 'protocol.json')
        if value['identity'] != protocol['identity']:
            raise ValueError('Status identity mismatch')
        if not value.get('artifacts'):
            raise ValueError('Complete status lacks artifacts')
        for name, expected in value['artifacts'].items():
            if digest(Path(run) / name) != expected:
                raise ValueError('Complete output corrupted: ' + name)
    return value


@contextmanager
def component(run, name, artifacts):
    protocol = contract(run)
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    path = state_file(run, name)
    row = {'status': 'running', 'started_at': now(), 'completed_at': None,
           'identity': protocol['identity']}
    atomic_json(path, row)
    try:
        yield protocol
        contract(run)
        row.update(status='complete', completed_at=now(),
                   artifacts={p: digest(Path(run) / p) for p in artifacts})
    except BaseException as exc:
        row.update(status='failed', completed_at=now(), error=repr(exc))
        raise
    finally:
        atomic_json(path, row)


def refresh_manifest(run, status=None):
    path = Path(run) / 'manifest.json'
    value = read_json(path) if path.exists() else {'stage': settings()['stage'], 'started_at': now()}
    if (Path(run) / 'protocol.json').exists():
        protocol = read_json(Path(run) / 'protocol.json')
        value.update(source_commit=None, source_hashes=protocol['source_hashes'],
                     **protocol['references'], project_clock=protocol['project_clock'])
    value['K0'] = component_status(run, 'K0_PRIME')
    value['K1_0'] = component_status(run, 'K1_0')
    value['status'] = status or value.get('status', 'pending')
    value['completed_at'] = now() if value['status'] in ('failed', 'complete') else None
    if value['status'] == 'complete':
        value['final_artifacts'] = {name: digest(Path(run) / name)
                                    for name in ('STAGE1_FINAL.json', 'STAGE1_FINAL.md')}
    atomic_json(path, value)


def save_condition(run, name, payload):
    protocol = contract(run)
    payload = {'status': 'complete', 'identity': protocol['identity'], **payload}
    atomic_json(Path(run) / 'conditions' / (name + '.json'),
                {'payload': payload, 'sha256': object_hash(payload)})


def load_condition(run, name):
    path = Path(run) / 'conditions' / (name + '.json')
    if not path.exists():
        return None
    row = read_json(path)
    value = row['payload']
    if (row['sha256'] != object_hash(value) or value['status'] != 'complete' or
            value['identity'] != read_json(Path(run) / 'protocol.json')['identity']):
        raise ValueError('Condition corrupted or identity differs: ' + name)
    return value


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--static', action='store_true')
    cli.add_argument('--run')
    cli.add_argument('--validate-run', action='store_true')
    cli.add_argument('--preflight', action='store_true')
    cli.add_argument('--state', choices=('running', 'failed', 'complete'))
    cli.add_argument('--lock-driver', action='store_true')
    args = cli.parse_args()
    if args.static:
        print(json.dumps(static_preflight(), ensure_ascii=False, indent=2))
        return
    run = external_run(args.run, create=True)
    if args.validate_run:
        return
    if args.lock_driver:
        import fcntl
        with (run / '.driver.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            env = {**os.environ, 'STAGE1_LOCK_HELD': '1', 'RUN': str(run)}
            child = subprocess.Popen(['sh', str(PACKAGE / 'run_stage1.sh')], env=env,
                                     pass_fds=(lock.fileno(),))
            signal.signal(signal.SIGTERM, lambda signum, frame: child.terminate())
            raise SystemExit(child.wait())
    if args.preflight:
        try:
            preflight(run)
        except BaseException:
            refresh_manifest(run, 'failed')
            traceback.print_exc()
            raise
    elif args.state:
        refresh_manifest(run, args.state)


if __name__ == '__main__':
    main()
