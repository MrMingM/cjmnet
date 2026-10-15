"""Atomic artifacts, immutable identities and separate worker ownership."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = Path(__file__).resolve().parent
KEEP = 'KEEP_SHARED'
WEATHERS = ('clean', 'fog', 'rain', 'snow')
ADVERSE = WEATHERS[1:]
FAMILIES = ('N1', 'N2', 'score-only')


def now():
    return datetime.now(timezone.utc).isoformat()


def safe_path(value, dataset=False):
    raw = Path(value).absolute()
    resolved = raw.resolve()
    # Check lexical aliases too: a directory named test cannot be used as an
    # alias for development. Resolved parents catch links to nested test data.
    if any(p.lower() == 'test' for path in (raw, resolved) for p in path.parts):
        raise ValueError('Formal test access prohibited: ' + str(value))
    if dataset and resolved.name not in ('train', 'validate', 'validation'):
        raise ValueError('Explicit development split required: ' + str(resolved))
    return resolved


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
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def settings():
    # JSON is a strict YAML subset: pure preflight has no PyYAML dependency.
    return read_json(PACKAGE / 'protocol.yaml')


def sources():
    from local_fusion_source_kill_stage1.common import source_hashes
    result = source_hashes()
    for folder in (PACKAGE, ROOT / 'local_fusion_source_kill_stage2_precheck'):
        for p in folder.iterdir():
            if p.suffix in ('.py', '.sh', '.yaml', '.md'):
                result[p.relative_to(ROOT).as_posix()] = digest(p)
    result['opencood/data_utils/post_processor/voxel_postprocessor.py'] = digest(
        ROOT / 'opencood/data_utils/post_processor/voxel_postprocessor.py')
    frozen = ROOT / 'gspr_communication/frozen_sources.json'
    result[frozen.relative_to(ROOT).as_posix()] = digest(frozen)
    for row in json.loads(frozen.read_text(encoding='utf-8-sig')):
        result[row['path']] = digest(ROOT / row['path'])
    return result


def resource_check(protocol, at=None):
    deadline = datetime.fromisoformat(protocol['resource_stop_at'])
    if (at or datetime.now(timezone.utc)) >= deadline:
        raise RuntimeError('RESOURCE_STOP: preserve results; no new workers; not hypothesis rejection')


def external_run(value):
    if os.name != 'posix':
        raise RuntimeError('Scientific workers run on the remote POSIX server only')
    path = safe_path(value)
    if Path('/data/cjm/datasets/logs').resolve() not in path.parents:
        raise ValueError('RUN must be below /data/cjm/datasets/logs')
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_contract(run):
    protocol = read_json(Path(run) / 'PROTOCOL.json')
    if protocol['source_hashes'] != sources():
        raise ValueError('Source hash drift: use a new RUN; do not overwrite scientific results')
    if protocol['settings'] != settings():
        raise ValueError('Frozen settings drift')
    from .preflight import section42
    if protocol['section42'] != section42():
        raise ValueError('Frozen section 42 changed: use a new RUN with explicit provenance')
    for path, expected in protocol['input_hashes'].items():
        if digest(path) != expected:
            raise ValueError('Input hash drift: ' + path)
    if protocol['identity'] != object_hash({k: v for k, v in protocol.items() if k != 'identity'}):
        raise ValueError('Protocol identity corrupted')
    for filename in ('protocol.json','protocol.yaml'):
        if read_json(Path(run)/filename)!=protocol:
            raise ValueError('Immutable protocol copy differs: '+filename)
    if read_json(Path(run)/'SOURCE_SNAPSHOT.json')!=protocol['source_hashes']:
        raise ValueError('Immutable source snapshot differs')
    return protocol


@contextmanager
def worker(run, name, protocol):
    import fcntl
    folder = Path(run) / name
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        resource_check(protocol)
        signal.signal(signal.SIGTERM, lambda sig, frame: sys.exit(128 + sig))
        state = dict(status='running', pid=os.getpid(), identity=protocol['identity'],
                     source_hash=object_hash(protocol['source_hashes']), input_hash=object_hash(protocol['input_hashes']),
                     config_hash=protocol['identity'], started_at=now(), completed_at=None)
        atomic_json(folder / 'manifest.json', state)
        try:
            yield folder
            load_contract(run)
            state.update(status='complete', completed_at=now())
        except BaseException as exc:
            state.update(status='failed', error=repr(exc), completed_at=now())
            raise
        finally:
            atomic_json(folder / 'manifest.json', state)


def resume(path, identity):
    path = safe_path(path)
    if not path.exists():
        return None
    row = read_json(path)
    if (row.get('envelope_hash') != object_hash({k:v for k,v in row.items() if k!='envelope_hash'}) or
            row['identity'] != identity or row['payload_hash'] != object_hash(row['payload'])):
        raise ValueError('Resume/hash mismatch: new RUN required: ' + str(path))
    if row['status'] != 'complete':
        return None
    for artifact, expected in row.get('artifacts', {}).items():
        target=safe_path(path.parent/artifact)
        if path.parent not in target.parents:
            raise ValueError('Artifact escapes condition directory: '+artifact)
        if digest(target) != expected:
            raise ValueError('Corrupted completed artifact: ' + artifact)
    return row['payload']


def commit_unit(path, identity, payload, artifacts=(), metadata=None):
    path = safe_path(path)
    row=dict(status='complete', identity=identity, config_hash=identity, completed_at=now(),
        payload_hash=object_hash(payload), payload=payload,
        artifacts={str(Path(p).relative_to(path.parent)): digest(p) for p in artifacts},
        metadata=metadata or {})
    row['envelope_hash']=object_hash(row)
    atomic_json(path,row)


def jsonlines(path):
    with safe_path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)
