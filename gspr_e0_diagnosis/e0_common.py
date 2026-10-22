"""E0 identity, immutable receipts and source checks. No research imports."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
WEATHERS = ('clean', 'fog', 'snow')
ARMS = ('R0', 'R1', 'R2', 'R3')
DEFAULT_OUTPUT = '/data/cjm/datasets/logs/gspr-e0-diagnosis-v1'
DEFAULT_ORIGINAL = '/data/cjm/datasets/logs/v2xr-weather-l012-v1'


def read(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def sha(path):
    h = hashlib.sha256()
    with safe(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def safe(path):
    path = Path(path).absolute()
    # Check lexical path AND resolved symlink target before any input is opened.
    for candidate in (path, path.resolve()):
        if any(part.lower() in ('test', 'testing') or 'opv2v-w' in part.lower() for part in candidate.parts):
            raise ValueError('Formal test access refused: ' + str(path))
    return path


def atomic(path, value, immutable=False):
    path = Path(path)
    if path.exists() and immutable:
        if read(path) != value:
            raise ValueError('Incompatible immutable artifact: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def rows(path):
    with safe(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sources():
    snapshot = read(HERE / 'source_snapshot.json')
    actual = {p.relative_to(HERE).as_posix(): sha(p) for p in HERE.rglob('*')
              if p.is_file() and '__pycache__' not in p.parts and p.name != 'source_snapshot.json'}
    if actual != snapshot['files']:
        raise ValueError('Upload the entire E0 package, including source_snapshot.json')
    for name, digest in read(HERE / 'PINS.json')['files'].items():
        if sha(PROJECT / name) != digest:
            raise ValueError('Read-only dependency changed: ' + name)
    return dict(package=actual, dependencies=read(HERE / 'PINS.json')['files'])


def historical():
    """Use the original weather adapter's explicit factory and official evaluator."""
    folder = PROJECT / 'v2xr_weather_l012'
    sys.path.insert(0, str(folder))
    import auditlib
    if Path(auditlib.__file__).resolve() != folder / 'auditlib.py':
        raise RuntimeError('Unexpected historical auditlib namespace')
    auditlib.sources()
    import evaluation
    import models
    return auditlib, evaluation, models


@contextmanager
def lock(path):
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def receipt(folder, request_id):
    path = Path(folder) / 'complete.json'
    if not path.exists():
        return None
    result = read(path)
    if result['status'] != 'complete' or result['request'] != request_id:
        raise ValueError('Stage identity differs: ' + str(path))
    for name, digest in result['files'].items():
        if sha(Path(folder) / name) != digest:
            raise ValueError('Completed artifact changed: ' + str(Path(folder) / name))
    return result


def finish(folder, request_id):
    folder = Path(folder)
    files = {p.relative_to(folder).as_posix(): sha(p) for p in folder.rglob('*')
             if p.is_file() and p.name not in ('complete.json', 'failure.json')
             and not p.name.endswith('.tmp')}
    atomic(folder / 'complete.json', dict(status='complete', request=request_id,
        files=files, finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())), immutable=True)


def frame_done(path, request_id):
    marker = Path(str(path) + '.sha.json')
    if not marker.exists():
        return False
    row = read(marker)
    if row['request'] != request_id or sha(path) != row['sha256']:
        raise ValueError('Frame evidence changed: ' + str(path))
    return True


def frame_commit(path, value, request_id):
    if frame_done(path, request_id):
        if read(path) != value:
            raise ValueError('Recomputed frame differs: ' + str(path))
        return
    atomic(path, value)
    atomic(str(path) + '.sha.json', dict(request=request_id, sha256=sha(path)), immutable=True)
