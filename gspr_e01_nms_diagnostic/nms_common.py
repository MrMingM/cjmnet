"""Read-only E01 provenance and independent diagnostic output helpers; no legacy name collision."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import os
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_RUN = '/data/cjm/datasets/logs/gspr-e01-v1'
DEFAULT_CHECKPOINT_SHA = '6b8abd19b6fd7db5f6e53e90fda7b628c68452ce00991de71d9de5bf9ac7f49d'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def safe(path):
    path = Path(path).absolute()
    for candidate in (path, path.resolve()):
        if any(p.lower() in ('test', 'testing') or 'opv2v-w' in p.lower() for p in candidate.parts):
            raise ValueError('Test paths are refused: ' + str(candidate))
    return path


def sha(path):
    result = hashlib.sha256()
    with safe(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def identity(value, compact=True):
    options = dict(sort_keys=True, ensure_ascii=False, allow_nan=False)
    if compact:
        options['separators'] = (',', ':')
    return hashlib.sha256(json.dumps(value, **options).encode()).hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True, ensure_ascii=False,
        allow_nan=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    os.replace(temporary, path)


@contextmanager
def swap(obj, name, replacement):
    # Unlike getattr/setattr alone, this also restores staticmethod descriptors exactly.
    previous = obj.__dict__[name] if isinstance(obj, type) else getattr(obj, name)
    setattr(obj, name, replacement)
    try:
        yield
    finally:
        setattr(obj, name, previous)


def verify_sources():
    expected = read(HERE / 'source_snapshot.json')['files']
    actual = {p.relative_to(HERE).as_posix(): sha(p) for p in HERE.rglob('*')
        if p.is_file() and '__pycache__' not in p.parts and p.name != 'source_snapshot.json'}
    if actual != expected:
        raise ValueError('Synchronize the whole gspr_e01_nms_diagnostic directory')
    for relative, digest in read(HERE / 'PINS.json')['files'].items():
        if sha(PROJECT / relative) != digest:
            raise ValueError('Original E01 code changed: ' + relative)
    return actual


@contextmanager
def existing_lock(path):
    """Take the original lease without creating/truncating/changing its PID contents."""
    import fcntl
    path = safe(path)
    with path.open('r+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Original E01 process still owns a lease: ' + str(path)
                               + '; no processes were killed and no lock was deleted') from error
        yield stream.fileno()


def original_e01():
    sys.path.insert(0, str(PROJECT / 'gspr_e01'))
    import e01_common
    e01_common.sources()
    return e01_common


def epoch_audit(model_root, contract, count=2):
    """Only complete.json and exact full-state hashes establish a recoverable epoch."""
    rows = []
    gap = False
    for epoch in range(1, count + 1):
        marker = Path(model_root) / 'epochs' / f'{epoch:02d}' / 'complete.json'
        if not marker.exists():
            gap = True
            continue
        if gap:
            raise ValueError('Noncontiguous original epoch commits')
        row = read(marker)
        if row['status'] != 'complete' or row['epoch'] != epoch or row['contract'] != contract['identity']:
            raise ValueError('Original epoch commit identity differs')
        for field in ('checkpoint', 'resume_state', 'clean_summary', 'trace'):
            if sha(row[field]) != row[field + '_sha256']:
                raise ValueError('Original committed bytes changed: ' + field)
        rows.append(dict(epoch=epoch, marker=str(marker), sha256=sha(marker),
                         resume_state=row['resume_state'], resume_state_sha256=row['resume_state_sha256']))
    return dict(last_committed_epoch=rows[-1]['epoch'] if rows else 0, verified_commits=rows,
        epoch_02_committed=any(r['epoch'] == 2 for r in rows),
        epoch_02_full_state_exists=(Path(model_root) / 'resume_states/02.pth').is_file(),
        interpretation='Original trainer writes scheduler/full state/complete.json after Clean AP. A detector-only checkpoint cannot replace a full Adam/scheduler/RNG state.')
