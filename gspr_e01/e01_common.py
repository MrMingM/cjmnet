"""E01 identity and process utilities; no research imports or legacy name collision."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import signal
import threading

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
SETTINGS = json.loads((HERE / 'settings.json').read_text(encoding='utf-8'))
WEATHERS = ('clean', 'fog', 'snow')
PROTOCOLS = ('O', 'O*', 'F')
METRICS = ('ap30', 'ap50', 'ap70')
STOP = threading.Event()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with safe(path).open('rb') as stream:
        for part in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def frozen_identity(value):
    """Training contracts retain the frozen helper's JSON serialization and hash."""
    import importlib.util
    key = '_e01_frozen_stdlib_common'
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, PROJECT / 'v2xr_official_repro/common.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key].identity(value)


def safe(path):
    path = Path(path).absolute()
    for p in (path, path.resolve()):
        if any(x.lower() in ('test', 'testing') or 'opv2v-w' in x.lower() for x in p.parts):
            raise ValueError('Formal test access refused: '+str(path))
    return path


def atomic(path, value, immutable=False):
    path = Path(path)
    if immutable and path.exists():
        if read(path) != value:
            raise ValueError('Immutable identity differs: '+str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    with tmp.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def sources():
    expected = read(HERE / 'source_snapshot.json')['files']
    actual = {p.relative_to(HERE).as_posix(): sha(p) for p in HERE.rglob('*')
        if p.is_file() and '__pycache__' not in p.parts and p.name != 'source_snapshot.json'}
    if actual != expected:
        raise ValueError('Synchronize the complete gspr_e01 package and its source snapshot')
    pins = read(HERE / 'PINS.json')['files']
    for name, digest in pins.items():
        if sha(PROJECT / name) != digest:
            raise ValueError('Read-only dependency changed: '+name)
    return dict(package=actual, dependencies=pins)


def reference_results():
    """Prefer the actual FINAL_RESULTS; fallback is its hash-bound original-byte E0 copy."""
    expected = read(HERE / 'PINS.json')['original_results_sha256']
    original = PROJECT / 'logs/train/FINAL_RESULTS.json'
    archived = PROJECT / 'gspr_e0_diagnosis/original_FINAL_RESULTS.json'
    source = original if original.is_file() else archived
    if sha(source) != expected or sha(archived) != expected:
        raise ValueError('Full-precision T0 archive identity differs')
    value = read(source)
    return value, dict(path=str(source), sha256=expected, original_path=str(original),
        original_summary_synchronized=original.is_file(),
        fallback='Original-byte archived copy; actual server C/D full summaries must equal it' if source == archived else None)


def server_only():
    if sys.platform != 'linux':
        raise RuntimeError('Scientific execution is server-only; local entry: check_static.py')


@contextmanager
def lock(path):
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stream.seek(0)
        stream.truncate()
        stream.write(str(os.getpid()))
        stream.flush()
        # Inherit the lease across workers, including an orphan surviving its controller.
        os.set_inheritable(stream.fileno(), True)
        yield stream.fileno()


def receipt(folder, request):
    marker = Path(folder) / 'complete.json'
    if not marker.exists():
        return None
    row = read(marker)
    if row['status'] != 'complete' or row['request'] != request:
        raise ValueError('Stage request differs: '+str(marker))
    for name, digest in row['files'].items():
        if sha(Path(folder) / name) != digest:
            raise ValueError('Committed stage artifact changed: '+name)
    return row


def finish(folder, request):
    folder = Path(folder)
    files = {p.relative_to(folder).as_posix(): sha(p) for p in folder.rglob('*')
        if p.is_file() and p != folder / 'complete.json' and not p.name.endswith(('.tmp', '.lock'))
        and p.name != 'last_failure.json' and 'logs' not in p.relative_to(folder).parts}
    atomic(folder / 'complete.json', dict(status='complete', request=request, files=files), immutable=True)


def run(command, log, env=None, leases=()):
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    print('START '+ ' '.join(map(str, command))+' log='+str(log), flush=True)
    started = time.monotonic()
    env = dict(os.environ if env is None else env)
    leases = leases or tuple(int(x) for x in env.get('E01_LEASE_FDS', '').split(',') if x)
    nested = env.get('E01_CHILD') == '1'
    env['E01_CHILD'] = '1'
    with log.open('a', encoding='utf-8') as stream:
        proc = subprocess.Popen(list(map(str, command)), env=env, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, pass_fds=tuple(leases), start_new_session=not nested)
        try:
            while True:
                try:
                    code = proc.wait(timeout=20)
                    break
                except subprocess.TimeoutExpired:
                    if STOP.is_set():
                        raise InterruptedError('Controller interrupted')
                    print(f'RUNNING pid={proc.pid} elapsed={int(time.monotonic()-started)}s log={log}', flush=True)
        except BaseException:
            if nested:
                proc.terminate()
            else:
                os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                if nested:
                    proc.kill()
                else:
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            raise
    if code:
        tail = log.read_text(encoding='utf-8', errors='replace')[-6000:]
        raise RuntimeError(f'Process failed returncode={code} log={log}\n{tail}')


def gpu_env(number):
    value = str(number)
    if not value.isdecimal() or int(value) > 63 or str(int(value)) != value:
        raise ValueError('GPU must be one physical number in 0..63')
    env = dict(os.environ, ROCR_VISIBLE_DEVICES=value, PYTHONHASHSEED='20261016',
        PYTHONDONTWRITEBYTECODE='1')
    env.pop('HIP_VISIBLE_DEVICES', None)
    env.pop('CUDA_VISIBLE_DEVICES', None)
    return env


def jsonl(path):
    with safe(path).open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)
