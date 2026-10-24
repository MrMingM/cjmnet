"""Source identities and an isolated process-local bridge to immutable E01 code."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
from contextlib import contextmanager
import hashlib

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
BASE = PROJECT / 'gspr_e01'
POLICY = json.loads((HERE / 'policy.json').read_text(encoding='utf-8'))
_BASE_SOURCES = None


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def own_sources():
    actual = {p.relative_to(HERE).as_posix(): sha(p) for p in HERE.rglob('*')
              if p.is_file() and '__pycache__' not in p.parts and p.name != 'source_snapshot.json'}
    if actual != read(HERE / 'source_snapshot.json')['files']:
        raise ValueError('Synchronize the whole gspr_e01_warmup directory')
    for name, digest in read(HERE / 'PINS.json')['files'].items():
        if sha(PROJECT / name) != digest:
            raise ValueError('Original E01 changed: ' + name)
    return actual


def bridge():
    global _BASE_SOURCES
    if _BASE_SOURCES is None:
        own_sources()
        sys.path.insert(0, str(BASE))
        import e01_common as base
        _BASE_SOURCES = base.sources
        _BASE_SOURCES()
        base.SETTINGS = dict(copy.deepcopy(base.SETTINGS), formal_startup=POLICY)
        base.sources = lambda: dict(frozen_e01=_BASE_SOURCES(), formal_startup=own_sources())
    import e01_common as base
    return base


def base_sources():
    bridge()
    return _BASE_SOURCES()


def frozen_worker():
    bridge()
    name = '_formal_warmup_frozen_worker'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, BASE / 'worker.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


@contextmanager
def short_temp():
    if sys.platform != 'linux':
        raise RuntimeError('Server execution only')
    root = Path('/data/cjm/datasets/tmp')
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='gw-', dir=root) as name:
        if len(os.fsencode(name + '/pymp-xxxxxxxx/listener-xxxxxxxx')) > 107:
            raise ValueError('IPC temporary path too long')
        previous = {key: os.environ.get(key) for key in ('TMPDIR', 'TEMP', 'TMP')}
        try:
            for key in previous:
                os.environ[key] = name
            yield name
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def ipc_probe():
    from multiprocessing.connection import Listener
    directory = tempfile.gettempdir()
    if any(os.environ.get(key) != directory for key in ('TMPDIR', 'TEMP', 'TMP')):
        raise ValueError('IPC temp environment/cache differs')
    with Listener(family='AF_UNIX') as listener:
        if len(os.fsencode(listener.address)) > 107 or Path(directory) not in Path(listener.address).parents:
            raise ValueError('IPC socket did not use the private short directory')
        return dict(status='complete', socket_path_bytes=len(os.fsencode(listener.address)))


@contextmanager
def signal_handlers(signal_api=signal):
    base = bridge()
    previous = {}
    stopping = False
    def stop(number, frame):
        nonlocal stopping
        if stopping:
            return  # Let the original process runner finish cleaning up its child group.
        stopping = True
        base.STOP.set()
        raise KeyboardInterrupt('Signal ' + signal_api.Signals(number).name)
    try:
        for number in (signal_api.SIGTERM, signal_api.SIGINT, signal_api.SIGHUP):
            handler = signal_api.getsignal(number)
            if handler != signal_api.SIG_IGN:
                previous[number] = handler
                signal_api.signal(number, stop)
        yield
    finally:
        for number, handler in previous.items():
            signal_api.signal(number, handler)
