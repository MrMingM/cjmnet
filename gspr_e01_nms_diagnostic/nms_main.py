"""Diagnose the saved T1/C epoch-2 first-frame failure without training or changing E01."""
import argparse
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

from nms_common import (DEFAULT_CHECKPOINT_SHA, DEFAULT_RUN, HERE, PROJECT, atomic,
    epoch_audit, existing_lock, identity, original_e01, read, safe, sha, verify_sources)


def unix_address_bytes(address):
    """Linux pathname sockets need room for a terminating NUL in sun_path[108]."""
    if not isinstance(address, str) or not address or '\0' in address:
        raise ValueError('Expected a nonempty pathname AF_UNIX address')
    count = len(os.fsencode(address))
    if count > 107:
        raise ValueError('AF_UNIX path too long: ' + str(count) + ' bytes: ' + address)
    return count


@contextmanager
def short_ipc_directory():
    """Own one short private server directory, independent of output and inherited TMPDIR."""
    if sys.platform != 'linux':
        raise RuntimeError('Real IPC temporary directories are server-only')
    with tempfile.TemporaryDirectory(prefix='gnms-', dir='/tmp') as directory:
        # Leave ample room for multiprocessing's pymp-* directory and listener-* name.
        unix_address_bytes(directory + '/pymp-' + 'x' * 16 + '/listener-' + 'x' * 16)
        yield directory


def ipc_environment(environment, directory):
    result = dict(environment)
    for key in ('TMPDIR', 'TEMP', 'TMP'):
        result[key] = str(directory)
    return result


def probe_ipc(directory, listener_factory=None):
    """Bind the same kind of socket as multiprocessing before importing any model code."""
    directory = str(directory)
    if any(os.environ.get(key) != directory for key in ('TMPDIR', 'TEMP', 'TMP')):
        raise ValueError('IPC temporary environment differs from the private request directory')
    if tempfile.gettempdir() != directory:
        raise ValueError('Cached Python temporary directory differs; use a fresh diagnostic child')
    if listener_factory is None:
        from multiprocessing.connection import Listener
        listener_factory = Listener
    with listener_factory(family='AF_UNIX') as listener:
        address = listener.address
        count = unix_address_bytes(address)
        if Path(directory) not in Path(address).parents:
            raise ValueError('Multiprocessing selected a socket outside the private IPC directory')
        return dict(status='complete', temporary_directory=directory, socket_path=address,
            socket_path_bytes=count, linux_pathname_limit_bytes=107,
            checked_before_model_import=True, no_training=True)


class DiagnosticInterrupted(Exception):
    def __init__(self, number):
        self.number = int(number)
        self.signal_name = signal.Signals(number).name
        super().__init__('Diagnostic interrupted by ' + self.signal_name)


@contextmanager
def termination_handlers(handler, signal_api=signal):
    """Respect dispositions inherited from nohup/background shells, including SIG_IGN."""
    previous = {}
    try:
        for number in (signal_api.SIGTERM, signal_api.SIGINT, signal_api.SIGHUP):
            inherited = signal_api.getsignal(number)
            if inherited == signal_api.SIG_IGN:
                continue
            previous[number] = inherited
            signal_api.signal(number, handler)
        yield
    finally:
        for number, original in previous.items():
            signal_api.signal(number, original)


def stop_child(process):
    """Terminate only this invocation's separate process group; wait before releasing leases."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def wait_for_worker(process, log, signal_api=signal):
    stopping = False
    def interrupted(number, frame):
        nonlocal stopping
        # A second signal during cleanup must not release leases while the child is still alive.
        if not stopping:
            stopping = True
            raise DiagnosticInterrupted(number)
    with termination_handlers(interrupted, signal_api):
        try:
            while True:
                try:
                    return process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    print('RUNNING first_frame log=' + str(log), flush=True)
        except DiagnosticInterrupted:
            stop_child(process)
            raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--run-root', type=Path, default=Path(DEFAULT_RUN))
    result.add_argument('--output-root', type=Path)
    result.add_argument('--checkpoint', type=Path)
    result.add_argument('--gpu')
    result.add_argument('--trace-layers', action='store_true', help='Passive first-frame layer observations only')
    result.add_argument('--compare-bn', action='store_true', help='Paired original/current-batch BN diagnosis; never scored')
    result.add_argument('--self-test', action='store_true')
    result.add_argument('--internal-worker', action='store_true', help=argparse.SUPPRESS)
    return result


def prepare(cli, e01):
    run_root = safe(cli.run_root).resolve(strict=True)
    original = read(run_root / 'request.json')
    if identity({k: v for k, v in original.items() if k != 'identity'}) != original['identity']:
        raise ValueError('Original controller request content hash differs')
    if original['source'] != e01.sources():
        raise ValueError('Original E01 run used different code; do not rewrite its source hashes')
    if Path(original['output']).resolve() != run_root:
        raise ValueError('Original output identity differs')
    gpu = str(original['E1_gpu'])
    if not gpu.isdecimal() or not 0 <= int(gpu) <= 63 or cli.gpu is not None and cli.gpu != gpu:
        raise ValueError('Diagnostic must use the physical E1 GPU registered in request.json: ' + gpu)
    if Path(original['executable']).resolve() != Path(sys.executable).resolve():
        raise ValueError('Use the original opencood Python executable')
    model = run_root / 'smoke/E1/T1/C'
    contract_path = model / 'contract.json'
    contract = read(contract_path)
    if (identity({k: v for k, v in contract.items() if k != 'identity'}, compact=False) != contract['identity']
            or contract['variant'] != 'C' or not contract['smoke_only']
            or contract['settings']['e01']['recipe'] != 'T1'
            or sha(model / 'config.yaml') != contract['config_sha256']
            or str(contract['environment']['physical_gpu']) != gpu):
        raise ValueError('Failed T1/C smoke model contract differs')
    checkpoint = safe(cli.checkpoint or model / 'epochs/02' / ('checkpoint_' + DEFAULT_CHECKPOINT_SHA + '.pth')).resolve(strict=True)
    if checkpoint.parent != model / 'epochs/02' or not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        raise ValueError('Use the saved failing epoch-02 checkpoint, not an initial/selected/other model')
    actual_sha = sha(checkpoint)
    if checkpoint.name != 'checkpoint_' + actual_sha + '.pth':
        raise ValueError('Immutable checkpoint filename and actual SHA256 differ')
    root = safe(cli.output_root or Path('/data/cjm/datasets/logs/gspr-e01-nms-diagnostic-v1') /
        (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + str(os.getpid()))).resolve()
    readonly = [PROJECT, run_root, Path(original['original']), Path(original['train']),
        Path('/data/cjm/datasets/v2xr-official-runtime-v1'),
        Path('/data/cjm/datasets/v2xr-official-validate-v1'),
        Path('/data/cjm/datasets/logs/v2xr-official-attfuse-model-v1'),
        Path('/data/cjm/datasets/logs/v2xr-official-attfuse-clean-repro-v1'),
        Path('/data/cjm/datasets/logs/v2xr-official-gt-audit-v1'),
        Path('/data/cjm/datasets/logs/v2xr-dminus1c-protocol-v1')]
    log_root = Path('/data/cjm/datasets/logs')
    if log_root not in root.parents or any(root == p.resolve() or root in p.resolve().parents or p.resolve() in root.parents for p in readonly):
        raise ValueError('Diagnostic output must be independent of every original output/runtime/data/source')
    if root.exists() and any(p.name not in ('launcher.log',) for p in root.iterdir()):
        raise ValueError('Diagnostic output already has artifacts; use a fresh directory, no overwrite')
    return root, dict(status='prepared', original_run_root=str(run_root), original_request_sha256=sha(run_root / 'request.json'),
        original_request_identity=original['identity'], original_E1_gpu=gpu,
        model_root=str(model), training_contract_sha256=sha(contract_path),
        checkpoint=str(checkpoint), checkpoint_sha256=actual_sha,
        sources=verify_sources(), original_threads=original['threads'],
        train_root=original['train'], no_training=True, no_protocol_change=True,
        resume_audit=epoch_audit(model, contract), diagnostic_output=str(root),
        trace_layers=cli.trace_layers or cli.compare_bn, bn_batch_stats=False)


def child(cli):
    root = safe(cli.output_root).resolve(strict=True)
    request = read(root / 'request.json')
    if request['sources'] != verify_sources():
        raise ValueError('Diagnostic sources changed before child start')
    # A bind failure must fail here, rather than being swallowed in a DataLoader queue thread.
    try:
        ipc = probe_ipc(request['ipc_temporary_directory'])
    except Exception as error:
        atomic(root / 'ipc_preflight.json', dict(status='failed', error=repr(error), no_training=True))
        raise
    atomic(root / 'ipc_preflight.json', ipc)
    print('IPC_PREFLIGHT=complete socket_path_bytes=' + str(ipc['socket_path_bytes']), flush=True)
    if sha(request['checkpoint']) != request['checkpoint_sha256'] or sha(Path(request['model_root']) / 'contract.json') != request['training_contract_sha256']:
        raise ValueError('Original checkpoint/contract changed before diagnostic')
    if os.environ.get('ROCR_VISIBLE_DEVICES') != request['original_E1_gpu'] or os.environ.get('PYTHONHASHSEED') != '20261016':
        raise ValueError('Diagnostic inference environment differs')
    original_e01()
    internal = argparse.Namespace(model_root=Path(request['model_root']), train_root=Path(request['train_root']),
        variant='C', recipe='T1', smoke=True, checkpoint=Path(request['checkpoint']),
        destination=root / 'observed_evaluation', clean_only=True, report_root=root,
        trace_layers=request.get('trace_layers', False), bn_batch_stats=request.get('bn_batch_stats', False))
    from nms_observe import execute
    execute(internal, request)
    if sha(request['checkpoint']) != request['checkpoint_sha256']:
        raise ValueError('Original detector checkpoint changed during diagnosis')
    report = read(root / 'report.json')
    changed = request.get('bn_batch_stats', False)
    if (report['eligible_for_AP_or_selection'] or report['protocol_changes'] != changed
            or report.get('BN_current_batch_counterfactual') != changed
            or report['score_GT_and_AP_functions_changed']):
        raise ValueError('Diagnostic must be unscored and precisely label its authorized BN intervention')


def run_diagnostic(root, request, e01, leases):
    request = dict(request)
    with ExitStack() as stack:
        root.mkdir(parents=True, exist_ok=True)
        ipc_directory = stack.enter_context(short_ipc_directory())
        request['ipc_temporary_directory'] = ipc_directory
        atomic(root / 'request.json', request)
        env = e01.gpu_env(request['original_E1_gpu'])
        for key, value in request['original_threads'].items():
            env.pop(key, None)
            if value is not None:
                env[key] = value
        env = ipc_environment(env, ipc_directory)
        env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
        command = [sys.executable, '-B', '-u', str(HERE / 'nms_main.py'),
                   '--internal-worker', '--output-root', str(root)]
        print('DIAGNOSTIC_OUTPUT=' + str(root), flush=True)
        print('PHYSICAL_GPU=' + request['original_E1_gpu'], flush=True)
        print('LAST_COMMITTED_EPOCH=' + str(request['resume_audit']['last_committed_epoch']), flush=True)
        print('BN_CURRENT_BATCH_COUNTERFACTUAL=' + str(request.get('bn_batch_stats', False)), flush=True)
        print('IPC_TMPDIR=' + ipc_directory, flush=True)
        log = root / 'first_frame.log'
        with log.open('w', encoding='utf-8') as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, env=env, cwd=str(PROJECT), pass_fds=tuple(leases), start_new_session=True)
            try:
                code = wait_for_worker(process, log)
            except DiagnosticInterrupted as error:
                atomic(root / 'diagnostic_status.json', dict(status='interrupted',
                    signal=error.signal_name, signal_number=error.number,
                    worker_returncode=process.returncode, log=str(log), no_training=True,
                    scientific_evaluation_complete=False))
                print('INTERRUPTED signal=' + error.signal_name + ' log=' + str(log)
                      + '; own diagnostic child stopped; original E01 files unchanged', flush=True)
                raise SystemExit(128 + error.number) from None
        if code:
            atomic(root / 'diagnostic_status.json', dict(status='failed', returncode=code, log=str(log)))
            raise RuntimeError('Diagnostic failed; inspect ' + str(log) + '\n' + log.read_text(encoding='utf-8')[-4000:])
        report = read(root / 'report.json')
        products = ['request.json', 'ipc_preflight.json', 'report.json', 'first_frame_arrays.npz']
        if request['trace_layers']:
            products.append('layer_trace.json')
        atomic(root / 'diagnostic_status.json', dict(status='diagnostic_complete', outcome=report['outcome'],
            scientific_evaluation_complete=False, no_training=True,
            output_hashes={n: sha(root / n) for n in products}))
        for line in log.read_text(encoding='utf-8').splitlines():
            if line.startswith(('SUMMARY=', 'LAYER_SUMMARY=')):
                print(line, flush=True)
        print('REPORT=' + str(root / 'report.json'), flush=True)
        return report


def run_comparison(root, request, e01, leases):
    from nms_bn import compare_pair
    root.mkdir(parents=True, exist_ok=True)
    atomic(root / 'comparison_request.json', dict(request,
        mode='original_vs_current_batch_BN_unscored_diagnostic',
        scope='Same checkpoint and first frame; two fresh processes; original E01 unchanged'))
    try:
        for name, changed in (('native', False), ('current_batch', True)):
            target = root / name
            print('START BN_COMPARE mode=' + name, flush=True)
            run_diagnostic(target, dict(request, trace_layers=True, bn_batch_stats=changed,
                no_protocol_change=not changed, diagnostic_output=str(target)), e01, leases)
        args = [read(root / name / filename) for filename in ('report.json', 'request.json', 'layer_trace.json')
                for name in ('native', 'current_batch')]
        result = compare_pair(*args)
        result['artifacts'] = {str(path.relative_to(root)): sha(path)
            for name in ('native', 'current_batch') for path in (root / name).iterdir()
            if path.is_file() and path.name in ('diagnostic_status.json', 'request.json', 'report.json',
                                                'ipc_preflight.json', 'layer_trace.json', 'first_frame_arrays.npz')}
        atomic(root / 'bn_comparison.json', result)
        atomic(root / 'comparison_status.json', dict(status='paired_diagnostic_complete',
            scientific_evaluation_complete=False, eligible_for_AP_or_selection=False,
            comparison_sha256=sha(root / 'bn_comparison.json')))
        print('BN_COMPARISON=' + __import__('json').dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
        print('BN_COMPARISON_REPORT=' + str(root / 'bn_comparison.json'), flush=True)
    except BaseException as error:
        atomic(root / 'comparison_status.json', dict(status='interrupted' if isinstance(error, SystemExit) else 'failed',
            error=repr(error), scientific_evaluation_complete=False, eligible_for_AP_or_selection=False))
        raise


def main():
    cli = parser().parse_args()
    if cli.self_test:
        verify_sources()
        from tests_synthetic import run_tests
        run_tests()
        return
    if sys.platform != 'linux':
        raise RuntimeError('Real checkpoint/first-frame diagnosis is server-only')
    verify_sources()
    if cli.internal_worker:
        child(cli)
        return
    e01 = original_e01()
    root, request = prepare(cli, e01)
    with ExitStack() as stack:
        # Hold all three original leases across both fresh processes in a BN comparison.
        leases = [stack.enter_context(existing_lock(Path(request['original_run_root']) / '.controller.lock')),
                  stack.enter_context(existing_lock(Path(request['model_root']) / '.worker.lock')),
                  stack.enter_context(existing_lock(Path('/data/cjm/datasets/logs/.gspr-e01-leases') /
                                                   ('gpu' + request['original_E1_gpu'] + '.lock')))]
        if cli.compare_bn:
            run_comparison(root, request, e01, leases)
        else:
            run_diagnostic(root, request, e01, leases)


if __name__ == '__main__':
    main()
