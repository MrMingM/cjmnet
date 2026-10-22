"""One controller, independent E0/E1 branches, immutable stage receipts."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from pathlib import Path
import os
import signal
import sys
import threading
import traceback
from e01_common import (HERE, PROJECT, SETTINGS, atomic, finish, gpu_env, identity,
    lock, read, receipt, run, server_only, sha, sources)


class Controller:
    def __init__(self, cli):
        self.cli, self.root = cli, cli.output_root
        self.status_lock = threading.Lock()
        self.states = {}
        self.request_id = None
        self.e0_checked = set()

    def state(self, name, status, **extra):
        with self.status_lock:
            self.states[name] = dict(status=status, **extra)
            atomic(self.root / 'current_stage.json', dict(request=self.request_id,
                mode=self.cli.run_mode, stages=self.states))

    def stage(self, name, command, output, gpu):
        folder = self.root / '.stages' / name
        request = identity(dict(parent=self.request_id, stage=name))
        if receipt(folder, request):
            self.verify_stage(name)
            self.state(name, 'reused_verified')
            print('REUSE verified '+name, flush=True)
            return
        self.state(name, 'running')
        try:
            run(command, self.root / 'logs' / (name+'.log'), env=gpu_env(gpu))
            folder.mkdir(parents=True, exist_ok=True)
            # A manifest of actual output bytes, never directory-existence completion.
            output = Path(output)
            files = {str(p): sha(p) for p in output.rglob('*') if p.is_file()
                and not p.name.endswith(('.lock', '.tmp')) and p.name != 'last_failure.json'
                and 'logs' not in p.relative_to(output).parts}
            if not files:
                raise ValueError('Stage produced no committed artifacts: '+name)
            atomic(folder / 'artifacts.json', dict(status='complete', files=files))
            finish(folder, request)
            self.state(name, 'complete')
        except BaseException:
            self.state(name, 'technical_failure', traceback=traceback.format_exc())
            raise

    def verify_stage(self, name):
        folder = self.root / '.stages' / name
        if receipt(folder, identity(dict(parent=self.request_id, stage=name))):
            for path, digest in read(folder / 'artifacts.json')['files'].items():
                if sha(path) != digest:
                    raise ValueError('Stage output changed: '+path)
            if name in ('E0_import', 'E0_full', 'E0_aliases'):
                source = self.cli.e0_existing_root if name == 'E0_import' else self.root / 'E0'
                if str(source) not in self.e0_checked:
                    from worker import verified_e0_source
                    verified_e0_source(source, self.cli.original_root)
                    self.e0_checked.add(str(source))

    def worker(self, stage, model_root, *args):
        return [sys.executable, '-B', HERE / 'worker.py', stage,
            '--model-root', model_root, '--train-root', self.cli.train_root,
            '--original-root', self.cli.original_root, *args]

    def e0(self):
        root, gpu = self.root / 'E0', self.cli.gpu_e0
        if self.cli.e0_existing_root:
            if self.cli.mode != 'all':
                raise ValueError('E0_EXISTING_ROOT is for completed full E0 only')
            self.stage('E0_import', self.worker('E0-import', root,
                '--existing', self.cli.e0_existing_root), root, gpu)
            return
        if self.cli.mode == 'all':
            command = ['sh', PROJECT / 'gspr_e0_diagnosis/run_all.sh']
            # Environment is passed explicitly, so E0/E1 cannot race over process globals.
            self.e0_command(command, root, gpu)
        else:
            stages = ('preflight', 'self-test') if self.cli.mode == 'self-test' else (
                'preflight', 'self-test', 'smoke-paired', 'smoke-R0', 'smoke-R1', 'smoke-R2', 'smoke-R3', 'smoke-analysis')
            for stage in stages:
                command = [sys.executable, '-B', PROJECT / 'gspr_e0_diagnosis/e0_main.py', stage,
                    '--output', root, '--original', self.cli.original_root]
                # e0_main verifies earlier receipts itself; their content hashes are also registered here.
                self.stage('E0_'+stage, command, root / 'stages', gpu)
        if self.cli.mode == 'all':
            self.stage('E0_aliases', self.worker('E0-import', root, '--existing', root), root, gpu)

    def e0_command(self, command, root, gpu):
        name = 'E0_full'
        folder = self.root / '.stages' / name
        request = identity(dict(parent=self.request_id, stage=name))
        if receipt(folder, request):
            self.verify_stage(name)
            self.state(name, 'reused_verified')
            return
        self.state(name, 'running')
        env = gpu_env(gpu)
        env.update(OUTPUT_ROOT=str(root), ORIGINAL_ROOT=str(self.cli.original_root),
            PHYSICAL_GPU=str(gpu), PYTHON=sys.executable)
        try:
            run(command, self.root / 'logs/E0_full.log', env=env)
            final = root / 'stages/final/complete.json'
            if read(final)['status'] != 'complete':
                raise ValueError('E0 final receipt missing')
            folder.mkdir(parents=True, exist_ok=True)
            # The E0 importer independently verifies every inner receipt and hash.
            atomic(folder / 'artifacts.json', dict(status='complete', files={str(final): sha(final)}), immutable=True)
            finish(folder, request)
            self.state(name, 'complete')
        except BaseException:
            self.state(name, 'technical_failure', traceback=traceback.format_exc())
            raise

    def e1(self):
        gpu = self.cli.gpu_e1
        self.stage('E1_self_test', self.worker('self-test', self.root / 'E1/self_test'), self.root / 'E1/self_test', gpu)
        if self.cli.mode == 'self-test':
            return
        self.models(self.root / 'smoke/E1', gpu, smoke=True)
        self.stage('E1_smoke_report', self.worker('E1-report', self.root / 'smoke/E1', '--smoke'), self.root / 'smoke/E1', gpu)
        if self.cli.mode == 'all':
            self.models(self.root / 'E1', gpu, smoke=False)
            self.stage('E1_report', self.worker('E1-report', self.root / 'E1'), self.root / 'E1', gpu)

    def models(self, root, gpu, smoke):
        for t in ('T1', 'T2'):
            for v in ('C', 'D'):
                model = root / t / v
                flags = ['--recipe', t, '--variant', v]+(['--smoke'] if smoke else [])
                prefix = ('smoke_' if smoke else '')+t+'_'+v
                self.stage(prefix+'_train', self.worker('train', model, *flags), model, gpu)
                selected = read(model / ('smoke_selected.json' if smoke else 'selected.json'))
                destination = model / 'final_evaluation'
                self.stage(prefix+'_evaluate', self.worker('evaluate', model, *flags,
                    '--checkpoint', selected['checkpoint'], '--destination', destination), destination, gpu)
                self.stage(prefix+'_targets', self.worker('targets', model, *flags,
                    '--destination', destination), model, gpu)


def run_branches(mode, e0, e1):
    errors = {}
    if mode == 'parallel':
        # Waiting for both futures preserves an independent E1 after an E0 technical failure.
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {pool.submit(fn): name for name, fn in (('E0', e0), ('E1', e1))}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as error:
                    errors[futures[future]] = repr(error)
    else:
        for name, fn in (('E0', e0), ('E1', e1)):
            try:
                fn()
            except Exception as error:
                errors[name] = repr(error)
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--original-root', type=Path, required=True)
    parser.add_argument('--train-root', type=Path, required=True)
    parser.add_argument('--e0-existing-root', type=Path)
    parser.add_argument('--run-mode', choices=('serial', 'parallel'), required=True)
    parser.add_argument('--mode', choices=('all', 'smoke', 'self-test'), default='all')
    parser.add_argument('--gpu-e0', required=True)
    parser.add_argument('--gpu-e1', required=True)
    parser.add_argument('--resume', choices=('0', '1'), required=True)
    cli = parser.parse_args()
    server_only()
    own_sources = sources()
    from preflight import validate_paths, resources
    cli.output_root, cli.original_root, cli.train_root = validate_paths(
        cli.output_root, cli.original_root, cli.train_root, cli.e0_existing_root)
    if cli.run_mode == 'parallel' and cli.gpu_e0 == cli.gpu_e1:
        raise ValueError('Parallel mode requires distinct physical GPUs')
    if cli.run_mode == 'serial' and cli.gpu_e0 != cli.gpu_e1:
        raise ValueError('Serial mode uses one physical GPU')
    root = cli.output_root
    root.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        fds = [stack.enter_context(lock(root / '.controller.lock'))]
        for gpu in sorted({cli.gpu_e0, cli.gpu_e1}, key=int):
            gpu_env(gpu)
            fds.append(stack.enter_context(lock(Path('/data/cjm/datasets/logs/.gspr-e01-leases') / ('gpu'+gpu+'.lock'))))
        os.environ['E01_LEASE_FDS'] = ','.join(map(str, fds))
        request = dict(family='GSPR_E01', source=own_sources, settings=SETTINGS,
            output=str(root), original=str(cli.original_root), train=str(cli.train_root),
            E0_existing=str(cli.e0_existing_root) if cli.e0_existing_root else None,
            run_mode=cli.run_mode, mode=cli.mode, E0_gpu=cli.gpu_e0, E1_gpu=cli.gpu_e1,
            executable=sys.executable, PYTHONHASHSEED=os.environ.get('PYTHONHASHSEED'),
            threads={k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')})
        request['identity'] = identity(request)
        marker = root / 'request.json'
        if marker.exists():
            if cli.resume != '1' or read(marker) != request:
                raise ValueError('RESUME=1 and identical complete request required; nothing overwritten')
        elif any(p.name not in ('logs', '.controller.lock') for p in root.iterdir()):
            raise ValueError('Nonempty unowned output; nothing overwritten')
        atomic(marker, request, immutable=True)
        control = Controller(cli)
        control.request_id = request['identity']
        # Recheck real resource/input identities on every launch, before reusing stage receipts.
        probes = {}
        for name, gpu in (('E0', cli.gpu_e0), ('E1', cli.gpu_e1)):
            path = root / 'preflight' / (name+'_gpu.json')
            run(control.worker('probe', root / 'preflight', '--destination', path),
                root / 'logs' / (name+'_probe.log'), env=gpu_env(gpu))
            probes[name] = read(path)
        resource = resources(root, cli.run_mode, probes)
        run(control.worker('preflight', root / 'preflight'), root / 'logs/preflight.log', env=gpu_env(cli.gpu_e1))
        atomic(root / 'preflight/resources.json', resource)
        for p in (root / '.stages').glob('*/complete.json'):
            control.verify_stage(p.parent.name)
        errors = run_branches(cli.run_mode, control.e0, control.e1)
        if errors:
            atomic(root / 'failure.json', dict(status='technical_failure', branches=errors,
                independently_finished_branches_preserved=True))
            raise RuntimeError('One or more independent branches failed: '+repr(errors))
        if cli.mode == 'all':
            control.stage('synthesis', control.worker('synthesis', root), root / 'reports', cli.gpu_e1)
        atomic(root / ('completion.json' if cli.mode == 'all' else cli.mode+'_completion.json'),
            dict(status='complete', request=request['identity'], mode=cli.mode,
                scientific_completion=cli.mode == 'all',
                stages={str(p.relative_to(root)): sha(p) for p in (root / '.stages').glob('*/complete.json')},
                final_report_sha256=sha(root / 'reports/E01_FINAL_CONCLUSIONS.md') if cli.mode == 'all' else None))
        print('COMPLETE mode='+cli.mode+' output='+str(root), flush=True)


if __name__ == '__main__':
    def interrupt(signum, frame):
        from e01_common import STOP
        STOP.set()
        raise KeyboardInterrupt('Signal '+str(signum))
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupt)
    main()
