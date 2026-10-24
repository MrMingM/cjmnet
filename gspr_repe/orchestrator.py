"""One launcher, serial subprocesses, verified stage journals and dependency identities."""
import argparse
from datetime import datetime, timezone
import os
import signal
from pathlib import Path
import subprocess
import sys
import time
import traceback

from auditlib import (CONFIG, DEFAULTS, HERE, atomic_json, bound_json, complete, identity,
                      gpu, lock, no_test, paths, read_json, server_only, sha256, sources)
from results import VARIANTS


def now():
    return datetime.now(timezone.utc).isoformat()


def verify_stage(path, request):
    row = complete(path / 'complete.json')
    if row['request_identity'] != identity(request):
        raise ValueError('Stage input/source/config/dependency changed: ' + str(path))
    for name, digest in row['artifacts'].items():
        if sha256(no_test(name)) != digest:
            raise ValueError('Completed stage artifact changed: ' + name)
    if row['stage'] == 'preflight':
        for name, digest in read_json(Path(row['output']) / 'external_inputs.json').items():
            if sha256(no_test(name)) != digest:
                raise ValueError('External input changed since preflight: ' + name)
        for name, expected in read_json(Path(row['output']) / 'input_trees.json').items():
            folder = no_test(name)
            actual = dict(files=sorted(p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()),
                directories=sorted(p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_dir()))
            if actual != expected:
                raise ValueError('Input file/directory inventory changed since preflight: ' + name)
    return row


def command(args, action, output, variant='B4', **flags):
    cmd = [sys.executable, '-B', str(HERE / 'main.py'), action, '--variant', variant, '--resume']
    for key in DEFAULTS:
        cmd += ['--' + key.replace('_', '-'), str(output if key == 'output_root' else getattr(args, key))]
    for key, value in flags.items():
        if value is True:
            cmd.append('--' + key.replace('_', '-'))
        elif value is not False and value is not None:
            cmd += ['--' + key.replace('_', '-'), str(value)]
    return cmd


class Pipeline:
    def __init__(self, args, run):
        self.args, self.run = args, run
        self.root = Path(args.output_root)
        self.previous = None

    def stage(self, name, output, cmd=None, callback=None):
        directory = self.root / 'stages' / name
        directory.mkdir(parents=True, exist_ok=True)
        request = dict(stage=name, run_identity=self.run['identity'], command=cmd,
            previous_complete=self.previous)
        try:
            if (directory / 'complete.json').exists():
                verify_stage(directory, request)
                print('REUSE COMPLETE ' + name, flush=True)
                self.previous = sha256(directory / 'complete.json')
                return
            bound_json(directory / 'request.json', request)
        except BaseException as error:
            failure = dict(stage=name, status='technical_failure', error=repr(error),
                traceback=traceback.format_exc(), duration_seconds=0, finished_utc=now(),
                request_identity=identity(request), output=str(output))
            atomic_json(directory / 'failure.json', failure)
            atomic_json(directory / 'stage_state.json', failure)
            atomic_json(self.root / 'current_stage.json', failure)
            raise
        started = time.monotonic()
        state = dict(stage=name, status='running', pid=os.getpid(), started_utc=now(),
            request_identity=identity(request), output=str(output))
        atomic_json(directory / 'stage_state.json', state)
        atomic_json(self.root / 'current_stage.json', state)
        print('START ' + name + ' -> ' + str(output), flush=True)
        process = None
        try:
            with (self.root / 'logs' / (name + '.log')).open('a', encoding='utf-8', newline='\n') as log:
                if cmd:
                    process = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', bufsize=1,
                        start_new_session=True)
                    state['child_pid'] = process.pid
                    atomic_json(directory / 'stage_state.json', state)
                    atomic_json(self.root / 'current_stage.json', state)
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        print(line, end='', flush=True)
                    if process.wait() != 0:
                        raise RuntimeError(f'{name} subprocess exit={process.returncode}; see {log.name}')
                extra_artifacts = callback() if callback else None
            artifacts = {str(p): sha256(p) for p in Path(output).rglob('*') if p.is_file()
                and not p.name.endswith('.tmp') and p.suffix not in ('.log', '.lock')
                and p.name not in ('failure.json', 'progress.json') and 'tensorboard' not in p.parts}
            if not artifacts:
                raise ValueError('Stage produced no verifiable artifact')
            if isinstance(extra_artifacts, list):
                artifacts.update({str(p): sha256(p) for p in extra_artifacts})
            state.update(status='complete', finished_utc=now(), duration_seconds=time.monotonic()-started)
            atomic_json(directory / 'stage_state.json', state)
            atomic_json(directory / 'complete.json', dict(**state, artifacts=artifacts))
            self.previous = sha256(directory / 'complete.json')
            print('COMPLETE ' + name, flush=True)
        except BaseException as error:
            if process is not None and sys.platform == 'linux':
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=30)
                except ProcessLookupError:
                    pass
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            state.update(status='technical_failure', finished_utc=now(), duration_seconds=time.monotonic()-started,
                error=repr(error), traceback=traceback.format_exc())
            atomic_json(directory / 'stage_state.json', state)
            atomic_json(directory / 'failure.json', state)
            atomic_json(self.root / 'current_stage.json', state)
            raise


def workflow(args, run):
    from results import final, stage1
    root = Path(args.output_root)
    pipeline = Pipeline(args, run)
    pipeline.stage('preflight', root/'preflight', command(args, 'preflight', root/'preflight'))
    if args.mode == 'preflight':
        return 0
    pipeline.stage('self-test', root/'self_test', command(args, 'self-test', root/'self_test'))
    for variant in VARIANTS:
        dest = root/'smoke'/variant
        pipeline.stage('smoke_'+variant, dest, command(args, 'train', dest, variant, smoke=True))
        weight = complete(dest/'smoke_checkpoint.json')['checkpoint']
        evaluation = dest/'evaluation_smoke12'
        pipeline.stage('smoke_eval_'+variant, evaluation, command(args, 'evaluate', dest, variant,
            checkpoint=weight, evaluation_root=evaluation, smoke=True))
    pipeline.stage('stage1_acceptance', root/'stage1', callback=lambda: stage1(root))
    if args.mode == 'smoke':
        return 0
    from evaluation import selection
    for variant in VARIANTS:
        dest = root/variant
        pipeline.stage(variant+'_train', dest, command(args, 'train', dest, variant))
        evaluation = dest/'final_evaluation'
        pipeline.stage(variant+'_evaluate', evaluation, command(args, 'evaluate', dest, variant,
            checkpoint=selection(dest)['checkpoint'], evaluation_root=evaluation))
    pipeline.stage('final', root/'reports', callback=lambda: final(root))
    atomic_json(root/'current_stage.json', dict(status='complete', stage='final'))
    return 0

def main():
    server_only()
    def interrupted(signum, frame):
        raise InterruptedError('Pipeline received signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('all', 'preflight', 'smoke'), default='all')
    parser.add_argument('--resume', action='store_true')
    for key, default in DEFAULTS.items():
        parser.add_argument('--' + key.replace('_', '-'), default=os.environ.get(key.upper(), default))
    args = parser.parse_args()
    # Accept TRAIN_ROOT as either extraction root or its accepted train/ subdirectory.
    train = Path(args.train_root)
    if train.name == 'train' and (train.parent / 'manifest.json').is_file():
        args.train_root = str(train.parent)
    paths(args)
    root = Path(args.output_root)
    existed = root.exists() and any(root.iterdir())
    if existed and (not args.resume or not ((root / 'run_identity.json').is_file() or (root / 'owner.json').is_file())):
        raise ValueError('Nonempty output requires RESUME=1 and this experiment\'s existing run_identity.json; refusing unrelated/historical output')
    root.mkdir(parents=True, exist_ok=True)
    bound_json(root / 'owner.json', dict(experiment='GSPR_REPE',
        backend=os.environ.get('BACKEND','rocm'), roots={k:getattr(args,k) for k in DEFAULTS},
        CUDA_INPUT_ROOT=os.environ.get('CUDA_INPUT_ROOT'),sources=sources()))
    (root / 'logs').mkdir(exist_ok=True)
    (root / 'reports').mkdir(exist_ok=True)
    try:
        gpu()
        run = dict(sources=sources(), settings=CONFIG,
            roots={k: getattr(args, k) for k in DEFAULTS},
            train_manifest_sha256=sha256(Path(args.train_root) / 'manifest.json'),
            backend=os.environ.get('BACKEND','rocm'), CUDA_INPUT_ROOT=os.environ.get('CUDA_INPUT_ROOT'),
            physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES',os.environ.get('ROCR_VISIBLE_DEVICES','platform default')), PYTHONHASHSEED=os.environ.get('PYTHONHASHSEED'))
        from importlib import metadata
        if os.environ.get('BACKEND') == 'cuda':
            from cuda_runtime import VERSIONS
            evidence_versions = VERSIONS
        else:
            evidence_versions = read_json(Path(args.d1a_root) / 'protocol.json')['versions']
        run['environment'] = dict(python=sys.version,
            versions={name: metadata.version(name) for name in evidence_versions},
            threads={k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')})
        run['identity'] = identity(run)
        with lock(root / '.pipeline.lock'):
            bound_json(root / 'run_identity.json', run)
            return workflow(args, run)
    except BaseException as error:
        atomic_json(root / 'STOP.json', dict(status='TECHNICAL_FAILURE', technical_failure=True,
            error=repr(error), traceback=traceback.format_exc(), stopped_utc=now()))
        raise


if __name__ == '__main__':
    sys.exit(main())
