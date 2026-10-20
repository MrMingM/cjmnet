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


def command(args, action, output, variant='C', **flags):
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


def checkpoint(root, variant, smoke=False):
    if variant.endswith('_BN'):
        return complete(root / 'artifact.json')['checkpoint']
    if smoke:
        return complete(root / 'smoke_checkpoint.json')['checkpoint']
    from evaluation import selection
    return selection(root)['checkpoint']


def gate_l0(root):
    from gates import l0
    from results import ap70, result
    summary = result(root, 'C')
    gate = l0(ap70(summary)['fog'])
    gate.update(summary_sha256=sha256(root / VARIANTS['C'] / 'final_evaluation/summary.json'))
    atomic_json(root / 'reports/L0_GATE.json', gate)
    if not gate['passed']:
        diagnosis(root, gate)


def diagnosis(root, gate):
    history = complete(root / VARIANTS['C'] / 'training_summary.json')
    rows = read_json(root / VARIANTS['C'] / 'history.json')['epochs']
    summary = read_json(root / VARIANTS['C'] / 'final_evaluation/summary.json')
    data = dict(status='performance_gate_not_met', gate=gate, confirmed_program_errors=[],
        technical_checks='preflight, self-test, smoke, full training commits, strict loading, frozen GT and exact O replay all passed',
        training=history, curves=rows, results=summary['results'],
        data=read_json(root / 'preflight/summary.json'),
        protocol=read_json(root / VARIANTS['C'] / 'contract.json'),
        evaluation_protocol_sha256=sha256(root / VARIANTS['C'] / 'final_evaluation/protocol.json'),
        interpretation='Failure to reach +5 pp alone does not confirm a pipeline bug. No formal L1/L2 started; isolated smoke checks already completed.')
    atomic_json(root / 'reports/L0_DIAGNOSIS.json', data)
    (root / 'reports/L0_DIAGNOSIS.md').write_text(
        '# L0 诊断\n\nFog 增益：' + str(gate['gain_pp']) + ' pp；预设门槛：+5 pp。\n\n'
        '本次性能门槛未达到。已确认的程序错误：无；这不能证明管线绝对无误。\n'
        '已通过数据配对/哈希、实际 Clean/Fog 读取、原始增强与GT过滤、损失工厂、模型严格加载、checkpoint身份和O原版回放检查。\n'
        '逐轮loss/Clean AP、实际路径与来源比例、GT/None统计见同名JSON及L0_C/epochs/*/sampling.jsonl。\n'
        '应结合这些证据诊断，不根据 Fog/Snow 改选 epoch 或调整本次门槛。\n', encoding='utf-8', newline='\n')


def gate_l1(root):
    from gates import l1
    from results import ap70, pairing, result
    gate = l1(ap70(result(root, 'C')), ap70(result(root, 'D')))
    gate['pairing'] = pairing(root)
    atomic_json(root / 'reports/L1_GATE.json', gate)
    (root / 'reports/L1_COMPARISON.md').write_text(
        '# D 与 C 的匹配对照\n\n三天气官方 O AP70 差值（pp）：`' + str(gate['gains_pp']) + '`\n\n'
        '门槛：`' + gate['status'] + '`。训练/选模/评测完整有效，继续 L2。\n\n'
        '采样与增强配对记录：`' + str(gate['pairing']) + '`\n', encoding='utf-8', newline='\n')


def workflow(args, run):
    from results import final
    root = Path(args.output_root)
    pipeline = Pipeline(args, run)
    pipeline.stage('preflight', root / 'preflight', command(args, 'preflight', root / 'preflight'))
    if args.mode == 'preflight':
        return
    pipeline.stage('self-test', root / 'self_test', command(args, 'self-test', root / 'self_test'))
    if args.mode == 'self-test':
        return
    for variant in ('C', 'D'):
        dest = root / 'smoke' / variant
        pipeline.stage('smoke_' + variant, dest, command(args, 'train', dest, variant, smoke=True))
    for method in ('BN', 'FiLM'):
        for base in ('C', 'D'):
            variant = base + '_' + method
            dest = root / 'smoke' / variant
            pipeline.stage('smoke_' + variant, dest, command(args, 'bn' if method == 'BN' else 'train',
                dest, variant, parent_root=root / 'smoke' / base, smoke=True))
            evaluation = dest / 'evaluation_smoke12'
            pipeline.stage('smoke_eval_' + variant, evaluation, command(args, 'evaluate', dest, variant,
                checkpoint=checkpoint(dest, variant, smoke=True), evaluation_root=evaluation, smoke=True))
    if args.mode == 'smoke':
        return
    for variant in ('C', 'D'):
        dest = root / VARIANTS[variant]
        pipeline.stage(variant + '_train', dest, command(args, 'train', dest, variant))
        evaluation = dest / 'final_evaluation'
        pipeline.stage(variant + '_evaluate', evaluation, command(args, 'evaluate', dest, variant,
            checkpoint=checkpoint(dest, variant), evaluation_root=evaluation))
        gate_dir = root / ('gate_L0' if variant == 'C' else 'gate_L1')
        gate_dir.mkdir(exist_ok=True)
        def gate_callback(v=variant, d=gate_dir):
            gate_l0(root) if v == 'C' else gate_l1(root)
            source = root / 'reports' / ('L0_GATE.json' if v == 'C' else 'L1_GATE.json')
            atomic_json(d / 'gate.json', read_json(source))
            paths = [source]
            if v == 'C' and not read_json(source)['passed']:
                paths += [root / 'reports/L0_DIAGNOSIS.json', root / 'reports/L0_DIAGNOSIS.md']
            if v == 'D':
                paths += [root / 'reports/L1_COMPARISON.md']
            return paths
        pipeline.stage('L0_gate' if variant == 'C' else 'L1_gate', gate_dir, callback=gate_callback)
        if variant == 'C' and not read_json(root / 'reports/L0_GATE.json')['passed']:
            atomic_json(root / 'STOP.json', dict(status='L0_GATE_FAILED', scientific_stop=True,
                technical_failure=False, stopped_utc=now()))
            final(root, partial=True)
            atomic_json(root / 'current_stage.json', dict(status='L0_GATE_FAILED', stage='L0_gate'))
            print('L0_GATE_FAILED: diagnosis and partial reports saved; L1/L2 not started.', flush=True)
            return 20
    for method in ('BN', 'FiLM'):
        for base in ('C', 'D'):
            variant = base + '_' + method
            dest = root / VARIANTS[variant]
            pipeline.stage(variant + '_adapt', dest, command(args, 'bn' if method == 'BN' else 'train',
                dest, variant, parent_root=root / VARIANTS[base]))
            evaluation = dest / 'final_evaluation'
            pipeline.stage(variant + '_evaluate', evaluation, command(args, 'evaluate', dest, variant,
                checkpoint=checkpoint(dest, variant), evaluation_root=evaluation))
    pipeline.stage('final', root / 'reports', callback=lambda: final(root))
    stop = root / 'STOP.json'
    if stop.exists():
        atomic_json(root / 'stop_history' / (sha256(stop) + '.json'), read_json(stop))
        stop.unlink()
    atomic_json(root / 'current_stage.json', dict(status='complete', stage='final', finished_utc=now()))
    print('ALL STAGES COMPLETE: ' + str(root / 'reports/FINAL_RESULTS.md'), flush=True)
    return 0


def main():
    server_only()
    def interrupted(signum, frame):
        raise InterruptedError('Pipeline received signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('all', 'preflight', 'self-test', 'smoke'), default='all')
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
    if existed and (not args.resume or not (root / 'run_identity.json').is_file()):
        raise ValueError('Nonempty output requires RESUME=1 and this experiment\'s existing run_identity.json; refusing unrelated/historical output')
    root.mkdir(parents=True, exist_ok=True)
    (root / 'logs').mkdir(exist_ok=True)
    (root / 'reports').mkdir(exist_ok=True)
    try:
        gpu()
        run = dict(sources=sources(), settings=CONFIG,
            roots={k: getattr(args, k) for k in DEFAULTS},
            train_manifest_sha256=sha256(Path(args.train_root) / 'manifest.json'),
            physical_gpu=os.environ.get('ROCR_VISIBLE_DEVICES'), PYTHONHASHSEED=os.environ.get('PYTHONHASHSEED'))
        from importlib import metadata
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
