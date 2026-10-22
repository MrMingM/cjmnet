"""Sequential inference-only controller. Every GPU stage uses a fresh worker."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from v2xr_author_fog_eval.core import (DEFAULTS, HERE, ROOT, bind, checkpoint_file, completed,
    identity, lock, no_test, read, reference_source, sha256, verify_sources, write)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--mode', choices=('self-test', 'smoke', 'all'), default='all')
    result.add_argument('--resume', type=int, choices=(0, 1), default=0)
    result.add_argument('--gpu', required=True)
    for key, value in DEFAULTS.items():
        result.add_argument('--' + key.replace('_', '-'), default=value)
    return result


def check_paths(args, require_checkpoint=True):
    if sys.platform != 'linux':
        raise RuntimeError('Real evaluation stages run on the Linux server only')
    if not args.gpu.isdecimal() or not 0 <= int(args.gpu) <= 63:
        raise ValueError('GPU must name one physical GPU number, 0..63')
    if os.environ.get('ROCR_VISIBLE_DEVICES') != args.gpu or os.environ.get('PYTHONHASHSEED') != '20261016':
        raise ValueError('Use run_server.sh to select the physical GPU and frozen Python hash seed')
    if any(os.environ.get(k) is not None for k in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES')):
        raise ValueError('Only ROCR_VISIBLE_DEVICES may be set')
    for key in DEFAULTS:
        setattr(args, key, str(no_test(getattr(args, key)).resolve()))
    output = Path(args.output_root)
    if Path('/data') not in output.parents:
        raise ValueError('OUTPUT_ROOT must be a new independent directory on /data')
    sources = [ROOT, Path(args.checkpoint), *[Path(getattr(args, k)) for k in DEFAULTS
                if k not in ('output_root', 'checkpoint')]]
    for source in sources:
        if output == source or output in source.parents or source in output.parents:
            raise ValueError('Output overlaps source/history: ' + str(source))
    temporary = no_test(os.environ.get('TMPDIR', '')).resolve()
    if Path('/data') not in temporary.parents or output not in temporary.parents:
        raise ValueError('TMPDIR must be a child of OUTPUT_ROOT on /data')
    if require_checkpoint:
        checkpoint_file(args.checkpoint)
    return output


def run_stage(root, stage, lock_fd):
    log = root / 'logs' / (stage + '.log')
    log.parent.mkdir(exist_ok=True)
    command = [sys.executable, '-B', '-u', '-m', 'v2xr_author_fog_eval.worker',
               '--root', str(root), '--stage', stage]
    print('START', stage, 'log=' + str(log), flush=True)
    started = time.monotonic()
    # Append on retry so the earlier failure remains inspectable.
    with log.open('a', encoding='utf-8') as stream:
        stream.write('\nATTEMPT ' + time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()) + '\n')
        stream.flush()
        process = subprocess.Popen(command, cwd=str(ROOT), stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True, pass_fds=(lock_fd,))
        previous = {}

        def interrupted(number, frame):
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise KeyboardInterrupt('Controller interrupted; worker process group stopped')

        for number in (signal.SIGTERM, signal.SIGINT):
            previous[number] = signal.signal(number, interrupted)
        try:
            while True:
                try:
                    code = process.wait(timeout=20)
                    break
                except subprocess.TimeoutExpired:
                    print(f'RUNNING {stage} elapsed={int(time.monotonic()-started)}s log={log}', flush=True)
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)
    print('END', stage, 'returncode=' + str(code), flush=True)
    if code:
        write(root / 'failure.json', dict(status='failed', stage=stage, command=command,
            returncode=code, log=str(log), run_identity=read(root / 'run_identity.json')['identity']))
        raise RuntimeError('Stage failed: ' + stage + '\n' + log.read_text(encoding='utf-8')[-5000:])


def workflow(mode):
    if mode == 'self-test':
        return ('self-test',)
    common = ('preflight', 'checkpoint-audit', 'smoke')
    return common if mode == 'smoke' else common + ('author-evaluation', 'report')


def main():
    args = parser().parse_args()
    if args.mode == 'self-test':
        from v2xr_author_fog_eval.tests import run_tests
        verify_sources()
        if sys.platform == 'linux':
            check_paths(args, require_checkpoint=False)
            Path(os.environ['TMPDIR']).mkdir(parents=True, exist_ok=True)
        run_tests()
        return
    root = check_paths(args)
    root.mkdir(parents=True, exist_ok=True)
    with lock(root / '.controller.lock') as lock_fd:
        sources = verify_sources()
        reference_source(args.cd_results)
        arguments = {k: getattr(args, k) for k in DEFAULTS}
        record = dict(kind='external_fixed_author_checkpoint_validation_v1', arguments=arguments,
            checkpoint_sha256=sha256(args.checkpoint), physical_gpu=args.gpu,
            python_executable=str(Path(sys.executable).resolve()),
            PYTHONHASHSEED='20261016', sources=sources)
        record['identity'] = identity(record)
        marker = root / 'run_identity.json'
        if marker.exists():
            if not args.resume:
                raise ValueError('Existing run requires RESUME=1; no overwrite is allowed')
            bind(marker, record)
        else:
            unexpected = set(p.name for p in root.iterdir()) - {'.controller.lock', 'logs', 'tmp', 'launcher.log'}
            if unexpected:
                raise ValueError('Unidentified nonempty output directory: ' + repr(sorted(unexpected)))
            bind(marker, record)
        if (root / 'contract.json').exists():
            raise ValueError('External fixed checkpoint output must not contain a local training contract')
        Path(os.environ['TMPDIR']).mkdir(parents=True, exist_ok=True)
        write(root / 'completion.json', dict(status='running', mode=args.mode,
            full_evaluation=False, run_identity=record['identity']))
        try:
            for stage in workflow(args.mode):
                run_stage(root, stage, lock_fd)
            if args.mode == 'all':
                completed(root / 'author_evaluation/summary.json')
                completed(root / 'reports/completion.json')
        except BaseException as error:
            write(root / 'completion.json', dict(status='failed', mode=args.mode,
                full_evaluation=False, run_identity=record['identity'], error=repr(error)))
            raise
        write(root / 'completion.json', dict(status='complete', mode=args.mode,
            full_evaluation=args.mode == 'all', run_identity=record['identity'],
            output_hashes={n: sha256(root / n) for n in (
                ('preflight.json', 'checkpoint_audit.json', 'smoke/summary.json') if args.mode == 'smoke' else
                ('preflight.json', 'checkpoint_audit.json', 'smoke/summary.json',
                 'author_evaluation/summary.json', 'reports/completion.json'))}))
        if (root / 'failure.json').exists():
            old = read(root / 'failure.json')
            old.update(resolved=True, resolution='Requested workflow completed; earlier failure retained for audit')
            write(root / 'failure.json', old)
        print('COMPLETE', args.mode, 'full_evaluation=' + str(args.mode == 'all'), flush=True)


if __name__ == '__main__':
    main()
