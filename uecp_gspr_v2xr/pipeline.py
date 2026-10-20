"""Serial stages with kernel locks, explicit commits, PID identity, isolated failure/recovery."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from uecp_gspr_v2xr.common import (HERE, ROOT, bound, complete, lock, read,
    server, settings, sha, sources, variant_path, write)
from uecp_gspr_v2xr.state_logic import valid_stage


class Pipeline:
    def __init__(self, root):
        self.root = root; self.child = None; self.stopping = False; self.current = None
        self.path = root / 'pipeline_status.json'
        self.status = read(self.path) if self.path.exists() else {}
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self.stop)

    def stop(self, sig, frame):
        self.stopping = True
        if self.child is not None and self.child.poll() is None:
            os.killpg(self.child.pid, sig)

    def run(self, name, action, artifacts, *extra, evaluation=False, expected_code=None):
        if self.stopping:
            raise InterruptedError('Stop requested')
        self.current = name
        old = self.status.get(name, dict(status='PENDING'))
        if valid_stage(old):
            print('REUSE '+name, flush=True)
            return
        cmd = [sys.executable, '-B', str(HERE / 'worker.py'), action, '--output', str(self.root), *map(str, extra)]
        env = dict(os.environ)
        if evaluation:
            env['PYTHONHASHSEED'] = '20261016'
        log = self.root / 'logs' / (name+'.log'); log.parent.mkdir(parents=True, exist_ok=True)
        self.status[name] = dict(status='RUNNING', started=time.time(), command=cmd,
            log=str(log), attempts=old.get('attempts', []) + ([old] if old.get('status') != 'PENDING' else []))
        write(self.path, self.status)
        write(self.root / 'current_stage.json', dict(stage=name, command=cmd, log=str(log), status='RUNNING'))
        print('START '+name+' log='+str(log), flush=True)
        with log.open('ab', buffering=0) as stream:
            self.child = subprocess.Popen(cmd, env=env, cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            while True:
                try:
                    code = self.child.wait(timeout=20); break
                except subprocess.TimeoutExpired:
                    print('RUNNING '+name+' log='+str(log), flush=True)
                    if self.stopping:
                        try:
                            code = self.child.wait(timeout=10); break
                        except subprocess.TimeoutExpired:
                            os.killpg(self.child.pid, signal.SIGKILL)
        self.child = None
        if code == 0:
            try:
                digests = {str(p): sha(p) for p in artifacts}
                for p in artifacts:
                    if p.name.endswith('complete.json'):
                        complete(p)
            except BaseException:
                code = 99
                print('Missing or corrupt committed stage outputs', flush=True)
        else:
            digests = {}
        self.status[name].update(status='COMPLETE' if code == 0 else 'SKIPPED' if code == expected_code else 'FAILED',
            ended=time.time(), returncode=code, artifacts=digests)
        write(self.path, self.status)
        if expected_code is not None and code == expected_code:
            self.status[name]['reason'] = 'Expected smoke fault injection; recovery stage follows'
            write(self.path, self.status)
            return
        if code:
            failure = dict(stage=name, command=cmd, returncode=code, time=time.time(), log=str(log),
                traceback=log.read_text(encoding='utf-8', errors='replace')[-12000:])
            write(self.root / 'failure.json', failure)
            write(self.root / 'failure_history' / (name+'_'+str(time.time_ns())+'.json'), failure)
            raise RuntimeError('Stage failed: '+name+'; see '+str(log))
        if self.stopping:
            raise InterruptedError('Stopped after latest committed optimizer/epoch boundary')

    def detector(self, variant, smoke=False):
        branch = variant_path(self.root, variant, smoke)
        tag = ('smoke_' if smoke else '')+variant
        extra = ['--variant', variant]+(['--smoke'] if smoke else [])
        n = 2 if smoke else read(self.root / 'config.json')['epochs']
        for e in range(1, n+1):
            ep = branch / 'epochs' / f'{e:02d}'
            if smoke and variant == 'E0' and e == 1 and not (ep / 'expected_train_interruption.json').exists() and not (ep / 'train_complete.json').exists():
                self.run('smoke_training_expected_interrupt', 'train', [], *extra, '--epoch', e,
                    '--interrupt-train-once', expected_code=44)
            self.run(f'{tag}_train_{e:02d}', 'train', [ep / 'train_complete.json'], *extra, '--epoch', e)
            self.run(f'{tag}_dev_{e:02d}', 'dev', [ep / 'dev_complete.json'], *extra, '--epoch', e, evaluation=True)
        self.run(tag+'_select', 'select', [branch / 'selected.json'], *extra)
        if smoke and variant == 'E0':
            marker = branch / 'final_evaluation/expected_interruption.json'
            if not marker.exists():
                self.run('smoke_prediction_expected_interrupt', 'final', [], *extra, '--interrupt-once', evaluation=True, expected_code=43)
        self.run(tag+'_final', 'final', [branch / 'final_evaluation/complete.json'], *extra, evaluation=True)

    def smoke(self, c):
        # All nine variants are trained independently, using tiny real frame subsets only.
        for v in c['variants']:
            self.detector(v, smoke=True)
        self.run('smoke_E1', 'diagnose', [self.root / 'smoke/E1/quality_diagnosis/complete.json'], '--smoke')
        # Replay epoch2 from exactly the committed epoch1 optimizer/RNG state.
        for v in ('E0', 'E2', 'E3_joint'):
            original = variant_path(self.root, v, True)
            replay = self.root / 'smoke_replay' / v
            replay.mkdir(parents=True, exist_ok=True)
            for name in ('initialization.json', 'config.json', 'parameters.json'):
                target = replay / name
                if not target.exists():
                    os.link(original / name, target)
                elif sha(original / name) != sha(target):
                    raise ValueError('Smoke recovery source changed')
            if v == 'E0':
                self.run('smoke_E0_fresh_epoch1', 'train', [replay / 'epochs/01/train_complete.json'],
                    '--variant', v, '--epoch', 1, '--smoke', '--replay')
                self.run('smoke_E0_mid_epoch_resume_check', 'resume-check', [self.root / 'smoke/E0_epoch1_resume_complete.json'],
                    '--variant', v, '--epoch', 1)
            else:
                target = replay / 'epochs/01'; target.mkdir(parents=True, exist_ok=True)
                for p in (original / 'epochs/01').iterdir():
                    if p.is_file() and not (target / p.name).exists():
                        os.link(p, target / p.name)
            self.run('smoke_'+v+'_resume_replay', 'train', [replay / 'epochs/02/train_complete.json'],
                '--variant', v, '--epoch', 2, '--smoke', '--replay')
            self.run('smoke_'+v+'_resume_check', 'resume-check', [self.root / 'smoke' / (v+'_epoch2_resume_complete.json')],
                '--variant', v, '--epoch', 2)
        self.run('smoke_report', 'report', [self.root / 'smoke/FINAL_RESULTS.json'], '--smoke')


def main():
    server(); p = argparse.ArgumentParser(); p.add_argument('--output', required=True)
    args = p.parse_args()
    from uecp_gspr_v2xr.paths import check
    root = check(args.output); c = settings(); mode = os.environ.get('MODE', 'all')
    if mode not in ('all', 'preflight', 'self-test', 'smoke', 'report'):
        raise ValueError('Unknown MODE')
    if os.environ.get('PYTHONHASHSEED') != str(c['seed']):
        raise ValueError('Set PYTHONHASHSEED before interpreter startup')
    request = dict(family='uecp_gspr_v2xr', config=c, sources=sources(),
        gpu=os.environ.get('GPU'), paths={k: os.environ.get(k) for k in
            ('PROJECT_ROOT', 'TRAIN_ROOT', 'VALIDATE_ROOT', 'PAIRED_VALIDATE_ROOT', 'RUNTIME_ROOT', 'MODEL_DIR',
             'D1A_ROOT', 'GT_AUDIT_ROOT', 'D1C_ROOT', 'PYTHON', 'QUALITY_LABEL_MANIFEST')})
    root.mkdir(parents=True, exist_ok=True)
    with lock(root / '.controller.lock'):
        marker = root / 'request.json'
        if marker.exists():
            if os.environ.get('RESUME') != '1' and mode != 'report':
                raise ValueError('Existing output requires RESUME=1')
            if read(marker) != request:
                raise ValueError('Resume source/config/data/GPU paths changed; use a new output')
        else:
            if mode == 'report':
                raise ValueError('No prior run to report')
            bound(marker, request)
        bound(root / 'config.json', c)
        protocol = read(HERE / 'protocol.json')
        protocol['effective_settings'] = c
        protocol['training'].update(seed=c['seed'], epochs=c['epochs'], microbatch=c['microbatch'])
        protocol['scene_split']['seed'] = c['seed']
        bound(root / 'protocol.json', protocol)
        write(root / 'controller.json', dict(pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().split()[21],
            output=str(root), controller=str(Path(__file__).resolve()), executable=sys.executable))
        pipeline = Pipeline(root)
        try:
            if mode == 'report':
                from uecp_gspr_v2xr.report import run
                run(root); return
            pipeline.run('preflight', 'preflight', [root / n for n in ('contract.json', 'data_manifest.json',
                'source_snapshot.json', 'preflight/data_check.json', 'preflight/input_stamps.json',
                'preflight/evidence.json', 'preflight/train_manifest.json', 'preflight/gspr_source_config.json')])
            from uecp_gspr_v2xr.runtime import input_stamps
            input_stamps(root)
            if mode == 'preflight':
                return
            pipeline.run('server_self_test', 'self-test', [root / 'self_test/complete.json'])
            if mode == 'self-test':
                return
            pipeline.smoke(c)
            if mode == 'smoke':
                return
            pipeline.run('accepted_training_content', 'content', [root / 'preflight/training_content.json'])
            for v in c['variants']:
                for e in range(1, c['epochs']+1):
                    for part in ('train', 'dev'):
                        pipeline.status.setdefault(f'{v}_{part}_{e:02d}', dict(status='PENDING'))
            write(pipeline.path, pipeline.status)
            pipeline.detector('E0')
            pipeline.run('E1_quality_diagnosis', 'diagnose', [root / 'E1/quality_diagnosis/complete.json'])
            for v in c['variants'][1:]:
                pipeline.detector(v)
            pipeline.run('final_report', 'report', [root / 'FINAL_RESULTS.json', root / 'FINAL_RESULTS.md', root / 'ABLATION_RESULTS.md'])
            bound(root / 'complete.json', dict(status='COMPLETE', contract_identity=read(root / 'contract.json')['identity'],
                final_report_sha256=sha(root / 'FINAL_RESULTS.json')))
        except BaseException:
            write(root / 'controller_failure.json', dict(stage=pipeline.current, traceback=traceback.format_exc(), time=time.time()))
            from uecp_gspr_v2xr.report import run
            try:
                run(root, partial=True)
            except Exception:
                traceback.print_exc()
            raise
        finally:
            if pipeline.child is not None and pipeline.child.poll() is None:
                os.killpg(pipeline.child.pid, signal.SIGTERM)
                try:
                    pipeline.child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(pipeline.child.pid, signal.SIGKILL); pipeline.child.wait()


if __name__ == '__main__':
    main()
