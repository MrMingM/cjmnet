"""Serial process supervisor, kernel locks, atomic stages, explicit epoch recovery boundaries."""
import argparse
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from v2xr_raygen_e0e3.common import (HERE, ROOT, atomic_bytes, bound, identity, lock, read,
                                   safe, server, settings, sha, verify_sources, write)


def make_request(config, sources):
    return dict(family='v2xr_raygen_e0e3', config=config, sources=sources,
                paths={k: v for k, v in os.environ.items() if k in
                    ('PROJECT_ROOT', 'TRAIN_ROOT', 'VALIDATE_ROOT', 'PAIRED_VALIDATE_ROOT', 'RUNTIME_ROOT', 'MODEL_DIR',
                     'D1A_ROOT', 'GT_AUDIT_ROOT', 'D1C_ROOT', 'PYTHON')})


def inherit_successor_paths(root):
    marker = root / 'data_reuse.json'
    if not marker.exists():
        return
    record = read(marker)
    contract = read(root / 'contract.json')
    if (record.get('status') != 'complete'
            or record['identity'] != identity({k: v for k, v in record.items() if k != 'identity'})
            or contract['identity'] != identity({k: v for k, v in contract.items() if k != 'identity'})
            or record['target_contract_identity'] != contract['identity']
            or record['inventory_sha256'] != sha(root / 'data_reuse_inventory.json')):
        raise ValueError('Successor data provenance changed')
    # The initializer is a separate process; reconstruct its accepted optional paths in this controller too.
    for key, value in read(root / 'request.json')['paths'].items():
        if key in os.environ and os.environ[key] != value:
            raise ValueError('Successor input environment differs: ' + key)
        os.environ[key] = value


class Controller:
    def __init__(self, root):
        self.root = root
        self.child = None
        self.stopping = False
        self.status_path = root / 'stage_status.json'
        self.status = read(self.status_path) if self.status_path.exists() else {}
        self.stage_name = None
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self.stop)

    def stop(self, sig, frame):
        self.stopping = True
        if self.child is not None and self.child.poll() is None:
            os.killpg(self.child.pid, sig)

    def run(self, name, action, *extra, evaluate=False, oom_allowed=False, expected_codes=()):
        self.stage_name = name
        if self.stopping:
            raise InterruptedError('Controller stop requested')
        if self.status.get(name, {}).get('status') == 'complete':
            print('Reuse stage ' + name, flush=True)
            return 0
        cmd = [sys.executable, '-B', str(HERE / 'worker.py'), action, '--output', str(self.root), *map(str, extra)]
        env = dict(os.environ)
        if evaluate:
            env['PYTHONHASHSEED'] = '20261016'
        start = time.time()
        prior = self.status.get(name, {})
        attempts = prior.get('attempts', []) + ([{k: v for k, v in prior.items() if k != 'attempts'}]
                                               if prior.get('status') in ('failed', 'running') else [])
        self.status[name] = dict(status='running', started=start, command=cmd, attempts=attempts)
        write(self.status_path, self.status)
        log = self.root / 'logs' / (name + '.log')
        log.parent.mkdir(parents=True, exist_ok=True)
        write(self.root / 'current_stage.json', dict(stage=name, command=cmd, log=str(log), status='running'))
        print('START ' + name + ' log=' + str(log), flush=True)
        with log.open('ab', buffering=0) as stream:
            self.child = subprocess.Popen(cmd, env=env, cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            while True:
                try:
                    code = self.child.wait(timeout=20)
                    break
                except subprocess.TimeoutExpired:
                    if self.stopping:
                        # Bound shutdown waiting, kill only our own still-live process group.
                        try:
                            code = self.child.wait(timeout=10)
                            break
                        except subprocess.TimeoutExpired:
                            os.killpg(self.child.pid, signal.SIGKILL)
                    print(f'RUNNING {name} elapsed={time.time()-start:.0f}s log={log}', flush=True)
        self.child = None
        self.status[name].update(status='complete' if code == 0 else 'failed', ended=time.time(),
                                 seconds=time.time()-start, returncode=code)
        write(self.status_path, self.status)
        if code in expected_codes:
            self.status[name]['expected_fault'] = True
            write(self.status_path, self.status)
        if code and not (oom_allowed and code == 42) and code not in expected_codes:
            tail = log.read_text(encoding='utf-8', errors='replace')[-12000:]
            failure = dict(status='failed', stage=name, command=cmd, returncode=code,
                  group=next((str(extra[i+1]) for i in range(len(extra)-1) if extra[i] == '--group'), None),
                  epoch=next((str(extra[i+1]) for i in range(len(extra)-1) if extra[i] == '--epoch'), None),
                  traceback=tail, log=str(log), time=time.time())
            write(self.root / 'failure.json', failure)
            write(self.root / 'failure_history' / (name + '_' + str(time.time_ns()) + '.json'), failure)
            raise RuntimeError('Stage failed: ' + name + ' returncode=' + str(code) + '\n' + tail[-4000:])
        if self.stopping:
            raise InterruptedError('Stopped; resume from committed training/evaluation boundaries')
        print('END ' + name + ' returncode=' + str(code), flush=True)
        return code

    def smoke(self):
        self.run('smoke_prepare', 'smoke-prepare')
        self.run('stable_calculation_checks', 'stable-check')
        execution = self.root / 'execution.json'
        if not execution.exists():
            chosen = None
            fixed_micro = read(self.root / 'data_reuse.json')['provenance']['execution']['microbatch'] if (self.root / 'data_reuse.json').exists() else None
            for micro in ((fixed_micro,) if fixed_micro else (8, 4, 2, 1)):
                code = self.run('capacity_micro' + str(micro), 'probe', '--group', 'E3', '--epoch', '1',
                                '--micro', micro, oom_allowed=True)
                if code == 0:
                    chosen = micro
                    break
            if chosen is None:
                raise RuntimeError('E3 capacity failed; an imported microbatch is never silently changed')
            measured = read(self.root / 'probe' / ('micro' + str(chosen)) / 'E3/epochs/01/train_complete.json')
            bound(execution, dict(microbatch=chosen, accumulation=8 // chosen, effective_batch=8,
                amp=False, FP32=True, batchnorm_note='BN uses microbatch statistics; common microbatch for all groups',
                physical_gpu=os.environ.get('GPU'), gpu_name=measured['gpu_name'],
                memory_probe='E3, all maximum 8 windows on largest-size real frames; empirical check only'))
        bound(self.root / 'execution_commit.json', dict(sha256=sha(execution),
            parent_contract=read(self.root / 'contract.json')['identity']))
        from v2xr_raygen_e0e3.resume_compat import (GPU_CAPACITY, MARKER, commit_gpu_capacity,
                                                   effective_execution)
        if (self.root / MARKER).exists():
            repair = read(self.root / MARKER)
            target = repair.get('requested_physical_gpu', read(execution)['physical_gpu'])
            if target != read(execution)['physical_gpu']:
                if target != os.environ.get('GPU'):
                    raise ValueError('Keep the enrolled target GPU: ' + target)
                if not (self.root / GPU_CAPACITY).exists():
                    micro = read(execution)['microbatch']
                    print('Recheck capacity on GPU ' + target + '; keep microbatch=' + str(micro), flush=True)
                    self.run('capacity_gpu' + target, 'probe', '--group', 'E3', '--epoch', '1',
                             '--micro', micro, '--capacity-gpu', target)
                    commit_gpu_capacity(self.root)
        if effective_execution(self.root)['physical_gpu'] != os.environ.get('GPU'):
            raise ValueError('Selected physical GPU differs from frozen execution contract')
        self.run('server_checks', 'checks')
        for group in ('E0', 'E1', 'E2', 'E3'):
            for epoch in (1, 2):
                self.run(f'smoke_{group}_train_{epoch}', 'train', '--group', group, '--epoch', epoch, '--smoke')
            original = self.root / 'smoke' / group / 'epochs/01'
            replay = self.root / 'smoke_replay' / group / 'epochs/01'
            replay.mkdir(parents=True, exist_ok=True)
            for p in original.iterdir():
                if p.is_file():
                    dest = replay / p.name
                    if dest.exists():
                        if sha(dest) != sha(p):
                            raise ValueError('Resume replay source changed')
                    else:
                        # Immutable epoch1 recovery state, copied without modifying the original.
                        os.link(p, dest)
            self.run(f'smoke_{group}_resume_replay', 'train', '--group', group, '--epoch', '2', '--smoke', '--replay')
            self.run(f'smoke_{group}_restore_check_v2', 'restore-check', '--group', group)
            self.run(f'smoke_{group}_resume_check', 'resume-check', '--group', group)
            # Check training recovery before spending time on three-weather inference.
            for epoch in (1, 2):
                if (group == 'E0' and epoch == 1
                        and self.status.get('smoke_E0_eval_1', {}).get('status') != 'complete'):
                    self.run('smoke_eval_interruption', 'evaluate', '--group', group, '--epoch', epoch,
                             '--smoke', '--interrupt-eval-once', evaluate=True, expected_codes=(43,))
                self.run(f'smoke_{group}_eval_{epoch}', 'evaluate', '--group', group, '--epoch', epoch, '--smoke', evaluate=True)
        self.run('smoke_report', 'report', '--smoke')
        train = read(self.root / 'smoke/E3/epochs/02/train_complete.json')
        sample_bytes = sum(p.stat().st_size for p in (self.root / 'labels').rglob('*.npz'))
        c = read(self.root / 'config.json')
        estimate = dict(smoke_only=True, E3_frames_per_second=train['frames_per_second'],
            approximate_total_training_hours=4 * c['epochs'] * 8080 / train['frames_per_second'] / 3600,
            note='short-run measured speed; excludes validation and full-label preparation; not an ETA guarantee',
            sampled_cache_bytes=sample_bytes, training_epoch_weight_bytes=(self.root / 'smoke/E3/epochs/02/detector.pth').stat().st_size)
        write(self.root / 'cost_estimate.json', estimate)
        print('MEASURED COST ' + repr(estimate), flush=True)


def main():
    server()
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    args = p.parse_args()
    inherit_successor_paths(safe(args.output).resolve())
    from v2xr_raygen_e0e3.paths import check
    root = check(args.output)
    if Path('/data') not in root.parents or root == ROOT or ROOT in root.parents:
        raise ValueError('OUTPUT_ROOT must be a separate directory on /data')
    mode = os.environ.get('MODE', 'all')
    if mode not in ('all', 'preflight', 'labels', 'smoke', 'report'):
        raise ValueError('Unknown MODE')
    c = settings()
    if os.environ.get('PYTHONHASHSEED') != str(c['seed']):
        raise ValueError('Launcher must set PYTHONHASHSEED before starting controller')
    own = verify_sources()
    request = make_request(c, own)
    root.mkdir(parents=True, exist_ok=True)
    with lock(root / '.controller.lock'):
        marker = root / 'request.json'
        repair = None
        if marker.exists():
            if os.environ.get('RESUME', '0') != '1' and mode != 'report':
                raise ValueError('Existing OUTPUT_ROOT requires RESUME=1')
            if mode != 'report' and read(marker) != request:
                raise ValueError('Stable computation is a new implementation: use a NEW OUTPUT_ROOT with REUSE_DATA_FROM pointing to the old run')
        else:
            allowed = {'.controller.lock', 'launcher.log', 'launcher.pid'}
            if any(p.name not in allowed for p in root.iterdir()):
                raise ValueError('Nonempty unowned output refused')
            if mode == 'report':
                raise ValueError('No existing run to report')
            bound(marker, request)
        if mode != 'report':
            bound(root / 'config.json', c)
            bound(root / 'protocol.json', read(HERE / 'protocol.json'))
        start_ticks = Path('/proc/self/stat').read_text().split()[21]
        write(root / 'controller.json', dict(pid=os.getpid(), start_ticks=start_ticks,
            output=str(root), executable=sys.executable, controller=str(Path(__file__).resolve())))
        control = Controller(root)
        try:
            if mode == 'report':
                control.run('report_refresh_' + str(time.time_ns()), 'report')
                return
            control.run('preflight', 'preflight')
            from v2xr_raygen_e0e3.runtime import input_stamps
            input_stamps(root)
            if repair is not None:
                print('Audited U-fixture repair: reuse prepared caches; recheck capacity if the GPU changed; retry server_checks.', flush=True)
                control.run('u_fix_regression', 'local-test')
                from v2xr_raygen_e0e3.resume_compat import NUMERIC_MARKER
                if (root / NUMERIC_MARKER).exists():
                    print('Audited resume-check v2: preserve committed E0 work; run exact restore check and retry validation.', flush=True)
                    control.run('resume_check_v2_regression', 'local-test')
                    from v2xr_raygen_e0e3.resume_compat import RMS_MARKER
                    if (root / RMS_MARKER).exists():
                        print('Audited RMS v3 repair: reuse E0/E1 training and exact-load checks; retry E1 comparison.', flush=True)
                        control.run('resume_rms_v3_regression', 'local-test')
            if mode == 'preflight':
                return
            control.run('synthetic_checks', 'local-test')
            if mode == 'labels':
                control.run('shared_content', 'content')
                control.run('shared_labels', 'labels')
                return
            # Decode selected real inputs via the loader during smoke. Full new weather decoding is shared afterwards.
            control.smoke()
            if mode == 'smoke':
                return
            control.run('shared_content', 'content')
            control.run('shared_labels', 'labels')
            for group in c['groups']:
                for epoch in range(1, c['epochs'] + 1):
                    for part in ('train', 'eval'):
                        control.status.setdefault(f'{group}_{part}_{epoch:02d}', dict(status='planned'))
            write(control.status_path, control.status)
            for group in c['groups']:
                for epoch in range(1, c['epochs'] + 1):
                    control.run(f'{group}_train_{epoch:02d}', 'train', '--group', group, '--epoch', epoch)
                    control.run(f'{group}_eval_{epoch:02d}', 'evaluate', '--group', group, '--epoch', epoch, evaluate=True)
                # Group result committed before starting the next group, no scientific gain gate.
                from v2xr_raygen_e0e3.report import group_report
                group_report(root, group)
            control.run('final_report', 'report')
            write(root / 'complete.json', dict(status='complete', contract_identity=read(root / 'contract.json')['identity'],
                execution_sha256=sha(root / 'execution.json'),
                gpu_capacity_sha256=sha(root / 'source_repair/gpu_capacity.json') if (root / 'source_repair/gpu_capacity.json').exists() else None,
                resume_check_repair_sha256=sha(root / 'source_repair/resume_check_v2.json') if (root / 'source_repair/resume_check_v2.json').exists() else None,
                resume_rms_repair_sha256=sha(root / 'source_repair/resume_rms_v3.json') if (root / 'source_repair/resume_rms_v3.json').exists() else None,
                summary_sha256=sha(root / 'summary.json'),
                compute_policy_sha256=sha(root / 'compute_policy.json'),
                stable_checks_sha256=sha(root / 'smoke/stable_calculation_checks.json'),
                data_reuse_sha256=sha(root / 'data_reuse.json') if (root / 'data_reuse.json').exists() else None,
                time=time.time()))
        except BaseException:
            if not (root / 'failure.json').exists() or control.child is not None:
                write(root / 'failure.json', dict(status='failed', stage=control.stage_name, traceback=traceback.format_exc(), time=time.time()))
            raise
        finally:
            if control.child is not None and control.child.poll() is None:
                os.killpg(control.child.pid, signal.SIGTERM)
                try:
                    control.child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(control.child.pid, signal.SIGKILL)
                    control.child.wait()


if __name__ == '__main__':
    main()
