"""Server orchestration; K1/K2 own separate locks, manifests and caches."""
import argparse
import os
import subprocess
import sys
from .common import atomic_json, external_run, now, resource_check
from .preflight import server_preflight


def finalize(run):
    import fcntl
    with (run / '.summary.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        from .reports import summarize
        return summarize(run)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--run', required=True)
    cli.add_argument('--component', choices=('k1','k2','both'), default='both')
    cli.add_argument('--p0-run')
    args = cli.parse_args()
    run = external_run(args.run)
    protocol = server_preflight(run, args.p0_run)
    resource_check(protocol)
    if args.component != 'both':
        try:
            if args.component == 'k1':
                from .k1_source_specificity import run as execute
            else:
                from .k2_label_stability import run as execute
            execute(run)
        finally:
            finalize(run)
        return
    import fcntl
    with (run / '.driver.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = dict(status='running', identity=protocol['identity'], pid=os.getpid(), started_at=now())
        atomic_json(run / 'manifest.json', state)
        children = []
        for component in ('k1','k2'):
            resource_check(protocol)
            env = os.environ.copy()
            env['ROCR_VISIBLE_DEVICES'] = env.get(component.upper()+'_GPU', env.get('ROCR_VISIBLE_DEVICES','2'))
            env.pop('HIP_VISIBLE_DEVICES', None)
            env.pop('CUDA_VISIBLE_DEVICES', None)
            log = (run / (component+'.log')).open('a', encoding='utf-8')
            process = subprocess.Popen([sys.executable, '-m', 'local_fusion_source_kill_stage2.driver',
                '--run', str(run), '--component', component], stdout=log, stderr=subprocess.STDOUT, env=env)
            atomic_json(run / (component+'_pid.json'), dict(pid=process.pid, physical_gpu=env['ROCR_VISIBLE_DEVICES']))
            children.append((process, log))
        codes = []
        try:
            for process, log in children:
                codes.append(process.wait())
                log.close()
        finally:
            finalize(run)
            state.update(status='complete' if len(codes)==2 and all(c==0 for c in codes) else 'failed',
                         completed_at=now(), worker_exit_codes=codes)
            atomic_json(run / 'manifest.json', state)
        if state['status'] != 'complete':
            raise SystemExit(1)
        print('STAGE2 COMPLETE; K3 remains frozen. Results:', run/'STAGE2_FINAL.md', flush=True)


if __name__ == '__main__':
    main()
