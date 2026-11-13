"""Run shared extraction once, then wait for three independent workers."""
import fcntl
import os
import subprocess
import sys

from v2xr_griq_experiments.dataset import OUT, ROOT, read, save, sha, smoke_root


def start(name, args, card='', parallel=False):
    target = OUT / 'logs'
    target.mkdir(exist_ok=True)
    with (target / f'{name}.log').open('ab') as log:
        proc = subprocess.Popen([sys.executable,'-uB','-m',f'v2xr_griq_experiments.{args[0]}',*args[1:]],
            env=dict(os.environ,ROCR_VISIBLE_DEVICES=card), stdout=log, stderr=subprocess.STDOUT)
    save(target/f'{name}.pid.json', dict(pid=proc.pid, device=card or 'CPU'))
    print('START',name,proc.pid,card or 'CPU',flush=True)
    return proc if parallel else finish(name,proc)


def finish(name, proc):
    code = proc.wait()
    save(OUT/'logs'/f'{name}.exit.json',dict(pid=proc.pid,exit_code=code))
    print('EXIT',name,code,flush=True)
    if code:
        raise RuntimeError(f'{name} failed with {code}. See logs/{name}.log')


def groups(smoke):
    names = ('M1','M2','M3')
    marker_root = smoke_root() if smoke else OUT
    procs = {}
    for name in names:
        marker = marker_root / name / 'COMPLETE.json'
        if marker.exists():
            row = read(marker)
            if (row['data'] != sha(marker_root/'DATA_AUDIT.json') or any(
                    sha(marker.parent/k) != v for k,v in row['artifacts'].items())):
                raise ValueError(f'Committed group changed: {name}')
            continue
        procs[name] = start(('smoke_' if smoke else '')+name,
                            ['worker',name]+(['--smoke'] if smoke else []),parallel=True)
    errors = []
    for name,proc in procs.items():
        try:
            finish(('smoke_' if smoke else '')+name,proc)
        except RuntimeError as exc:
            errors.append(str(exc))
    if errors:
        raise RuntimeError('; '.join(errors))
    if any(not (marker_root/n/'COMPLETE.json').is_file() for n in names):
        raise RuntimeError('Incomplete parallel group set')


def main():
    subprocess.run(['sh','-n',str(ROOT/'run_all.sh')],check=True)
    if b'\r' in (ROOT/'run_all.sh').read_bytes():
        raise ValueError('Expected LF shell script')
    start('preflight',['preflight'])
    start('design_tests',['tests.test_design'])
    card = os.environ['GRIQ_GPU']
    if card not in ('1','2','3'):
        raise ValueError('GRIQ_GPU must identify physical card 1, 2 or 3')
    for smoke in (True, False):
        root = smoke_root() if smoke else OUT
        suffix = ['--smoke'] if smoke else []
        for split in ('train','validate'):
            if not (root/f'{split}_COMPLETE.json').is_file():
                start(f'{"smoke_" if smoke else ""}extract_{split}', ['prepare_candidates',split,*suffix],card)
        start(f'{"smoke_" if smoke else ""}audit',['audit',*suffix])
        groups(smoke)
        if '--smoke' in sys.argv:
            return
    start('compare',['compare'])
    print('COMPLETE GRIQ',flush=True)


if __name__ == '__main__':
    with (OUT/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
