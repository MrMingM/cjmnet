"""Switch the original sequential process at a committed P1 epoch."""
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time


ROOT = Path('/data/cjm/datasets/logs/v2xr-fusion-pot-v1')
HERE = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text())


def main():
    pid = int((ROOT / 'launch.pid').read_text())
    proc = Path('/proc') / str(pid)
    def process_identity():
        try:
            fields = proc.joinpath('stat').read_text().rsplit(')', 1)[1].split()
            return fields[19] if fields[0] != 'Z' else None
        except FileNotFoundError:
            return None

    # Keep an open proc directory for the lifetime of the identity check.
    fd = os.open(proc, os.O_DIRECTORY)
    try:
        started = process_identity()
        if (proc.joinpath('cwd').resolve() != HERE.parent
                or proc.joinpath('cmdline').read_bytes().split(b'\0')[1:-1] != [b'-uB', b'run.py']
                or b'ROCR_VISIBLE_DEVICES=4' not in proc.joinpath('environ').read_bytes().split(b'\0')):
            raise RuntimeError('Legacy P1 process identity differs')
        status = read('STATUS.json')
        if (status.get('group'), status.get('stage')) != ('P1', 'formal_train'):
            raise RuntimeError('Legacy process is outside P1 training')
        log = ROOT / 'launch.log'
        epoch = int(re.findall(r'^TRAIN START (\d+)/20$', log.read_text(), re.M)[-1])
        marker = ROOT / 'P1' / 'epochs' / f'{epoch:02d}' / 'complete.json'
        print(f'WAIT FOR P1 EPOCH {epoch} COMMIT; legacy PID {pid}', flush=True)
        while not marker.exists():
            if process_identity() != started:
                raise RuntimeError('Legacy P1 exited before the requested commit')
            time.sleep(.5)
        row = json.loads(marker.read_text())
        if row['epoch'] != epoch or row['status'] != 'complete':
            raise RuntimeError('Checkpoint marker failed validation')
        status = read('STATUS.json')
        if status.get('group') != 'P1' or started is None or process_identity() != started:
            raise RuntimeError('Legacy process no longer owns P1')
        os.kill(pid, signal.SIGTERM)
        for _ in range(120):
            if process_identity() != started:
                break
            time.sleep(.5)
        else:
            raise RuntimeError('Legacy P1 did not exit after the epoch commit')
        print(f'LEGACY STOPPED AFTER EPOCH {epoch}', flush=True)
    finally:
        os.close(fd)

    # Prevent the old sequential entry point from starting while split jobs run.
    import fcntl
    with (ROOT / '.lock').open('a') as lock:
        for attempt in range(30):
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(2)
        else:
            raise RuntimeError('Legacy workers still hold the original lock')
        time.sleep(5)
        with (ROOT / 'P1.parallel.log').open('ab', buffering=0) as log:
            result = subprocess.Popen(['sh', str(HERE / 'run_parallel.sh'), 'P1'],
                                      stdout=log, stderr=subprocess.STDOUT)
            (ROOT / 'P1.parallel.pid').write_text(f'{result.pid}\n')
            if result.wait():
                raise RuntimeError('P1 split worker failed')
        while True:
            statuses = [read(f'{name}/PARALLEL_STATUS.json')['status'] for name in ('P1', 'P2', 'P3')]
            if 'failed' in statuses:
                raise RuntimeError(f'A formal worker failed: {statuses}')
            if statuses == ['complete'] * 3:
                break
            time.sleep(60)
        with (ROOT / 'report.parallel.log').open('ab', buffering=0) as log:
            subprocess.run(['sh', str(HERE / 'run_parallel.sh'), 'report'],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        print('ALL PARALLEL GROUPS AND REPORT COMPLETE', flush=True)


if __name__ == '__main__':
    main()
