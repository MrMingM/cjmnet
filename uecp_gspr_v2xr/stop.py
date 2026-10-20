"""Verify /proc PID start identity; signal only this run's controller and its own children."""
import argparse
import os
from pathlib import Path
import signal
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from uecp_gspr_v2xr.common import read, server


def main():
    server(); p = argparse.ArgumentParser(); p.add_argument('output'); args = p.parse_args()
    row = read(Path(args.output) / 'controller.json')
    proc = Path('/proc') / str(row['pid'])
    if not proc.exists():
        print('Controller already exited'); return
    cmd = (proc / 'cmdline').read_bytes().split(b'\0')
    if ((proc / 'stat').read_text().split()[21] != row['start_ticks']
            or row['controller'].encode() not in cmd or row['output'].encode() not in cmd):
        raise ValueError('PID has been reused; refusing to signal an unrelated process')
    os.kill(row['pid'], signal.SIGTERM)
    print('TERM sent to verified controller; recovery uses last committed optimizer boundary')


if __name__ == '__main__':
    main()
