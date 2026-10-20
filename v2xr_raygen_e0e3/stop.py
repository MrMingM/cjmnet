"""Signal only the controller whose kernel process identity matches this run."""
import argparse
import os
from pathlib import Path
import signal
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from v2xr_raygen_e0e3.common import read, server


def main():
    server()
    p = argparse.ArgumentParser()
    p.add_argument('output')
    args = p.parse_args()
    r = read(Path(args.output).resolve() / 'controller.json')
    proc = Path('/proc') / str(r['pid'])
    if not proc.exists():
        print('Controller has already exited')
        return
    stat = (proc / 'stat').read_text().split()[21]
    cmd = (proc / 'cmdline').read_bytes().split(b'\0')
    if stat != r['start_ticks'] or r['controller'].encode() not in cmd or r['output'].encode() not in cmd:
        raise ValueError('PID belongs to another process; refusing to signal')
    os.kill(r['pid'], signal.SIGTERM)
    print('TERM sent to verified task controller; it forwards only to its own child process group')


if __name__ == '__main__':
    main()
