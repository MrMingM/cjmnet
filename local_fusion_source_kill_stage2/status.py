"""Read-only status; detects dead worker PIDs and validates completed artifacts."""
import argparse
import os
from pathlib import Path
from .common import read_json, load_contract, resume


def status(run):
    run = Path(run)
    protocol = load_contract(run) if (run/'PROTOCOL.json').exists() else None
    result = {}
    for name in ('k1','k2'):
        path = run/name/'manifest.json'
        state = read_json(path) if path.exists() else {'status':'pending'}
        value = state['status']
        if value == 'running':
            try:
                os.kill(state['pid'], 0)
            except ProcessLookupError:
                value = 'failed'
        if value == 'complete' and protocol is not None:
            if resume(run/name/'result.json', protocol['identity']) is None:
                value = 'failed'
        result[name.upper()] = value
    root = run/'STAGE2_FINAL.json'
    if 'failed' in result.values():
        result['STAGE2'] = 'failed'
    elif root.exists() and all(v == 'complete' for v in result.values()):
        result['STAGE2'] = read_json(root)['status']
    else:
        result['STAGE2'] = 'running' if 'running' in result.values() else 'pending'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    for name, value in status(args.run).items():
        print(name+' = '+value)


if __name__ == '__main__':
    main()
