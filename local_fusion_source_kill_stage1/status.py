"""Read independent worker states without racing their manifest updates."""
import argparse
import os
from .common import component_status, external_run, read_json, digest


def statuses(run):
    result = {name: component_status(run, name)['status'] for name in ('K0_PRIME', 'K1_0')}
    path = run / 'manifest.json'
    manifest = read_json(path) if path.exists() else {}
    stage = manifest.get('status', 'pending')
    if 'failed' in result.values():
        stage = 'failed'
    if stage == 'running':
        # A killed driver must not look alive forever; report status, do not alter artifacts.
        try:
            pid = int((run / 'pid').read_text().strip())
            os.kill(pid, 0)
        except (FileNotFoundError, ProcessLookupError, ValueError):
            stage = 'failed'
    if stage == 'complete':
        if any(value != 'complete' for value in result.values()):
            raise ValueError('Stage completion conflicts with component status')
        if not manifest.get('final_artifacts'):
            raise ValueError('Completed Stage 1 lacks final report hashes')
        for filename, expected in manifest['final_artifacts'].items():
            if digest(run / filename) != expected:
                raise ValueError('Final report corrupted')
    return {**result, 'STAGE1': stage}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--run', required=True)
    args = cli.parse_args()
    for key, value in statuses(external_run(args.run)).items():
        print(f'{key} = {value}')


if __name__ == '__main__':
    main()
