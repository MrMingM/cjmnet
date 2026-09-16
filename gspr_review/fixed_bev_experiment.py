"""No training: compare fixed BEV gains with learned gains using existing wire packets."""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import yaml
from . import runtime as rt
from .audit_location import differences, read_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--location-run', required=True, help='Completed canonical-message location evaluation')
    parser.add_argument('--output-dir')
    args = parser.parse_args()
    location = Path(args.location_run).resolve()
    paired = json.loads((location/'paired_evaluation.json').read_text(encoding='utf-8'))
    bev = Path(paired['bev_run'])
    options = yaml.safe_load((bev/'experiment.yaml').read_text(encoding='utf-8'))
    if options['review'].get('architecture') != 'bev_contrast':
        raise ValueError('Expected trained BEV contrast architecture')
    manifest = json.loads((bev/'manifest.json').read_text(encoding='utf-8'))
    frontend = Path(manifest['frontend_checkpoint'])
    weights = bev/'reviewer_best.pth'
    if rt.sha256(weights) != paired['bev_checkpoint_sha256'] or rt.sha256(frontend) != manifest['frontend_sha256']:
        raise ValueError('Original checkpoints changed')
    watched = [weights, frontend, bev/'experiment.yaml', bev/'frontend_config.yaml']
    for branch in ('processed_lidar', 'processed_lidar_weather'):
        metadata = json.loads((location/f'{branch}_protocol/protocol.json').read_text(encoding='utf-8'))
        if metadata['options'] != options or metadata['frontend_sha256'] != manifest['frontend_sha256']:
            raise ValueError('Cached evaluation protocol differs from training configuration')
        packets = sorted((location/f'{branch}_packets').glob('*.json'))
        if not packets:
            raise ValueError('Missing canonical packet cache for '+branch)
        watched.extend(packets)
    hashes = {str(p): rt.sha256(p) for p in watched}
    rt.verify_frozen()
    out = rt.new_output(args.output_dir or str(location)+'_fixed_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    rt.write_json(out/'fixed_protocol.json', {'location_run': str(location), 'input_sha256': hashes,
        'fixed_constants': [0., .05, .10], 'scope': 'Development-only rule control. Same received messages and support. No learning or AP-based coefficient choice.'})
    rows = []
    step = 0
    for branch in ('processed_lidar', 'processed_lidar_weather'):
        reference = read_frames(location/f'{branch}_protocol')
        for control, constant in (('zero', 0.), ('fixed_005', .05), ('fixed_010', .10), ('learned', None)):
            step += 1
            folder = out/f'{branch}_{control}'
            print(f'{datetime.now().isoformat(timespec="seconds")} FIXED CONTROL {step}/8: {branch} {control}', flush=True)
            command = [sys.executable, '-u', '-m', 'gspr_review.evaluate',
                '--config', str(bev/'experiment.yaml'), '--frontend-config', str(bev/'frontend_config.yaml'),
                '--frontend-checkpoint', str(frontend), '--reviewer', str(weights), '--mode', 'review',
                '--lidar-key', branch, '--packet-cache', str(location/f'{branch}_packets'),
                '--packet-cache-mode', 'replay', '--compare-ego', '--output-dir', str(folder)]
            if constant is not None:
                command += ['--fixed-bev-adjustment', str(constant)]
            subprocess.run(command, check=True, cwd=rt.ROOT)
            comparison = differences(reference, read_frames(folder))
            if not comparison['matches']:
                rt.write_json(out/'fixed_mismatch.json', comparison)
                raise RuntimeError('Control differs in sample, bytes, original support or messages; see fixed_mismatch.json')
            metrics = yaml.safe_load((folder/'eval.yaml').read_text(encoding='utf-8'))
            comm = json.loads((folder/'communication.json').read_text(encoding='utf-8'))
            rows.append(dict(branch=branch, control=control, fixed_adjustment=constant,
                             **{k: metrics[k] for k in ('ap30', 'ap_50', 'ap_70')}, **comm))
            rt.write_json(out/'fixed_partial.json', rows)
    for path, digest in hashes.items():
        if rt.sha256(path) != digest:
            raise RuntimeError('Original checkpoint/config/cache changed: '+path)
    rt.verify_frozen()
    rt.write_json(out/'fixed_comparison.json', rows)
    print('Send back:', out/'fixed_comparison.json', flush=True)


if __name__ == '__main__':
    main()
