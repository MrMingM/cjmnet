"""Read-only checkpoint sweep; subprocess evaluation retains existing packet path."""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import yaml
from . import runtime as rt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True, help='Completed contrast training directory')
    parser.add_argument('--output-dir')
    args = parser.parse_args()
    run = Path(args.run_dir).resolve()
    config = run/'experiment.yaml'
    frontend = run/'frontend_config.yaml'
    reviewer = run/'reviewer_best.pth'
    manifest = json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    checkpoint = Path(manifest['frontend_checkpoint'])
    options = yaml.safe_load(config.read_text(encoding='utf-8'))
    if options['review'].get('architecture') != 'contrast':
        parser.error('Use the contrast run, not legacy or contrast_gain')
    watched = [config, frontend, reviewer, checkpoint]
    hashes = {str(p): rt.sha256(p) for p in watched}
    if hashes[str(checkpoint)] != manifest['frontend_sha256']:
        raise ValueError('Frontend checksum does not match training manifest')
    rt.verify_frozen()
    out = rt.new_output(args.output_dir or str(run)+'_amplitude_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    cases = [(0, 'normal')] + [(scale, intervention) for scale in (1, 10, 50, 100)
                              for intervention in ('normal', 'permuted')]
    rt.write_json(out/'scan_protocol.json', {'run': str(run), 'input_sha256': hashes,
        'cases': cases, 'scope': 'Development amplitude diagnostic, no training. Post-tanh scaling capped at saved max_adjustment and original weight bounds.'})
    rows = []
    completed = 0
    for branch in ('processed_lidar', 'processed_lidar_weather'):
        signature = None
        for scale, intervention in cases:
            completed += 1
            folder = out/f'{branch}_x{scale}_{intervention}'
            print(f'[{datetime.now().isoformat(timespec="seconds")}] {completed}/18: {branch} x{scale} {intervention}', flush=True)
            command = [sys.executable, '-u', '-m', 'gspr_review.evaluate',
                '--config', str(config), '--frontend-config', str(frontend),
                '--frontend-checkpoint', str(checkpoint), '--reviewer', str(reviewer),
                '--mode', 'review', '--lidar-key', branch, '--correction-scale', str(scale),
                '--evidence-intervention', intervention, '--compare-ego', '--output-dir', str(folder)]
            subprocess.run(command, check=True, cwd=rt.ROOT)
            frames = [json.loads(line) for line in (folder/'frames.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
            current = [(r['sample_index'], r['total_bytes'], r['eligible_points']) for r in frames]
            if signature is None:
                signature = current
            elif current != signature:
                raise RuntimeError('Amplitude controls differ in frames, original eligibility or bytes')
            ap = yaml.safe_load((folder/'eval.yaml').read_text(encoding='utf-8'))
            comm = json.loads((folder/'communication.json').read_text(encoding='utf-8'))
            rows.append(dict(branch=branch, scale=scale, intervention=intervention,
                             **{k: ap[k] for k in ('ap30', 'ap_50', 'ap_70')}, **comm))
            rt.write_json(out/'scan_partial.json', rows)
    for path, expected in hashes.items():
        if rt.sha256(path) != expected:
            raise RuntimeError('Input file changed during scan: ' + path)
    rt.verify_frozen()
    rt.write_json(out/'scan_comparison.json', rows)
    print('Send back:', out/'scan_comparison.json', flush=True)


if __name__ == '__main__':
    main()
