import argparse
import json
from pathlib import Path
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    root = Path(args.output_dir)
    rows = []
    for architecture in ('legacy', 'contrast', 'contrast_gain'):
        for branch in ('processed_lidar', 'processed_lidar_weather'):
            for control in ('normal', 'neutral', 'permuted', 'protocol', 'a0b0'):
                folder = root/f'{architecture}_{branch}_{control}'
                ap = yaml.safe_load((folder/'eval.yaml').read_text(encoding='utf-8'))
                comm = json.loads((folder/'communication.json').read_text(encoding='utf-8'))
                row = dict(architecture=architecture, branch=branch, control=control,
                           **{k: ap[k] for k in ('ap30', 'ap_50', 'ap_70')}, **comm)
                rows.append(row)
                print(json.dumps(row), flush=True)
    # These controls must use the same frame set and actual byte count.
    for architecture in ('legacy', 'contrast', 'contrast_gain'):
        for branch in ('processed_lidar', 'processed_lidar_weather'):
            controls = [r for r in rows if r['architecture'] == architecture and r['branch'] == branch
                        and r['control'] != 'a0b0']
            if len({(r['frames'], r['mean_total_bytes']) for r in controls}) != 1:
                raise RuntimeError('Content controls have different frame counts or byte costs')
            signatures = []
            for control in ('normal', 'neutral', 'permuted', 'protocol'):
                folder = root/f'{architecture}_{branch}_{control}'
                frames = [json.loads(line) for line in (folder/'frames.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
                signatures.append([(r['sample_index'], r['total_bytes'], r['eligible_points']) for r in frames])
            if any(s != signatures[0] for s in signatures[1:]):
                raise RuntimeError('Content controls differ in per-frame identity, bytes or original eligibility')
    (root/'comparison.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
