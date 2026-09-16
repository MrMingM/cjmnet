"""Collect the eight fixed-budget controls without guessing latest directories."""
import argparse
import json
from pathlib import Path
import yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', required=True)
    args = p.parse_args()
    rows = []
    for branch in ('processed_lidar', 'processed_lidar_weather'):
        for mode in ('a0b0', 'protocol', 'review', 'shuffled'):
            path = Path(f'{args.run_dir}_{branch}_{mode}')
            metrics = yaml.safe_load((path/'eval.yaml').read_text())
            comm = json.loads((path/'communication.json').read_text())
            row = {'branch': branch, 'mode': mode, **{k: metrics[k] for k in ('ap30', 'ap_50', 'ap_70')}, **comm}
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False))
    (Path(args.run_dir)/'controls_summary.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
