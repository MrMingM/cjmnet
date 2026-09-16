"""Sweep the conservative selector margin using saved Oracle frame records.

This analysis uses per-frame AP stored by oracle_eval.py.  It is a cheap
diagnostic proxy and does not replace a final global-AP rerun.
"""

import argparse
import csv
import json
import os

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input', nargs='+', required=True,
        help='Entries formatted as domain=/path/to/oracle_selection.jsonl')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--checkpoint_margin', type=float, default=None)
    parser.add_argument('--quantiles', type=int, default=201)
    return parser.parse_args()


def load_domain(specification):
    if '=' not in specification:
        raise ValueError('--input requires domain=path, got %s' % specification)
    domain, path = specification.split('=', 1)
    rows = []
    with open(path, encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():
                continue
            raw = json.loads(line)
            learned = raw.get('learned_selector')
            if learned is None or 'scores' not in learned:
                raise ValueError('%s lacks learned-selector scores' % path)
            candidates = raw['candidates']
            names = [item['name'] for item in candidates]
            scores = np.asarray([
                float(learned['scores'][name]) for name in names
            ], dtype=np.float64)
            frame_ap = np.asarray([
                float(item['metrics']['frame_ap']) for item in candidates
            ], dtype=np.float64)
            all_index = names.index('all')
            alternatives = [index for index in range(len(names))
                            if index != all_index]
            best_alternative = max(
                alternatives, key=lambda index: scores[index])
            rows.append({
                'names': names,
                'scores': scores,
                'frame_ap': frame_ap,
                'all_index': all_index,
                'best_alternative': best_alternative,
                'predicted_margin': float(
                    scores[best_alternative] - scores[all_index])
            })
    if not rows:
        raise RuntimeError('No records in %s' % path)
    return domain, path, rows


def evaluate(rows, threshold):
    selected_ap = []
    all_ap = []
    oracle_ap = []
    switches = 0
    harmful = 0
    counts = {}
    for row in rows:
        all_index = row['all_index']
        selected = all_index
        if row['predicted_margin'] > threshold:
            selected = row['best_alternative']
        switches += int(selected != all_index)
        harmful += int(
            selected != all_index and
            row['frame_ap'][selected] < row['frame_ap'][all_index])
        name = row['names'][selected]
        counts[name] = counts.get(name, 0) + 1
        selected_ap.append(row['frame_ap'][selected])
        all_ap.append(row['frame_ap'][all_index])
        oracle_ap.append(float(np.max(row['frame_ap'])))
    selected_mean = float(np.mean(selected_ap))
    all_mean = float(np.mean(all_ap))
    oracle_mean = float(np.mean(oracle_ap))
    return {
        'frames': len(rows),
        'threshold': float(threshold),
        'selected_frame_ap': selected_mean,
        'all_frame_ap': all_mean,
        'gain_vs_all': selected_mean - all_mean,
        'oracle_headroom': oracle_mean - all_mean,
        'switch_ratio': switches / len(rows),
        'harmful_switch_ratio': harmful / len(rows),
        'selection_counts': counts
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    domains = [load_domain(item) for item in args.input]
    margins = np.asarray([
        row['predicted_margin'] for _, _, rows in domains for row in rows
    ], dtype=np.float64)
    thresholds = [0.0, float(margins.max() + 1.0e-6)]
    thresholds.extend(float(x) for x in np.quantile(
        margins, np.linspace(0.0, 1.0, args.quantiles)))
    if args.checkpoint_margin is not None:
        thresholds.append(float(args.checkpoint_margin))
    thresholds = sorted(set(max(0.0, value) for value in thresholds))

    table = []
    combined = []
    for threshold in thresholds:
        domain_results = {}
        for domain, _, rows in domains:
            result = evaluate(rows, threshold)
            domain_results[domain] = result
            table.append({'domain': domain, **{
                key: value for key, value in result.items()
                if key != 'selection_counts'
            }})
        gains = [item['gain_vs_all'] for item in domain_results.values()]
        combined.append({
            'threshold': threshold,
            'macro_mean_gain': float(np.mean(gains)),
            'worst_domain_gain': float(np.min(gains)),
            'mean_switch_ratio': float(np.mean([
                item['switch_ratio'] for item in domain_results.values()])),
            'mean_harmful_switch_ratio': float(np.mean([
                item['harmful_switch_ratio']
                for item in domain_results.values()])),
            'domains': domain_results
        })

    safe = [item for item in combined if item['worst_domain_gain'] >= -1.0e-10]
    best_safe = max(
        safe, key=lambda item: (item['macro_mean_gain'], item['threshold'])) \
        if safe else max(combined, key=lambda item: item['worst_domain_gain'])
    best_macro = max(
        combined, key=lambda item: (item['macro_mean_gain'],
                                    item['worst_domain_gain']))
    summary = {
        'notes': {
            'metric': 'Mean per-frame AP proxy, not global dataset AP.',
            'usage': 'Tune only on diagnostic/development data, then freeze.'
        },
        'inputs': {domain: os.path.abspath(path)
                   for domain, path, _ in domains},
        'checkpoint_margin': args.checkpoint_margin,
        'best_safe_threshold': best_safe,
        'best_macro_threshold': best_macro
    }
    with open(os.path.join(args.output_dir, 'margin_sweep_summary.json'),
              'w', encoding='utf-8') as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    with open(os.path.join(args.output_dir, 'margin_sweep.csv'), 'w',
              newline='', encoding='utf-8') as stream:
        fields = list(table[0])
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)
    print('Best safe threshold:')
    print(json.dumps(best_safe, indent=2, ensure_ascii=False))
    print('Best macro threshold:')
    print(json.dumps(best_macro, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
