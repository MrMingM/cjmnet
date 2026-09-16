"""Consolidate complete full-validation runs; never round away small AP changes."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    import yaml
    root = Path(args.run)
    lines = ['# GSPR evidence experiment', '',
             'See evaluation protocol below. Full/original_full are unbudgeted references.', '',
             '| Variant | Weather | Control | Frames | AP30 | AP50 | AP70 | Bytes/frame | Replaced | Corrected / lost |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    found = set()
    incomplete = []
    protocols = set()
    for folder in sorted(root.glob('eval_*')):
        if not (folder/'protocol.json').is_file() or not (folder/'summary.json').is_file():
            incomplete.append(folder.name)
            continue
        protocol = json.loads((folder/'protocol.json').read_text())
        summary = json.loads((folder/'summary.json').read_text())
        protocols.add(protocol.get('evaluation_protocol', 'development'))
        if len(protocols) > 1:
            raise ValueError('Do not mix historical test and development validation in one report')
        found.add((protocol['variant'], protocol['weather']))
        for mode, row in summary.items():
            if row['frames'] != len(protocol['sample_indices']):
                raise ValueError(f'Incomplete validation: {folder}/{mode}')
            filename = 'eval_global_sort.yaml' if protocol['global_sort'] else 'eval.yaml'
            values = yaml.safe_load((folder/mode/filename).read_text())
            lines.append(f'| {protocol["variant"]} | {protocol["weather"]} | {mode} | {row["frames"]} | '
                f'{values["ap30"]:.6f} | {values["ap_50"]:.6f} | {values["ap_70"]:.6f} | '
                f'{row["mean_total_bytes"]:.1f} | {row["replacement_fraction"]:.4f} | '
                f'{row["corrected_misses"]} / {row["lost_detections"]} |')
    expected = {(v, w) for v in ('matching', 'concat', 'no_u') for w in ('clean', 'fog', 'rain', 'snow')}
    if found != expected:
        lines += ['', 'Incomplete suite; missing: '+str(sorted(expected-found))]
    if incomplete:
        lines += ['', 'Ignored unfinished directories: '+str(incomplete)]
    lines += ['', 'Evaluation protocol: '+str(sorted(protocols)),
              'benchmark = OPV2V clean test + pre-generated OPV2V-W fog/rain/snow test; '
              'no online weather; non-global AP matching the original experiment. '
              'development = validation scenes + online simulated weather + globally sorted AP.']
    lines += ['', 'Inspect each model history.jsonl: candidate_regret, oracle_over_rule, predicted_over_rule, '
              'pair_accuracy. These describe sampled fixed-context teacher candidates, not an oracle AP ceiling.',
              '', 'Each frames.jsonl contains scene IDs and per-frame TP/FP/scores for scene-wise reanalysis. '
              'A single training seed does not establish cross-dataset generalization.']
    (root/'results.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(root/'results.md')


if __name__ == '__main__':
    main()
