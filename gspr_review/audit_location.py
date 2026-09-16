"""Read completed control files to locate the first mismatch; no GPU needed."""
import argparse
import json
from pathlib import Path
import yaml

FIELDS = ('sample_index', 'total_bytes', 'eligible_points', 'message_sha256')


def read_frames(folder):
    return [json.loads(s) for s in (Path(folder)/'frames.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]


def differences(reference, current):
    counts = {key: 0 for key in FIELDS}
    examples = []
    for index, (a, b) in enumerate(zip(reference, current)):
        changes = {key: {'reference': a.get(key), 'current': b.get(key)} for key in FIELDS
                   if key not in a or key not in b or a[key] != b[key]}
        for key in changes:
            counts[key] += 1
        if changes and len(examples) < 5:
            examples.append({'row': index, 'sample_index': a.get('sample_index'), 'differences': changes})
    return {'matches': len(reference) == len(current) and not any(counts.values()),
            'reference_frames': len(reference), 'current_frames': len(current),
            'mismatch_counts': counts, 'first_examples': examples}


def audit(root):
    root = Path(root)
    checks = []
    for branch in ('processed_lidar', 'processed_lidar_weather'):
        ref = root/f'{branch}_protocol'
        if not (ref/'communication.json').is_file():
            continue
        reference = read_frames(ref)
        for name in ('point_normal', 'point_permuted', 'bev_normal', 'bev_permuted'):
            folder = root/f'{branch}_{name}'
            if not (folder/'communication.json').is_file():
                continue
            result = differences(reference, read_frames(folder))
            result.update(branch=branch, control=name)
            # Report protocol option drift independently of numerical messages.
            a = json.loads((ref/'protocol.json').read_text(encoding='utf-8'))
            b = json.loads((folder/'protocol.json').read_text(encoding='utf-8'))
            result['protocol_differences'] = {k: {'reference': a.get(k), 'current': b.get(k)}
                for k in ('root', 'sample_indices', 'frontend_sha256', 'lidar_key') if a.get(k) != b.get(k)}
            checks.append(result)
    return {'run': str(root), 'checks': checks,
            'note': 'Hash-only mismatch does NOT prove roundoff. Original packets are not saved, so their numeric difference cannot be recovered from a hash.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir')
    parser.add_argument('--point-run')
    args = parser.parse_args()
    if args.output_dir:
        root = Path(args.output_dir).resolve()
    elif args.point_run:
        point = Path(args.point_run).resolve()
        choices = sorted(p for p in point.parent.glob(point.name+'_location_*') if (p/'location_protocol.json').is_file())
        if len(choices) != 1:
            parser.error('Specify --output-dir; candidates: '+str([str(p) for p in choices]))
        root = choices[0]
    else:
        parser.error('Provide --output-dir or --point-run')
    report = audit(root)
    # Only a diagnostic file is added; no model, configuration or result is changed.
    (root/'location_mismatch.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    print('Send back:', root/'location_mismatch.json', flush=True)


if __name__ == '__main__':
    main()
