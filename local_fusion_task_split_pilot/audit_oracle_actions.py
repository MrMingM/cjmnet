"""Read an existing direction-B Oracle run without running a detector.

The Same and Task paths in the original run are independent and cumulative.
Their per-target records are useful diagnostics, but a difference at one step
is not the causal effect of task separation from an identical starting state.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


WEATHERS = ('clean', 'fog', 'rain', 'snow')


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _detected(record, phase):
    return bool(record[f'focal_{phase}']['detected'])


def summarize_frame(frame):
    same = frame['same']
    task = frame['task']
    if len(same) != len(task):
        raise ValueError('Same/Task target counts differ')
    same_by_id = {int(row['target_index']): row for row in same}
    task_by_id = {int(row['target_index']): row for row in task}
    if len(same_by_id) != len(same) or same_by_id.keys() != task_by_id.keys():
        raise ValueError('Same/Task target identities differ or repeat')

    counts = Counter()
    examples = []
    for target in sorted(same_by_id):
        a, b = same_by_id[target], task_by_id[target]
        if bool(a['was_shared_missed']) != bool(b['was_shared_missed']):
            raise ValueError('Shared target state differs between paths')
        counts['targets'] += 1
        if b['changed']:
            counts['task_changed'] += 1
        if b['task_separated']:
            counts['task_separated_selected'] += 1
        if not _detected(a, 'before') and _detected(a, 'after'):
            counts['same_path_focal_rescues'] += 1
        if not _detected(b, 'before') and _detected(b, 'after'):
            counts['task_path_focal_rescues'] += 1
        if int(b['matched_after']) > int(b['matched_before']):
            counts['task_path_match_count_increases'] += 1
        if _detected(b, 'after') and not _detected(a, 'after'):
            counts['task_path_focal_only_at_this_step'] += 1
            examples.append({
                'sample_index': int(frame['sample_index']),
                'target_index': target,
                'task_sources': [b['cls_source'], b['reg_source']],
                'same_sources': [a['cls_source'], a['reg_source']],
                'task_separated': bool(b['task_separated']),
                'shared_missed': bool(b['was_shared_missed']),
            })
        if _detected(a, 'after') and not _detected(b, 'after'):
            counts['same_path_focal_only_at_this_step'] += 1
    return counts, examples


def analyze_run(run):
    run = Path(run)
    required = [run / 'protocol.json', run / 'death_test_results.json']
    required += [run / weather / 'frames.jsonl' for weather in WEATHERS]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    protocol = _read_json(required[0])
    death = _read_json(required[1])
    expected_indices = [int(x) for x in protocol['validation_indices']]
    if len(expected_indices) != len(set(expected_indices)):
        raise ValueError('Protocol validation indices repeat')
    result = {
        'scope': 'read-only screening of independent, cumulative B Oracle paths',
        'limitations': [
            'Same and Task paths start from Shared but diverge after the first action.',
            'A per-target path difference is not a same-state causal task benefit.',
            'The original frames.jsonl does not contain final matched-GT identities.',
            'GT determined each target ROI and action during the original Oracle run.',
        ],
        'input_sha256': {str(path): _sha256(path) for path in required},
        'conditions': {},
    }
    for weather in WEATHERS:
        counts = Counter()
        samples = []
        examples = []
        frame_path = run / weather / 'frames.jsonl'
        with frame_path.open(encoding='utf-8') as stream:
            for line_no, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                frame = json.loads(line)
                samples.append(int(frame['sample_index']))
                row_counts, row_examples = summarize_frame(frame)
                counts.update(row_counts)
                examples.extend(row_examples)
        if samples != expected_indices:
            raise ValueError(f'{weather}: frames differ from protocol indices')
        saved = death['conditions'][weather]
        if len(samples) != int(saved['frames']):
            raise ValueError(f'{weather}: frame count differs from death report')
        if counts['targets'] != int(saved['target_action_counts']['Oracle-Task']['targets']):
            raise ValueError(f'{weather}: target count differs from death report')
        for local_key, saved_key in (
            ('task_changed', 'changed'),
            ('task_separated_selected', 'task_separated'),
        ):
            if counts[local_key] != int(saved['target_action_counts']['Oracle-Task'][saved_key]):
                raise ValueError(f'{weather}: {local_key} differs from death report')
        result['conditions'][weather] = {
            'frames': len(samples),
            'counts': dict(counts),
            'final_ap70': {name: float(saved['results'][name]['ap70'])
                           for name in ('Shared', 'Oracle-Same', 'Oracle-Task')},
            'path_local_examples': examples,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = analyze_run(args.run_dir)
    destination = args.output or args.run_dir / 'oracle_action_audit.json'
    if destination.exists():
        parser.error(f'output already exists: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding='utf-8')
    print(destination)


if __name__ == '__main__':
    main()
