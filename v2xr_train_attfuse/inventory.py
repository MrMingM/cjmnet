"""Read-only frame-count evidence from exact archive member names, never invent frames."""
from collections import Counter
import json
from pathlib import Path

from auditlib import REPRO, atomic_json, identity, read_json, sha256

OFFICIAL_REFERENCE_FILES = ('readme.md', 'opencood/data_utils/datasets/opv2v/basedataset.py')


def official_reference_hashes():
    """Validate case against the frozen manifest, including on case-insensitive Windows."""
    expected = read_json(REPRO / 'UPSTREAM.json')['files']
    result = {}
    for name in OFFICIAL_REFERENCE_FILES:
        if name not in expected:
            raise ValueError('Official reference spelling/case absent from UPSTREAM.json: ' + name)
        digest = sha256(REPRO / 'upstream' / name)
        if digest != expected[name]:
            raise ValueError('Official reference differs from frozen source: ' + name)
        result[name] = digest
    return result


def timestamp_overview(timestamps):
    ordered = sorted(timestamps, key=lambda t: (int(t), t))
    differences = [int(b) - int(a) for a, b in zip(ordered, ordered[1:])]
    positive = Counter(x for x in differences if x > 0)
    step = min(positive, key=lambda x: (-positive[x], x)) if positive else None
    gaps = []
    for a, b, delta in zip(ordered, ordered[1:], differences):
        if step is not None and delta > step:
            plausible = delta % step == 0
            candidates = list(range(int(a) + step, min(int(b), int(a) + step * 21), step)) if plausible else []
            gaps.append(dict(before=a, after=b, delta=delta, modal_step=step,
                candidate_missing_count=delta // step - 1 if plausible else None,
                candidate_timestamps=[str(t).zfill(len(a)) for t in candidates],
                interpretation='Candidate gap only; variable sampling is possible. No missing frame is established without a reference.'))
    aliases = Counter(int(t) for t in ordered)
    return dict(count=len(timestamps), first=ordered[0] if ordered else None,
        last=ordered[-1] if ordered else None, modal_step=step,
        step_histogram={str(k): v for k, v in sorted(Counter(differences).items())},
        numeric_timestamp_aliases={str(k): [t for t in ordered if int(t) == k]
                                   for k, n in aliases.items() if n > 1}, candidate_gaps=gaps)


def write_inventory(output, args, scenes, empty_scenes, archives, scene_metadata, raw_scene_count):
    per_archive = []
    for archive in archives:
        names = list(archive['entries'])
        official_yaml = sorted(Path(n).stem for n in names if n.endswith('.yaml') and 'additional' not in n)
        ignored = sorted(n for n in names if n.lower().endswith(('.yaml', '.yml'))
                         and not (n.endswith('.yaml') and 'additional' not in n))
        per_archive.append(dict(scene=archive['scene'], cav=archive['cav'], archive=archive['path'],
            archive_sha256=archive['sha256'], archive_size=archive['size'], kind=archive['kind'],
            excluded_from_loader=archive['excluded_from_loader'],
            official_yaml_timestamps=official_yaml, official_yaml_count=len(official_yaml),
            selected_timestamps=archive['timestamps'], ignored_yaml_names=ignored,
            member_names=sorted(names), timestamp_overview=timestamp_overview(official_yaml)))
    by_key = {(r['scene'], r['cav']): r for r in per_archive}
    rows, frame_keys = [], []
    for scene in scenes:
        cav_rows = [by_key[(scene['scene'], c)] for c in scene['agents']]
        sets = [set(r['official_yaml_timestamps']) for r in cav_rows]
        union = set.union(*sets)
        intersection = set.intersection(*sets)
        ego_count = by_key[(scene['scene'], scene['ego'])]['official_yaml_count']
        rows.append(dict(scene=scene['scene'], ego=scene['ego'], agents=scene['agents'],
            registered_cavs=scene['agents'][:7], empty_agents=scene['empty_agents'],
            selected_frame_count=len(scene['timestamps']), official_ego_yaml_count=ego_count,
            cav_yaml_counts={r['cav']: r['official_yaml_count'] for r in cav_rows},
            cav_timestamp_sets_equal=all(s == sets[0] for s in sets),
            union_count=len(union), intersection_count=len(intersection),
            timestamp_overview=timestamp_overview(by_key[(scene['scene'], scene['ego'])]['official_yaml_timestamps'])))
        frame_keys += [dict(scene=scene['scene'], timestamp=t, ego=scene['ego'],
            available_cavs=scene['agents'], official_registered_cavs=scene['agents'][:7]) for t in scene['timestamps']]
    actual = len(frame_keys)
    official_count = sum(r['official_ego_yaml_count'] for r in rows)
    unique = len({(r['scene'], r['timestamp']) for r in frame_keys})
    report = dict(stage='D-1D_INPUT_INVENTORY', scan_completed=True,
        status='smoke_inventory' if args.smoke else 'count_matches' if actual == 8084 else 'count_mismatch',
        expected_cooperative_frames=8084, actual_cooperative_frames=actual, difference_from_reference=actual - 8084,
        official_filename_rule_ego_count=official_count, unique_physical_frame_keys=unique,
        counter_agrees_with_official_filename_rule=actual == official_count,
        count_check_scope='smoke prefix' if args.smoke else 'all original train scenes',
        source_scene_count=len(scenes) + len(empty_scenes), raw_source_scene_count=raw_scene_count,
        data_scene_count=len(scenes), empty_scene_count=len(empty_scenes),
        source_archive_count=len(archives), empty_archive_count=sum(a['excluded_from_loader'] for a in archives),
        candidate_gap_scene_count=sum(bool(r['timestamp_overview']['candidate_gaps']) for r in rows),
        candidate_gap_scenes=[dict(scene=r['scene'], ego=r['ego'],
            actual_count=r['official_ego_yaml_count'], gaps=r['timestamp_overview']['candidate_gaps'])
            for r in rows if r['timestamp_overview']['candidate_gaps']],
        ignored_yaml_name_count=sum(len(r['ignored_yaml_names']) for r in per_archive),
        ignored_yaml_files=[dict(scene=r['scene'], cav=r['cav'], names=r['ignored_yaml_names'])
                            for r in per_archive if r['ignored_yaml_names']],
        extraction_performed=False, CRC_tests_performed=False, training_allowed=False,
        official_counter='sum of ego YAML counts after official lexical CAV ordering/RSU-last; filename ends .yaml and excludes additional',
        official_references=official_reference_hashes(),
        warning='A temporal gap is only a candidate. 8083 vs 8084 alone cannot identify a missing frame or prove an official typo.')
    metadata = []
    for row in scene_metadata:
        blob = Path(row['path']).read_bytes()
        metadata.append(dict(**row, text_preview=blob[:65536].decode('utf-8', errors='replace'),
            preview_truncated=len(blob) > 65536))
    inventory = dict(summary=report, scenes=rows, empty_scenes=empty_scenes,
        archives=per_archive, frame_keys=frame_keys, scene_metadata=metadata)
    inventory['identity'] = identity(inventory)
    atomic_json(output / 'inventory.json', inventory)
    with (output / 'frame_keys.jsonl').open('w', encoding='utf-8', newline='\n') as stream:
        for row in frame_keys:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    lines = ['scene\tego\tselected_frames\tofficial_ego_yaml\tcav_count\tfirst\tlast\tmodal_step\tcandidate_gaps']
    for row in rows:
        t = row['timestamp_overview']
        lines.append('\t'.join(map(str, (row['scene'], row['ego'], row['selected_frame_count'],
            row['official_ego_yaml_count'], len(row['agents']), t['first'], t['last'], t['modal_step'], len(t['candidate_gaps'])))))
    (output / 'scene_counts.tsv').write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    archive_lines = ['scene\tcav\tkind\tyaml_count\tfirst\tlast\tignored_yaml_count\tarchive_sha256']
    for row in per_archive:
        t = row['timestamp_overview']
        archive_lines.append('\t'.join(map(str, (row['scene'], row['cav'], row['kind'], row['official_yaml_count'],
            t['first'], t['last'], len(row['ignored_yaml_names']), row['archive_sha256']))))
    (output / 'archive_counts.tsv').write_text('\n'.join(archive_lines) + '\n', encoding='utf-8', newline='\n')
    report['inventory_identity'] = inventory['identity']
    report['output_hashes'] = {n: sha256(output / n) for n in
        ('inventory.json', 'frame_keys.jsonl', 'scene_counts.tsv', 'archive_counts.tsv')}
    atomic_json(output / 'inventory_summary.json', report)
    print('TRAIN FRAME INVENTORY:', output / 'inventory_summary.json',
          f'actual={actual}, reference=8084, official-filename-rule={official_count}, training_allowed=False', flush=True)
    return report
