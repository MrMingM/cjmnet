"""Build immutable 64 candidate + 24 successful-control frame manifest on CPU."""
import argparse
import json
from pathlib import Path
from collections import Counter
from gspr_evidence import stage3_runtime as sr
from gspr_evidence.stage3_analysis import read_json, write_json
from qa_observation_diagnostic.metrics import distance_bin
from .common import COUNTS, SEED, sample_rows


def check_b(root, weather):
    p = read_json(root/weather/'protocol.json'); s = read_json(root/weather/'summary.json')
    if p['stage'] != '3B' or p['smoke'] or not p['development_only'] or p['test_data_used'] or not s['complete']:
        raise ValueError('Require completed full development Stage-3B')
    errors = []
    for name, old in p['implementation_sha256'].items():
        path = sr.ROOT/name
        actual = sr.source_sha(path) if path.is_file() else None
        if actual != old:
            errors.append(dict(file=name, stage3b_sha256=old, current_sha256=actual))
    if errors:
        raise ValueError('Historical implementation mismatches (no files overwritten): '+json.dumps(errors))
    if sr.validate_sources() != p['audited_sources']:
        raise ValueError('Historical source manifest mismatch')
    if sr.sha(root/weather/'diagnostics.jsonl') != s['diagnostics_sha256']:
        raise ValueError('Stage-3B diagnostics changed')
    return p


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--stage3b-root', required=True); p.add_argument('--stage2-root', required=True)
    p.add_argument('--output-dir', required=True); p.add_argument('--seed', type=int, default=SEED)
    args = p.parse_args()
    out = sr.safe_output(args.output_dir, create=True)
    root = sr.safe_output(args.stage3b_root)
    plan = dict(schema=1, development_only=True, test_data_used=False, seed=args.seed,
        stage3b_root=str(root), stage2_root=str(Path(args.stage2_root).resolve()), weathers={}, input_files={},
        code_sha256={p.relative_to(sr.ROOT).as_posix():sr.source_sha(p) for p in Path(__file__).parent.glob('*.py')},
        selection='75% primary failure stage + remainder other stages (availability fallback); within pool balance global recoverability then scene then distance; seeded hash ties; one focal target/frame',
        controls='Stage-2 any-peer-valid + full-detected targets in frames with NO Stage-3B candidate; clean-ego-hit/weather-ego-miss audit subset, not all healthy frames',
        interpretation='GT-localized upper-bound experiment; selected diagnostic sample, not AP or population rate')
    for weather, n in COUNTS.items():
        protocol = check_b(root, weather)
        if protocol['stage2_root'] != plan['stage2_root']:
            raise ValueError('Stage-2 root mismatch')
        lineage = sr.load_lineage(args.stage2_root, weather)
        plan['input_files'].update(lineage['input_files'])
        for path, digest in {**protocol.get('input_files', {}), **protocol['stage3a_input_files']}.items():
            if sr.sha(path) != digest:
                raise ValueError('Historical Stage-3B input changed: '+path)
            plan['input_files'][path] = digest
        for name in ('protocol.json', 'summary.json', 'diagnostics.jsonl'):
            path = root/weather/name; plan['input_files'][str(path)] = sr.sha(path)
        rows, seen = [], set()
        with (root/weather/'diagnostics.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                f = json.loads(line); index = f['sample_index']
                if index in seen:
                    raise ValueError('Duplicate Stage-3B frame')
                seen.add(index)
                for j in f['candidate_targets']:
                    meta = f['target_metadata'][str(j)]
                    original = lineage['rows'][index, j]
                    if not original['source_valid_full_miss']:
                        raise ValueError('Invalid source-valid candidate')
                    recoverable = any(j in b['recovered_candidates'] for b in f['branches']
                        if b['family'] in ('peer_score_full_geometry', 'full_score_peer_geometry',
                                          'peer_query_scale_all', 'peer_query_scale_0', 'peer_query_scale_1'))
                    rows.append(dict(sample_index=index, target_index=j, cohort='candidate', **meta,
                        distance_bin=distance_bin(meta['distance']), global_recoverable=recoverable,
                        valid_peers=original['detected_peer_indices'], input_sha256=f['input_sha256']))
        if sorted(seen) != protocol['selected_candidate_frames'] or len(rows) != protocol['candidate_targets']:
            raise ValueError('Incomplete Stage-3B frame/target coverage')
        primary = 'score_filtered' if weather == 'snow' else 'nms_suppressed'
        pool = [r for r in rows if r['failure_stage'] == primary]
        count = min(3*n//4, len({r['sample_index'] for r in pool}))
        selected = sample_rows(pool, count, args.seed, weather)
        used = {r['sample_index'] for r in selected}
        others = [r for r in rows if r['failure_stage'] != primary and r['sample_index'] not in used]
        extra = min(n-count, len({r['sample_index'] for r in others}))
        selected += sample_rows(others, extra, args.seed, weather, used)
        if len(selected) < n:
            selected += sample_rows(rows, n-len(selected), args.seed, weather, {r['sample_index'] for r in selected})
        controls = []
        for (index, j), r in lineage['rows'].items():
            if index not in seen and r['any_peer_alone_detected'] and r['full']['matched']:
                meta = lineage['stats'][index, j]
                controls.append(dict(sample_index=index, target_index=j, cohort='control', scene=meta['scene'],
                    distance=meta['distance'], distance_bin=distance_bin(meta['distance']),
                    failure_stage='detected', global_recoverable=False, valid_peers=r['detected_peer_indices'],
                    input_sha256=None))
        selected += sample_rows(controls, 8, args.seed, weather, seen)
        plan['weathers'][weather] = dict(rows=sorted(selected, key=lambda r: r['sample_index']),
            available_candidate_frames=len(seen), available_control_frames=len({r['sample_index'] for r in controls}),
            by_cohort=dict(Counter(r['cohort'] for r in selected)), by_scene=dict(Counter(str(r['scene']) for r in selected)),
            by_failure=dict(Counter(r['failure_stage'] for r in selected)),
            by_global_recoverability=dict(Counter(str(r['global_recoverable']) for r in selected if r['cohort']=='candidate')))
        print(weather, plan['weathers'][weather]['by_cohort'], plan['weathers'][weather]['by_failure'], flush=True)
    for path, old in plan['input_files'].items():
        if sr.sha(path) != old:
            raise ValueError('Input changed during sampling: '+path)
    write_json(out/'sampling_plan.json', plan)
    print('PLAN:', out/'sampling_plan.json', flush=True)


if __name__ == '__main__':
    main()
