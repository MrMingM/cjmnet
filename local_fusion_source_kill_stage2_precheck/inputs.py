"""Resolve existing evidence and hash-check it before CPU deserialization."""
from collections import Counter
import json
import os
from pathlib import Path
from .common import (ROOT, HISTORICAL, WEATHERS, digest, read_json, safe_path, object_hash)
from .source_redundancy import analyze_group
from .data_adequacy import separation_proxy
from .oracle_records import analyze_target


def load_reports(folder):
    folder = safe_path(folder)
    hashes = {name: digest(folder / name) for name in HISTORICAL}
    k0 = read_json(folder / 'K0_GROUP_MARGIN.json')
    k10 = read_json(folder / 'K1_0_ORACLE_DECOMPOSITION.json')
    final = read_json(folder / 'STAGE1_FINAL.json')
    if final['status'] != 'complete' or final['K0_PRIME'] != k0 or final['K1_0'] != k10:
        raise ValueError('Standalone Stage 1 reports conflict with final report')
    if not final['k3_frozen'] or final['test_data_used'] or not k10['k3_frozen']:
        raise ValueError('Unexpected Stage 1 scope')
    gate = k10['reproduction_gate']
    if gate['status'] != 'complete' or gate['absolute_ap_tolerance'] != 1e-6 or len(gate['checks']) != 8:
        raise ValueError('Original FULL reproduction gate missing')
    for row in gate['checks'].values():
        if any(value > 1e-6 for metrics in row.values() for value in metrics.values()):
            raise ValueError('Original FULL reproduction gate failed')
    return k0, k10, final, hashes


def discover_stage1(reports_folder, explicit=None):
    """Select by matching completed report hashes, never by most recent timestamp."""
    if explicit:
        candidates = [safe_path(explicit)]
    else:
        candidates = list(Path('/data/cjm/datasets/logs').glob('source_kill_stage1_*'))
    expected = {name: digest(Path(reports_folder) / name) for name in HISTORICAL}
    found = []
    for candidate in candidates:
        candidate = safe_path(candidate)
        if not (candidate / 'manifest.json').is_file():
            continue
        if read_json(candidate / 'manifest.json').get('status') != 'complete':
            continue
        if all((candidate / name).is_file() and digest(candidate / name) == value for name, value in expected.items()):
            found.append(candidate)
    if len(found) != 1:
        raise ValueError(f'Expected one completed Stage 1 run matching report hashes, found {len(found)}; '
                         'set STAGE1_RUN explicitly (no latest-run guessing)')
    return found[0]


def full_inputs(stage1, reports_hashes):
    stage1 = safe_path(stage1)
    protocol = read_json(stage1 / 'protocol.json')
    manifest = read_json(stage1 / 'manifest.json')
    if manifest['status'] != 'complete':
        raise ValueError('Stage 1 run incomplete')
    for name, expected in reports_hashes.items():
        if digest(stage1 / name) != expected:
            raise ValueError('Stage 1 run/report mismatch: ' + name)
    # Only hash-check the dependencies used by this read-only audit. No current
    # detector/config/checkpoint or dataset access is required.
    for name in ('local_fusion_action_utility_audit/counterfactual.py',
                 'local_fusion_source_kill_stage1/oracle_adapter.py'):
        if digest(ROOT / name) != protocol['source_hashes'][name]:
            raise ValueError('Original key/partition semantics changed: ' + name)
    refs = protocol['references']
    s0_run = safe_path(refs['s0_run'])
    if digest(s0_run / 'protocol.json') != refs['s0_protocol_hash'] or digest(s0_run / 'manifest.json') != refs['s0_manifest_hash']:
        raise ValueError('S0 protocol/manifest changed since Stage 1')
    s0 = read_json(s0_run / 'protocol.json')
    s0_manifest = read_json(s0_run / 'manifest.json')
    if s0_manifest['identity'] != s0['identity']:
        raise ValueError('S0 identity mismatch')
    if s0['validation_indices'] != protocol['validation_indices']:
        raise ValueError('S0/Stage 1 validation split mismatch')
    for key in ('root_dir', 'validate_dir'):
        safe_path(s0['dataset_paths'][key])  # Metadata only; never enumerate a dataset.
    for split in ('train', 'validation'):
        indices = s0[split + '_indices']
        if not indices or len(indices) != len(set(indices)):
            raise ValueError('S0 split indices invalid')
    return protocol, s0_run, s0, s0_manifest


def read_s0(stage1, reports_hashes, k0):
    protocol, run, s0, manifest = full_inputs(stage1, reports_hashes)
    # Explicitly CPU-only, before importing Torch to deserialize existing .pt.
    for name in ('ROCR_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES'):
        os.environ[name] = ''
    import torch
    from local_fusion_source_kill_stage1.k0_group_margin import groups_from_frame
    rows, pairs, scene_map = [], [], {}
    for split in ('train', 'validation'):
        for weather in WEATHERS:
            for index in s0[split + '_indices']:
                name = f'cache/{split}/{weather}/{index:08d}.pt'
                entry = manifest['entries'].get(f'S0-frame/{split}/{weather}/{index}', {})
                path = safe_path(run / name)
                if run not in path.parents or entry.get('status') != 'complete' or digest(path) != entry['artifacts'].get(name):
                    raise ValueError('Original S0 cache incomplete/corrupted: ' + name)
                frame = torch.load(path, map_location='cpu', weights_only=True)
                if (frame['split'], frame['weather'], frame['frame']) != (split, weather, index):
                    raise ValueError('Cache group identity changed')
                if frame['scene'] not in s0[split + '_scenes']:
                    raise ValueError('Cache scene outside recorded split')
                scene = split + ':' + str(frame['scene'])
                if (split, index) in scene_map and scene_map[split, index] != scene:
                    raise ValueError('Scene changes across weather')
                scene_map[split, index] = scene
                # Reuse original Stage 1 schema guards; no detector is imported.
                guarded = list(groups_from_frame(frame))
                computed = []
                for label in frame['labels']:
                    anchor = frame['proposals']['anchor_ids'][label['proposal_position']]
                    task_rows = {}
                    for task in ('cls', 'reg'):
                        result = analyze_group(label['outcomes'][task], frame['source_names'])
                        row = dict(split=split, weather=weather, frame=index, scene_id=scene,
                                   task=task, stratum='background' if label['background'] else 'associated',
                                   proposal=anchor, focal=label['target_index'],
                                   physical_id=[split, index, anchor, label['target_index']], **result)
                        rows.append(row)
                        computed.append(row)
                        task_rows[task] = row
                    pairs.append({**{k: task_rows['cls'][k] for k in (
                        'split', 'weather', 'frame', 'scene_id', 'stratum', 'physical_id')},
                        **separation_proxy(task_rows['cls'], task_rows['reg'])})
                if [r['beneficial'] for r in computed] != [r['has_discrete_beneficial_action'] for r in guarded]:
                    raise ValueError('P0 beneficial groups disagree with original K0 definition')
            print(f'P0 CPU read-only S0 {split}/{weather} complete', flush=True)
    # Exact count AND scene-map reproduction of the original K0 summaries.
    for split in ('train', 'validation'):
        for weather in WEATHERS:
            for task in ('cls', 'reg'):
                for stratum in ('associated', 'background'):
                    key = f'{split}/{weather}/{task}/{stratum}'
                    selected = [r for r in rows if (r['split'], r['weather'], r['task'], r['stratum']) ==
                                (split, weather, task, stratum)]
                    old = k0['conditions'][key]
                    counts = Counter(r['scene_id'].split(':', 1)[1] for r in selected if r['beneficial'])
                    if len(selected) != old['group_count'] or dict(counts) != old['beneficial_set_scene_coverage']['counts']:
                        raise ValueError('P0 cannot reproduce K0 counts/scenes: ' + key)
    return rows, pairs, scene_map, protocol


def read_oracle_details(stage1, k10, scene_map, protocol):
    rows = []
    for variant in ('FULL', 'A_ONLY'):
        for weather in WEATHERS:
            for mode in ('same', 'task'):
                name = f'{weather}_{mode}_{variant}'
                condition = k10['conditions'][name]
                path = safe_path(stage1 / condition['frame_artifact'])
                if stage1 not in path.parents or digest(path) != condition['frame_sha256']:
                    raise ValueError('Original Oracle frame records changed: ' + name)
                frames = []
                counts = Counter()
                with path.open(encoding='utf-8') as stream:
                    for line in stream:
                        frame = json.loads(line)
                        index = frame['sample_index']
                        frames.append(index)
                        if len(frame['records']) != len(frame['current_state_audit']):
                            raise ValueError('Oracle record/audit length mismatch')
                        targets = []
                        for record, audit in zip(frame['records'], frame['current_state_audit']):
                            result = analyze_target(record, audit, mode, variant)
                            targets.append(result['target_index'])
                            rows.append(dict(split='validation', weather=weather, mode=mode, variant=variant,
                                             frame=index, scene_id=scene_map['validation', index], **result))
                            counts['targets'] += 1
                            counts['changed'] += result['changed']
                            counts['task_separated'] += result['task_separated']
                            counts['selected_A'] += result['selected_group'] == 'A'
                            for field in ('step_recovered', 'step_lost', 'step_new_fp'):
                                counts[field] += result[field]
                        if len(targets) != len(set(targets)):
                            raise ValueError('Duplicate target in saved Oracle trajectory')
                if frames != protocol['validation_indices'] or len(frames) != condition['frames']:
                    raise ValueError('Incomplete Oracle frame records')
                if any(counts[k] != condition['counts'][k] for k in counts):
                    raise ValueError('Oracle counts do not reproduce Stage 1 summary: ' + name)
    return rows
