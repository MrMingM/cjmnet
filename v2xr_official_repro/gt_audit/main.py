"""Server-only D-1B, using the successful D-1A runtime without predictions."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
from importlib import metadata
import json
import os
from pathlib import Path
import random
import sys
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(1, str(HERE.parent))
from common import (COMMIT, PACKAGE, atomic_json, identity, jsonable, lock, no_test,
                    read_json, server_only, sha256)
from package_check import verify_package
from inputs import WEATHER, prepare_view, verify_d1a
from observe_gt import observe, ids
from report import TOLERANCE, compare_frame, summarize, write_results


def verify_dataset(dataset, condition, manifest):
    if (len(dataset) != 829 or dataset.train or dataset.training or not dataset.visualize
            or not dataset.fov or not dataset.proj_first or not dataset.use_Lidar
            or dataset.use_Radar or dataset.use_DeLidar or dataset.max_cav != 7
            or dataset.async_flag or dataset.loc_err_flag or dataset.train_sim
            or dataset.eval_sim != (condition != 'clean')
            or (condition != 'clean' and dataset.sim_weather != WEATHER[condition])):
        raise ValueError('Actual official dataset flags differ from D-1A/weather-only changes')
    scenes, keys = [], []
    for content in dataset.scenario_database.values():
        agents = list(content)
        ego = agents[0]
        stamps = [t for t in content[ego] if t != 'ego']
        scene = Path(content[ego][stamps[0]]['yaml']).parent.parent.name
        scenes.append(dict(scene=scene, agents=agents, ego=ego, timestamps=stamps))
        keys.extend([scene, t] for t in stamps)
        for cav, records in content.items():
            if [t for t in records if t != 'ego'] != stamps:
                raise ValueError('Official CAV timestamp alignment changed')
            for stamp in stamps:
                expected = Path(manifest['overlay']) / scene / cav / (stamp + (WEATHER[condition] or '') + '.pcd')
                if Path(records[stamp]['lidar']) != expected or not expected.is_file():
                    raise ValueError('Official eval_sim/sim_weather selected an unexpected point cloud')
    if scenes != manifest['scenes'] or keys != manifest['frame_keys']:
        raise ValueError('Official ego/CAV/frame order differs from D-1A')


def run(args, output, attempt, source_hashes):
    evidence, d1a_frames = verify_d1a(args)
    manifest = prepare_view(args, output, evidence, d1a_frames)
    # Isolation is the identical D-1A bootstrap helper. No runtime installation/build.
    from observe import bootstrap, module_inventory
    runtime = bootstrap(args.runtime_root)
    import numpy as np
    import torch
    import yaml
    from opencood.data_utils import datasets
    from opencood.hypes_yaml import yaml_utils
    from run import config_difference
    if os.environ.get('PYTHONHASHSEED') != '20261016':
        raise ValueError('Use run_server.sh to fix PYTHONHASHSEED before Python starts')
    if datasets.COM_RANGE != 70:
        raise ValueError('Official COM_RANGE differs')
    reference_path = runtime / 'opencood/hypes_yaml/V2X-R/LiDAR/V2XR_attfuse.yaml'
    supplied = yaml.safe_load((Path(args.model_dir) / 'config.yaml').read_text())
    reference = yaml.safe_load(reference_path.read_text())
    if any(k not in ('root_dir', 'validate_dir') for k in config_difference(reference, supplied)):
        raise ValueError('D-1A config differs from the pinned official template')
    base = yaml_utils.load_yaml(str(Path(args.model_dir) / 'config.yaml'))
    if jsonable(base) != evidence['protocol']['evaluated_hypes']:
        raise ValueError('Parsed D-1A config changed from successful run')
    conditions = {}
    evaluated_configs = {}
    for condition, suffix in WEATHER.items():
        config = copy.deepcopy(base)
        config['validate_dir'] = manifest['overlay']
        if condition != 'clean':
            config['eval_sim'] = True
            config['sim_weather'] = suffix
        permitted = {'validate_dir'} if condition == 'clean' else {'validate_dir', 'eval_sim', 'sim_weather'}
        if any(k not in permitted for k in config_difference(jsonable(base), jsonable(config))):
            raise ValueError('Unexpected config adaptation')
        dataset = datasets.build_dataset(config, visualize=True, train=False)
        verify_dataset(dataset, condition, manifest)
        if type(dataset).__module__ + '.' + type(dataset).__name__ != evidence['protocol']['actual_dataset_class']:
            raise ValueError('Actual dataset class differs from D-1A')
        conditions[condition] = dataset
        evaluated_configs[condition] = config
    modules = module_inventory(runtime)
    for name, row in modules.items():
        previous = evidence['protocol']['imported_modules'].get(name)
        if row['path'].endswith('.so') and previous != row:
            raise ValueError(f'Native extension differs from successful D-1A: {name}')
    versions = {n: metadata.version(n) for n in evidence['protocol']['versions']}
    if versions != evidence['protocol']['versions'] or torch.version.hip != evidence['protocol']['hip_version']:
        raise ValueError('Runtime dependency versions differ from successful D-1A')
    selected = manifest['frame_keys'][:args.limit_frames] if args.limit_frames else manifest['frame_keys']
    contract = dict(stage='D-1B', source_hashes=source_hashes,
        input_identity=manifest['identity'], d1a_evidence=evidence['artifacts'],
        runtime_modules=modules, evaluated_configs=evaluated_configs, versions=versions,
        hip_version=torch.version.hip, selected_frame_keys=selected, geometry_tolerance=TOLERANCE,
        seed=20261016, rng_rule='reset Python/NumPy/Torch to 20261016 + frame_index before each condition',
        loader_workers=0, dataset_flags=dict(train=False, visualize=True),
        detector_used=False, official_empty_voxel_fallback='preserved and observed')
    contract = jsonable(contract)
    contract['identity'] = identity(contract)
    contract_path = output / 'contract.json'
    if contract_path.exists() and read_json(contract_path) != contract:
        raise ValueError('Source/input/runtime/config changed; use a new OUTPUT_ROOT')
    atomic_json(contract_path, contract)
    protocol = dict(**contract, upstream_commit=COMMIT,
        upstream_source_hashes=read_json(PACKAGE / 'UPSTREAM.json')['files'],
        d1a_summary=evidence['summary'], d1a_model_provenance=evidence['model'],
        native_build_selection=evidence['protocol']['native_build_selection'],
        eval_utils='opencood.utils.eval_utils_opv2v',
        AP='not computed; inherited BEV IoU/VOC2010/global_sort=False protocol is unchanged',
        evaluation_gt='original post_processor.generate_gt_bbx(original collate_batch_test([dataset[index]]))',
        id_mapping='ID list dedup order from original generate_gt_bbx, checked against its exact returned corners',
        box_order='x,y,z,h,w,l,yaw (radians), ego LiDAR coordinates',
        physical_evidence='same native ROI on actual post-FOV/pre-empty-voxel-fallback LiDAR; diagnostic only',
        random_caveat='D-1A DataLoader used 16 workers; audit uses resumable per-frame matched seeds. '
                      'All Clean per-frame GT counts must equal D-1A. Random fallback occurrences are reported.',
        view='output-only symlink overlay of original D-1A files and verified paired weather payload',
        prediction_none_skip=False, input_manifest_sha256=sha256(output / 'input_manifest.json'))
    if (output / 'summary.json').exists():
        summary = read_json(output / 'summary.json')
        if summary['status'] == 'complete' and summary.get('contract_identity') == contract['identity']:
            for name, digest in summary['output_hashes'].items():
                if sha256(output / name) != digest:
                    raise ValueError(f'Completed output changed: {name}')
            print('Matching completed audit verified; no frames rerun.', flush=True)
            return
        raise ValueError('Existing summary has an incompatible completion identity')
    atomic_json(output / 'protocol.json', protocol)
    frame_dir = output / 'frames'
    frame_dir.mkdir(exist_ok=True)
    frames, removed = [], []
    observer_checks = []
    for index, key in enumerate(selected):
        path = frame_dir / f'{index:06d}.json'
        if path.exists():
            record = read_json(path)
            frame = record['frame']
            if (record['contract_identity'] != contract['identity']
                    or record['frame_sha256'] != identity(frame)
                    or frame['index'] != index or frame['frame_key'] != key):
                raise ValueError(f'Invalid resumable frame: {path}')
        else:
            frame = dict(index=index, frame_key=key, conditions={}, observer_checks=[])
            for condition, dataset in conditions.items():
                seed = 20261016 + index
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                states = random.getstate(), np.random.get_state(), torch.get_rng_state()
                row = observe(dataset, index, key)
                # First 12 frames: rerun uninstrumented official methods from identical RNG
                # states, so smoke checks that observers do not change final GT.
                if index < 12:
                    random.setstate(states[0])
                    np.random.set_state(states[1])
                    torch.set_rng_state(states[2])
                    plain = dataset[index]
                    plain_gt = dataset.post_processor.generate_gt_bbx(dataset.collate_batch_test([plain]))
                    if (set(ids(plain['ego']['object_ids'])) != set(row['dataset_gt_ids'])
                            or not np.array_equal(plain_gt.numpy(), np.asarray([r['corners'] for r in row['final_gt']]).reshape(-1, 8, 3))):
                        raise ValueError('Observer changed official GT output')
                    frame['observer_checks'].append(condition)
                if condition == 'clean' and row['final_gt_count'] != d1a_frames[index]['gt_count']:
                    raise ValueError(f'Clean GT count differs from D-1A at {key}; stop without reporting weather conclusions')
                if len(row['context']['participating_cavs']) != d1a_frames[index]['participating_agents']:
                    raise ValueError('Actual participating CAV count differs from D-1A')
                frame['conditions'][condition] = row
            compare_frame(frame)
            atomic_json(path, dict(contract_identity=contract['identity'], frame_sha256=identity(frame), frame=frame))
        # Recompute comparisons even on resume; never trust a partial JSONL tail.
        removed.extend(compare_frame(frame))
        frames.append(frame)
        observer_checks.extend(dict(index=index, condition=c) for c in frame['observer_checks'])
        atomic_json(output / 'progress.json', dict(status='running', completed_frames=len(frames),
            requested_frames=len(selected), latest_frame_key=key, contract_identity=contract['identity']))
        if index % 5 == 0 or len(selected) <= 20 or index + 1 == len(selected):
            print(f'D-1B {index + 1}/{len(selected)} {key}: ' + ' '.join(
                f'{c}={frame["conditions"][c]["final_gt_count"]}' for c in WEATHER), flush=True)
    for name, rows in (('per_frame.jsonl', frames), ('removed_gt.jsonl', removed)):
        tmp = output / (name + '.tmp')
        with tmp.open('w', encoding='utf-8', newline='\n') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        os.replace(tmp, output / name)
    summary = summarize(frames, removed)
    if summary['full_validation'] and summary['Clean_GT_total'] != 6224:
        raise ValueError('Full Clean GT total differs from accepted D-1A')
    summary.update(contract_identity=contract['identity'], clean_per_frame_counts_match_d1a=True,
                   clean_total_expected_d1a=6224, observer_equivalence_checks=observer_checks)
    protocol['imported_modules_final'] = module_inventory(runtime)
    protocol['observer_equivalence_checks'] = observer_checks
    atomic_json(output / 'protocol.json', protocol)
    # Completion marker last, after reports and output hashes.
    write_results(output, summary)
    summary['output_hashes'] = {n: sha256(output / n) for n in
        ('per_frame.jsonl', 'removed_gt.jsonl', 'RESULTS.md', 'protocol.json', 'input_manifest.json')}
    atomic_json(output / 'progress.json', dict(status='complete', completed_frames=len(frames), requested_frames=len(selected)))
    atomic_json(output / 'summary.json', summary)
    print((output / 'RESULTS.md').read_text(), flush=True)


def main():
    server_only()
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = dict(runtime_root='/data/cjm/datasets/v2xr-official-runtime-v1',
        model_dir='/data/cjm/datasets/logs/v2xr-official-attfuse-model-v1',
        view_root='/data/cjm/datasets/v2xr-official-validate-v1',
        paired_root='/data/cjm/datasets/v2xr-lidar-weather-validate-v1',
        d1a_root='/data/cjm/datasets/logs/v2xr-official-attfuse-clean-repro-v1',
        output_root='/data/cjm/datasets/logs/v2xr-official-gt-audit-v1')
    for name, default in defaults.items():
        parser.add_argument('--' + name.replace('_', '-'), default=default)
    parser.add_argument('--limit-frames', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not 0 <= args.limit_frames <= 829:
        raise ValueError('Frame limit must be 0 (all) or 1..829')
    sources = verify_package()
    for name in defaults:
        setattr(args, name, str(no_test(getattr(args, name)).resolve()))
    output = Path(args.output_root)
    for source in [PACKAGE.parent] + [Path(getattr(args, n)) for n in defaults if n != 'output_root']:
        if output == source or source in output.parents or output in source.parents:
            raise ValueError('Use an independent output directory on the data disk')
    if Path('/data') not in output.parents or Path('/data') not in Path(os.environ.get('TMPDIR', '')).resolve().parents:
        raise ValueError('OUTPUT_ROOT and TMPDIR must be on /data')
    request = dict(arguments={k: v for k, v in vars(args).items() if k != 'resume'}, source_hashes=sources)
    with lock(output.with_suffix('.lock')), lock(Path(args.runtime_root).with_suffix('.run.lock')):
        marker = output / 'request.json'
        if output.exists() and any(output.iterdir()):
            if not args.resume or not marker.exists() or read_json(marker) != request:
                raise ValueError('Existing output requires RESUME=1 and identical request/source')
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(marker, request)
        attempt = output / 'attempts' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        attempt.mkdir(parents=True)
        try:
            run(args, output, attempt, sources)
        except Exception as error:
            failure = dict(status='failed', error=repr(error), traceback=traceback.format_exc())
            atomic_json(attempt / 'failure.json', failure)
            atomic_json(output / 'last_failure.json', failure)
            raise


if __name__ == '__main__':
    main()
