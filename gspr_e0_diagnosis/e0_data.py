"""Paired validation geometry using the original loader, masks and voxelizer."""
from pathlib import Path
import random
import numpy as np
from e0_common import WEATHERS, atomic, frame_commit, frame_done, identity, read, sha
from e0_numeric import coverage, nearest_proxy, range_group
from e0_runtime import npz_write, points_in_targets, smoke_indices


def audit_pairs(cli, folder, request_id, evidence, limit=False):
    from opencood.data_utils.datasets import build_dataset
    from predict import live_gt_observer
    from support import compare_gt
    from opencood.utils import pcd_utils, box_utils
    import torch
    import copy
    datasets = {w: build_dataset(copy.deepcopy(evidence['b']['evaluated_configs'][w]), visualize=True, train=False) for w in WEATHERS}
    for weather, ds in datasets.items():
        if (len(ds) != 829 or ds.train or ds.async_flag or ds.loc_err_flag or not ds.proj_first
                or ds.use_Radar or not ds.use_Lidar or ds.use_DeLidar):
            raise ValueError('Unexpected paired data configuration')
        suffix = {'clean': '', 'fog': '_fog_0.060', 'snow': '_snow_2.5_2.0'}[weather]
        recovered = []
        for scene_index, scene in enumerate(evidence['input_manifest']['scenes']):
            actual = ds.scenario_database[scene_index]
            if list(actual) != scene['agents']:
                raise ValueError('Paired vehicle order changed')
            for cav, stamps in actual.items():
                if [s for s in stamps if s != 'ego'] != scene['timestamps']:
                    raise ValueError('Paired timestamps differ')
                for stamp in scene['timestamps']:
                    expected = Path(evidence['input_manifest']['overlay']) / scene['scene'] / cav / (stamp+suffix+'.pcd')
                    if Path(stamps[stamp]['lidar']) != expected:
                        raise ValueError('Unverified weather suffix or file pairing')
            recovered.extend([scene['scene'], s] for s in scene['timestamps'])
        if recovered != evidence['b']['selected_frame_keys']:
            raise ValueError('Paired frame order changed')
    indices = sorted(set(smoke_indices(evidence)) | set(range(12))) if limit else list(range(829))
    for index in indices:
        output = folder / 'frames' / f'{index:06d}.json'
        if frame_done(output, request_id):
            previous = read(output)
            if sha(folder / previous['clean_cloud']) != previous['clean_cloud_sha256']:
                raise ValueError('Clean spatial proxy cloud changed')
            for artifact in previous['author_mask_clouds'].values():
                if sha(folder / artifact['path']) != artifact['sha256']:
                    raise ValueError('Author color-mask evidence changed')
            continue
        fixed = evidence['frames'][index]['conditions']['clean']['final_gt']
        key = evidence['b']['selected_frame_keys'][index]
        observations = {}
        for weather, ds in datasets.items():
            # Analysis voxel statistics have their own explicit per-frame order.
            # Original inference workers/order remain untouched in all model runs.
            # Exact per-frame rule of the frozen D-1B GT audit, including fallback RNG.
            seed = 20261016 + index
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            captured = []
            original_car = ds.get_item_single_car
            original_pcd = pcd_utils.pcd_to_np
            observed_green = {}
            loaded_raw = {}
            paths_by_cav = {str(r['cav']): r['lidar_file'] for r in evidence['frames'][index]['conditions'][weather]['context']['cavs']}

            def observe_pcd(path, *args, **kwargs):
                result = original_pcd(path, *args, **kwargs)
                if kwargs.get('get_weather', args[0] if args else False):
                    observed_green[str(Path(path))] = result[1].copy()
                    loaded_raw[str(Path(path))] = dict(raw_points=len(result[0]),
                        raw_range_counts=np.bincount(range_group(np.linalg.norm(result[0][:, :3], axis=1)), minlength=4).tolist(),
                        author_color_zero_points=int((result[1] == 0).sum()),
                        author_color_nonzero_points=int((result[1] != 0).sum()))
                return result

            def observe_car(base, *args):
                calls = []
                native_preprocess = ds.pre_processor.preprocess
                def preprocess(points):
                    result = native_preprocess(points)
                    calls.append(dict(points=points.copy(), voxels=len(result['voxel_num_points'])))
                    return result
                ds.pre_processor.preprocess = preprocess
                try:
                    result = original_car(base, *args)
                finally:
                    ds.pre_processor.preprocess = native_preprocess
                # Reviewed LiDAR-only function calls radar then lidar, before fallback loops.
                if len(calls) < 2:
                    raise ValueError('Official preprocessing call structure changed')
                cav = str(base['id'])
                green = observed_green[str(Path(paths_by_cav[cav]))]
                if green.shape != (len(base['lidar_np']),) or not np.isfinite(green).all():
                    raise ValueError('Author color channel does not align with loaded points')
                if not np.array_equal(base['de_lidar_np'], base['lidar_np'][green == 0]):
                    raise ValueError('Observed author mask differs from native de_lidar_np')
                captured.append(dict(cav=cav, raw=base['lidar_np'].copy(), kept=calls[1]['points'],
                    author_xyz=box_utils.project_points_by_matrix_torch(base['lidar_np'][:, :3], base['params']['transformation_matrix']),
                    author_nonzero=(green != 0),
                    initial_voxels=calls[1]['voxels'],
                    coords=result['processed_lidar_features']['voxel_coords'].copy(),
                    counts=result['processed_lidar_features']['voxel_num_points'].copy(),
                    pose=list(base['params']['lidar_pose']),
                    transform=np.asarray(base['params']['transformation_matrix']).copy()))
                return result

            ds.get_item_single_car = observe_car
            pcd_utils.pcd_to_np = observe_pcd
            try:
                sample = ds[index]
            finally:
                ds.get_item_single_car = original_car
                pcd_utils.pcd_to_np = original_pcd
            expected = evidence['frames'][index]['conditions'][weather]
            cavs = expected['context']['participating_cavs']
            if len(captured) != len(cavs) or len(sample['ego']['processed_lidar']['voxel_features']) != len(cavs):
                raise ValueError('Paired participating CAV alignment changed')
            if [r['cav'] for r in captured] != list(map(str, cavs)):
                raise ValueError('Actual participating CAV identities/order differ')
            batch = ds.collate_batch_test([sample])
            _, gt = live_gt_observer(ds.post_processor, batch, ds.post_processor.generate_gt_bbx)
            compare_gt(gt, expected['final_gt'], 'paired live official GT')
            excluded = []
            for item in expected['context']['cavs']:
                if str(item['cav']) not in list(map(str, cavs)):
                    excluded.append(dict(cav=item['cav'], distance_to_ego_m=item['distance_to_ego_m'],
                        reason='loaded by native dataset but excluded from model participation',
                        **loaded_raw[str(Path(item['lidar_file']))]))
            observations[weather] = dict(cavs={str(c): o for c, o in zip(cavs, captured)}, gt=gt,
                excluded_loaded_cavs=excluded,
                kept=np.concatenate([o['kept'][:, :3] for o in captured]),
                author_xyz=np.concatenate([o['author_xyz'] for o in captured]),
                author_nonzero=np.concatenate([o['author_nonzero'] for o in captured]),
                author_agents=np.concatenate([np.full(len(o['author_xyz']), a, np.int16) for a, o in enumerate(captured)]),
                agents=np.concatenate([np.full(len(o['kept']), a, np.int16) for a, o in enumerate(captured)]))
        clean = observations['clean']
        record = dict(index=index, frame_key=key, conditions={}, author_mask_clouds={})
        for weather, observation in observations.items():
            if observation['cavs'].keys() != clean['cavs'].keys():
                raise ValueError('Weather participating vehicles differ')
            agents = []
            for cav, target in observation['cavs'].items():
                base = clean['cavs'][cav]
                if not np.array_equal(base['pose'], target['pose']) or not np.array_equal(base['transform'], target['transform']):
                    raise ValueError('Paired sensor pose/transformation differs')
                proxy_raw, _ = nearest_proxy(base['raw'][:, :3], target['raw'][:, :3])
                proxy_kept, _ = nearest_proxy(base['kept'][:, :3], target['kept'][:, :3])
                counts = target['counts']
                base_cells = {tuple(c[-2:]) for c in base['coords']} if base['initial_voxels'] else set()
                target_cells = {tuple(c[-2:]) for c in target['coords']} if target['initial_voxels'] else set()
                agents.append(dict(cav=cav, pose=target['pose'], raw_points=len(target['raw']),
                    author_color_zero_points=int((~target['author_nonzero']).sum()),
                    author_color_nonzero_points=int(target['author_nonzero'].sum()),
                    raw_range_counts=np.bincount(range_group(np.linalg.norm(target['raw'][:, :3], axis=1)), minlength=4).tolist(),
                    retained_real_points=len(target['kept']),
                    official_voxels=len(counts), initial_real_voxels=target['initial_voxels'],
                    synthetic_lidar_fallback=target['initial_voxels'] == 0,
                    occupied_pillars=len({tuple(c[-2:]) for c in target['coords']}),
                    few_point_pillars=int((counts <= 3).sum()), high_density_pillars=int((counts >= 16).sum()),
                    voxel_count_histogram=np.bincount(counts.astype(int), minlength=33).tolist(),
                    spatial_coverage=coverage(base['kept'], target['kept']), raw_nearest_proxy=proxy_raw,
                    retained_nearest_proxy=proxy_kept,
                    native_voxel_coverage=dict(clean_occupied=len(base_cells), weather_occupied=len(target_cells),
                        intersection=len(base_cells & target_cells), missing_clean_cells=len(base_cells-target_cells),
                        additional_weather_cells=len(target_cells-base_cells))))
            inside = points_in_targets(observation['kept'], fixed)
            nearby = points_in_targets(observation['kept'], fixed, 1.5)
            clean_inside = points_in_targets(clean['kept'], fixed)
            clean_nearby = points_in_targets(clean['kept'], fixed, 1.5)
            targets = []
            for j, obj in enumerate(fixed):
                a = observation['kept'][inside[j]]
                b = clean['kept'][clean_inside[j]]
                targets.append(dict(object_id=obj['object_id'], box=obj['box'], points=len(a), clean_points=len(b),
                    coverage=coverage(b, a),
                    nearby_points=int(nearby[j].sum()), clean_nearby_points=int(clean_nearby[j].sum()),
                    nearby_coverage=coverage(clean['kept'][clean_nearby[j]], observation['kept'][nearby[j]])))
            record['conditions'][weather] = dict(agents=agents, targets=targets,
                excluded_loaded_cavs=observation['excluded_loaded_cavs'],
                total_retained_real_points=len(observation['kept']),
                union_coverage=coverage(clean['kept'], observation['kept']),
                gt_identity=identity(observation['gt']))
            color_path = folder / 'author_mask_clouds' / weather / f'{index:06d}.npz'
            npz_write(color_path, xyz=observation['author_xyz'].astype(np.float32),
                nonzero=observation['author_nonzero'], agents=observation['author_agents'])
            record['author_mask_clouds'][weather] = dict(path=color_path.relative_to(folder).as_posix(), sha256=sha(color_path))
        cloud = folder / 'clean_clouds' / f'{index:06d}.npz'
        npz_write(cloud, xyz=clean['kept'].astype(np.float32), agents=clean['agents'])
        record.update(clean_cloud=cloud.relative_to(folder).as_posix(), clean_cloud_sha256=sha(cloud))
        frame_commit(output, record, request_id)
        print(f'PAIRED {index+1}/829 {key}', flush=True)
    summary = dict(status='complete', indices=indices, frame_count=len(indices),
        point_labels='spatial proxies only; no simulator origin/return identity available',
        author_mask='additional original green==0 vs nonzero mask, used to form de_lidar_np by official loader; origin semantics unverified; do not equate with physical truth',
        counts_are='per CAV counts sum observations; occupied union is in common ego XY coordinates',
        cav_scope='participating CAV geometry and model diagnostics; additionally list raw counts/ranges/author masks for native-loaded but nonparticipating CAVs',
        coordinate_checks='same scene/timestamp/CAV, shared YAML, equal lidar poses and actual native transformations',
        voxelization='official inference voxelizer; reset Python/NumPy/Torch to 20261016+frame_index before each condition, matching frozen D-1B; inference-slot statistics collected separately',
        geometry_scope='real post-range/mask-ego/projection/FOV points presented to voxelizer, before its range/cap selection; native_voxel_coverage separately measures actual grid occupancy',
        actual_preprocess=evidence['b']['evaluated_configs']['clean']['preprocess'],
        empty_cells='missing_clean_cells counts Clean-occupied cells absent in weather; no invented total free-space denominator',
        fallback='artificial LiDAR fallback excluded from real-point geometric coverage; official voxel counts reported separately',
        frames=[str((folder / 'frames' / f'{i:06d}.json').relative_to(folder)) for i in indices])
    atomic(folder / 'E0_POINT_DEGRADATION.json', summary, immutable=True)
