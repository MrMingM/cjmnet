"""Observe original dataset.__getitem__, native ROI, collate and generate_gt_bbx.

All observers return the original results unchanged. No detector is created.
"""
import copy
from contextlib import contextmanager
import math


def oid(value):
    # Official np.array(object_ids) may convert integral YAML keys to np scalars.
    if hasattr(value, 'item'):
        value = value.item()
    if not isinstance(value, (str, int)) or isinstance(value, bool):
        raise ValueError(f'Unsupported object ID type: {type(value)}')
    return str(value)


def ids(values):
    result = [oid(v) for v in values]
    if len(set(result)) != len(result):
        raise ValueError('Duplicate/colliding object IDs within one CAV or final frame')
    return result


@contextmanager
def patched(obj, name, replacement):
    original = getattr(obj, name)
    setattr(obj, name, replacement)
    try:
        yield original
    finally:
        setattr(obj, name, original)


def observe(dataset, index, expected_key):
    import numpy as np
    from opencood.data_utils.datasets.opv2v import intermediate_fusion_dataset as module
    from opencood.hypes_yaml import yaml_utils
    from opencood.utils import box_utils
    original_retrieve = dataset.retrieve_base_data
    original_car = dataset.get_item_single_car
    original_fov = dataset.mask_ego_fov_flag
    original_roi = module.points_in_boxes_cpu
    original_mask = box_utils.get_mask_for_boxes_within_range_torch
    cav_rows, active, roi_calls, raw = [], {}, [], []
    context = {}

    def retrieve(*args, **kwargs):
        result = original_retrieve(*args, **kwargs)
        data, scene_index, stamp = result
        database = dataset.scenario_database[scene_index]
        scene = next(iter(database.values()))[stamp]['yaml']
        from pathlib import Path
        scene = Path(scene).parent.parent.name
        if [scene, stamp] != expected_key:
            raise ValueError('Physical frame changed in retrieve_base_data')
        ego = next(k for k, v in data.items() if v['ego'])
        ego_pose = data[ego]['params']['lidar_pose']
        context.update(scene=scene, timestamp=stamp, ego=str(ego), ego_pose=copy.deepcopy(ego_pose), cavs=[])
        for cav, content in data.items():
            record = database[cav][stamp]
            yaml = yaml_utils.load_yaml(record['yaml'])
            raw_ids = ids(yaml['vehicles'])
            if set(raw_ids) != set(ids(content['params']['vehicles'])):
                raise ValueError('Current raw YAML IDs differ from reformed official params')
            raw.append(dict(cav=str(cav), yaml=record['yaml'], object_ids=raw_ids, count=len(raw_ids)))
            pose = content['params']['lidar_pose']
            distance = math.hypot(pose[0] - ego_pose[0], pose[1] - ego_pose[1])
            context['cavs'].append(dict(cav=str(cav), ego=content['ego'], lidar_pose=copy.deepcopy(pose),
                distance_to_ego_m=distance, time_delay=content['time_delay'],
                transformation_matrix=content['params']['transformation_matrix'].tolist(),
                lidar_file=record['lidar'], yaml_file=record['yaml']))
        return result

    def fov(selected, points, ego_params):
        result = original_fov(selected, points, ego_params)
        # First call is LiDAR, second is radar (fixed LiDAR modality, DeLidar=False).
        active.setdefault('fov', []).append(dict(before=len(points), after=len(result), points=result.copy()))
        return result

    def car(selected, *args, **kwargs):
        active.clear()
        result = original_car(selected, *args, **kwargs)
        object_ids = ids(result['object_ids'])
        boxes = result['object_bbx_center']
        if boxes.shape != (len(object_ids), 7) or len(active['fov']) != 2:
            raise ValueError('Unexpected official single-car/FOV path')
        lidar = result['projected_lidar']
        fov_lidar = active['fov'][0]['points']
        fallback = not np.array_equal(lidar, fov_lidar)
        # Diagnostic uses the same native operator on actual post-FOV points.
        # It never supplies a mask or points back to the dataset.
        physical_counts = original_roi(fov_lidar[:, :3], boxes[:, [0, 1, 2, 5, 4, 3, 6]]).sum(axis=1)
        row = dict(cav=str(selected['id']), ego=selected['ego'],
            before_point_filter_ids=object_ids, before_point_filter_boxes=boxes.tolist(),
            candidate_range_removed_ids=sorted(set(ids(selected['params']['vehicles'])) - set(object_ids)),
            physical_post_fov_points_in_box=[int(n) for n in physical_counts],
            lidar_before_fov_count=active['fov'][0]['before'], lidar_after_fov_count=len(fov_lidar),
            lidar_at_gt_filter_count=len(lidar), official_empty_voxel_fallback=fallback,
            radar_empty_voxel_fallback=not np.array_equal(result['projected_radar'], active['fov'][1]['points']))
        cav_rows.append(row)
        return result

    def roi(points, boxes):
        result = original_roi(points, boxes)
        i = len(roi_calls)
        row = cav_rows[i]
        if not np.array_equal(boxes, np.asarray(row['before_point_filter_boxes']).reshape(-1, 7)[:, [0, 1, 2, 5, 4, 3, 6]]):
            raise ValueError('Native ROI call does not match observed CAV/box IDs')
        counts = [int(n) for n in result.sum(axis=1)]
        row['official_points_in_box'] = counts
        row['after_point_filter_ids'] = [k for k, n in zip(row['before_point_filter_ids'], counts) if n > 0]
        row['removed_zero_point_ids'] = [k for k, n in zip(row['before_point_filter_ids'], counts) if n == 0]
        roi_calls.append(row['cav'])
        return result

    with patched(dataset, 'retrieve_base_data', retrieve), patched(dataset, 'get_item_single_car', car), \
            patched(dataset, 'mask_ego_fov_flag', fov), patched(module, 'points_in_boxes_cpu', roi):
        item = dataset[index]
    if len(roi_calls) != len(cav_rows) or not cav_rows:
        raise ValueError('Incomplete official point-filter observations')
    ego = item['ego']
    object_ids = ids(ego['object_ids'])
    centers = ego['object_bbx_center'][ego['object_bbx_mask'] == 1]
    if len(object_ids) != len(centers) or not np.isfinite(centers).all():
        raise ValueError('Official final ID/box alignment changed')
    survivor_union = {k for r in cav_rows for k in r['after_point_filter_ids']}
    if survivor_union != set(object_ids):
        raise ValueError('Observed native keep decisions differ from official dataset GT')
    center_by_id = dict(zip(object_ids, centers.tolist()))
    # Capture the mask actually applied by generate_gt_bbx, including its ID dedup order.
    batch = dataset.collate_batch_test([item])
    native_ids = batch['ego']['object_ids']
    dedup_indices = [native_ids.index(x) for x in set(native_ids)]
    evaluation_ids = [oid(native_ids[i]) for i in dedup_indices]
    captured = []

    def final_mask(boxes, *args, **kwargs):
        result = original_mask(boxes, *args, **kwargs)
        captured.append((boxes.detach().cpu().numpy().copy(), result.detach().cpu().numpy().copy()))
        return result

    with patched(box_utils, 'get_mask_for_boxes_within_range_torch', final_mask):
        gt = dataset.post_processor.generate_gt_bbx(batch)
    if len(captured) != 1:
        raise ValueError('Unexpected official evaluation GT range-filter path')
    corners, keep = captured[0]
    if (len(corners) != len(evaluation_ids) or not np.isfinite(corners).all()
            or not np.array_equal(gt.cpu().numpy(), corners[keep])):
        raise ValueError('Observed evaluation GT does not match original generate_gt_bbx return')
    final = [dict(object_id=k, box=center_by_id[k], corners=c.tolist())
             for k, c, valid in zip(evaluation_ids, corners, keep) if valid]
    candidates = {k for r in cav_rows for k in r['before_point_filter_ids']}
    raw_union = sorted({k for r in raw for k in r['object_ids']})
    context['participating_cavs'] = [r['cav'] for r in cav_rows]
    return dict(context=context, raw_yaml=raw, raw_yaml_unique_object_ids=raw_union,
        raw_yaml_unique_count=len(raw_union), raw_yaml_cav_count_sum=sum(r['count'] for r in raw),
        cavs=cav_rows, final_gt=final, final_gt_count=len(final),
        dataset_gt_ids=object_ids, evaluation_range_removed_ids=[k for k, valid in zip(evaluation_ids, keep) if not valid],
        all_cav_zero_point_removed_ids=sorted(candidates - survivor_union))
