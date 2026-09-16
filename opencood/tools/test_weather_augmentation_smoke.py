"""Small end-to-end smoke test for pre-voxel weather augmentation.

The script reads 8-12 training samples, verifies that the clean branch is
unchanged, prints per-agent point/pillar statistics, runs teacher/clean/weather
forwards, and performs one backward pass. It does not update weights or save a
checkpoint.
"""

import argparse
import copy
import random
from types import SimpleNamespace

import numpy as np
import torch

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.tools.train_weather_consistency import (
    build_teacher_student_inputs,
    clean_communication_loss,
    consistency_losses,
    freeze_batchnorm,
    keep_batchnorm_eval,
    load_model_from_dir,
    print_weather_augmentation_stats,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Smoke test deterministic pre-voxel weather augmentation.')
    parser.add_argument('--hypes_yaml', required=True)
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--samples', default=8, type=int)
    parser.add_argument('--start_index', default=0, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--rain_rate', default=None, type=float,
                        help='Override a fixed Physics-Rain rate in mm/h.')
    parser.add_argument('--weather_augmentation_seed', default=None, type=int)
    parser.add_argument(
        '--weather_augmentation_mode', default='',
        choices=['', 'v2x_dgw_awa', 'pre_voxel_matched_dropout',
                 'physics_rain', 'physics_fog', 'physics_snow',
                 'mixed_weather', 'mixed_physics_weather'])
    parser.add_argument('--fog_lookup_dir', default='')
    parser.add_argument('--fog_alpha', default=None, type=float)
    parser.add_argument('--snowfall_rate', default=None, type=float)
    return parser.parse_args()


def assert_numpy_equal(left, right, name):
    if not np.array_equal(np.asarray(left), np.asarray(right)):
        raise AssertionError('%s changed between none and AWA clean branch' %
                             name)


def assert_processed_lidar_equal(left, right, branch='clean'):
    if set(left.keys()) != set(right.keys()):
        raise AssertionError('%s processed_lidar keys changed' % branch)
    for key in left:
        if len(left[key]) != len(right[key]):
            raise AssertionError(
                '%s %s agent count changed' % (branch, key))
        for agent_index, (left_value, right_value) in enumerate(
                zip(left[key], right[key])):
            if not np.array_equal(left_value, right_value):
                raise AssertionError(
                    '%s %s changed for agent %d' %
                    (branch, key, agent_index))


def processed_lidar_differs(left, right):
    """Return whether two processed branches differ in shape or content."""
    if set(left.keys()) != set(right.keys()):
        return True
    for key in left:
        if len(left[key]) != len(right[key]):
            return True
        for left_value, right_value in zip(left[key], right[key]):
            if not np.array_equal(left_value, right_value):
                return True
    return False


def assert_finite_tensor(value, name):
    if not torch.is_tensor(value) or not torch.isfinite(value).all():
        raise AssertionError('%s is missing or contains NaN/Inf' % name)


def main():
    opt = parse_args()
    if opt.samples < 1 or opt.samples > 12:
        raise ValueError('--samples must be in [1, 12]')

    hypes = yaml_utils.load_yaml(opt.hypes_yaml)
    # Frozen detector configurations need not contain training-time weather
    # fields.  Insert the branch explicitly instead of mutating a detached
    # temporary dictionary returned by ``dict.get``.
    augmentation_cfg = hypes.setdefault('weather_augmentation', {})
    if opt.weather_augmentation_mode:
        augmentation_cfg['mode'] = opt.weather_augmentation_mode
    point_level_modes = [
        'v2x_dgw_awa', 'pre_voxel_matched_dropout', 'physics_rain',
        'physics_fog', 'physics_snow', 'mixed_weather',
        'mixed_physics_weather']
    augmentation_mode = augmentation_cfg.get('mode')
    if augmentation_mode not in point_level_modes:
        raise ValueError('smoke test requires a pre-voxel augmentation mode')
    if opt.weather_augmentation_seed is not None:
        augmentation_cfg['seed'] = opt.weather_augmentation_seed
    if opt.rain_rate is not None:
        if augmentation_mode != 'physics_rain':
            raise ValueError('--rain_rate requires physics_rain mode')
        if opt.rain_rate <= 0:
            raise ValueError('--rain_rate must be positive')
        augmentation_cfg.setdefault('physics_rain', {})[
            'fixed_rain_rate'] = opt.rain_rate
    if augmentation_mode == 'physics_fog':
        fog_cfg = augmentation_cfg.setdefault('physics_fog', {})
        if opt.fog_lookup_dir:
            fog_cfg['lookup_dir'] = opt.fog_lookup_dir
        if opt.fog_alpha is not None:
            if opt.fog_alpha <= 0:
                raise ValueError('--fog_alpha must be positive')
            fog_cfg['fixed_alpha'] = opt.fog_alpha
        if not fog_cfg.get('lookup_dir'):
            raise ValueError('physics_fog requires --fog_lookup_dir')
    if opt.snowfall_rate is not None:
        if augmentation_mode != 'physics_snow':
            raise ValueError(
                '--snowfall_rate requires physics_snow mode')
        if opt.snowfall_rate <= 0:
            raise ValueError('--snowfall_rate must be positive')
        augmentation_cfg.setdefault('physics_snow', {})[
            'fixed_snowfall_rate'] = opt.snowfall_rate
    default_frame = (
        'local_sensor'
        if augmentation_mode in [
            'physics_rain', 'physics_fog', 'physics_snow']
        else 'projected_ego')
    expected_frame = None if augmentation_mode in [
        'mixed_weather', 'mixed_physics_weather'] else \
        augmentation_cfg.get(augmentation_mode, {}).get(
            'perturbation_frame', default_frame)

    reference_hypes = copy.deepcopy(hypes)
    reference_hypes.setdefault('weather_augmentation', {})['mode'] = 'none'
    awa_dataset = build_dataset(hypes, visualize=False, train=True)
    reference_dataset = build_dataset(
        reference_hypes, visualize=False, train=True)
    awa_dataset.set_weather_augmentation_epoch(0)

    device = torch.device(opt.device if torch.cuda.is_available() else 'cpu')
    teacher = load_model_from_dir(
        hypes, opt.model_dir, device, trainable=False)
    student = load_model_from_dir(
        hypes, opt.model_dir, device, trainable=True)
    frozen_bn_count = freeze_batchnorm(student)
    criterion = train_utils.create_loss(hypes)
    loss_opt = SimpleNamespace(fg_thresh=0.2, lambda_reliability=1.0)
    input_opt = SimpleNamespace()

    print('Smoke device: %s' % device)
    print('Frozen BatchNorm layers: %d' % frozen_bn_count)
    stop_index = min(len(awa_dataset), opt.start_index + opt.samples)
    if stop_index <= opt.start_index:
        raise ValueError('start_index is outside the dataset')
    rain_stats = []

    for local_index, dataset_index in enumerate(
            range(opt.start_index, stop_index)):
        deterministic_seed = int(
            augmentation_cfg.get('seed', 20)) + dataset_index
        np.random.seed(deterministic_seed)
        random.seed(deterministic_seed)
        reference_sample = reference_dataset[dataset_index]
        np.random.seed(deterministic_seed)
        random.seed(deterministic_seed)
        awa_sample = awa_dataset[dataset_index]

        # The exact same weather key must reproduce the exact same voxelized
        # student input.  This is stronger than comparing aggregate counts.
        np.random.seed(deterministic_seed)
        random.seed(deterministic_seed)
        repeat_sample = awa_dataset[dataset_index]
        assert_processed_lidar_equal(
            awa_sample['ego']['processed_lidar_weather'],
            repeat_sample['ego']['processed_lidar_weather'],
            branch='same-key weather')
        if awa_sample['ego']['weather_augmentation_stats'] != \
                repeat_sample['ego']['weather_augmentation_stats']:
            raise AssertionError(
                'same weather key changed augmentation statistics')

        # On the first sample, changing only the epoch stream must produce a
        # different stochastic weather realization.  Restore epoch zero
        # immediately so the forward/backward smoke still uses the frozen key.
        if local_index == 0:
            awa_dataset.set_weather_augmentation_epoch(1)
            np.random.seed(deterministic_seed)
            random.seed(deterministic_seed)
            changed_key_sample = awa_dataset[dataset_index]
            awa_dataset.set_weather_augmentation_epoch(0)
            if not processed_lidar_differs(
                    awa_sample['ego']['processed_lidar_weather'],
                    changed_key_sample['ego']['processed_lidar_weather']):
                raise AssertionError(
                    'changing weather epoch did not change student voxels')

        reference_ego = reference_sample['ego']
        awa_ego = awa_sample['ego']
        assert reference_ego['cav_num'] == awa_ego['cav_num']
        assert_processed_lidar_equal(
            reference_ego['processed_lidar'], awa_ego['processed_lidar'])
        for key in ['object_bbx_center', 'object_bbx_mask',
                    'pairwise_t_matrix', 'spatial_correction_matrix']:
            assert_numpy_equal(reference_ego[key], awa_ego[key], key)

        collated = awa_dataset.collate_batch_train([awa_sample])
        augmentation_stats = collated['ego'].pop(
            'weather_augmentation_stats')
        for scene_stats in augmentation_stats:
            for stats in scene_stats:
                stats_mode = stats.get(
                    'augmentation_mode', augmentation_mode)
                stats_expected_frame = expected_frame
                if stats_expected_frame is None:
                    stats_expected_frame = augmentation_cfg.get(
                        stats_mode, {}).get(
                            'perturbation_frame', 'projected_ego')
                if stats.get('perturbation_frame') != stats_expected_frame:
                    raise AssertionError(
                        'weather perturbation ran in the wrong frame')
                if stats_expected_frame == 'local_sensor' and \
                        not stats.get('voxel_fallback_used', False) and \
                        stats['local_augmented_point_count'] < \
                        stats['final_point_count']:
                    raise AssertionError(
                        'post-projection filtering increased point count')
                if stats_mode == 'physics_rain':
                    rain_partition = (
                        stats['retained_original_point_count'] +
                        stats['false_return_count'] +
                        stats['lost_point_count'])
                    if rain_partition != stats['input_point_count']:
                        raise AssertionError(
                            'rain retained/false/lost partition is invalid')
                    if not 0.0 <= stats['output_intensity_mean'] <= 1.0:
                        raise AssertionError(
                            'rain output intensity is outside [0, 1]')
                    rain_stats.append(stats)
                if stats_mode == 'physics_fog':
                    fog_partition = (
                        stats['retained_original_point_count'] +
                        stats['fog_return_count'] +
                        stats['lost_point_count'])
                    if fog_partition != stats['input_point_count']:
                        raise AssertionError(
                            'fog retained/return/lost partition is invalid')
                if stats_mode == 'physics_snow':
                    snow_partition = (
                        stats['retained_original_point_count'] +
                        stats['snow_return_count'] +
                        stats['lost_point_count'])
                    if snow_partition != stats['input_point_count']:
                        raise AssertionError(
                            'snow retained/return/lost partition is invalid')
                    if not 0.0 <= \
                            stats['output_intensity_mean'] <= 1.0:
                        raise AssertionError(
                            'snow output intensity is outside [0, 1]')
        print_weather_augmentation_stats(augmentation_stats)
        collated = train_utils.to_device(collated, device)
        teacher_ego, clean_ego, weather_ego = build_teacher_student_inputs(
            collated['ego'], hypes, input_opt)
        for branch in [teacher_ego, clean_ego, weather_ego]:
            branch['return_aux'] = True

        student.train()
        keep_batchnorm_eval(student)
        student.zero_grad()
        with torch.no_grad():
            teacher_out = teacher(teacher_ego)
        clean_out = student(clean_ego)
        weather_out = student(weather_ego)

        for branch_name, output in [
                ('teacher', teacher_out), ('clean', clean_out),
                ('weather', weather_out)]:
            # Detection heads are the cross-backbone contract.  Communication
            # confidence and exposed fused features are Where2comm-specific
            # auxiliaries and are intentionally optional for AttFuse.
            for key in ['psm', 'rm']:
                assert_finite_tensor(
                    output.get(key), '%s.%s' % (branch_name, key))
            for key in ['single_confidence', 'fused_feature', 'com']:
                if key in output:
                    assert_finite_tensor(
                        output[key], '%s.%s' % (branch_name, key))
        if clean_out['psm'].shape != weather_out['psm'].shape or \
                clean_out['rm'].shape != weather_out['rm'].shape:
            raise AssertionError('clean/weather model output shapes differ')

        auxiliary_keys = {'single_confidence', 'fused_feature'}
        auxiliary_presence = [
            auxiliary_keys.issubset(output)
            for output in (teacher_out, clean_out, weather_out)]
        if len(set(auxiliary_presence)) != 1:
            raise AssertionError(
                'teacher/clean/weather auxiliary output contracts differ')
        supports_consistency_aux = auxiliary_presence[0]
        if supports_consistency_aux:
            for key in sorted(auxiliary_keys):
                if clean_out[key].shape != weather_out[key].shape:
                    raise AssertionError(
                        'clean/weather %s shapes differ' % key)

        clean_det = criterion(clean_out, collated['ego']['label_dict'])
        weather_det = criterion(weather_out, collated['ego']['label_dict'])
        if supports_consistency_aux:
            fuse_loss, conf_loss = consistency_losses(
                weather_out, teacher_out, weather_ego, loss_opt)
            clean_conf = clean_communication_loss(clean_out, teacher_out)
            auxiliary_contract = 'where2comm_consistency'
        else:
            # AttFuse exposes only the official detection heads.  Its smoke
            # still verifies both detector losses and a finite backward pass;
            # no unavailable auxiliary objective is fabricated.
            fuse_loss = clean_det.new_zeros(())
            conf_loss = clean_det.new_zeros(())
            clean_conf = clean_det.new_zeros(())
            auxiliary_contract = 'detection_heads_only'
        total_loss = 0.5 * clean_det + 0.3 * weather_det + \
            0.5 * fuse_loss + 0.2 * conf_loss + clean_conf
        assert_finite_tensor(total_loss, 'total_loss')

        if local_index == 0:
            total_loss.backward()
            for name, parameter in student.named_parameters():
                if parameter.grad is not None and not torch.isfinite(
                        parameter.grad).all():
                    raise AssertionError('non-finite gradient in %s' % name)

        auxiliary_shapes = {
            key: tuple(clean_out[key].shape)
            for key in sorted(auxiliary_keys) if key in clean_out}
        print(
            'sample=%d record_len=%s psm=%s rm=%s aux_contract=%s '
            'aux_shapes=%s clean_det=%.4f weather_det=%.4f fuse=%.4f '
            'conf=%.4f clean_conf=%.4f total=%.4f' %
            (dataset_index, collated['ego']['record_len'].tolist(),
             tuple(clean_out['psm'].shape), tuple(clean_out['rm'].shape),
             auxiliary_contract, auxiliary_shapes, clean_det.item(),
             weather_det.item(), fuse_loss.item(), conf_loss.item(),
             clean_conf.item(), total_loss.item()))

    if rain_stats:
        input_count = sum(s['input_point_count'] for s in rain_stats)
        retained = sum(
            s['retained_original_point_count'] for s in rain_stats)
        false_returns = sum(s['false_return_count'] for s in rain_stats)
        lost = sum(s['lost_point_count'] for s in rain_stats)
        clean_voxels = sum(s['clean_voxel_count'] for s in rain_stats)
        weather_voxels = sum(s['weather_voxel_count'] for s in rain_stats)
        clean_foreground = sum(
            s.get('clean_foreground_point_count', 0) for s in rain_stats)
        weather_foreground = sum(
            s.get('weather_foreground_point_count', 0) for s in rain_stats)
        weighted_output_intensity = sum(
            s['output_intensity_mean'] *
            s['local_augmented_point_count'] for s in rain_stats)
        output_count = retained + false_returns
        print(
            'PhysicsRain summary agents=%d input=%d retained=%d (%.4f) '
            'false=%d (%.6f) lost=%d (%.4f) clean_voxels=%d '
            'weather_voxels=%d voxel_ratio=%.4f foreground=%d/%d '
            'foreground_ratio=%.4f output_I=%.6f' %
            (len(rain_stats), input_count, retained,
             retained / max(input_count, 1), false_returns,
             false_returns / max(input_count, 1), lost,
             lost / max(input_count, 1), clean_voxels, weather_voxels,
             weather_voxels / max(clean_voxels, 1),
             clean_foreground, weather_foreground,
             weather_foreground / max(clean_foreground, 1),
             weighted_output_intensity / max(output_count, 1)))
    print('Weather augmentation smoke test passed for %d samples' %
          (stop_index - opt.start_index))
    print('WEATHER_AUGMENTATION_DETERMINISM_AND_KEY_CHANGE_PASSED')


if __name__ == '__main__':
    main()
