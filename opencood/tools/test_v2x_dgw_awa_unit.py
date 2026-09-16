"""Lightweight unit tests for pre-voxel V2X-DGW AWA."""

import numpy as np

from opencood.data_utils.augmentor.v2x_dgw_awa import \
    V2XDGAdaptiveWeatherAugmentor


def make_points(count=1000):
    rng = np.random.default_rng(7)
    xyz = rng.uniform(
        low=[-120.0, -35.0, -2.0],
        high=[120.0, 35.0, 0.5],
        size=(count, 3)).astype(np.float32)
    intensity = rng.uniform(0.0, 1.0, size=(count, 1)).astype(np.float32)
    return np.concatenate((xyz, intensity), axis=1)


def run_tests():
    config = {
        'dynamic_range': True,
        'range_proportion_low': 0.5,
        'range_proportion_high': 0.8,
        'random_drop_out': True,
        'keep_ratio_low': 0.8,
        'keep_ratio_high': 1.0,
        'random_gaussian_noise': True,
        'max_noise_points': 2000,
        'noise_resolution': 0.01,
        'noise_intensity_std': 0.5,
        'random_jittering': True,
        'jitter_std': 0.01,
        'jitter_clip': 0.05
    }
    lidar_range = [-140.8, -38.4, -3.0, 140.8, 38.4, 1.0]
    augmentor = V2XDGAdaptiveWeatherAugmentor(
        config, lidar_range, seed=20)
    assert augmentor.perturbation_frame == 'projected_ego'
    points = make_points()

    proportion_a = augmentor.sample_range_proportion(0, 11)
    proportion_b = augmentor.sample_range_proportion(0, 11)
    assert proportion_a == proportion_b
    ranged = augmentor.apply_dynamic_range(points, proportion_a)
    assert ranged.shape[1] == 4
    assert 0 < ranged.shape[0] < points.shape[0]

    rng_a = augmentor.make_rng(0, 11, 2, stream=1)
    rng_b = augmentor.make_rng(0, 11, 2, stream=1)
    out_a, stats_a = augmentor.perturb_projected_points(
        ranged, points, rng_a, proportion_a)
    out_b, stats_b = augmentor.perturb_projected_points(
        ranged, points, rng_b, proportion_a)
    assert np.array_equal(out_a, out_b)
    assert stats_a == stats_b
    assert out_a.dtype == np.float32
    assert out_a.ndim == 2 and out_a.shape[1] == 4
    assert out_a.shape[0] > 0
    assert np.isfinite(out_a).all()

    different_rng = augmentor.make_rng(0, 11, 3, stream=1)
    out_c, _ = augmentor.perturb_projected_points(
        ranged, points, different_rng, proportion_a)
    assert not np.array_equal(out_a, out_c)

    empty = np.empty((0, 4), dtype=np.float32)
    fallback_rng = augmentor.make_rng(0, 12, 1, stream=1)
    fallback_out, fallback_stats = augmentor.perturb_projected_points(
        empty, points, fallback_rng, proportion_a)
    assert fallback_out.shape[0] >= 1
    assert fallback_stats['fallback_used']
    assert np.isfinite(fallback_out).all()

    empty_fallback_rng = augmentor.make_rng(0, 13, 1, stream=1)
    empty_out, empty_stats = augmentor.perturb_projected_points(
        empty, empty, empty_fallback_rng, proportion_a)
    assert empty_out.shape == (0, 4)
    assert empty_out.dtype == np.float32
    assert empty_stats['empty_input']
    assert not empty_stats['fallback_used']
    assert empty_stats['final_point_count'] == 0
    assert empty_stats['mean_distance'] == 0.0
    assert empty_stats['max_distance'] == 0.0

    local_config = dict(config)
    local_config['perturbation_frame'] = 'local_sensor'
    local_augmentor = V2XDGAdaptiveWeatherAugmentor(
        local_config, lidar_range, seed=20)
    local_rng = local_augmentor.make_rng(0, 11, 2, stream=1)
    local_out, local_stats = local_augmentor.perturb_points(
        ranged, points, local_rng, proportion_a)
    assert local_out.shape[0] > 0
    assert local_stats['perturbation_frame'] == 'local_sensor'
    assert np.isfinite(local_out).all()

    invalid_frame_config = dict(config)
    invalid_frame_config['perturbation_frame'] = 'world'
    try:
        V2XDGAdaptiveWeatherAugmentor(
            invalid_frame_config, lidar_range, seed=20)
        raise AssertionError('invalid perturbation_frame was accepted')
    except ValueError as error:
        assert 'perturbation_frame' in str(error)

    matched_config = {
        'dynamic_range': False,
        'random_drop_out': True,
        'dropout_strategy': 'matched_range',
        'dropout_base': 0.02,
        'dropout_range': 0.15,
        'dropout_max': 0.35,
        'random_gaussian_noise': False,
        'random_jittering': False
    }
    matched_augmentor = V2XDGAdaptiveWeatherAugmentor(
        matched_config, lidar_range, seed=20)
    assert matched_augmentor.sample_range_proportion(0, 11) == 1.0
    matched_rng_a = matched_augmentor.make_rng(0, 11, 2, stream=1)
    matched_rng_b = matched_augmentor.make_rng(0, 11, 2, stream=1)
    matched_a, matched_stats_a = \
        matched_augmentor.perturb_projected_points(
            points, points, matched_rng_a, 1.0)
    matched_b, matched_stats_b = \
        matched_augmentor.perturb_projected_points(
            points, points, matched_rng_b, 1.0)
    assert np.array_equal(matched_a, matched_b)
    assert matched_stats_a == matched_stats_b
    assert 0 < matched_a.shape[0] < points.shape[0]
    assert matched_stats_a['noise_point_count'] == 0
    assert matched_stats_a['drop_probability_min'] >= 0.02
    assert matched_stats_a['drop_probability_max'] <= 0.35
    original_rows = {row.tobytes() for row in points}
    assert all(row.tobytes() in original_rows for row in matched_a)

    next_epoch = augmentor.sample_range_proportion(1, 11)
    assert next_epoch != proportion_a
    print('V2X-DGW AWA unit tests passed')


if __name__ == '__main__':
    run_tests()
