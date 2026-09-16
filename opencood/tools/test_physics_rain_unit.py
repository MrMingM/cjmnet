"""Fast CPU checks for the clean-room Physics-Rain augmentor.

This test uses synthetic rays and does not require an OPV2V dataset or model.
Run it before dataset smoke tests and AP evaluation.
"""

import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from opencood.data_utils.augmentor.physics_rain import PhysicsRainAugmentor


LIDAR_RANGE = [-140.8, -38.4, -3.0, 140.8, 38.4, 1.0]


def make_points(count=50000, seed=2026):
    rng = np.random.default_rng(seed)
    azimuth = rng.uniform(-np.pi, np.pi, count)
    elevation = rng.uniform(-0.15, 0.05, count)
    distance = rng.uniform(2.0, 140.0, count)
    cos_elevation = np.cos(elevation)
    xyz = np.column_stack((
        distance * cos_elevation * np.cos(azimuth),
        distance * cos_elevation * np.sin(azimuth),
        distance * np.sin(elevation)))
    intensity = rng.uniform(0.02, 1.0, count)
    return np.ascontiguousarray(
        np.column_stack((xyz, intensity)), dtype=np.float32)


def run_once(augmentor, points, rate):
    output, stats = augmentor.augment(
        points, augmentor.make_rng(0, 17, 2, stream=1), rate)
    if output.ndim != 2 or output.shape[1] != 4:
        raise AssertionError('Physics-Rain output must have shape [N, 4]')
    if output.dtype != np.float32 or not np.isfinite(output).all():
        raise AssertionError('Physics-Rain output must be finite float32')
    if output.size and (output[:, 3].min() < 0 or output[:, 3].max() > 1):
        raise AssertionError('Physics-Rain intensity escaped [0, 1]')
    partition = (stats['retained_original_point_count'] +
                 stats['false_return_count'] + stats['lost_point_count'])
    if partition != points.shape[0]:
        raise AssertionError('rain outcome labels do not partition input')
    if output.shape[0] != (stats['retained_original_point_count'] +
                           stats['false_return_count']):
        raise AssertionError('rain output count disagrees with outcome labels')
    if sum(stats['input_distance_histogram']) != points.shape[0]:
        raise AssertionError('input distance histogram lost points')
    if sum(stats['output_distance_histogram']) != output.shape[0]:
        raise AssertionError('output distance histogram lost points')
    return output, stats


def main():
    points = make_points()
    augmentor = PhysicsRainAugmentor({
        'rain_rate_low': 5.0,
        'rain_rate_high': 30.0,
        'min_range': 1.5,
        'beam_divergence': 0.003,
        'min_droplet_diameter_mm': 0.05,
        'range_accuracy': 0.09,
        'water_refractive_index': 1.328,
        'power_threshold_scale': 0.9,
    }, LIDAR_RANGE, seed=20)

    results = []
    for rate in [5.0, 15.0, 30.0, 40.0]:
        output, stats = run_once(augmentor, points, rate)
        repeated, repeated_stats = run_once(augmentor, points, rate)
        if not np.array_equal(output, repeated) or stats != repeated_stats:
            raise AssertionError('same seed/rate is not exactly reproducible')
        results.append(stats)
        print(
            'rate=%5.1f input=%d retained=%d false=%d lost=%d output=%d '
            'mean_intensity=%.6f alpha=%.8f' %
            (rate, stats['input_point_count'],
             stats['retained_original_point_count'],
             stats['false_return_count'], stats['lost_point_count'],
             stats['local_augmented_point_count'],
             stats['output_intensity_mean'],
             stats['extinction_coefficient']))

    alphas = [stats['extinction_coefficient'] for stats in results]
    output_counts = [stats['local_augmented_point_count'] for stats in results]
    intensities = [stats['output_intensity_mean'] for stats in results]
    if not all(a < b for a, b in zip(alphas, alphas[1:])):
        raise AssertionError('extinction must increase with rain rate')
    if not all(a >= b for a, b in zip(output_counts, output_counts[1:])):
        raise AssertionError('synthetic retained count is not severity-monotonic')
    if not all(a >= b for a, b in zip(intensities, intensities[1:])):
        raise AssertionError('synthetic intensity is not severity-monotonic')

    empty, empty_stats = augmentor.augment(
        np.empty((0, 4), dtype=np.float32),
        augmentor.make_rng(0, 0, 0), 15.0)
    if empty.shape != (0, 4) or not empty_stats['empty_input']:
        raise AssertionError('empty input handling failed')
    print('Physics-Rain unit checks passed.')


if __name__ == '__main__':
    main()
