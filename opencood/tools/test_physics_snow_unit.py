"""CPU unit checks for the online Physics-Snow augmentor."""

import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from opencood.data_utils.augmentor.physics_snow import PhysicsSnowAugmentor


def main():
    augmentor = PhysicsSnowAugmentor({
        'snowfall_rate_low': 0.5,
        'snowfall_rate_high': 2.5,
        'terminal_velocity': 1.0,
        'snow_density': 0.1,
        'beam_divergence': 0.003,
        'intensity_threshold': 0.01,
    }, [-140.8, -38.4, -3.0, 140.8, 38.4, 1.0], seed=31)

    rng = np.random.default_rng(19)
    xyz = rng.normal(size=(50000, 3)).astype(np.float32)
    xyz /= np.maximum(
        np.linalg.norm(xyz, axis=1, keepdims=True), 1.0e-6)
    xyz *= rng.uniform(2.0, 100.0, size=(50000, 1)).astype(np.float32)
    intensity = rng.uniform(
        0.02, 1.0, size=(50000, 1)).astype(np.float32)
    points = np.concatenate([xyz, intensity], axis=1)

    intercepted_counts = []
    for rate in [0.5, 2.5, 5.0]:
        out_a, stats_a = augmentor.augment(
            points, augmentor.make_rng(0, 7, 0), rate)
        out_b, stats_b = augmentor.augment(
            points, augmentor.make_rng(0, 7, 0), rate)
        if not np.array_equal(out_a, out_b) or stats_a != stats_b:
            raise AssertionError('snow augmentation is not deterministic')
        if out_a.ndim != 2 or out_a.shape[1] != 4:
            raise AssertionError('invalid snow output shape')
        if not np.isfinite(out_a).all():
            raise AssertionError('snow output contains NaN or Inf')
        if out_a.shape[0] > points.shape[0]:
            raise AssertionError('snow augmentation increased ray count')
        if out_a.shape[0] and (
                out_a[:, 3].min() < 0 or out_a[:, 3].max() > 1):
            raise AssertionError('snow intensity is outside [0, 1]')
        partition = (
            stats_a['retained_original_point_count'] +
            stats_a['snow_return_count'] +
            stats_a['lost_point_count'])
        if partition != points.shape[0]:
            raise AssertionError('snow point partition is inconsistent')
        intercepted_counts.append(stats_a['intercepted_point_count'])
        print(
            'rate=%.1f input=%d intercepted=%d attenuated=%d snow=%d '
            'retained=%d lost=%d output=%d occupancy=%.3e '
            'hazard=%.6f mean_I=%.4f' %
            (rate, stats_a['input_point_count'],
             stats_a['intercepted_point_count'],
             stats_a['attenuated_point_count'],
             stats_a['snow_return_count'],
             stats_a['retained_original_point_count'],
             stats_a['lost_point_count'], out_a.shape[0],
             stats_a['occupancy_ratio'],
             stats_a['intercept_hazard'],
             stats_a['output_intensity_mean']))

    if not intercepted_counts[0] < intercepted_counts[1] < \
            intercepted_counts[2]:
        raise AssertionError(
            'snow interception is not monotonic with snowfall rate')
    print('Physics-Snow unit checks passed.')


if __name__ == '__main__':
    main()
