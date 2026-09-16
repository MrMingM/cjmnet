"""CPU unit checks for the vectorized Physics-Fog augmentor."""

import os
import pickle
import sys
import tempfile

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from opencood.data_utils.augmentor.physics_fog import PhysicsFogAugmentor


def make_lookup(path, alpha):
    table = {}
    for index in range(2001):
        distance = index / 10.0
        table[distance] = (
            min(max(distance * 0.25, 0.1), 20.0),
            2.0e-8 * (1.0 - np.exp(-alpha * distance)))
    with open(os.path.join(
            path,
            'integral_0m_to_200m_stepsize_0.1m_tau_h_20ns_'
            'alpha_%.3f.pickle' % alpha), 'wb') as stream:
        pickle.dump(table, stream)


def main():
    with tempfile.TemporaryDirectory() as directory:
        for alpha in [0.010, 0.050]:
            make_lookup(directory, alpha)
        augmentor = PhysicsFogAugmentor({
            'lookup_dir': directory,
            'fixed_alpha': 0.05,
            'intensity_threshold': 0.01,
            'distance_noise_scale': 0.0,
        }, [-140.8, -38.4, -3.0, 140.8, 38.4, 1.0], seed=17)
        sampling_augmentor = PhysicsFogAugmentor({
            'lookup_dir': directory,
            'alpha_low': 0.01,
            'alpha_high': 0.05,
            'alpha_sampling': 'lookup_discrete',
        }, [-140.8, -38.4, -3.0, 140.8, 38.4, 1.0], seed=17)
        sampled_alphas = {
            sampling_augmentor.sample_alpha(0, index)
            for index in range(32)
        }
        if not sampled_alphas.issubset({0.01, 0.05}):
            raise AssertionError(
                'lookup-discrete sampling produced an unsupported alpha')
        if sampled_alphas != {0.01, 0.05}:
            raise AssertionError(
                'lookup-discrete sampling did not exercise both tables')

        rng = np.random.default_rng(11)
        xyz = rng.normal(size=(2000, 3)).astype(np.float32)
        xyz /= np.maximum(
            np.linalg.norm(xyz, axis=1, keepdims=True), 1.0e-6)
        xyz *= rng.uniform(2.0, 100.0, size=(2000, 1)).astype(np.float32)
        intensity = rng.uniform(
            0.05, 1.0, size=(2000, 1)).astype(np.float32)
        points = np.concatenate([xyz, intensity], axis=1)

        out_a, stats_a = augmentor.augment(
            points, augmentor.make_rng(0, 3, 0), 0.05)
        out_b, stats_b = augmentor.augment(
            points, augmentor.make_rng(0, 3, 0), 0.05)
        if not np.array_equal(out_a, out_b) or stats_a != stats_b:
            raise AssertionError('fog augmentation is not deterministic')
        if out_a.ndim != 2 or out_a.shape[1] != 4:
            raise AssertionError('invalid fog output shape')
        if not np.isfinite(out_a).all():
            raise AssertionError('fog output contains NaN or Inf')
        if out_a.shape[0] > points.shape[0]:
            raise AssertionError('fog augmentation increased ray count')
        if out_a.shape[0] and (
                out_a[:, 3].min() < 0 or out_a[:, 3].max() > 1):
            raise AssertionError('fog intensity is outside [0, 1]')
        if stats_a['fog_return_count'] + \
                stats_a['lost_point_count'] + \
                stats_a['retained_original_point_count'] != points.shape[0]:
            raise AssertionError('fog point partition is inconsistent')
        print(
            'alpha=%.3f lookup=%.3f input=%d retained=%d fog=%d lost=%d '
            'output=%d MOR=%.2f mean_I=%.4f' %
            (stats_a['fog_alpha'], stats_a['fog_lookup_alpha'],
             stats_a['input_point_count'],
             stats_a['retained_original_point_count'],
             stats_a['fog_return_count'], stats_a['lost_point_count'],
             out_a.shape[0], stats_a['meteorological_optical_range'],
             stats_a['output_intensity_mean']))
    print('Physics-Fog unit checks passed.')


if __name__ == '__main__':
    main()
