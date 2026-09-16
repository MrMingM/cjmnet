"""Vectorized physics-guided LiDAR fog augmentation.

The hard-target return follows Beer-Lambert two-way attenuation.  Soft fog
returns use the precomputed beam-integral lookup tables released with
Hahner et al., ICCV 2021.  This implementation is original OpenCOOD glue and
does not copy the reference simulator's per-point loop.
"""

import glob
import os
import pickle
import re

import numpy as np


class PhysicsFogAugmentor:
    """Scene-shared extinction with independent per-agent realizations."""

    DISTANCE_STEP = 0.1
    MAX_LOOKUP_DISTANCE = 200.0

    def __init__(self, config, lidar_range, seed=20):
        self.config = dict(config or {})
        self.lidar_range = np.asarray(lidar_range, dtype=np.float32)
        if self.lidar_range.shape != (6,):
            raise ValueError('lidar_range must contain 6 values')
        self.seed = int(seed)
        self.perturbation_frame = 'local_sensor'

        self.lookup_dir = os.path.abspath(os.path.expanduser(
            self.config.get('lookup_dir', '')))
        self.fixed_alpha = self.config.get('fixed_alpha', None)
        if self.fixed_alpha is not None:
            self.fixed_alpha = float(self.fixed_alpha)
        self.alpha_low = float(self.config.get('alpha_low', 0.005))
        self.alpha_high = float(self.config.get('alpha_high', 0.03))
        self.alpha_sampling = str(
            self.config.get('alpha_sampling', 'lookup_discrete')).lower()
        self.gamma = float(self.config.get('gamma', 1.0e-6))
        self.beta_scale = float(self.config.get('beta_scale', 1.0))
        self.intensity_threshold = float(
            self.config.get('intensity_threshold', 0.01))
        self.distance_noise_scale = float(
            self.config.get('distance_noise_scale', 2.0))
        self.max_soft_response = float(
            self.config.get('max_soft_response', 1.0))

        self._tables = {}
        self._table_paths = self._discover_tables()
        self.available_alphas = sorted(self._table_paths)
        self._validate_config()

    def _discover_tables(self):
        if not self.lookup_dir or not os.path.isdir(self.lookup_dir):
            raise FileNotFoundError(
                'Physics-Fog lookup_dir not found: %s' % self.lookup_dir)
        paths = {}
        pattern = re.compile(r'alpha_([0-9]+(?:\.[0-9]+)?)\.pickle$')
        for path in glob.glob(os.path.join(self.lookup_dir, '*.pickle')):
            match = pattern.search(os.path.basename(path))
            if match:
                paths[float(match.group(1))] = path
        if not paths:
            raise FileNotFoundError(
                'No alpha_*.pickle lookup tables in %s' % self.lookup_dir)
        return paths

    def _validate_config(self):
        if not 0.0 < self.alpha_low <= self.alpha_high:
            raise ValueError('fog alpha must satisfy 0 < low <= high')
        if self.alpha_sampling not in ['lookup_discrete', 'uniform']:
            raise ValueError(
                'alpha_sampling must be lookup_discrete or uniform')
        if self.fixed_alpha is not None and self.fixed_alpha <= 0:
            raise ValueError('fixed_alpha must be positive')
        self.training_alphas = [
            value for value in self.available_alphas
            if self.alpha_low <= value <= self.alpha_high
        ]
        if self.fixed_alpha is None and \
                self.alpha_sampling == 'lookup_discrete' and \
                not self.training_alphas:
            raise ValueError(
                'No fog lookup alpha lies inside [%.6f, %.6f]' %
                (self.alpha_low, self.alpha_high))
        if self.gamma <= 0 or self.beta_scale <= 0:
            raise ValueError('gamma and beta_scale must be positive')
        if not 0.0 <= self.intensity_threshold < 1.0:
            raise ValueError('intensity_threshold must be in [0, 1)')
        if self.distance_noise_scale < 0:
            raise ValueError('distance_noise_scale must be non-negative')
        if self.max_soft_response <= 0:
            raise ValueError('max_soft_response must be positive')

    @staticmethod
    def validate_points(points):
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] != 4:
            raise ValueError(
                'fog points must have shape [N, 4], got %s' %
                (tuple(points.shape),))
        if not np.issubdtype(points.dtype, np.floating):
            raise TypeError('fog points must use a floating dtype')
        if not np.isfinite(points).all():
            raise ValueError('fog points contain NaN or Inf')
        return np.ascontiguousarray(points, dtype=np.float32)

    def make_rng(self, epoch, sample_index, agent_index, stream=0):
        sequence = np.random.SeedSequence([
            self.seed,
            int(epoch),
            int(sample_index),
            int(agent_index),
            int(stream),
        ])
        return np.random.default_rng(sequence)

    def sample_alpha(self, epoch, sample_index):
        if self.fixed_alpha is not None:
            return self.fixed_alpha
        rng = self.make_rng(epoch, sample_index, 0, stream=11)
        if self.alpha_sampling == 'lookup_discrete':
            index = int(rng.integers(0, len(self.training_alphas)))
            return float(self.training_alphas[index])
        return float(rng.uniform(self.alpha_low, self.alpha_high))

    def _nearest_available_alpha(self, requested_alpha):
        return min(
            self.available_alphas,
            key=lambda value: abs(value - requested_alpha))

    def _load_table(self, requested_alpha):
        table_alpha = self._nearest_available_alpha(requested_alpha)
        if table_alpha in self._tables:
            return table_alpha, self._tables[table_alpha]
        with open(self._table_paths[table_alpha], 'rb') as stream:
            source = pickle.load(stream)

        count = int(round(
            self.MAX_LOOKUP_DISTANCE / self.DISTANCE_STEP)) + 1
        fog_distance = np.zeros(count, dtype=np.float32)
        fog_response = np.zeros(count, dtype=np.float32)
        for index in range(count):
            key = round(index * self.DISTANCE_STEP, 1)
            value = source.get(key)
            if value is None:
                value = source.get(float(str(key)))
            if value is None:
                raise KeyError(
                    'fog lookup missing distance %.1f in %s' %
                    (key, self._table_paths[table_alpha]))
            fog_distance[index] = float(value[0])
            fog_response[index] = float(value[1])
        self._tables[table_alpha] = (fog_distance, fog_response)
        return table_alpha, self._tables[table_alpha]

    def augment(self, points, rng, alpha):
        points = self.validate_points(points)
        requested_alpha = float(alpha)
        if requested_alpha <= 0:
            raise ValueError('fog alpha must be positive')
        table_alpha, lookup = self._load_table(requested_alpha)

        input_count = int(points.shape[0])
        if input_count == 0:
            return points.copy(), self._empty_stats(
                requested_alpha, table_alpha)

        intensity = np.clip(
            points[:, 3].astype(np.float64), 0.0, 1.0)
        xyz = points[:, :3].astype(np.float64)
        distance = np.linalg.norm(xyz, axis=1)
        hard_intensity = intensity * np.exp(
            -2.0 * requested_alpha * distance)

        lookup_index = np.rint(
            np.minimum(distance, self.MAX_LOOKUP_DISTANCE) /
            self.DISTANCE_STEP).astype(np.int64)
        fog_distance = lookup[0][lookup_index].astype(np.float64)
        base_response = lookup[1][lookup_index].astype(np.float64)

        meteorological_range = np.log(20.0) / requested_alpha
        beta = self.beta_scale * 0.046 / meteorological_range
        beta_0 = self.gamma / np.pi
        soft_response = (
            base_response * intensity * np.square(distance) *
            beta / beta_0)
        soft_response = np.clip(
            soft_response, 0.0, self.max_soft_response)

        valid_soft = (
            (fog_distance > 0.0) &
            (fog_distance < distance) &
            (soft_response > hard_intensity))
        noisy_fog_distance = fog_distance.copy()
        if self.distance_noise_scale > 0 and np.any(valid_soft):
            additive = self.distance_noise_scale * rng.beta(
                2.0, 20.0, size=int(valid_soft.sum()))
            noisy_fog_distance[valid_soft] += additive
            noisy_fog_distance = np.minimum(
                noisy_fog_distance, np.maximum(distance - 1.0e-3, 0.0))

        output = points.copy()
        scale = np.ones_like(distance)
        nonzero = distance > 1.0e-6
        scale[valid_soft & nonzero] = (
            noisy_fog_distance[valid_soft & nonzero] /
            distance[valid_soft & nonzero])
        output[:, :3] = (xyz * scale[:, None]).astype(np.float32)
        output[:, 3] = hard_intensity.astype(np.float32)
        output[valid_soft, 3] = soft_response[valid_soft].astype(np.float32)

        lost_mask = (~valid_soft) & (
            output[:, 3] < self.intensity_threshold)
        output = np.ascontiguousarray(output[~lost_mask], dtype=np.float32)
        fog_count = int(valid_soft.sum())
        lost_count = int(lost_mask.sum())
        retained_count = input_count - fog_count - lost_count
        stats = {
            'perturbation_frame': self.perturbation_frame,
            'fog_alpha': requested_alpha,
            'fog_lookup_alpha': table_alpha,
            'meteorological_optical_range': meteorological_range,
            'input_point_count': input_count,
            'retained_original_point_count': retained_count,
            'fog_return_count': fog_count,
            'lost_point_count': lost_count,
            'local_augmented_point_count': int(output.shape[0]),
            'input_intensity_mean': float(intensity.mean()),
            'output_intensity_mean': float(output[:, 3].mean())
            if output.shape[0] else 0.0,
            'after_range_point_count': input_count,
            'noise_point_count': fog_count,
            'keep_ratio': float(output.shape[0] / max(input_count, 1)),
            'after_dropout_point_count': int(output.shape[0]),
            'drop_probability_mean': -1.0,
            'fallback_used': False,
            'empty_input': False,
            'mean_distance': float(
                np.linalg.norm(output[:, :3], axis=1).mean())
            if output.shape[0] else 0.0,
            'max_distance': float(
                np.linalg.norm(output[:, :3], axis=1).max())
            if output.shape[0] else 0.0,
        }
        return output, stats

    def _empty_stats(self, requested_alpha, table_alpha):
        return {
            'perturbation_frame': self.perturbation_frame,
            'fog_alpha': requested_alpha,
            'fog_lookup_alpha': table_alpha,
            'meteorological_optical_range': (
                np.log(20.0) / requested_alpha),
            'input_point_count': 0,
            'retained_original_point_count': 0,
            'fog_return_count': 0,
            'lost_point_count': 0,
            'local_augmented_point_count': 0,
            'input_intensity_mean': 0.0,
            'output_intensity_mean': 0.0,
            'after_range_point_count': 0,
            'noise_point_count': 0,
            'keep_ratio': 1.0,
            'after_dropout_point_count': 0,
            'drop_probability_mean': -1.0,
            'fallback_used': False,
            'empty_input': True,
            'mean_distance': 0.0,
            'max_distance': 0.0,
        }
