"""Point-level Adaptive Weather Augmentation adapted from V2X-DGW.

The augmentor operates on an individual agent point cloud before
voxelization. Input and output arrays have shape ``[N, 4]`` with columns
``x, y, z, intensity``. Randomness is supplied through explicit NumPy
generators so DataLoader order does not change the sampled corruption.
"""

import numpy as np


class V2XDGAdaptiveWeatherAugmentor:
    """Reproducible point-level weather augmentation for one agent."""

    def __init__(self, config, lidar_range, seed=20):
        self.config = dict(config or {})
        self.lidar_range = np.asarray(lidar_range, dtype=np.float32)
        if self.lidar_range.shape != (6,):
            raise ValueError('lidar_range must contain 6 values')
        self.seed = int(seed)
        self.perturbation_frame = self.config.get(
            'perturbation_frame', 'projected_ego')

        self.dynamic_range = bool(self.config.get('dynamic_range', True))
        self.range_low = float(
            self.config.get('range_proportion_low', 0.5))
        self.range_high = float(
            self.config.get('range_proportion_high', 0.8))
        self.random_drop_out = bool(
            self.config.get('random_drop_out', True))
        self.dropout_strategy = self.config.get(
            'dropout_strategy', 'uniform_keep_ratio')
        self.keep_low = float(self.config.get('keep_ratio_low', 0.8))
        self.keep_high = float(self.config.get('keep_ratio_high', 1.0))
        self.matched_dropout_base = float(
            self.config.get('dropout_base', 0.02))
        self.matched_dropout_range = float(
            self.config.get('dropout_range', 0.15))
        self.matched_dropout_max = float(
            self.config.get('dropout_max', 0.35))
        self.random_gaussian_noise = bool(
            self.config.get('random_gaussian_noise', True))
        self.max_noise_points = int(
            self.config.get('max_noise_points', 2000))
        self.noise_resolution = float(
            self.config.get('noise_resolution', 0.01))
        self.noise_intensity_std = float(
            self.config.get('noise_intensity_std', 0.5))
        self.random_jittering = bool(
            self.config.get('random_jittering', True))
        self.jitter_std = float(self.config.get('jitter_std', 0.01))
        self.jitter_clip = float(self.config.get('jitter_clip', 0.05))

        self._validate_config()

    def _validate_config(self):
        if self.perturbation_frame not in [
                'projected_ego', 'local_sensor']:
            raise ValueError('unsupported perturbation_frame: %s' %
                             self.perturbation_frame)
        valid_dropout_strategies = [
            'uniform_keep_ratio', 'matched_range']
        if self.dropout_strategy not in valid_dropout_strategies:
            raise ValueError('unsupported dropout_strategy: %s' %
                             self.dropout_strategy)
        if not 0.0 < self.range_low <= self.range_high <= 1.0:
            raise ValueError('range proportions must satisfy 0 < low <= high <= 1')
        if not 0.0 < self.keep_low <= self.keep_high <= 1.0:
            raise ValueError('keep ratios must satisfy 0 < low <= high <= 1')
        if not 0.0 <= self.matched_dropout_base <= 1.0 or \
                not 0.0 <= self.matched_dropout_range <= 1.0 or \
                not 0.0 <= self.matched_dropout_max <= 1.0:
            raise ValueError(
                'matched dropout base/range/max must be in [0, 1]')
        if self.max_noise_points < 0:
            raise ValueError('max_noise_points must be non-negative')
        if self.noise_resolution <= 0:
            raise ValueError('noise_resolution must be positive')
        if self.noise_intensity_std < 0:
            raise ValueError('noise_intensity_std must be non-negative')
        if self.jitter_std < 0 or self.jitter_clip < 0:
            raise ValueError('jitter_std and jitter_clip must be non-negative')

    @staticmethod
    def validate_points(points, name='points', allow_empty=True):
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] != 4:
            raise ValueError('%s must have shape [N, 4], got %s' %
                             (name, tuple(points.shape)))
        if not allow_empty and points.shape[0] == 0:
            raise ValueError('%s must contain at least one point' % name)
        if not np.issubdtype(points.dtype, np.floating):
            raise TypeError('%s must use a floating dtype, got %s' %
                            (name, points.dtype))
        if not np.isfinite(points).all():
            raise ValueError('%s contains NaN or Inf' % name)
        return np.ascontiguousarray(points, dtype=np.float32)

    def make_rng(self, epoch, sample_index, agent_index, stream=0):
        """Create a deterministic independent RNG for one data stream."""
        sequence = np.random.SeedSequence([
            self.seed,
            int(epoch),
            int(sample_index),
            int(agent_index),
            int(stream)
        ])
        return np.random.default_rng(sequence)

    def sample_range_proportion(self, epoch, sample_index):
        if not self.dynamic_range:
            return 1.0
        rng = self.make_rng(epoch, sample_index, 0, stream=0)
        return float(rng.uniform(self.range_low, self.range_high))

    def apply_dynamic_range(self, points, range_proportion):
        """Apply V2X-DGW scene-level xy range reduction in sensor space."""
        points = self.validate_points(points, name='raw_points')
        if not self.dynamic_range:
            return points.copy()

        proportion = float(range_proportion)
        if not self.range_low <= proportion <= self.range_high:
            raise ValueError('range_proportion is outside the configured range')
        reduced_range = self.lidar_range.copy()
        reduced_range[[0, 1, 3, 4]] *= proportion
        mask = (
            (points[:, 0] > reduced_range[0]) &
            (points[:, 0] < reduced_range[3]) &
            (points[:, 1] > reduced_range[1]) &
            (points[:, 1] < reduced_range[4]) &
            (points[:, 2] > reduced_range[2]) &
            (points[:, 2] < reduced_range[5])
        )
        return np.ascontiguousarray(points[mask], dtype=np.float32)

    @staticmethod
    def _nearest_point(points):
        distance = np.linalg.norm(points[:, :3], axis=1)
        return points[[int(np.argmin(distance))]].copy()

    def _sample_noise_axis(self, rng, minimum, maximum, count):
        values = np.arange(
            minimum, maximum, self.noise_resolution, dtype=np.float32)
        if values.size == 0:
            return None
        return rng.choice(values, count, replace=True).astype(np.float32)

    def _matched_range_dropout(self, points, rng):
        """Apply the old voxel-dropout probability law to raw points.

        This is the controlled ablation for augmentation placement. The
        probability formula and range normalization match
        ``corrupt_student_voxels``; only the unit being sampled changes from
        an occupied voxel to an individual pre-voxel point.
        """
        distance = np.linalg.norm(points[:, :2], axis=1)
        x_max = float(self.lidar_range[3])
        y_max = float(self.lidar_range[4])
        max_distance = max((x_max * x_max + y_max * y_max) ** 0.5, 1.0)
        range_ratio = np.clip(distance / max_distance, 0.0, 1.0)
        drop_probability = self.matched_dropout_base + \
            self.matched_dropout_range * range_ratio
        drop_probability = np.clip(
            drop_probability, 0.0, self.matched_dropout_max)
        keep_mask = rng.random(points.shape[0]) > drop_probability
        if not np.any(keep_mask):
            keep_mask[int(rng.integers(0, points.shape[0]))] = True
        stats = {
            'drop_probability_mean': float(drop_probability.mean()),
            'drop_probability_min': float(drop_probability.min()),
            'drop_probability_max': float(drop_probability.max())
        }
        return points[keep_mask], stats

    def perturb_points(self, points, fallback_points, rng,
                       range_proportion):
        """Apply dropout, noise insertion, and jitter in one frame.

        ``points`` and ``fallback_points`` must already use the same coordinate
        frame. If range reduction removes every point, the nearest clean point
        is kept so clean/weather agent order and ``record_len`` remain equal.
        """
        points = self.validate_points(points, name='weather_points')
        fallback_points = self.validate_points(
            fallback_points, name='fallback_points')
        stats = {
            'perturbation_frame': self.perturbation_frame,
            'range_proportion': float(range_proportion),
            'after_range_point_count': int(points.shape[0]),
            'fallback_used': False,
            'empty_input': False,
            'noise_point_count': 0
        }

        if points.shape[0] == 0:
            if fallback_points.shape[0] > 0:
                points = self._nearest_point(fallback_points)
                stats['fallback_used'] = True
            else:
                # A projected non-ego CAV can legitimately have no point in
                # the ego LiDAR range. Preserve that clean-branch behaviour
                # instead of inventing a synthetic point or crashing a
                # DataLoader worker. Collation later verifies that clean and
                # weather branches omit exactly the same agent indices.
                stats.update({
                    'empty_input': True,
                    'keep_ratio': 1.0,
                    'after_dropout_point_count': 0,
                    'final_point_count': 0,
                    'mean_distance': 0.0,
                    'max_distance': 0.0
                })
                return points.copy(), stats

        if self.random_drop_out:
            input_point_count = points.shape[0]
            if self.dropout_strategy == 'matched_range':
                points, dropout_stats = self._matched_range_dropout(
                    points, rng)
                stats.update(dropout_stats)
                keep_ratio = float(points.shape[0] / input_point_count)
            else:
                keep_ratio = float(rng.uniform(
                    self.keep_low, self.keep_high))
                keep_count = max(1, int(
                    keep_ratio * input_point_count))
                indices = rng.choice(
                    input_point_count, keep_count, replace=False)
                points = points[indices]
        else:
            keep_ratio = 1.0
        stats['keep_ratio'] = keep_ratio
        stats['after_dropout_point_count'] = int(points.shape[0])

        if self.random_gaussian_noise and self.max_noise_points > 0:
            noise_count = int(rng.integers(0, self.max_noise_points))
            mins = points.min(axis=0)
            maxs = points.max(axis=0)
            if noise_count > 0 and np.all(
                    mins[:3] < maxs[:3] - self.noise_resolution) and \
                    mins[3] < maxs[3] - self.noise_resolution:
                noise_x = self._sample_noise_axis(
                    rng, mins[0], maxs[0], noise_count)
                noise_y = self._sample_noise_axis(
                    rng, mins[1], maxs[1], noise_count)
                noise_z = self._sample_noise_axis(
                    rng, mins[2], maxs[2], noise_count)
                if noise_x is not None and noise_y is not None and \
                        noise_z is not None:
                    noise_i = rng.normal(
                        loc=float((mins[3] + maxs[3]) / 2.0),
                        scale=self.noise_intensity_std,
                        size=noise_count).astype(np.float32)
                    noise = np.stack(
                        (noise_x, noise_y, noise_z, noise_i), axis=1)
                    points = np.concatenate((points, noise), axis=0)
                    stats['noise_point_count'] = noise_count

        if self.random_jittering and points.shape[0] > 0:
            jitter = rng.normal(
                loc=0.0, scale=self.jitter_std,
                size=(points.shape[0], 3))
            jitter = np.clip(
                jitter, -self.jitter_clip, self.jitter_clip).astype(np.float32)
            points = points.copy()
            points[:, :3] += jitter

        points = self.validate_points(
            points, name='augmented_points', allow_empty=False)
        stats['final_point_count'] = int(points.shape[0])
        stats['mean_distance'] = float(
            np.linalg.norm(points[:, :3], axis=1).mean())
        stats['max_distance'] = float(
            np.linalg.norm(points[:, :3], axis=1).max())
        return points, stats

    def perturb_projected_points(self, points, fallback_points, rng,
                                 range_proportion):
        """Backward-compatible alias for older callers and checkpoints."""
        return self.perturb_points(
            points, fallback_points, rng, range_proportion)
