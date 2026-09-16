"""Deterministic local-frame physics-inspired LiDAR rain augmentation.

The implementation follows the physical structure of LISA (Kilic et al.,
arXiv:2107.07004) without copying its GPL-3.0 source code. It keeps the
published Beer-Lambert attenuation, Marshall-Palmer droplet distribution,
strongest-return comparison, ray-preserving false returns, and range
uncertainty. The expensive per-point enumeration of every droplet is replaced
with a vectorized approximation that samples the largest droplet in a Poisson
beam population. It is therefore LISA-inspired, not bit-exact LISA.

Input and output arrays use ``[x, y, z, reflectivity]`` with coordinates in
meters and reflectivity in ``[0, 1]``. All operations are performed in the
transmitting CAV's local sensor frame.
"""

import numpy as np


class PhysicsRainAugmentor:
    """Scene-shared rain rate with independent per-agent realizations."""

    DISTANCE_BINS = np.asarray(
        [0.0, 20.0, 40.0, 60.0, 80.0, 120.0, np.inf],
        dtype=np.float32)

    def __init__(self, config, lidar_range, seed=20):
        self.config = dict(config or {})
        self.lidar_range = np.asarray(lidar_range, dtype=np.float32)
        if self.lidar_range.shape != (6,):
            raise ValueError('lidar_range must contain 6 values')
        self.seed = int(seed)
        self.perturbation_frame = 'local_sensor'

        self.fixed_rain_rate = self.config.get('fixed_rain_rate', None)
        if self.fixed_rain_rate is not None:
            self.fixed_rain_rate = float(self.fixed_rain_rate)
        self.rain_rate_low = float(
            self.config.get('rain_rate_low', 5.0))
        self.rain_rate_high = float(
            self.config.get('rain_rate_high', 30.0))
        self.min_range = float(self.config.get('min_range', 1.5))
        self.beam_divergence = float(
            self.config.get('beam_divergence', 3.0e-3))
        self.min_droplet_diameter_mm = float(
            self.config.get('min_droplet_diameter_mm', 0.05))
        self.range_accuracy = float(
            self.config.get('range_accuracy', 0.09))
        self.water_refractive_index = float(
            self.config.get('water_refractive_index', 1.328))
        self.power_threshold_scale = float(
            self.config.get('power_threshold_scale', 0.9))
        self.max_poisson_mean = float(
            self.config.get('max_poisson_mean', 1.0e6))
        self.intensity_tolerance = float(
            self.config.get('intensity_tolerance', 1.0e-4))

        x_extent = max(abs(float(self.lidar_range[0])),
                       abs(float(self.lidar_range[3])))
        y_extent = max(abs(float(self.lidar_range[1])),
                       abs(float(self.lidar_range[4])))
        self.max_range = float(self.config.get(
            'max_range', (x_extent * x_extent +
                          y_extent * y_extent) ** 0.5))
        self._validate_config()

    def _validate_config(self):
        if not 0.0 < self.rain_rate_low <= self.rain_rate_high:
            raise ValueError('rain rates must satisfy 0 < low <= high')
        if self.fixed_rain_rate is not None and self.fixed_rain_rate <= 0:
            raise ValueError('fixed_rain_rate must be positive')
        if self.min_range <= 0 or self.max_range <= self.min_range:
            raise ValueError('invalid LiDAR min/max range')
        if self.beam_divergence <= 0:
            raise ValueError('beam_divergence must be positive')
        if self.min_droplet_diameter_mm <= 0:
            raise ValueError('min droplet diameter must be positive')
        if self.range_accuracy < 0:
            raise ValueError('range_accuracy must be non-negative')
        if self.water_refractive_index <= 1.0:
            raise ValueError('water_refractive_index must exceed 1')
        if self.power_threshold_scale <= 0 or self.max_poisson_mean <= 0:
            raise ValueError('power and Poisson limits must be positive')

    @staticmethod
    def validate_points(points, name='points'):
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] != 4:
            raise ValueError('%s must have shape [N, 4], got %s' %
                             (name, tuple(points.shape)))
        if not np.issubdtype(points.dtype, np.floating):
            raise TypeError('%s must use a floating dtype' % name)
        if not np.isfinite(points).all():
            raise ValueError('%s contains NaN or Inf' % name)
        return np.ascontiguousarray(points, dtype=np.float32)

    def make_rng(self, epoch, sample_index, agent_index, stream=0):
        sequence = np.random.SeedSequence([
            self.seed,
            int(epoch),
            int(sample_index),
            int(agent_index),
            int(stream)
        ])
        return np.random.default_rng(sequence)

    def sample_rain_rate(self, epoch, sample_index):
        """Sample one deterministic scene-shared rain rate in mm/hr."""
        if self.fixed_rain_rate is not None:
            return self.fixed_rain_rate
        rng = self.make_rng(epoch, sample_index, 0, stream=7)
        return float(rng.uniform(self.rain_rate_low, self.rain_rate_high))

    @staticmethod
    def extinction_coefficient(rain_rate):
        """Convert the LISA rain asymptote from dB/km to 1/m.

        The published relation is ``alpha_db = 1.45 * Rr**0.64``. Dividing
        by ``4.343 * 1000`` converts power attenuation from dB/km to Nepers/m.
        The two-way path is applied separately as ``exp(-2*alpha*range)``.
        """
        alpha_db_per_km = 1.45 * float(rain_rate) ** 0.64
        return alpha_db_per_km / (4.343 * 1000.0)

    @staticmethod
    def _distance_histogram(ranges):
        counts, _ = np.histogram(ranges, bins=PhysicsRainAugmentor.DISTANCE_BINS)
        return [int(value) for value in counts]

    def augment(self, points, rng, rain_rate):
        """Return corrupted local points and provenance statistics."""
        points = self.validate_points(points, name='rain_input_points')
        rate = float(rain_rate)
        if rate <= 0:
            raise ValueError('rain_rate must be positive')

        input_count = int(points.shape[0])
        empty_stats = {
            'perturbation_frame': self.perturbation_frame,
            'rain_rate': rate,
            'input_point_count': input_count,
            'retained_original_point_count': 0,
            'false_return_count': 0,
            'lost_point_count': 0,
            'local_augmented_point_count': 0,
            'input_intensity_mean': 0.0,
            'input_intensity_std': 0.0,
            'output_intensity_mean': 0.0,
            'output_intensity_std': 0.0,
            'input_distance_histogram': [0] * 6,
            'output_distance_histogram': [0] * 6,
            'extinction_coefficient': self.extinction_coefficient(rate),
            'fallback_used': False,
            'empty_input': input_count == 0,
            'noise_point_count': 0,
            'after_range_point_count': input_count,
            'keep_ratio': 1.0,
            'after_dropout_point_count': input_count,
            'drop_probability_mean': -1.0
        }
        if input_count == 0:
            empty_stats.update({
                'final_point_count': 0,
                'mean_distance': 0.0,
                'max_distance': 0.0
            })
            return points.copy(), empty_stats

        intensity = points[:, 3].astype(np.float64)
        if intensity.min() < -self.intensity_tolerance or \
                intensity.max() > 1.0 + self.intensity_tolerance:
            raise ValueError(
                'rain reflectivity must be normalized to [0, 1], got '
                '[%.6f, %.6f]' % (intensity.min(), intensity.max()))
        intensity = np.clip(intensity, 0.0, 1.0)

        xyz = points[:, :3].astype(np.float64)
        ranges = np.linalg.norm(xyz, axis=1)
        valid_ray = ranges > 1.0e-6
        alpha = self.extinction_coefficient(rate)
        attenuation = np.exp(-2.0 * alpha * ranges)
        attenuated_intensity = intensity * attenuation

        power_min = self.power_threshold_scale / (self.max_range ** 2)
        target_power = np.zeros_like(ranges)
        target_power[valid_ray] = attenuated_intensity[valid_ray] / \
            np.maximum(ranges[valid_ray] ** 2, 1.0e-12)

        # Marshall-Palmer distribution N(D)=8000*exp(-lambda*D).
        mp_lambda = 4.1 * rate ** (-0.21)
        droplet_density = 8000.0 * np.exp(
            -mp_lambda * self.min_droplet_diameter_mm) / mp_lambda
        beam_radius = 0.5 * np.tan(self.beam_divergence) * ranges
        beam_volume = (np.pi / 3.0) * ranges * beam_radius ** 2
        poisson_mean = np.clip(
            droplet_density * beam_volume, 0.0, self.max_poisson_mean)
        particle_count = rng.poisson(poisson_mean)

        scatter_candidate = np.logical_and(
            particle_count > 0, ranges > self.min_range)
        false_range = np.zeros_like(ranges)
        false_intensity = np.zeros_like(ranges)
        false_power = np.zeros_like(ranges)
        candidate_indices = np.flatnonzero(scatter_candidate)
        if candidate_indices.size > 0:
            n = particle_count[candidate_indices].astype(np.float64)
            u_diameter = np.clip(
                rng.random(candidate_indices.size), 1.0e-12, 1.0 - 1.0e-12)
            # Maximum of N shifted-exponential droplet diameters.
            tail = -np.expm1(np.log(u_diameter) / n)
            max_diameter = self.min_droplet_diameter_mm - \
                np.log(np.maximum(tail, 1.0e-15)) / mp_lambda

            u_range = rng.random(candidate_indices.size)
            target_range = ranges[candidate_indices]
            sampled_range = (
                self.min_range ** 3 + u_range *
                (target_range ** 3 - self.min_range ** 3)) ** (1.0 / 3.0)
            false_range[candidate_indices] = sampled_range

            beam_diameter_mm = 1000.0 * np.tan(
                self.beam_divergence) * sampled_range
            fresnel_reflectivity = abs(
                (self.water_refractive_index - 1.0) /
                (self.water_refractive_index + 1.0)) ** 2
            fill_fraction = np.minimum(
                (max_diameter /
                 np.maximum(beam_diameter_mm, 1.0e-9)) ** 2,
                1.0)
            sampled_intensity = fresnel_reflectivity * \
                np.exp(-2.0 * alpha * sampled_range) * fill_fraction
            false_intensity[candidate_indices] = sampled_intensity
            false_power[candidate_indices] = sampled_intensity / \
                np.maximum(sampled_range ** 2, 1.0e-12)

        use_false = np.logical_and(
            false_power > target_power, false_power >= power_min)
        target_detectable = target_power >= power_min
        retained = np.logical_and(target_detectable, np.logical_not(use_false))
        lost = np.logical_not(np.logical_or(use_false, retained))

        new_range = np.zeros_like(ranges)
        new_intensity = np.zeros_like(intensity)
        if np.any(retained):
            retained_indices = np.flatnonzero(retained)
            snr = target_power[retained_indices] / power_min
            sigma = self.range_accuracy / np.sqrt(
                2.0 * np.maximum(snr, 1.0e-12))
            noisy_range = ranges[retained_indices] + rng.normal(
                0.0, sigma)
            new_range[retained_indices] = np.maximum(
                noisy_range, self.min_range)
            new_intensity[retained_indices] = \
                attenuated_intensity[retained_indices]
        new_range[use_false] = false_range[use_false]
        new_intensity[use_false] = false_intensity[use_false]

        kept = np.logical_and(np.logical_not(lost), valid_ray)
        scale = np.zeros_like(ranges)
        scale[kept] = new_range[kept] / ranges[kept]
        output_xyz = xyz[kept] * scale[kept, None]
        output_intensity = np.clip(
            new_intensity[kept], 0.0, 1.0)
        output = np.concatenate(
            (output_xyz, output_intensity[:, None]), axis=1)
        output = np.ascontiguousarray(output, dtype=np.float32)

        output_ranges = np.linalg.norm(output[:, :3], axis=1) \
            if output.shape[0] > 0 else np.empty((0,), dtype=np.float32)
        stats = dict(empty_stats)
        stats.update({
            'retained_original_point_count': int(retained.sum()),
            'false_return_count': int(use_false.sum()),
            'lost_point_count': int(lost.sum()),
            'local_augmented_point_count': int(output.shape[0]),
            'input_intensity_mean': float(intensity.mean()),
            'input_intensity_std': float(intensity.std()),
            'output_intensity_mean': float(output_intensity.mean())
            if output_intensity.size else 0.0,
            'output_intensity_std': float(output_intensity.std())
            if output_intensity.size else 0.0,
            'input_distance_histogram': self._distance_histogram(ranges),
            'output_distance_histogram': self._distance_histogram(
                output_ranges),
            'keep_ratio': float(output.shape[0] / input_count),
            'after_dropout_point_count': int(output.shape[0]),
            'final_point_count': int(output.shape[0]),
            'mean_distance': float(output_ranges.mean())
            if output_ranges.size else 0.0,
            'max_distance': float(output_ranges.max())
            if output_ranges.size else 0.0
        })
        return output, stats
