"""Physics-guided online LiDAR snowfall augmentation.

This is an original, vectorized OpenCOOD adapter inspired by the snowfall
rate, particle-size, beam-occlusion and return-replacement model of Hahner
et al. (CVPR 2022, SysCV/LiDAR_snow_sim).  It is deliberately an online
surrogate rather than a copy of the reference simulator: OPV2V point clouds
do not contain HDL-64 channel IDs and online training cannot depend on
precomputed particle fields.
"""

import numpy as np


class PhysicsSnowAugmentor:
    """Scene-shared snowfall with independent per-agent particle fields."""

    def __init__(self, config, lidar_range, seed=20):
        self.config = dict(config or {})
        self.lidar_range = np.asarray(lidar_range, dtype=np.float32)
        if self.lidar_range.shape != (6,):
            raise ValueError('lidar_range must contain 6 values')
        self.seed = int(seed)
        self.perturbation_frame = 'local_sensor'

        self.fixed_snowfall_rate = self.config.get(
            'fixed_snowfall_rate', None)
        if self.fixed_snowfall_rate is not None:
            self.fixed_snowfall_rate = float(self.fixed_snowfall_rate)
        self.snowfall_rate_low = float(
            self.config.get('snowfall_rate_low', 0.5))
        self.snowfall_rate_high = float(
            self.config.get('snowfall_rate_high', 2.5))
        self.terminal_velocity = float(
            self.config.get('terminal_velocity', 1.0))
        self.snow_density = float(self.config.get('snow_density', 0.1))
        # The reference simulator exposes beam divergence in degrees and
        # converts it with np.radians before constructing beam limits.
        self.beam_divergence_deg = float(
            self.config.get('beam_divergence', 0.003))
        self.snow_reflectivity = float(
            self.config.get('snow_reflectivity', 0.9))
        self.intensity_decay_range = float(
            self.config.get('intensity_decay_range', 15.0))
        self.intensity_threshold = float(
            self.config.get('intensity_threshold', 0.01))
        self.max_particle_diameter = float(
            self.config.get('max_particle_diameter', 0.02))
        self.intercept_scale = float(
            self.config.get('intercept_scale', 1.0))
        self._validate_config()

    def _validate_config(self):
        if not 0.0 < self.snowfall_rate_low <= self.snowfall_rate_high:
            raise ValueError(
                'snowfall rate must satisfy 0 < low <= high')
        if self.fixed_snowfall_rate is not None and \
                self.fixed_snowfall_rate <= 0:
            raise ValueError('fixed_snowfall_rate must be positive')
        if self.terminal_velocity <= 0 or self.snow_density <= 0:
            raise ValueError(
                'terminal_velocity and snow_density must be positive')
        if self.beam_divergence_deg <= 0:
            raise ValueError('beam_divergence must be positive')
        if not 0 < self.snow_reflectivity <= 1:
            raise ValueError('snow_reflectivity must be in (0, 1]')
        if self.intensity_decay_range <= 0:
            raise ValueError('intensity_decay_range must be positive')
        if not 0 <= self.intensity_threshold < 1:
            raise ValueError('intensity_threshold must be in [0, 1)')
        if self.max_particle_diameter <= 0 or self.intercept_scale <= 0:
            raise ValueError(
                'particle diameter and intercept_scale must be positive')

    @staticmethod
    def validate_points(points):
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] != 4:
            raise ValueError(
                'snow points must have shape [N, 4], got %s' %
                (tuple(points.shape),))
        if not np.issubdtype(points.dtype, np.floating):
            raise TypeError('snow points must use a floating dtype')
        if not np.isfinite(points).all():
            raise ValueError('snow points contain NaN or Inf')
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

    def sample_snowfall_rate(self, epoch, sample_index):
        if self.fixed_snowfall_rate is not None:
            return self.fixed_snowfall_rate
        rng = self.make_rng(epoch, sample_index, 0, stream=13)
        return float(rng.uniform(
            self.snowfall_rate_low, self.snowfall_rate_high))

    def physical_parameters(self, snowfall_rate):
        """Return occupancy, particle scale and ray-interception hazard.

        Occupancy and the Sekhon--Srivastava rate follow the public reference
        implementation.  For exponential particle diameters, converting
        volume occupancy to a Poisson line-interception process gives the
        hazard ``occupancy / (2 * diameter_scale)``.
        """
        snowfall_rate = float(snowfall_rate)
        if snowfall_rate <= 0:
            raise ValueError('snowfall_rate must be positive')
        occupancy = snowfall_rate / (
            3.6e6 * self.snow_density * self.terminal_velocity)
        diameter_rate_per_cm = 22.9 * snowfall_rate ** -0.45
        diameter_scale_m = 1.0 / diameter_rate_per_cm / 100.0
        intercept_hazard = (
            self.intercept_scale * occupancy /
            max(2.0 * diameter_scale_m, np.finfo(np.float64).eps))
        return occupancy, diameter_scale_m, intercept_hazard

    def augment(self, points, rng, snowfall_rate):
        points = self.validate_points(points)
        snowfall_rate = float(snowfall_rate)
        occupancy, diameter_scale, hazard = self.physical_parameters(
            snowfall_rate)
        input_count = int(points.shape[0])
        if input_count == 0:
            return points.copy(), self._empty_stats(
                snowfall_rate, occupancy, diameter_scale, hazard)

        xyz = points[:, :3].astype(np.float64)
        intensity = np.clip(
            points[:, 3].astype(np.float64), 0.0, 1.0)
        distance = np.linalg.norm(xyz, axis=1)

        hit_range = rng.exponential(
            1.0 / max(hazard, np.finfo(np.float64).eps),
            size=input_count)
        valid_ray = distance > 1.0e-6
        intercepted = valid_ray & (hit_range < distance)

        # Intersected particles are cross-section size biased.  An
        # exponential diameter law therefore becomes Gamma(k=3, theta).
        particle_diameter = np.minimum(
            rng.gamma(shape=3.0, scale=diameter_scale, size=input_count),
            self.max_particle_diameter)
        beam_divergence_rad = np.deg2rad(self.beam_divergence_deg)
        beam_radius = np.maximum(
            0.5 * beam_divergence_rad * hit_range,
            0.5 * particle_diameter)
        occlusion = np.clip(
            np.square(particle_diameter / (2.0 * beam_radius)),
            0.0, 1.0)
        occlusion[~intercepted] = 0.0

        receiver_overlap = np.clip(
            (hit_range - 0.9) / 0.1, 0.0, 1.0)
        snow_intensity = (
            self.snow_reflectivity * occlusion * receiver_overlap /
            (1.0 + np.square(
                hit_range / self.intensity_decay_range)))
        # The Poisson interception hazard is also the particle-field
        # extinction coefficient.  Apply two-way transmittance to the hard
        # target, then account for the explicitly sampled dominant flake.
        transmittance = np.exp(-2.0 * hazard * distance)
        hard_intensity = intensity * transmittance * (1.0 - occlusion)
        snow_return = intercepted & (snow_intensity > hard_intensity)
        attenuated = ~snow_return & (
            hard_intensity < intensity)

        output = points.copy()
        output[:, 3] = hard_intensity.astype(np.float32)
        if np.any(snow_return):
            scaling = hit_range[snow_return] / distance[snow_return]
            output[snow_return, :3] = (
                xyz[snow_return] * scaling[:, None]).astype(np.float32)
            output[snow_return, 3] = \
                snow_intensity[snow_return].astype(np.float32)

        lost = (~snow_return) & (
            output[:, 3] < self.intensity_threshold)
        output = np.ascontiguousarray(output[~lost], dtype=np.float32)

        snow_count = int(snow_return.sum())
        lost_count = int(lost.sum())
        retained_count = input_count - snow_count - lost_count
        stats = {
            'perturbation_frame': self.perturbation_frame,
            'snowfall_rate': snowfall_rate,
            'terminal_velocity': self.terminal_velocity,
            'snow_density': self.snow_density,
            'beam_divergence_deg': self.beam_divergence_deg,
            'occupancy_ratio': occupancy,
            'particle_diameter_scale': diameter_scale,
            'intercept_hazard': hazard,
            'mean_two_way_transmittance': float(transmittance.mean()),
            'intercepted_point_count': int(intercepted.sum()),
            'attenuated_point_count': int(attenuated.sum()),
            'retained_original_point_count': retained_count,
            'snow_return_count': snow_count,
            'lost_point_count': lost_count,
            'input_point_count': input_count,
            'local_augmented_point_count': int(output.shape[0]),
            'input_intensity_mean': float(intensity.mean()),
            'output_intensity_mean': (
                float(output[:, 3].mean()) if output.shape[0] else 0.0),
            'after_range_point_count': input_count,
            'noise_point_count': snow_count,
            'keep_ratio': float(output.shape[0] / max(input_count, 1)),
            'after_dropout_point_count': int(output.shape[0]),
            'drop_probability_mean': -1.0,
            'fallback_used': False,
            'empty_input': False,
            'mean_distance': (
                float(np.linalg.norm(output[:, :3], axis=1).mean())
                if output.shape[0] else 0.0),
            'max_distance': (
                float(np.linalg.norm(output[:, :3], axis=1).max())
                if output.shape[0] else 0.0),
        }
        return output, stats

    def _empty_stats(
            self, snowfall_rate, occupancy, diameter_scale, hazard):
        return {
            'perturbation_frame': self.perturbation_frame,
            'snowfall_rate': snowfall_rate,
            'terminal_velocity': self.terminal_velocity,
            'snow_density': self.snow_density,
            'beam_divergence_deg': self.beam_divergence_deg,
            'occupancy_ratio': occupancy,
            'particle_diameter_scale': diameter_scale,
            'intercept_hazard': hazard,
            'mean_two_way_transmittance': 1.0,
            'intercepted_point_count': 0,
            'attenuated_point_count': 0,
            'retained_original_point_count': 0,
            'snow_return_count': 0,
            'lost_point_count': 0,
            'input_point_count': 0,
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
