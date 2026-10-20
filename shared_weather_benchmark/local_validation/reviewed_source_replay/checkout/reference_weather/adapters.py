"""Explicit XYZI adapters around the published physical simulation kernels."""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import random

import numpy as np

from reference_weather.common import atomic_json


@contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def infer_channels(points, tolerance, max_channels):
    """Cluster observed elevation rays; never pretend reconstructed IDs are sensor IDs."""
    xyz = np.asarray(points)[:, :3]
    angles = np.degrees(np.arctan2(xyz[:, 2], np.linalg.norm(xyz[:, :2], axis=1)))
    order = np.argsort(angles)
    sorted_angles = angles[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_angles) > tolerance) + 1, len(angles)]
    centers, assignments = [], np.full(len(angles), -1, dtype=np.int32)
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end - start < 3:
            continue
        group = sorted_angles[start:end]
        if group[-1] - group[0] > tolerance * 2:
            raise ValueError('Elevation rays cannot be separated reliably; supply a verified sensor profile.')
        assignments[order[start:end]] = len(centers)
        centers.append(float(np.mean(group)))
    if not 4 <= len(centers) <= max_channels:
        raise ValueError(f'Expected 4..{max_channels} elevation groups, found {len(centers)}')
    return assignments, centers


def ground_plane(points, seed):
    """Fit OPV2V's observed road instead of STF's fixed mounting-height window."""
    from sklearn.linear_model import RANSACRegressor
    xyz = np.asarray(points)[:, :3]
    radius = np.linalg.norm(xyz[:, :2], axis=1)
    candidates = xyz[(radius > 5) & (radius < 70) & (xyz[:, 2] < -0.5)]
    if len(candidates) < 30:
        raise ValueError('Insufficient road candidates for reference Snow intensity estimation')
    # A lower envelope per 5m XY cell reduces roofs/vehicle surfaces in the fit.
    cells = np.floor(candidates[:, :2] / 5).astype(np.int32)
    _, inverse = np.unique(cells, axis=0, return_inverse=True)
    envelope = []
    for cell in range(int(inverse.max()) + 1):
        group = candidates[inverse == cell]
        low = np.quantile(group[:, 2], .2)
        envelope.extend(group[group[:, 2] <= low + .10])
    envelope = np.asarray(envelope)
    if len(envelope) < 20:
        raise ValueError('Insufficient lower-envelope road points')
    reg = RANSACRegressor(loss='squared_error', residual_threshold=.15,
                          max_trials=200, random_state=int(seed)).fit(envelope[:, :2], envelope[:, 2])
    normal = np.r_[reg.estimator_.coef_, -1.0]
    norm = np.linalg.norm(normal)
    normal /= norm
    offset = float(reg.estimator_.intercept_ / norm)
    if abs(normal[2]) < .8 or np.count_nonzero(reg.inlier_mask_) < 20:
        raise ValueError('Unreliable OPV2V road plane; Snow generation stopped without a clean fallback')
    return normal, offset


class ReferenceSimulator:
    def __init__(self, mode, config, assets_root):
        self.mode, self.config, self.assets = mode, config, Path(assets_root)
        self.backend = None
        if mode == 'fog':
            from reference_weather.vendor.fog import fog_simulation as fog
            self.backend = fog
            fog.INTEGRAL_PATH = self.assets / 'fog' / 'integral_lookup_tables' / 'original'
            s = config['fog']
            if s['beta_policy'] != 'upstream_constructor':
                raise ValueError('Unknown Fog beta policy')
            self.parameters = fog.ParameterSet(alpha=float(s['alpha']), gamma=float(s['gamma']))
            if s['alpha'] not in fog.get_available_alphas():
                raise ValueError('Fog alpha has no exact upstream integral table')
        elif mode == 'rain':
            import scipy.integrate
            # Compatibility alias for old LISA/PyMieScatt; same trapezoidal rule.
            if not hasattr(scipy.integrate, 'trapz'):
                scipy.integrate.trapz = np.trapz
            from reference_weather.vendor.lisa.atmos_models import LISA
            params = {k: v for k, v in config['rain'].items() if k != 'rain_rate_mm_h'}
            with working_directory(self.assets / 'lisa'):
                self.backend = LISA(saved_model=True, atm_model='rain', **params)
        elif mode == 'snow':
            from reference_weather.vendor.snow import simulation as snow
            self.backend = snow
            s = config['snow']
            self.sensor_path = self.assets / 'snow' / 'opv2v_virtual_sensor.yaml'
            if not self.sensor_path.is_file():
                raise FileNotFoundError('Run the assets stage before constructing Snow')
            snow.SENSOR_INFO_PATH = self.sensor_path
            snow.CHANNEL_WORKERS = int(s['channel_workers'])
        else:
            raise ValueError(mode)

    def augment(self, raw, seed):
        points = np.asarray(raw, dtype=np.float64).copy()
        # Zero-range points make ray normalization undefined; record their removal.
        valid = np.linalg.norm(points[:, :3], axis=1) > 1e-5
        points = points[valid]
        np.random.seed(int(seed))
        random.seed(int(seed))
        stats = dict(zero_range_removed=int((~valid).sum()))
        if not len(points):
            return np.empty((0, 4), np.float32), stats
        if self.mode == 'fog':
            s = self.config['fog']
            self.backend.RNG = np.random.default_rng(int(seed))
            points[:, 3] *= 255.0
            output, _, info = self.backend.simulate_fog(
                self.parameters, points, int(s['noise']), gain=bool(s['gain']), noise_variant=s['noise_variant'])
            stats.update(fog_replaced=int(info['num_fog_responses']),
                         effective_alpha=float(self.parameters.alpha), effective_beta=float(self.parameters.beta),
                         effective_gamma=float(self.parameters.gamma))
            keep = output[:, 3] > 0
            stats['lost'] = int((~keep).sum())
            output = output[keep, :4]
            output[:, 3] /= 255.0
        elif self.mode == 'rain':
            output = self.backend.augment(points, float(self.config['rain']['rain_rate_mm_h']))
            labels = output[:, 4]
            stats.update(lost=int((labels == 0).sum()), rain_replaced=int((labels == 1).sum()),
                         target_returns=int((labels == 2).sum()))
            output = output[labels != 0, :4]
        else:
            s = self.config['snow']
            assignments, centers = infer_channels(points, float(s['elevation_cluster_tolerance_deg']),
                                                  int(s['max_channels']))
            unmatched = assignments < 0
            if unmatched.mean() > float(s['max_unmatched_fraction']):
                raise ValueError('Too many points lack a reconstructed Snow channel')
            stats.update(reconstructed_elevation_centers_deg=centers,
                         unmatched_channel_points=int(unmatched.sum()),
                         points_beyond_particle_radius=int((np.linalg.norm(points[:, :3], axis=1)
                                                            > s['particle_radius_m']).sum()))
            points, assignments = points[~unmatched], assignments[~unmatched]
            points[:, 3] *= 255.0
            pc = np.column_stack((points, assignments))
            normal, offset = ground_plane(pc, seed)
            # Original augment() retains its intensity/noise fit; only its road
            # plane input uses OPV2V geometry instead of the STF mounting window.
            self.backend.calculate_plane = lambda _: (normal, offset)
            stats.update(ground_normal=normal.tolist(), ground_offset=offset)
            original_stats, output = self.backend.augment(
                pc, particle_file_prefix='particles', beam_divergence=float(np.degrees(s['beam_divergence_rad'])),
                shuffle=True, show_progressbar=False, only_camera_fov=False,
                noise_floor=float(s['noise_floor']), root_path=str(self.assets / 'snow'))
            stats.update(snow_attenuated=int(original_stats[0]), lost=int(original_stats[1]),
                         snow_replaced=int((output[:, 4] == 2).sum()))
            output = output[:, :4]
            output[:, 3] /= 255.0
        if not np.isfinite(output).all():
            raise ValueError(f'{self.mode} returned NaN/Inf; generation stopped')
        if len(output) and ((output[:, 3] < 0).any() or (output[:, 3] > 1 + 1e-6).any()):
            raise ValueError(f'{self.mode} returned invalid intensity')
        stats['empty_output'] = len(output) == 0
        return np.ascontiguousarray(output, dtype=np.float32), stats
