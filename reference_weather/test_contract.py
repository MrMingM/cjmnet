"""Server-side adapter/identity checks; no detector experiments run locally."""
from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from reference_weather.adapters import ReferenceSimulator, infer_channels
from reference_weather.common import (atomic_json, development_path, iter_frames,
                                     read_json, stable_seed, verify_sources)
from reference_weather.generate import metadata
from reference_weather.prepare_assets import particle_parameters, select_particle_members


class ContractTests(unittest.TestCase):
    def test_source_snapshot(self):
        self.assertTrue(verify_sources())

    def test_no_test_split(self):
        with self.assertRaises(ValueError):
            development_path('/unread/test', 'test')

    def test_seed_is_per_file_and_independent_of_task_order(self):
        keys = ['fog', 'validate', 'scene', '100', '000010']
        self.assertEqual(stable_seed(20, *keys), stable_seed(20, *keys))
        self.assertNotEqual(stable_seed(20, *keys), stable_seed(20, *keys[:-1], '000011'))
        self.assertNotEqual(stable_seed(20, *keys), stable_seed(20, 'rain', *keys[1:]))

    def test_reconstructed_channels_preserve_ray_groups(self):
        angles = np.radians([-20, -10, -5, 2])
        points = np.array([[10 * np.cos(a), 0, 10 * np.sin(a), .5]
                           for a in angles for _ in range(6)])
        labels, centers = infer_channels(points, .03, 64)
        self.assertEqual(labels.tolist(), np.repeat(np.arange(4), 6).tolist())
        np.testing.assert_allclose(centers, np.degrees(angles), atol=1e-6)

    def test_no_silent_channel_overflow(self):
        angles = np.radians(np.arange(-20, 45))
        points = np.array([[10 * np.cos(a), 0, 10 * np.sin(a), .5]
                           for a in angles for _ in range(3)])
        with self.assertRaises(ValueError):
            infer_channels(points, .03, 64)

    def test_fog_intensity_conversion_and_zero_return_removal(self):
        sim = ReferenceSimulator.__new__(ReferenceSimulator)
        sim.mode = 'fog'
        sim.config = dict(fog=dict(noise=0, gain=False, noise_variant='v1'))
        sim.parameters = SimpleNamespace(alpha=.06, beta=.001, gamma=1e-6)
        def fake_fog(params, points, noise, **kwargs):
            np.testing.assert_allclose(points[:, 3], [127.5, 255])
            output = points.copy()
            output[:, 3] = [51, 0]
            return output, None, dict(num_fog_responses=0)
        sim.backend = SimpleNamespace(simulate_fog=fake_fog)
        output, stats = sim.augment(np.array([[10, 0, 0, .5], [20, 0, 0, 1]]), 20)
        np.testing.assert_allclose(output, [[10, 0, 0, .2]])
        self.assertEqual(stats['lost'], 1)

    def test_lisa_lost_labels_are_removed_and_extra_label_is_dropped(self):
        sim = ReferenceSimulator.__new__(ReferenceSimulator)
        sim.mode, sim.config = 'rain', dict(rain=dict(rain_rate_mm_h=33))
        sim.backend = SimpleNamespace(augment=lambda points, rate: np.array(
            [[0, 0, 0, 0, 0], [3, 0, 0, .1, 1], [30, 0, 0, .5, 2]]))
        out, stats = sim.augment(np.array([[10, 0, 0, .5]] * 3), 20)
        self.assertEqual(out.shape, (2, 4))
        self.assertEqual(stats['lost'], 1)
        self.assertEqual(stats['rain_replaced'], 1)

    def test_particle_selection_requires_exact_64_pattern_bank(self):
        config = dict(snow=dict(snowfall_rate_mm_h=1.5, terminal_velocity_m_s=1.,
                      snow_density_g_cm3=.1, snowflake_diameter_m=.003, distribution='gunn'))
        rate, occ = particle_parameters(config)
        names = [f'npy/gunn_{rate}_{occ}_{i}.npy' for i in range(1, 65)]
        self.assertEqual(len(select_particle_members(names, config)), 64)
        with self.assertRaises(ValueError):
            select_particle_members(names[:-1], config)

    def test_smoke_metadata_does_not_expand_to_unselected_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, target = root / 'validate', root / 'generated'
            cav = source / 'scene' / '100'
            cav.mkdir(parents=True)
            for stamp in ('000001', '000002'):
                (cav / f'{stamp}.yaml').write_text('vehicles: {}')
                (cav / f'{stamp}.pcd').write_text('placeholder')
            metadata(cav, target / 'scene' / '100', '000001')
            self.assertTrue((target / 'scene' / '100' / '000001.yaml').is_symlink())
            self.assertFalse((target / 'scene' / '100' / '000002.yaml').exists())
            self.assertEqual(len(list(iter_frames(source))), 2)


if __name__ == '__main__':
    unittest.main()
