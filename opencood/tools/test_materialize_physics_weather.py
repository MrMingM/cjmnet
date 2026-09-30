"""Small server-side checks for the reusable physics-weather dataset tool."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from opencood.tools.materialize_physics_weather import (
    cav_order, frames, in_range, load_pose, validate_output_location, write_pcd,
)
from opencood.utils.pcd_utils import pcd_to_np


class MaterializePhysicsWeatherTest(unittest.TestCase):
    def test_sample_and_agent_indices_follow_basedataset_order(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for scene in ('scene_b', 'scene_a'):
                for cav in ('-1', '10', '2'):
                    path = root / scene / cav
                    path.mkdir(parents=True)
                    for stamp in ('000002', '000001'):
                        (path / (stamp + '.yaml')).touch()
                        (path / (stamp + '.pcd')).touch()
            self.assertEqual(cav_order(root / 'scene_a'), ['10', '2', '-1'])
            rows = list(frames(root))
            self.assertEqual(len(rows), 12)
            self.assertEqual([(r[0], r[1], r[2], r[3], r[4]) for r in rows[:3]], [
                (0, 0, 'scene_a', '10', '000001'),
                (0, 1, 'scene_a', '2', '000001'),
                (0, 2, 'scene_a', '-1', '000001'),
            ])
            self.assertEqual(rows[-1][:5], (3, 2, 'scene_b', '-1', '000002'))

    def test_pcd_round_trip_matches_project_reader(self):
        with tempfile.TemporaryDirectory() as folder:
            points = np.array([[1.25, -2.5, .125, .3],
                               [5., 7., -1., .8]], dtype=np.float32)
            target = Path(folder) / 'frame.pcd'
            write_pcd(target, points, verify=True)
            loaded = pcd_to_np(str(target))
            np.testing.assert_allclose(loaded[:, :3], points[:, :3], atol=2e-5)
            np.testing.assert_allclose(loaded[:, 3], points[:, 3], atol=1/255 + 1e-5)

    def test_empty_pcd_can_be_read(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'empty.pcd'
            write_pcd(target, np.empty((0, 4), dtype=np.float32), verify=True)
            self.assertEqual(pcd_to_np(str(target)).shape, (0, 4))

    def test_output_cannot_replace_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'source'
            source.mkdir()
            with self.assertRaises(ValueError):
                validate_output_location(source / 'generated', {'train': str(source)})

    def test_projected_range_uses_strict_voxel_bounds(self):
        points = np.array([[0., 0., 0., .5], [1., 0., 0., .5]],
                          dtype=np.float32)
        transform = np.eye(4)
        transform[0, 3] = 1.
        inside, xyz = in_range(points, transform, [0., -1., -1., 2., 1., 1.])
        np.testing.assert_array_equal(inside, [True, False])
        np.testing.assert_allclose(xyz[:, 0], [1., 2.])

    def test_numpy_tagged_official_pose_yaml(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'frame.yaml'
            pose = [np.float32(value) for value in (1, 2, 3, 0, 90, 0)]
            content = yaml.dump({'lidar_pose': pose})
            self.assertIn('!!python/object/apply:numpy.core.multiarray.scalar',
                          content)
            path.write_text(content, encoding='utf-8')
            self.assertEqual(load_pose(path), [1., 2., 3., 0., 90., 0.])


if __name__ == '__main__':
    unittest.main()
