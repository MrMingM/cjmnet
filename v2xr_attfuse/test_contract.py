"""Meaningful reader/label and geometry checks; execute on the research server."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

from v2xr_attfuse.data import frame_indices, validate_annotation
from v2xr_attfuse.geometry import CheckedGeometry


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not sys.platform.startswith('linux'):
            raise RuntimeError('Checks run on the Linux research server')

    def annotation(self):
        return dict(lidar_pose=[0, 0, 0, 0, 0, 0], ego_speed=0, vehicles={1:
            dict(location=[5, 0, 0], angle=[0, 0, 0], center=[0, 0, 0], extent=[1, 2, 1])})

    def test_reject_malformed_pose_and_nonfinite_extent(self):
        value = self.annotation()
        self.assertEqual(validate_annotation(value), 1)
        value['lidar_pose'] = [0] * 5
        with self.assertRaises(ValueError):
            validate_annotation(value)
        value = self.annotation()
        value['vehicles'][1]['extent'][0] = float('nan')
        with self.assertRaises(ValueError):
            validate_annotation(value)

    def test_existing_world_projection_uses_half_extents(self):
        import numpy as np
        from opencood.utils.box_utils import project_world_objects
        value = self.annotation()
        output = {}
        project_world_objects(value['vehicles'], output, value['lidar_pose'], [-100] * 3 + [100] * 3, 'hwl')
        np.testing.assert_allclose(output[1][0, :6], [5, 0, 0, 2, 4, 2], atol=1e-6)

    def test_rotated_ego_pose_projects_world_forward_to_local_x(self):
        import numpy as np
        from opencood.utils.box_utils import project_world_objects
        value = self.annotation()
        value['lidar_pose'] = [10, 20, 0, 0, 90, 0]
        value['vehicles'][1]['location'] = [10, 25, 0]
        value['vehicles'][1]['angle'] = [0, 90, 0]
        output = {}
        project_world_objects(value['vehicles'], output, value['lidar_pose'], [-100] * 3 + [100] * 3, 'hwl')
        np.testing.assert_allclose(output[1][0, :3], [5, 0, 0], atol=1e-6)

    def test_current_reader_preserves_red_intensity_in_ascii_and_binary(self):
        import numpy as np
        import open3d as o3d
        from opencood.utils.pcd_utils import pcd_to_np
        xyz = np.array([[10., 0, -1], [20, 1, 0], [30, 2, 1]])
        intensity = np.array([51., 153, 204]) / 255
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(xyz)
        cloud.colors = o3d.utility.Vector3dVector(np.column_stack((intensity, np.ones(3), np.zeros(3))))
        with tempfile.TemporaryDirectory(prefix='v2xr_reader_') as tmp:
            for ascii_mode in (True, False):
                path = Path(tmp) / ('ascii.pcd' if ascii_mode else 'binary.pcd')
                self.assertTrue(o3d.io.write_point_cloud(str(path), cloud, write_ascii=ascii_mode, compressed=False))
                np.testing.assert_allclose(pcd_to_np(str(path)), np.column_stack((xyz, intensity)), atol=1e-6)

    def test_smoke_spans_all_scenes_and_selection_is_global(self):
        keys = [['sceneA', f'{i:06}'] for i in range(4)] + [['sceneB', f'{i:06}'] for i in range(6)]
        self.assertEqual(frame_indices(keys, 2), [0, 3, 4, 9])
        self.assertEqual(frame_indices(keys, 0), list(range(10)))

    def test_parsed_model_grid_is_json_serializable(self):
        import numpy as np
        from v2xr_attfuse.run import jsonable
        value = jsonable(dict(grid_size=np.array([704, 200, 1], dtype=np.int64), scalar=np.float64(.4)))
        self.assertEqual(json.loads(json.dumps(value)), dict(grid_size=[704, 200, 1], scalar=.4))

    def test_invalid_iou_is_rejected_before_matching(self):
        import numpy as np
        common = SimpleNamespace(convert_format=lambda x: [], compute_iou=lambda a, b: np.array([float('nan')]))
        original = common.compute_iou
        with CheckedGeometry(common, np):
            with self.assertRaises(ValueError):
                common.compute_iou(None, [None])
        self.assertIs(common.compute_iou, original)

    def test_invalid_polygon_is_rejected(self):
        import numpy as np
        from shapely.geometry import Polygon
        common = SimpleNamespace(convert_format=lambda x: [Polygon([(0, 0), (1, 1), (0, 1), (1, 0)])],
            compute_iou=lambda a, b: np.zeros(len(b)))
        with CheckedGeometry(common, np):
            with self.assertRaises(ValueError):
                common.convert_format(np.zeros((1, 4, 2)))


if __name__ == '__main__':
    unittest.main()
