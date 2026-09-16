import unittest
import numpy as np
from qa_observation_diagnostic.collect import _box_geometry, _angular_occlusion, _grid_target_cells


def box(cx, cy, length=4.0, width=2.0, z0=-1.0, z1=1.0):
    xy = np.array([[cx+length/2, cy+width/2], [cx+length/2, cy-width/2],
                   [cx-length/2, cy-width/2], [cx-length/2, cy+width/2]])
    low = np.c_[xy, np.full(4, z0)]
    high = np.c_[xy, np.full(4, z1)]
    return np.r_[low, high]


class GeometryTest(unittest.TestCase):
    def test_geometry(self):
        g = _box_geometry(box(10, 0))
        self.assertAlmostEqual(g['length'], 4.0, places=5)
        self.assertAlmostEqual(g['width'], 2.0, places=5)
        ids = _grid_target_cells(g, (10, 10), (-20, -20, -3, 20, 20, 3))
        self.assertGreater(len(ids), 0)

    def test_occlusion_proxy(self):
        front = _box_geometry(box(10, 0))
        back = _box_geometry(box(20, 0))
        side = _box_geometry(box(10, 10))
        occ = _angular_occlusion([front, back], 1)
        side_occ = _angular_occlusion([front, side], 1)
        self.assertGreater(occ, side_occ)


if __name__ == '__main__':
    unittest.main()
