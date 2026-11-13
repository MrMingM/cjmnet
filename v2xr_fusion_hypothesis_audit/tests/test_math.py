import unittest

import numpy as np
from shared.geometry import cosine, error, wrap_yaw
from shared.statistics import correlation_blocks, ratio_interval


class DiagnosticsTests(unittest.TestCase):
    def test_yaw(self):
        self.assertAlmostEqual(wrap_yaw(np.pi-.1),-.1)
        self.assertAlmostEqual(error(np.array([1,0,0,0,0,0,np.pi-.1]),np.zeros(7))[6],-.1)

    def test_cosine(self):
        self.assertAlmostEqual(cosine(np.array([2.,0]),np.array([1.,0])),1)
        self.assertEqual(cosine(np.zeros(2),np.ones(2)),0)

    def test_scene_resampling(self):
        out=ratio_interval({'a':[1,4],'b':[2,3]})
        self.assertAlmostEqual(out['value'],3/7)
        self.assertEqual(out['scenes'],2)
        x=np.arange(8).reshape(-1,1)
        result=correlation_blocks(x,-x,['a']*4+['b']*4)
        self.assertAlmostEqual(result['r'][0],-1)
