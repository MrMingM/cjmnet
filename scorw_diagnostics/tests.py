"""Run only on server: AP invariants, voxel capacity/duplicates and transactions."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from .core import stage, write, read
from .ap_audit import ap
from .voxel_trace import Tracer, REASONS, trace


class Checks(unittest.TestCase):
    def test_sorting(self):
        r = [dict(tp=[0],fp=[1],score=[.1],gt=1),dict(tp=[1],fp=[0],score=[.9],gt=0)]
        self.assertEqual(ap(r, True), 1.)
        self.assertEqual(ap(r, False), .5)

    def test_empty_ap(self):
        self.assertEqual(ap([dict(tp=[],fp=[],score=[],gt=3)],True),0.)
        self.assertIsNone(ap([], True))

    def test_scene_pair_scope(self):
        self.assertNotEqual(('sceneA','001','1'),('sceneB','001','1'))

    def test_transaction(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'s'; calls=[]
            def work():
                calls.append(1); write(p/'result.json',dict(ok=True))
            stage(p,dict(a=1),work); stage(p,dict(a=1),work)
            self.assertEqual(len(calls),1)
            with self.assertRaises(ValueError): stage(p,dict(a=2),work)
            write(p/'result.json',dict(ok=False))
            with self.assertRaises(ValueError): stage(p,dict(a=1),work)

    def test_interrupted_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'s'
            def fail():
                write(p/'result.json',dict(ok=False)); raise RuntimeError('fixture interruption')
            with self.assertRaises(RuntimeError): stage(p,dict(a=1),fail)
            stage(p,dict(a=1),lambda:write(p/'result.json',dict(ok=True)))
            self.assertTrue(read(p/'result.json')['ok'])

    def pre(self):
        from opencood.data_utils.pre_processor import build_preprocessor
        return build_preprocessor(dict(core_method='SpVoxelPreprocessor',cav_lidar_range=[0,0,0,4,4,4],
            args=dict(voxel_size=[1,1,1],max_points_per_voxel=2,max_voxel_test=2,max_voxel_train=2)),False)

    def test_duplicate_and_capacity(self):
        p = np.array([[.1,.1,.1,.5]]*3+[[1.1,.1,.1,.6],[2.1,.1,.1,.7],[4.,0,0,.8]],dtype=np.float32)
        reason, s4, audit = Tracer(self.pre()).slots(p)
        self.assertEqual(list(reason[:3]),[7,7,6])
        self.assertEqual(reason[4],REASONS.index('voxel_total_limit'))
        self.assertEqual(reason[5],REASONS.index('voxel_boundary'))
        self.assertEqual(audit['slots'],3)

    def test_empty_voxel(self):
        reason,s4,audit=Tracer(self.pre()).slots(np.empty((0,4),dtype=np.float32))
        self.assertEqual(len(reason),0);self.assertEqual(audit['slots'],0)

    def test_input_order_changes_ID(self):
        pc=np.asarray([[.1,.1,.1,.1],[.1,.1,.1,.2],[.1,.1,.1,.3]],dtype=np.float32)
        a=Tracer(self.pre()).slots(pc)[0];b=Tracer(self.pre()).slots(pc[::-1].copy())[0]
        self.assertTrue(np.array_equal(a,b))
        self.assertFalse(np.array_equal(pc[a==7],pc[::-1][b==7]))

    def test_original_position_membership(self):
        from .point_flow import membership
        from shared_weather_benchmark.geometry import box_corners
        obj=dict(location=[50,0,0],center=[0,0,0],angle=[0,0,0],extent=[2,1,1])
        box=np.array([box_corners(obj)],dtype=np.float32)
        clean=np.array([[50,0,0,.4],[0,0,0,.4]],dtype=np.float32)
        hit,m=membership(clean,[0]*6,[0]*6,box,1e-6)
        self.assertEqual(hit[0].tolist(),[0])
        _,m=membership(clean,[0]*6,[0]*6,np.repeat(box,2,axis=0),1e-6)
        self.assertEqual(m.tolist(),[2,0])

    def test_inactive_is_not_ego_loss(self):
        from opencood.utils import pcd_utils
        reasons,_,audit,_=trace(np.array([[0,0,0,.5]],dtype=np.float32),[0]*6,[0]*6,
                                Tracer(self.pre()),pcd_utils,False)
        self.assertEqual(reasons.tolist(),[REASONS.index('inactive_CAV')])

    def test_actual_runtime_project_equivalence(self):
        from scorw_r3.runtime import Runtime
        from opencood.utils import pcd_utils
        from .voxel_trace import project
        runtime=Runtime.__new__(Runtime);runtime.pcd=pcd_utils
        limit=[-140,-40,-3,140,40,1];runtime.h={'preprocess':{'cav_lidar_range':limit}}
        cloud=np.array([[0,0,0,.1],[3,2,0,.2],[30,-4,-1,.3],[300,0,0,.4]],dtype=np.float32)
        pose=[10,4,0,2,19,-3];ego=[3,-6,0,-1,-8,2]
        a,ids=runtime.project(cloud,pose,ego)
        b,_,other=project(cloud,pose,ego,pcd_utils,limit)
        self.assertTrue(np.array_equal(a,b));self.assertTrue(np.array_equal(ids,other))


def run(out):
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    write(out/'UNIT_TESTS.json',dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
          status='passed' if result.wasSuccessful() else 'failed',server_only=True,GPU_used=False))
    if not result.wasSuccessful():raise ValueError('Diagnostic unit tests failed')
