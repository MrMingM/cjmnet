"""Synthetic pure NumPy/unit tests only. Never import torch or read research data."""
import ast
import copy
import importlib.util
import json
import os
import sys
from pathlib import Path
import tempfile
import unittest
import numpy as np
from .common import PACKAGE,config,select_frames,Manifest,object_hash,atomic_json,safe_path
from .coordinates import (local_xy,world_xy,rotation,box_corners,corners_to_box,
                          crop,retained_points,angle_error)
from .observability import faces,residual,jacobian,finite_jacobian,information,complementarity
from .statistics import rho,association,incremental,ridge_oof,cluster_ci

class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.cfg=config(PACKAGE/'experiment.yaml')
        self.theta=np.array([3.,-2.,4.,2.,.37])
        t=np.linspace(-.8,.8,31)
        self.front=np.c_[np.full(len(t),2.),t]
        self.side=np.c_[t*2,np.full(len(t),1.)]

    def info(self,local):
        p=world_xy(local,self.theta)
        return information(p,self.theta,np.ones(len(p)),self.cfg)['geo']['variants']['raw']

    def test_round_trip(self):
        p=np.random.default_rng(2).normal(size=(50,2))
        np.testing.assert_allclose(local_xy(world_xy(p,self.theta),self.theta),p,atol=1e-13)
        b=np.array([3.,-2.,1.,4.,2.,1.5,.37])
        np.testing.assert_allclose(corners_to_box(box_corners(b)),b,atol=1e-12)

    def test_jacobian(self):
        p=world_xy(np.r_[self.front,self.side],self.theta)
        j=jacobian(p,self.theta)
        numerical=finite_jacobian(p,self.theta,self.cfg['finite_difference_steps'])
        np.testing.assert_allclose(j,numerical,rtol=2e-7,atol=1e-8)
        finer=finite_jacobian(p,self.theta,np.array(self.cfg['finite_difference_steps'])/10)
        np.testing.assert_allclose(j,finer,rtol=1e-7,atol=1e-8)
        fixed=faces(p,self.theta)
        np.testing.assert_allclose(residual(p,self.theta,fixed),0,atol=1e-14)

    def test_single_face_and_adjacent_faces(self):
        a,b=self.info(self.front),self.info(self.side)
        both=self.info(np.r_[self.front,self.side])
        self.assertEqual(a['rank'],2)
        self.assertEqual(b['rank'],2)
        self.assertEqual(both['rank'],3)
        self.assertFalse(both['valid'])
        self.assertTrue(both['directional_valid'])
        self.assertEqual(both['min_eigenvalue'],0)
        # Width is genuinely unconstrained by an x-face; a side adds that direction.
        self.assertEqual(a['matrix'][3,3],0)
        self.assertGreater(both['matrix'][3,3],0)
        c=complementarity(a['matrix'],b['matrix'],self.cfg['matrix_ridge'])
        self.assertGreater(c['C_logdet'],0)
        self.assertAlmostEqual(c['C_weak'],0,places=10)

    def test_all_faces_complete_nullspace(self):
        # Opposite faces are required to separate center from size.
        a=self.info(np.r_[self.front,-self.front])
        b=self.info(np.r_[self.side,-self.side])
        both=self.info(np.r_[self.front,-self.front,self.side,-self.side])
        self.assertEqual(both['rank'],5)
        self.assertGreater(both['min_eigenvalue'],max(a['min_eigenvalue'],b['min_eigenvalue']))

    def test_rotation(self):
        # Unequal face support avoids repeated largest eigenvalues; otherwise only
        # the eigenspace projector, not a particular basis vector, is identifiable.
        p=world_xy(np.r_[self.front,self.side[:11]],self.theta)
        phi=.61; r=rotation(phi)
        target=self.theta.copy(); target[:2]=r@target[:2]; target[4]+=phi
        a=information(p,self.theta,np.ones(len(p)),self.cfg)['geo']['variants']['raw']
        b=information(p@r.T,target,np.ones(len(p)),self.cfg)['geo']['variants']['raw']
        t=np.eye(5); t[:2,:2]=r
        np.testing.assert_allclose(b['matrix'],t@a['matrix']@t.T,atol=1e-11)
        np.testing.assert_allclose(b['eigenvalues'],a['eigenvalues'],atol=1e-11)
        self.assertAlmostEqual(abs(np.asarray(b['strongest_eigenvector'])@(t@a['strongest_eigenvector'])),1,places=10)

    def test_units(self):
        p=world_xy(np.r_[self.front,self.side],self.theta)
        j=jacobian(p,self.theta)
        # Change state length coordinates from m to cm, preserving meter residual.
        units=np.array([100.,100.,100.,100.,1.])
        s=np.array(self.cfg['parameter_scales'])
        a=(j*s).T@(j*s)
        b=(j/units*(s*units)).T@(j/units*(s*units))
        np.testing.assert_allclose(a,b,atol=1e-12)

    def test_empty_one_line(self):
        for p in (np.empty((0,2)),self.front[:1],self.front):
            v=self.info(p)
            self.assertFalse(v['valid'])
            self.assertTrue(np.isfinite(v['matrix']).all())
            self.assertTrue(np.isfinite(v['eigenvalues']).all())
            self.assertIsNone(v['condition_number'])
            self.assertIsNone(v['logdet'])

    def test_weight_and_count(self):
        p=world_xy(self.front,self.theta)
        a=information(p,self.theta,np.full(len(p),.5),self.cfg)
        np.testing.assert_allclose(a['gspr']['I_physical'],a['geo']['I_physical']*.5)
        repeated=information(np.repeat(p,2,0),self.theta,np.ones(2*len(p)),self.cfg)
        np.testing.assert_allclose(a['geo']['variants']['count_normalized']['matrix'],repeated['geo']['variants']['count_normalized']['matrix'],atol=1e-12)
        np.testing.assert_allclose(a['geo']['variants']['shape']['matrix'],repeated['geo']['variants']['shape']['matrix'],atol=1e-12)

    def test_provenance(self):
        f=np.zeros((2,3,4)); f[0,0,:3]=[.1,.1,.1]; f[1,0,:3]=[1.1,.1,.1]
        proc={'voxel_features':f,'voxel_coords':np.array([[0,0,0,0],[1,0,0,1]]),'voxel_num_points':np.array([1,1])}
        rel={'point_valid_mask':np.array([[1,0,0],[1,0,0]],bool),'point_reliability':np.ones((2,3))*.7,'point_uncertainty':np.ones((2,3))*.1}
        result=retained_points(proc,rel,2,[1,1,1],[0,0,0,2,2,2])
        self.assertEqual(len(result[0][0]),1)
        rel['point_valid_mask'][0,1]=True
        with self.assertRaises(ValueError):
            retained_points(proc,rel,2,[1,1,1],[0,0,0,2,2,2])

    def test_angle_and_crop(self):
        self.assertAlmostEqual(angle_error(.01,np.pi-.01),.02)
        box=np.array([3,-2,1,4,2,2,.37])
        points=np.c_[world_xy([[0,0],[3,0]],self.theta),[1,1]]
        np.testing.assert_array_equal(crop(points,box,1.15),[True,False])

class InfrastructureTests(unittest.TestCase):
    def test_inventory_ignores_protocol_metadata(self):
        from .data import inventory
        with tempfile.TemporaryDirectory(dir=PACKAGE) as root:
            source=Path(root)/'validate'/'scene'/'0'
            source.mkdir(parents=True)
            for name in ('000002.yaml','000001.yaml','data_protocol.yaml','000001_additional.yaml'):
                (source/name).write_text('{}',encoding='utf-8')
            records=inventory(source.parent.parent,5)
            self.assertEqual([(r['index'],r['frame_id']) for r in records],[(0,'000001'),(1,'000002')])

    def test_reviewed_source_variant_is_exact_and_frozen(self):
        from .common import snapshot_changes,source_identity,read_json
        current=source_identity()
        variants=read_json(PACKAGE/'reviewed_source_variants.json')
        key='opencood/data_utils/datasets/basedataset.py'
        variant=variants[key]
        current[key]=variant['local_sha256']
        changed=dict(current); changed[key]=variant['server_sha256']
        self.assertEqual(snapshot_changes(current,changed),([],[key]))
        changed[key]='unreviewed'
        self.assertIn(key,snapshot_changes(current,changed)[0])
        changed[key]=variant['server_sha256']
        changed['audit/reviewed_source_variants.json']='tampered'
        self.assertIn(key,snapshot_changes(current,changed)[0])
        changed=dict(current); changed[key]=variant['server_sha256']
        changed['audit/'+variant['reference']]='tampered'
        self.assertIn(key,snapshot_changes(current,changed)[0])

    def test_source_scope_preserves_runtime_code(self):
        from .common import include_source
        for name in ('data_utils/datasets/basedataset.py',
                     'data_utils/datasets/intermediate_fusion_dataset.py',
                     'models/sub_modules/point_pillar_scatter.py','hypes_yaml/yaml_utils.py'):
            self.assertTrue(include_source('opencood',name))
        for name in ('logs/old/config.yaml','logs/old/script.py',
                     'hypes_yaml/point_pillar_where2comm.yaml',
                     'data_utils/datasets/intermediate_fusion_dataset_back.py'):
            self.assertFalse(include_source('opencood',name))
        self.assertTrue(include_source('audit','experiment.yaml'))
        self.assertTrue(include_source('audit','reference_sources/basedataset.py.txt'))

    def test_s4_process_classification(self):
        from .device_guard import process_kind
        for args in ('tail -f /data/logs/source_evidence_audit_123/driver.log',
                     'grep source_evidence_audit',
                     '/usr/bin/pgrep -f source_evidence_audit'):
            self.assertEqual(process_kind(args),'observer')
        self.assertEqual(process_kind('/env/bin/python -u -m local_fusion_source_evidence_audit.pipeline'),'s4_python')
        self.assertEqual(process_kind('python3.10 -m local_fusion_source_evidence_audit.pipeline'),'s4_python')
        self.assertEqual(process_kind('sh local_fusion_source_evidence_audit/run_all.sh'),'unknown_s4')
        self.assertEqual(process_kind('python train.py'),'python')
        self.assertEqual(process_kind('sh -c "python -m local_fusion_source_evidence_audit.pipeline"'),'unknown_s4')

    def test_rocm_server_hcu_and_standard_gpu(self):
        from .device_guard import parse_stats
        # Exact field names and values supplied from the server.
        server={'card0':{'HCU use (%)':'0.0','vram Total Memory (MiB)':'65520',
                         'vram Total Used Memory (MiB)':'4'},
                'card4':{'HCU use (%)':'99.2','vram Total Memory (MiB)':'65520',
                         'vram Total Used Memory (MiB)':'47396'}}
        _,use,used=parse_stats(json.dumps(server),0)
        self.assertEqual((use,used),(0.,4*1024**2))
        _,use,used=parse_stats(json.dumps(server),4)
        cfg=config(PACKAGE/'experiment.yaml')['gpu']
        self.assertGreater(use,cfg['maximum_idle_use_percent'])
        self.assertGreater(used,cfg['maximum_idle_vram_mb']*1024**2)
        standard={'card0':{'GPU use (%)':'0%','VRAM Total Used Memory (B)':'4194304'}}
        self.assertEqual(parse_stats(json.dumps(standard),0)[1:],(0.,4194304.))

    def test_rocm_unknown_or_invalid_fails_closed(self):
        from .device_guard import parse_stats
        for values in ({'HCU use (%)':'N/A','vram Total Used Memory (MiB)':'4'},
                       {'HCU use (%)':'NaN','vram Total Used Memory (MiB)':'4'},
                       {'HCU use (%)':'0','vram Total Used Memory (MiB)':'Infinity'},
                       {'HCU use (%)':'0','vram Total Used Memory (MiB)':'-1'},
                       {'HCU use (%)':'101','vram Total Used Memory (MiB)':'4'},
                       {'HCU use (%)':'0','vram Total Used Memory (unknown)':'4'},
                       {'HCU use (%)':'0','GPU use (%)':'0','vram Total Used Memory (MiB)':'4'},
                       {'vram Total Used Memory (MiB)':'4'}):
            with self.subTest(values=values),self.assertRaises(RuntimeError):
                parse_stats(json.dumps({'card0':values}),0)
        for raw in ('{}','not JSON'):
            with self.assertRaises(RuntimeError):
                parse_stats(raw,0)

    def test_subset_plan(self):
        from .subset_forward import plans
        p=plans(5)
        self.assertEqual(len(p),11)
        self.assertTrue(all(s[0]==0 and len(s)==len(set(s)) for s in p))

    @unittest.skipUnless(importlib.util.find_spec('shapely'), 'Optional local Shapely missing; mandatory on server')
    def test_matching_and_cohorts(self):
        from .extract import contributions,match_predictions
        cfg=config(PACKAGE/'experiment.yaml')
        gt=np.array([[0.,0.,0.,4.,2.,2.,0.]])
        def pred(dx):
            b=gt[0].copy(); b[0]+=dx
            return {'corners':np.array([box_corners(b)]),'scores':[.9]}
        predictions={'0':pred(.5),'0,1':pred(.3),'0,2':pred(.2),'0,1,2':pred(.1)}
        c=contributions(predictions,gt,['car'],3,cfg)
        self.assertAlmostEqual(c['singles'][0]['utility'][0],.2)
        self.assertAlmostEqual(c['pairs'][0]['utility'][0],.1)
        self.assertAlmostEqual(c['pairs'][0]['gain'],.02)
        predictions['0']={'corners':[],'scores':[]}
        c=contributions(predictions,gt,['car'],3,cfg)
        self.assertTrue(c['singles'][0]['recovery'])
        self.assertFalse(c['singles'][0]['regression_valid'])
        self.assertIsNone(c['singles'][0]['utility'])
        duplicated={'corners':np.repeat(pred(.1)['corners'],2,axis=0),'scores':[.9,.8]}
        self.assertEqual(len(match_predictions(duplicated,gt,cfg)),1)

    def test_nonempty_report_pipeline(self):
        # Small synthetic fixture exercises analysis/serialization, not a research experiment.
        from .evaluate import evaluate
        from .report import write_reports
        cfg=config(PACKAGE/'experiment.yaml')
        cfg.update(bootstrap_replicates=4,permutation_replicates=2,minimum_rows=6)
        rng=np.random.default_rng(71); frames=[]
        for s in range(6):
            for weather in cfg['weathers']:
                rows=[]
                for i in (0,1,2):
                    box=np.array([0.,0.,0.,4.,2.,2.,.1*s])
                    local=np.c_[np.full(5+s+i,2.),rng.uniform(-.8,.8,5+s+i)]
                    obs=information(world_xy(local,box[[0,1,3,4,6]]),box[[0,1,3,4,6]],np.full(len(local),.8),cfg)
                    rows.append(dict(scene_id=str(s),frame_id='f',weather=weather,gt_object_id='car',
                        source_id=str(i),source_index=i,is_ego=i==0,gt_box=box,point_count=len(local)+2,
                        valid_point_count=len(local),mean_gspr_reliability=.8,mean_gspr_uncertainty=.1,
                        source_distance=10+s,viewing_angle=.2*s,observability=obs))
                singles=[dict(gt_object_id='car',source_index=i,regression_valid=True,recovery=False,lost=False,
                              utility=rng.normal(size=5)) for i in (1,2)]
                pairs=[dict(gt_object_id='car',i=1,j=2,regression_valid=True,gain=float(rng.normal()),utility=rng.normal(size=5))]
                frames.append({'sources':rows,'contributions':{'singles':singles,'pairs':pairs}})
        result=evaluate(frames,cfg,'smoke')
        self.assertEqual(result['decision']['verdict'],'SMOKE_ONLY')
        self.assertIn('geo/count_normalized',result['H2'])
        self.assertIn('geo/raw_C_weak',result['H3'])
        with tempfile.TemporaryDirectory(dir=PACKAGE) as root:
            hashes=write_reports(Path(root),result,{'identity':'unit','config':cfg})
            self.assertEqual(len(hashes),6)

    def test_sampling(self):
        rows=[{'scene_id':s,'frame_id':str(f),'index':i*10+f} for i,s in enumerate('abc') for f in range(10)]
        a=select_frames(rows,12,20261012)
        b=select_frames(list(reversed(rows)),12,20261012)
        self.assertEqual(a,b)
        self.assertEqual([sum(r['scene_id']==s for r in a) for s in 'abc'],[4,4,4])

    def test_cache_corruption_and_identity(self):
        with tempfile.TemporaryDirectory(dir=PACKAGE) as root:
            m=Manifest(root,'abc'); m.put('test.json',{'x':1})
            self.assertEqual(m.get('test.json'),{'x':1})
            with self.assertRaises(ValueError):
                Manifest(root,'xyz')
            atomic_json(Path(root)/'test.json',{'identity':'abc','payload':{'x':2}})
            with self.assertRaises(ValueError):
                m.get('test.json')

    def test_prohibited_paths(self):
        for p in ('/data/opv2v/test/a.pcd','/data/logs/source_evidence_audit_123/cache.pt'):
            with self.assertRaises(ValueError):
                safe_path(p)

    def test_no_s4_imports(self):
        for path in PACKAGE.glob('*.py'):
            tree=ast.parse(path.read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                if isinstance(node,ast.Import):
                    names=[n.name for n in node.names]
                elif isinstance(node,ast.ImportFrom):
                    names=[node.module or '']
                else:
                    continue
                self.assertFalse(any(n.startswith('local_fusion_') for n in names),(path,names))

    def test_scene_statistics(self):
        cfg=config(PACKAGE/'experiment.yaml'); cfg['bootstrap_replicates']=40
        rng=np.random.default_rng(3); n=120
        scenes=np.repeat(np.arange(6),20); x=rng.normal(size=n); y=x+.03*rng.normal(size=n)
        a=association(x,y,scenes,cfg,3)
        self.assertTrue(a['valid']); self.assertGreater(a['ci95'][0],.9)
        self.assertIsNone(rho(np.ones(n),y))
        # Weighted-multiset bootstrap must equal explicit resampling even with ties.
        tied=np.round(x); tied_y=np.round(y)
        fast=association(tied,tied_y,scenes,cfg,11)
        exact=cluster_ci(np.c_[tied,tied_y],scenes,lambda v:rho(v[:,0],v[:,1]),cfg,11)
        np.testing.assert_allclose(fast['ci95'],exact['ci95'],atol=1e-12)
        b=incremental(rng.normal(size=(n,2)),x[:,None],y,scenes,cfg,4)
        self.assertGreater(b['delta_r2']['estimate'],.5)
        # OOF outputs cannot fit a held-out scene's arbitrary label offset.
        predictions=ridge_oof(np.ones((n,1)),scenes.astype(float),scenes,10)
        self.assertAlmostEqual(predictions[0],3.)

    def test_empty_evaluation(self):
        from .evaluate import evaluate
        cfg=config(PACKAGE/'experiment.yaml'); cfg['bootstrap_replicates']=4; cfg['permutation_replicates']=2
        result=evaluate([],cfg,'O0')
        self.assertEqual(result['decision']['verdict'],'INCONCLUSIVE')

if __name__=='__main__':
    if '--require-server' in sys.argv:
        sys.argv.remove('--require-server')
        if os.name!='posix' or importlib.util.find_spec('shapely') is None:
            raise RuntimeError('Server unit tests require Linux and Shapely; no skipping matching tests')
    unittest.main(verbosity=2)
