"""Synthetic tests only. No experiment, real dataset, checkpoint or network."""
import ast
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
from .common import (ROOT, KEEP, atomic_json, commit_unit,
                     read_json, resource_check, resume, safe_path, settings)
from .bootstrap import (ap70, interval, k1_bootstrap, metric_bootstrap, ordered_resample, scene_draws)
from .verdicts import (final_verdict, k1_family, k2_family, sample_gate)
from .nulls import (alpha_distance, assert_fair, delta_vector, dirichlet_weights,
                    rng_for, sample_vectors)
from .stability import (action_id, aggregate, compare_labels, label_summary, perturb_corners)
from .source_adapter import run_with_hooks
from .fixtures import Dataset, Tensor, ap_fixture, oracle_fixture


def action(cls, reg, discrete=1, score=.5, expansion=1.):
    keep = cls == reg == KEEP
    return dict(cls_source=cls, reg_source=reg, expansion=None if keep else expansion,
        key=[discrete,0,0,int(discrete>0),discrete,0,score,.8,2 if keep else 1 if cls==reg else 0],
        group='KEEP' if keep else 'A' if discrete else 'B')


def evaluation(rows):
    keep = action(KEEP,KEEP,0,0)
    return {'actions':[keep]+rows,'keep_key':keep['key']}


class Rules(unittest.TestCase):
    def test_k1_ratio_boundary(self):
        self.assertEqual(k1_family(5,3,dict(lower=.1,upper=4))['verdict'],'FAIL')
    def test_k1_gap_boundary(self):
        self.assertEqual(k1_family(1,.51,dict(lower=.1,upper=.9))['verdict'],'FAIL')
    def test_k1_ci_zero_boundary(self):
        self.assertEqual(k1_family(5,0,dict(lower=0,upper=6))['verdict'],'FAIL')
    def test_k1_pass(self):
        self.assertEqual(k1_family(5,1,dict(lower=1,upper=7))['verdict'],'PASS')
    def test_k1_nonpositive_inconclusive(self):
        self.assertEqual(k1_family(0,0,dict(lower=1,upper=2))['verdict'],'INCONCLUSIVE')
    def test_k1_unfair_inconclusive(self):
        self.assertEqual(k1_family(5,0,dict(lower=1,upper=2),valid=False)['verdict'],'INCONCLUSIVE')
    def test_k1_missing_bootstrap(self):
        self.assertEqual(k1_family(5,0,None)['verdict'],'INCONCLUSIVE')
    def test_h1_stop(self):
        self.assertEqual(final_verdict('PASS','PASS','INCONCLUSIVE','PASS')['H1_SOURCE_SPECIFIC'],'STOP')
    def test_h2_conditional_stop(self):
        self.assertEqual(final_verdict('PASS','FAIL','PASS','PASS')['decision'],'TASK SEPARATION STOP')
    def test_h2_not_applicable(self):
        self.assertEqual(final_verdict('FAIL','PASS','PASS','PASS')['H2_TASK_SPECIFIC'],'NOT_APPLICABLE')
    def test_all_pass_keeps_k3_frozen(self):
        self.assertTrue(final_verdict('PASS','PASS','PASS','PASS')['k3_frozen'])
    def test_same_sample_threshold(self):
        rows = [dict(scene_id=str(i%7)) for i in range(200)]
        self.assertEqual(sample_gate(rows,'same',False)['verdict'],'PASS')
        self.assertEqual(sample_gate(rows[:-1],'same',False)['verdict'],'INCONCLUSIVE')
    def test_task_sample_threshold(self):
        self.assertEqual(sample_gate([dict(scene_id=str(i%5)) for i in range(50)],'task',False)['verdict'],'PASS')
    def test_sample_concentration(self):
        rows = [dict(scene_id='a')]*101 + [dict(scene_id=str(i%9)) for i in range(99)]
        self.assertEqual(sample_gate(rows,'same',False)['verdict'],'INCONCLUSIVE')
    def test_provenance_count_mismatch(self):
        with self.assertRaisesRegex(ValueError,'provenance'):
            sample_gate([dict(scene_id='a')],'same')
    def test_same_frozen_thresholds(self):
        m = dict(keep_agreement=.9,source_retention=.7,membership_agreement=.8)
        ci = dict(keep_agreement=dict(lower=.8),source_retention=dict(lower=.6))
        self.assertEqual(k2_family('same',m,ci)['verdict'],'PASS')
        m['membership_agreement'] = .79999
        self.assertEqual(k2_family('same',m,ci)['verdict'],'FAIL')
    def test_task_frozen_thresholds(self):
        self.assertEqual(k2_family('task',dict(positive_retention=.7,pair_retention=.6),
            dict(positive_retention=dict(lower=.5)))['verdict'],'PASS')
    def test_task_bootstrap_failure(self):
        self.assertEqual(k2_family('task',dict(positive_retention=.9,pair_retention=.9),
            dict(positive_retention=dict(lower=.499)))['verdict'],'FAIL')


class Nulls(unittest.TestCase):
    def test_counts_for_variable_cavs(self):
        for count in (1,2,5,8):
            weights = dirichlet_weights(count,2*count,.2,np.random.default_rng(3))
            self.assertEqual(weights.shape,(2*count,count))
            np.testing.assert_allclose(weights.sum(1),1)
    def test_unfair_count_rejected(self):
        with self.assertRaises(ValueError):
            dirichlet_weights(5,6,.2,np.random.default_rng(0))
    def test_same_keep_required(self):
        shared = object()
        assert_fair({KEEP:shared,'single:0':0},{KEEP:shared,'null:0':1})
        with self.assertRaises(ValueError):
            assert_fair({KEEP:shared,'x':0},{KEEP:object(),'n':1})
    def test_rng_reproducible(self):
        np.testing.assert_equal(rng_for(1,'fog',2,'N2').normal(size=10),rng_for(1,'fog',2,'N2').normal(size=10))
    def test_rng_seed_and_weather_independent(self):
        self.assertNotEqual(rng_for(1,'fog',2,'N2').random(),rng_for(2,'fog',2,'N2').random())
    def test_whole_vector_sampling(self):
        v = np.column_stack([np.arange(300)*k for k in range(1,9)])
        pool = {'fog/single':v,'fog/query':v}
        selected,label = sample_vectors(pool,'fog','single',500,np.random.default_rng(3))
        for k in range(8):
            np.testing.assert_equal(selected[:,k],selected[:,0]*(k+1))
        self.assertEqual(label,'fog/single')
    def test_pool_fallback(self):
        v = np.ones((300,8))
        selected,label = sample_vectors({'fog/single':v,'fog/query':v},'snow','single',3,np.random.default_rng(1))
        self.assertEqual(label,'single')
    def test_pool_global_fallback(self):
        v=np.ones((300,8))
        _,label=sample_vectors({'fog/query':v},'snow','single',3,np.random.default_rng(1))
        self.assertEqual(label,'global')
    def test_delta_order_and_wrapped_yaw(self):
        a = np.zeros((1,7)); b = a.copy(); a[0,6] = math.radians(359)
        vector = delta_vector([.2],a,[.1],b)
        self.assertAlmostEqual(vector[0,7],math.radians(-1))
        self.assertAlmostEqual(vector[0,0],.1)
    def test_alpha_metric_zero_safe(self):
        metrics = {k:dict(median=0,p90=0) for k in ('abs_score','center_shift','size_change','yaw_change','candidate_iou_change')}
        self.assertEqual(alpha_distance(metrics,metrics),0)
    def test_alpha_grid_fixed(self):
        self.assertEqual(settings()['calibration']['alpha_grid'],[.05,.1,.2,.5,1.,2.,5.])
    def test_train_only_calibration_boundary(self):
        tree = ast.parse((Path(__file__).parent/'calibration.py').read_text(encoding='utf-8'))
        calls = [n for n in ast.walk(tree) if isinstance(n,ast.Call)]
        forbidden = ('post_process','caluclate_tp_fp','_state','_action_key','run_oracle')
        self.assertFalse(any(isinstance(n.func,ast.Attribute) and n.func.attr in forbidden for n in calls))


class Stability(unittest.TestCase):
    def test_equivalent_cross_source_retention(self):
        base = label_summary(evaluation([action('single:1','single:1'),action('query:2','query:2')]))
        altered = label_summary(evaluation([action('query:2','query:2',score=.8)]))
        result = compare_labels(base,altered)
        self.assertTrue(result['source_retention'])
        self.assertFalse(result['exact_source_stable'])
    def test_task_pair_set_intersection(self):
        base = label_summary(evaluation([action('single:1','query:2',2),action('query:3','single:4',2)]))
        changed = label_summary(evaluation([action('query:3','single:4',2)]))
        self.assertTrue(compare_labels(base,changed)['pair_retention'])
    def test_task_positive_not_source_disagreement(self):
        x = label_summary(evaluation([action('single:1','query:2',1),action('single:1','single:1',1)]))
        self.assertFalse(x['task_positive'])
    def test_task_margin_loss_separate_from_literal_same_choice(self):
        baseline=label_summary(evaluation([action('single:1','query:2',2)]))
        changed=label_summary(evaluation([action('single:1','query:2',1,score=.9),action('single:1','single:1',1,score=.2)]))
        metrics=compare_labels(baseline,changed)
        self.assertTrue(metrics['task_positive_lost'])
        self.assertFalse(metrics['task_to_same_flip'])
    def test_task_positive_real_discrete_margin(self):
        x = label_summary(evaluation([action('single:1','query:2',2),action('single:1','single:1',1)]))
        self.assertTrue(x['task_positive'])
    def test_keep_only(self):
        result = compare_labels(label_summary(evaluation([])),label_summary(evaluation([])))
        self.assertTrue(result['keep_agreement'])
        self.assertFalse(result['source_retention'])
    def test_membership_enter_leave_disagreement(self):
        x = label_summary(evaluation([action('single:0','single:0')]))
        y = label_summary(evaluation([action('single:1','single:1')]))
        self.assertEqual(compare_labels(x,y)['membership_agreement'],0)
    def test_center_variants(self):
        original = np.array([[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]],dtype=float)
        for name,value in settings()['k2']['perturbations'].items():
            changed,scale = perturb_corners(original,value)
            np.testing.assert_allclose(changed.mean(0)[:2],value[1:3],atol=1e-7)
        np.testing.assert_equal(original[0],[-1,-1,0])
    def test_yaw_two_degrees_radians(self):
        x = np.array([[1.,0,0],[-1.,0,0]])
        y,_ = perturb_corners(x,['yaw',0,0,2,1])
        self.assertAlmostEqual(float(y[0,1]),math.sin(math.radians(2)),places=7)
    def test_roi_multiplier_exact(self):
        scale = settings()['k2']['perturbations']['roi_minus'][-1]
        self.assertAlmostEqual(1.5*scale,1.455)
        self.assertAlmostEqual(1.5*1.03,1.545)
        self.assertEqual(action_id(action('single:0','single:0',expansion=1.455),.97)[0],1.5)
    def test_actual_roi_shrink_mask_without_historical_guard_change(self):
        from .source_adapter import masks_at_multiplier
        from qa_local_intervention.operators import masks_for_frame
        class Anchors:
            def __init__(self):
                self.value=np.zeros((3,3,1,7))
                x,y=np.meshgrid([-.98,0,.98],[-.98,0,.98])
                self.value[:,:,0,:2]=np.stack((x,y),axis=-1)
            def detach(self): return self
            def cpu(self): return self
            def numpy(self): return self.value
        a=Anchors()
        c=np.array([[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],dtype=float)
        original=c.copy()
        levels=[np.zeros((1,1,3,3))]
        lidar_range=[-1.47,-1.47,-1,1.47,1.47,1]
        with self.assertRaises(ValueError):
            masks_for_frame(a,c,levels,lidar_range,.97)
        baseline,_,_=masks_at_multiplier(a,c,levels,lidar_range,1.)
        shrunk,context,stats=masks_at_multiplier(a,c,levels,lidar_range,.97)
        self.assertEqual(int(baseline.sum()),9)
        self.assertEqual(int(shrunk.sum()),1)
        self.assertEqual(int(context[0].sum()),1)
        self.assertEqual(stats['effective_roi_multiplier'],.97)
        np.testing.assert_equal(original,c)
    def test_eight_variants_three_families(self):
        p = settings()['k2']['perturbations']
        self.assertEqual(len(p),8)
        self.assertEqual([sum(v[0]==f for v in p.values()) for f in ('center','yaw','roi')],[4,2,2])
    def test_family_macro_and_worst(self):
        rows = []
        base = label_summary(evaluation([action('single:0','single:0')]))
        metric = compare_labels(base,base)
        for variant in settings()['k2']['perturbations']:
            for scene in ('a','b'):
                rows.append(dict(scene_id=scene,variant=variant,**metric))
        result,_ = aggregate(rows,'same',repetitions=10)
        self.assertEqual(result['center']['verdict'],'PASS')
        self.assertEqual(result['center']['worst_variant']['keep_agreement'],1)


class Bootstrap(unittest.TestCase):
    def test_scene_draw_determinism(self):
        self.assertEqual(list(scene_draws(['a','b'],5,1)),list(scene_draws(['b','a'],5,1)))
    def test_scene_not_target_iid(self):
        rows = [dict(scene_id='a',variant='v',m=1)]*20+[dict(scene_id='b',variant='v',m=0)]*20
        result = metric_bootstrap(rows,'m',['v'],repetitions=50,seed=1)
        self.assertEqual(set(result['distribution']),{0.,.5,1.})
    def test_scene_multiplicity_preserved(self):
        rows = [dict(scene_id='a',i=1),dict(scene_id='a',i=2),dict(scene_id='b',i=3)]
        self.assertEqual([r['i'] for r in ordered_resample(rows,['a','a'])],[1,2,1,2])
    def test_undefined_ci_rejected(self):
        with self.assertRaises(ValueError):
            interval([float('nan')])
    def test_ap_recompute_not_mean_scene_ap(self):
        rows = [dict(scene_id='a',stats={'0.7':dict(tp=[0],fp=[1],score=[.9],gt=1)}),
                dict(scene_id='b',stats={'0.7':dict(tp=[1],fp=[0],score=[.8],gt=1)})]
        self.assertAlmostEqual(ap70(rows,ap_fixture),.25)
    def test_paired_weather_scene_bootstrap(self):
        rows = [dict(sample_index=i,scene_id=str(i),stats={'0.7':dict(tp=[1],fp=[0],score=[.9],gt=1)}) for i in (0,1)]
        table = {w:{m:rows for m in ('shared','same','task')} for w in ('fog','rain','snow')}
        result = k1_bootstrap(table,[table,table,table],ap_fixture,repetitions=10)
        self.assertEqual(result['same']['lower'],0)
    def test_unpaired_bootstrap_rejected(self):
        rows = [dict(sample_index=i,scene_id=str(i),stats={'0.7':dict(tp=[1],fp=[0],score=[.9],gt=1)}) for i in (0,1)]
        table = {w:{m:rows for m in ('shared','same','task')} for w in ('fog','rain','snow')}
        wrong = {w:{m:list(reversed(rows)) for m in ('shared','same','task')} for w in table}
        with self.assertRaisesRegex(ValueError,'Unpaired'):
            k1_bootstrap(table,[wrong],ap_fixture,repetitions=1)


class OracleAndIO(unittest.TestCase):
    def setUp(self):
        (ROOT/'.tmp').mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT/'.tmp',prefix='stage2_unit_')
        self.folder = Path(self.temp.name)
    def tearDown(self):
        self.temp.cleanup()
    def test_atomic_resume_and_drift(self):
        p = self.folder/'r.json'
        commit_unit(p,'identity',{'value':1})
        self.assertEqual(resume(p,'identity'),{'value':1})
        with self.assertRaises(ValueError):
            resume(p,'changed')
    def test_corrupt_result_rejected(self):
        p = self.folder/'r.json'
        commit_unit(p,'identity',{'value':1})
        x = read_json(p); x['payload']['value']=2; atomic_json(p,x)
        with self.assertRaises(ValueError):
            resume(p,'identity')
    def test_artifact_hash_mismatch(self):
        artifact = self.folder/'f.json'; atomic_json(artifact,{'x':1})
        p = self.folder/'r.json'; commit_unit(p,'i',{},[artifact]); atomic_json(artifact,{'x':2})
        with self.assertRaises(ValueError):
            resume(p,'i')
    def test_formal_test_paths_rejected(self):
        for path in ('/data/scd/datasets/opv2v_official_data_dumping/test',
                     '/data/cjm/datasets/opv2v-w/fog/test/nested/a.pcd',
                     '/data/cjm/datasets/opv2v-w/snow/test'):
            with self.assertRaises(ValueError):
                safe_path(path)
    @unittest.skipUnless(os.name=='posix','Symlink guard verified on server POSIX')
    def test_symlink_test_path(self):
        target = self.folder/'test'; target.mkdir(); link=self.folder/'alias'; link.symlink_to(target,target_is_directory=True)
        with self.assertRaises(ValueError):
            safe_path(link/'nested')
    def test_resource_deadline(self):
        p = settings()
        resource_check(p,datetime(2026,10,14,tzinfo=timezone.utc))
        with self.assertRaisesRegex(RuntimeError,'RESOURCE_STOP'):
            resource_check(p,datetime(2026,10,27,3,23,57,tzinfo=timezone.utc))
    def test_original_task_shortlist_not_fixed_six(self):
        b=oracle_fixture(); ds=Dataset(); current=dict(psm=Tensor(.1),rm=Tensor(.1))
        pool={KEEP:current,**{f'single:{i}':dict(psm=Tensor(.8),rm=Tensor(.8)) for i in range(5)}}
        proxy={k:b._proxy_cache(ds,{},v) for k,v in pool.items() if k!=KEEP}
        args=(b._proxy_cache(ds,{},current),proxy,np.ones((8,3)),{x:(np.ones((1,8),dtype=bool),{}) for x in (1.,1.5)},tuple(sorted(pool)),(1.,1.5))
        task,info=b._shortlist_actions(*args,'task',.7,6)
        same,_=b._shortlist_actions(*args,'same',.7,6)
        self.assertGreater(len(task),6)
        self.assertEqual(len(same),6)
    def test_original_full_and_adapter_equal(self):
        b=oracle_fixture(); ds=Dataset(); shared=dict(psm=Tensor(.1),rm=Tensor(.1))
        pool={KEEP:shared,'single:0':dict(psm=Tensor(.8),rm=Tensor(.8))}
        proxy={'single:0':b._proxy_cache(ds,{},pool['single:0'])}
        args=(ds,{'ego':{'anchor_box':None}},None,[np.zeros((1,1,1,8))],shared,pool,proxy,None,(1.,1.5),'same',.7,6)
        _,_,old=b.run_oracle(*args)
        _,_,new,audit=run_with_hooks(b,'FULL',args)
        self.assertEqual(old,new)
        self.assertEqual(len(audit),len(old))
        self.assertEqual(shared['psm'].value,.1)
        from .source_adapter import assert_shortlist_contract
        assert_shortlist_contract(new,audit,len(pool),(1.,1.5),'same',6)
        with self.assertRaises(ValueError):
            assert_shortlist_contract(new,audit,len(pool)+1,(1.,1.5),'same',6)
    def test_a_only_state_replay_independence(self):
        b=oracle_fixture(); ds=Dataset(); shared=dict(psm=Tensor(.1),rm=Tensor(.1))
        pool={KEEP:shared,'single:0':dict(psm=Tensor(.8),rm=Tensor(.8))}
        proxy={'single:0':b._proxy_cache(ds,{},pool['single:0'])}
        args=(ds,{'ego':{'anchor_box':None}},None,[np.zeros((1,1,1,8))],shared,pool,proxy,None,(1.,1.5),'same',.7,6)
        states=[]
        first=run_with_hooks(b,'A_ONLY',args,lambda target,current,*rest:states.append(current['psm'].value))
        second=run_with_hooks(b,'A_ONLY',args)
        self.assertEqual(states,[.1,.8])
        self.assertEqual(first[2],second[2])
        self.assertEqual(shared['psm'].value,.1)
    def test_p0_primary_schema_is_a_only_adverse_validation(self):
        from .preflight import baseline_rows
        p=self.folder/'p0.jsonl'
        base=dict(split='validation',weather='fog',variant='A_ONLY',frame=1,mode='same',target_index=0,
                  selected_discrete_positive=True,true_task_improvement=False)
        values=[base,{**base,'weather':'clean'}, {**base,'split':'train'}, {**base,'variant':'FULL'},
                {**base,'mode':'task','true_task_improvement':True}]
        p.write_text(''.join(json.dumps(x)+'\n' for x in values),encoding='utf-8')
        self.assertEqual({k:len(v) for k,v in baseline_rows(p).items()},{'same':1,'task':1})
    def test_k1_report_fixture(self):
        from unittest.mock import patch
        from .common import FAMILIES,WEATHERS
        from .k1_source_specificity import condition_path
        from .reports import k1_report
        from .nulls import MAGNITUDES
        spec=settings(); spec['bootstrap']['repetitions']=10
        conditions={}
        def frame(n,index):
            return dict(sample_index=index,scene_id=str(index),stats={'0.7':dict(tp=[1]*n,fp=[0]*n,score=[.9]*n,gt=3)})
        for weather in WEATHERS:
            for family in ('Source',)+FAMILIES:
                seeds=[0] if family=='Source' else spec['seeds'][:1] if weather=='clean' else spec['seeds']
                for seed in seeds:
                    for mode in ('same','task'):
                        count=2 if mode=='same' else 3
                        if family!='Source': count=1
                        path=condition_path(self.folder,weather,family,seed,mode)
                        path.parent.mkdir(parents=True,exist_ok=True)
                        raw=path.with_suffix('.jsonl')
                        raw.write_text(''.join(json.dumps(dict(frame=frame(count,i),shared=frame(1,i),
                            shared_prediction_sha256='fixture',model_input_sha256='fixture',source_count=2))+'\n' for i in (0,1)),encoding='utf-8')
                        ap={k:count/3 for k in ('ap30','ap50','ap70')}
                        conditions[f'{weather}/{family}/{seed}/{mode}']=dict(weather=weather,family=family,seed=seed,mode=mode,
                            AP=ap,Shared={k:1/3 for k in ap},stats_file=raw.name,
                            output_magnitude={k:dict(median=.1,p90=.2) for k in MAGNITUDES})
        with patch('local_fusion_source_kill_stage2.reports.settings',return_value=spec):
            result=k1_report(self.folder,{'identity':'fixture'},conditions,ap_fixture)
        self.assertEqual(result['verdict'],{'SAME':'PASS','TASK':'PASS'})
        self.assertEqual(len(result['families']),3)
    def test_k2_evaluation_current_state_not_s0(self):
        tree=ast.parse((Path(__file__).parent/'k2_label_stability.py').read_text(encoding='utf-8'))
        self.assertFalse(any(isinstance(n,ast.Attribute) and n.attr=='load_cache' for n in ast.walk(tree)))
        self.assertIn('evaluate_target',(Path(__file__).parent/'k2_label_stability.py').read_text(encoding='utf-8'))
    def test_input_fingerprint_excludes_gt(self):
        from .source_adapter import inference_input_hash
        a={'processed_lidar':{'voxels':np.ones((2,3),dtype=np.float32)},'record_len':[2], 'gt_boxes':[1]}
        b={**a,'gt_boxes':[99]}
        self.assertEqual(inference_input_hash(a),inference_input_hash(b))
        b['record_len']=[3]
        self.assertNotEqual(inference_input_hash(a),inference_input_hash(b))
    def test_synthetic_smoke(self):
        from .smoke import smoke
        self.assertEqual(smoke(self.folder)['status'],'complete')


@unittest.skipUnless(os.name=='posix','Synthetic Torch representation fixtures run on server only')
class ServerRepresentation(unittest.TestCase):
    def test_actual_decoder_roundtrip_and_score_only_geometry(self):
        import torch
        from types import SimpleNamespace
        from .nulls import decode,encode,empirical_predictions
        tree=ast.parse((ROOT/'opencood/data_utils/post_processor/voxel_postprocessor.py').read_text(encoding='utf-8'))
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='VoxelPostprocessor')
        decoder=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='delta_to_boxes3d')
        decoder.decorator_list=[]
        scope={'torch':torch}
        exec(compile(ast.Module(body=[decoder],type_ignores=[]),'<actual decoder fixture>','exec'),scope)
        ds=SimpleNamespace(post_processor=SimpleNamespace(delta_to_boxes3d=scope['delta_to_boxes3d']))
        anchors=torch.zeros(2,3,2,7)
        anchors[...,3:6]=torch.tensor([1.5,1.8,4.])
        anchors[...,0]=torch.arange(6).reshape(2,3,1)
        torch.manual_seed(42)
        shared={'psm':torch.randn(1,2,2,3)*.1,'rm':torch.randn(1,14,2,3)*.01}
        batch={'ego':{'anchor_box':anchors}}
        s,box=decode(ds,batch,shared)
        result=encode(ds,batch,shared,s,box)
        self.assertTrue(torch.allclose(result['rm'],shared['rm'],atol=1e-6))
        self.assertTrue(torch.allclose(result['psm'],shared['psm'],atol=1e-6))
        vector=np.tile([.01,.1,.2,0,.05,.05,.05,.02],(300,1)).astype(np.float32)
        pools={'fog/single':vector,'fog/query':vector}
        pool,_=empirical_predictions(ds,batch,shared,['single:0','query:0'],pools,'fog',np.random.default_rng(4),True)
        self.assertEqual(len(pool),3)
        self.assertTrue(all(torch.equal(v['rm'],shared['rm']) for v in pool.values()))
        altered,_=empirical_predictions(ds,batch,shared,['single:0','query:0'],pools,'fog',np.random.default_rng(4),False)
        _,changed=decode(ds,batch,altered['empirical:000'])
        self.assertTrue(torch.allclose(changed-box,torch.tensor(vector[0,1:]),atol=1e-5))
    def test_n2_uses_exact_valid_source_count_and_frozen_heads(self):
        import torch
        from types import SimpleNamespace
        from .nulls import mix_predictions
        b=SimpleNamespace(torch=torch,predict_from_levels=lambda base,fused:dict(
            psm=fused[0].mean(1,keepdim=True),rm=fused[0].repeat(1,7,1,1)))
        shared={'psm':torch.zeros(1,1,2,3),'rm':torch.zeros(1,14,2,3)}
        for n in (1,2,5):
            levels=[torch.arange(n,dtype=torch.float32)[:,None,None,None].expand(n,2,2,3)]
            pool=mix_predictions(b,None,levels,shared,.2,np.random.default_rng(3))
            self.assertEqual(len(pool),2*n+1)
            self.assertIs(pool[KEEP],shared)
            self.assertTrue(all(0<=float(v['psm'].max())<=n-1 for k,v in pool.items() if k!=KEEP))


if __name__ == '__main__':
    unittest.main()
