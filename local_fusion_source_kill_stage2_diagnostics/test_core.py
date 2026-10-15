"""Only fabricated records and original AST fixtures; no scientific data."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from local_fusion_source_kill_stage2.common import (KEEP,ROOT,PACKAGE,commit_unit,read_json,sources)
from local_fusion_source_kill_stage2.fixtures import Dataset,Tensor,oracle_fixture
from local_fusion_source_kill_stage2.source_adapter import evaluate_target,run_with_hooks
from local_fusion_source_kill_stage1.oracle_adapter import run_restricted
from .differences import exact_differences,render_summary
from .replay import select_frame,trajectory
from .inspect import continuous_path,summarize_pair


class Differences(unittest.TestCase):
    def test_identical_native_json_records(self):
        value={'records':[{'focal_before':{'score':.1},'changed':False}],
               'current_state_audit':[{'selected_key':[1,0,0,1,0,0,.1,.9,2]}]}
        self.assertTrue(exact_differences(value,json.loads(json.dumps(value)))['exact_equal'])
    def test_small_float_difference_is_not_ignored(self):
        a={'current_state_audit':[{'selected_key':[1,0,0,1,0,0,.10000001,.9,2]}]}
        b={'current_state_audit':[{'selected_key':[1,0,0,1,0,0,.1,.9,2]}]}
        result=exact_differences(a,b)
        self.assertFalse(result['exact_equal'])
        first=result['first_differences'][0]
        self.assertEqual(first['category'],'FOCAL_SCORE_OR_IOU')
        self.assertGreater(first['absolute_difference'],0)
    def test_discrete_key_difference_identified(self):
        result=exact_differences({'current_state_audit':[{'keep_key':[1,0,0,1,0,0,.1,.9,2]}]},
                                 {'current_state_audit':[{'keep_key':[0,0,0,1,0,0,.1,.9,2]}]})
        self.assertEqual(result['categories'],{'DISCRETE_KEY':1})
    def test_selected_action_difference(self):
        result=exact_differences({'records':[{'cls_source':'single:1'}]},{'records':[{'cls_source':'single:2'}]})
        self.assertEqual(result['first_differences'][0]['category'],'SELECTED_ACTION')
    def test_detected_flip_is_discrete_not_continuous(self):
        result=exact_differences({'records':[{'focal_after':{'detected':True}}]},
                                 {'records':[{'focal_after':{'detected':False}}]})
        self.assertEqual(result['first_differences'][0]['category'],'DETECTION_STATE')
    def test_structure_mismatch_and_limit(self):
        result=exact_differences({'a':list(range(20))},{'a':list(range(1,21)),'b':None},limit=2)
        self.assertEqual(result['total_differences'],21)
        self.assertEqual(len(result['first_differences']),2)
    def test_tuple_list_mismatch_is_reported_not_accepted(self):
        result=exact_differences({'mask_stats':{'shape':(2,3)}},{'mask_stats':{'shape':[2,3]}})
        self.assertFalse(result['exact_equal'])
        self.assertEqual(result['first_differences'][0]['category'],'ROI_MASK_METADATA')
    def test_missing_none_key_still_detected(self):
        self.assertFalse(exact_differences({}, {'a':None})['exact_equal'])
    def test_integer_float_equality_matches_python_gate(self):
        self.assertTrue(exact_differences({'x':1},{'x':1.})['exact_equal'])
    def test_summary_stays_diagnostic(self):
        report=dict(conclusion='fixture',weather='fog',mode='same',frame=1,
            environment={'ROCR_VISIBLE_DEVICES':'3'},comparisons={'a':exact_differences({'x':1},{'x':2})})
        result=render_summary(report)
        self.assertIn('未修改协议',result)
        self.assertIn('trajectory.x',result)
    def test_continuous_whitelist_rejects_discrete_and_roi(self):
        self.assertTrue(continuous_path('trajectory.current_state_audit[0].actions[0].key[6]'))
        self.assertFalse(continuous_path('trajectory.current_state_audit[0].actions[0].key[3]'))
        self.assertFalse(continuous_path('trajectory.records[0].focal_after.detected'))
        self.assertFalse(continuous_path('trajectory.records[0].shortlist.best_proxy[0]'))
        self.assertFalse(continuous_path('trajectory.records[0].expansion'))
    def test_continuous_drift_reported_without_automatic_acceptance(self):
        original={'records':[],'current_state_audit':[{'keep_key':[1,0,0,1,0,0,.1,.7,2],'actions':[]}]}
        actual=json.loads(json.dumps(original)); actual['current_state_audit'][0]['keep_key'][6]+=.0000002
        result=summarize_pair(actual,original)
        self.assertTrue(result['only_named_continuous_fields_differ'])
        self.assertEqual(result['full9_order_changes'],[])
        self.assertNotIn('verdict',result)
    def test_tiny_float_can_change_ranking(self):
        key=[1,0,0,1,0,0,.1,.7,2]
        original={'records':[],'current_state_audit':[{'keep_key':key,'actions':[{'key':[1,0,0,1,0,0,.10000001,.7,1]}]}]}
        actual=json.loads(json.dumps(original)); actual['current_state_audit'][0]['actions'][0]['key'][6]=.09999999
        result=summarize_pair(actual,original)
        self.assertTrue(result['only_named_continuous_fields_differ'])
        self.assertEqual(len(result['full9_order_changes']),1)
    def test_ineligible_ranking_change_is_separate(self):
        key=[1,0,0,1,0,0,.1,.7,2]
        original={'records':[],'current_state_audit':[{'keep_key':key,'actions':[{'key':[1,0,0,1,0,0,.10000001,.7,1],'allowed':False}]}]}
        actual=json.loads(json.dumps(original)); actual['current_state_audit'][0]['actions'][0]['key'][6]=.09999999
        result=summarize_pair(actual,original)
        self.assertEqual(len(result['full9_order_changes']),1)
        self.assertEqual(result['a_only_legal_order_changes'],[])


class ReplayFixtures(unittest.TestCase):
    def test_original_adapter_and_hook_with_callback_match_saved_json(self):
        b=oracle_fixture(); ds=Dataset()
        shared={'psm':Tensor(.1),'rm':Tensor(.1)}
        pool={KEEP:shared,'single:0':{'psm':Tensor(.8),'rm':Tensor(.8)}}
        proxy={'single:0':b._proxy_cache(ds,{},pool['single:0'])}
        args=(ds,{'ego':{'anchor_box':None}},None,[np.zeros((1,1,1,8))],shared,pool,proxy,None,(1.,1.5),'same',.7,6)
        _,_,records,audits=run_restricted(b,'A_ONLY',*args)
        expected=json.loads(json.dumps(trajectory(records,audits)))
        calls=[]
        def callback(target,current,*unused):
            evaluation=evaluate_target(b,ds,{},current,pool,proxy,np.ones((8,3)),
                {x:(np.ones((1,8),dtype=bool),{}) for x in (1.,1.5)},(1.,1.5),'same',target)
            calls.append(evaluation)
        _,_,records,audits=run_with_hooks(b,'A_ONLY',args,callback)
        self.assertTrue(exact_differences(trajectory(records,audits),expected)['exact_equal'])
        self.assertEqual(len(calls),2)
        self.assertEqual(shared['psm'].value,.1)
    def test_existing_stage2_source_snapshot_still_matches(self):
        self.assertEqual(read_json(PACKAGE/'source_snapshot.json')['source_hashes'],sources())
    def test_first_pending_frame_follows_original_loader_order(self):
        (ROOT/'.tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp',prefix='k2_diagnostic_fixture_') as value:
            run=Path(value)
            row=dict(weather='fog',frame=20,target_index=1)
            targets={'same':[row,{**row,'frame':10}],'task':[]}
            protocol={'p0_run':str(run),'identity':'fixture','validation_indices':[20,10]}
            with patch('local_fusion_source_kill_stage2_diagnostics.replay.baseline_rows',return_value=targets):
                self.assertEqual(select_frame(run,protocol)[2],20)
                commit_unit(run/'k2/conditions/fog/same/00000020.json','fixture',[])
                self.assertEqual(select_frame(run,protocol)[2],10)


if __name__=='__main__':
    unittest.main()
