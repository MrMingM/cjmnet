"""Synthetic records and function adapters only; no detector or real samples."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from local_fusion_action_utility_audit.reproducibility import POLICY
from local_fusion_source_kill_stage2.common import (
    PACKAGE, atomic_json, commit_unit, digest, object_hash, read_json, resume, sources)
from local_fusion_source_kill_stage2.verdicts import final_verdict
from .finalize import assert_inherited
from .reproduction import MESSAGE, adapt_evaluate_frame, adapt_run, compare_trajectory


def fixture():
    return dict(records=[dict(cls_source='single:1', reg_source='single:1', changed=True,
        focal_after=dict(score=.8, iou=.75, detected=True), shortlist=dict(best_proxy=[1, .8, .75, .7]))],
        current_state_audit=[dict(keep_key=[1,0,0,1,0,0,.4,.7,2],
            selected_key=[2,0,0,1,1,0,.8,.75,0], actions=[dict(
                key=[2,0,0,1,1,0,.8,.75,0], group='A', allowed=True, expansion=1.5)])])


def dummy_evaluate(protocol, runtime, dataset, batch, weather, index, mode, positives):
    records, audits, saved = batch
    if saved is None or records != saved['records'] or audits != saved['current_state_audit']:
        raise ValueError('REPRODUCTION_FAIL: K2 original A_ONLY current-state trajectory differs')
    return positives


def make_runtime(value):
    raise AssertionError('Original dummy runtime must not run')


def evaluate_frame(value):
    raise AssertionError('Original dummy evaluation must not run')


def dummy_run(value):
    return evaluate_frame(make_runtime(value))


def dummy_pipeline(run):
    protocol = read_json(run/'PROTOCOL.json')
    path = run/'k2/conditions/fog/same/00000000.json'
    saved_payload = resume(path, protocol['identity'])
    if saved_payload is not None:
        return saved_payload
    saved = fixture()
    actual = deepcopy(saved)
    actual['records'][0]['focal_after']['score'] += 2e-7
    payload = evaluate_frame(protocol,None,None,
        (actual['records'],actual['current_state_audit'],saved),'fog',0,'same',[{'fixture':True}])
    commit_unit(path,protocol['identity'],payload)
    return payload


class Reproduction(unittest.TestCase):
    def test_exact(self):
        result = compare_trajectory(fixture(), fixture())
        self.assertTrue(result['accepted'])
        self.assertEqual(result['legacy_numeric_policy'], POLICY)

    def test_diagnosed_scale_accepted_only_on_continuous_fields(self):
        a, b = fixture(), fixture()
        a['current_state_audit'][0]['actions'][0]['key'][6] += 1.6689300537109375e-6
        a['records'][0]['focal_after']['score'] += 2e-7
        a['records'][0]['shortlist']['best_proxy'][1] += 2e-7
        result = compare_trajectory(a,b)
        self.assertTrue(result['accepted'])
        self.assertEqual(result['differences']['total_differences'], 3)
        self.assertEqual(len(result['differences']['first_differences']), 3)

    def test_exceeds_existing_policy_rejected(self):
        a,b = fixture(),fixture()
        a['current_state_audit'][0]['actions'][0]['key'][6] += .001
        self.assertFalse(compare_trajectory(a,b)['accepted'])

    def test_discrete_identity_group_roi_candidate_changes_rejected(self):
        for field,value in (('group','B'),('allowed',False),('expansion',1.),('source','single:2')):
            with self.subTest(field=field):
                a,b = fixture(),fixture()
                a['current_state_audit'][0]['actions'][0][field] = value
                self.assertFalse(compare_trajectory(a,b)['accepted'])
        a,b = fixture(),fixture()
        a['current_state_audit'][0]['actions'][0]['key'][0] += 1
        self.assertFalse(compare_trajectory(a,b)['accepted'])
        a,b = fixture(),fixture()
        a['records'][0]['cls_source'] = 'single:2'
        self.assertFalse(compare_trajectory(a,b)['accepted'])

    def test_proxy_discrete_coordinate_strict(self):
        a,b = fixture(),fixture()
        a['records'][0]['shortlist']['best_proxy'][0] = 0
        self.assertFalse(compare_trajectory(a,b)['accepted'])

    def test_focal_detected_strict(self):
        a,b = fixture(),fixture()
        a['records'][0]['focal_after']['detected'] = False
        self.assertFalse(compare_trajectory(a,b)['accepted'])

    def test_nan_and_inf_even_when_equal_rejected(self):
        for value in (float('nan'),float('inf')):
            a,b = fixture(),fixture()
            a['current_state_audit'][0]['actions'][0]['key'][6] = value
            b['current_state_audit'][0]['actions'][0]['key'][6] = value
            with self.assertRaisesRegex(ValueError,'nonfinite'):
                compare_trajectory(a,b)

    def test_tiny_order_flip_rejected(self):
        b = fixture()
        key = b['current_state_audit'][0]['keep_key'][:]
        key[8] = 0
        key[6] += 1e-7
        b['current_state_audit'][0]['actions'][0]['key'] = key
        a = deepcopy(b)
        a['current_state_audit'][0]['actions'][0]['key'][6] -= 2e-7
        self.assertFalse(compare_trajectory(a,b)['accepted'])

    def test_breaking_exact_outcome_tie_rejected_even_keep_preference_masks_rank(self):
        # Group E->B must remain exact even when full9 order would not change.
        a,b = fixture(),fixture()
        a['current_state_audit'][0]['actions'][0]['group'] = 'B'
        b['current_state_audit'][0]['actions'][0]['group'] = 'E'
        self.assertFalse(compare_trajectory(a,b)['accepted'])

    def test_candidate_sequence_strict(self):
        a,b = fixture(),fixture()
        a['current_state_audit'][0]['actions'].append(deepcopy(a['current_state_audit'][0]['actions'][0]))
        self.assertFalse(compare_trajectory(a,b)['accepted'])


class Adapters(unittest.TestCase):
    def test_known_guard_replaced_only_in_namespace(self):
        calls=[]
        adapted = adapt_evaluate_frame(dummy_evaluate, lambda *args: calls.append(args))
        args=({},None,None,([],[],None),'fog',0,'same',['result'])
        self.assertEqual(adapted(*args), ['result'])
        self.assertEqual(len(calls),1)
        with self.assertRaisesRegex(ValueError,MESSAGE):
            dummy_evaluate(*args)

    def test_real_function_adapter_leaves_code_and_globals_unchanged(self):
        from local_fusion_source_kill_stage2.k2_label_stability import evaluate_frame
        before=digest(Path(evaluate_frame.__code__.co_filename))
        adapted = adapt_evaluate_frame(evaluate_frame, lambda *args: None)
        self.assertNotEqual(adapted.__code__,evaluate_frame.__code__)
        self.assertEqual(before,digest(Path(evaluate_frame.__code__.co_filename)))
        self.assertNotIn('_verify_replayed_trajectory',evaluate_frame.__globals__)

    def test_unknown_guard_refused(self):
        with self.assertRaisesRegex(ValueError,'exactly one'):
            adapt_evaluate_frame(fixture,lambda *args: None)

    def test_run_namespace_isolated(self):
        original_runtime = dummy_run.__globals__['make_runtime']
        original_evaluate = dummy_run.__globals__['evaluate_frame']
        adapted = adapt_run(dummy_run,lambda r:r+1,41,lambda *args: None)
        self.assertEqual(adapted({}),42)
        self.assertIs(original_runtime,dummy_run.__globals__['make_runtime'])
        self.assertIs(original_evaluate,dummy_run.__globals__['evaluate_frame'])


class Integrity(unittest.TestCase):
    def test_new_protocol_keeps_science_and_hashes_repair_inputs(self):
        from .contract import derive
        with tempfile.TemporaryDirectory() as directory:
            parent_run=Path(directory)
            parent=dict(identity='old', settings={'threshold':.6}, input_hashes={'original':'sha'},
                        source_hashes={'original-code':'sha'}, resource_stop_at='fixed')
            atomic_json(parent_run/'PROTOCOL.json',parent)
            repaired=derive(parent,parent_run,parent_run/'diagnosis',{'diagnostic':'sha'},
                            {'repair-code':'new-sha'},{'device':'fixture'})
            assert_inherited(parent,repaired)
            self.assertNotEqual(repaired['identity'],parent['identity'])
            self.assertEqual(repaired['identity'],object_hash({k:v for k,v in repaired.items() if k!='identity'}))
            self.assertEqual(repaired['input_hashes']['repair-code'],'new-sha')
            self.assertEqual(parent['input_hashes'],{'original':'sha'})

    def test_synthetic_evaluate_save_resume_proof_chain(self):
        from .driver import execute_k2
        from local_fusion_source_kill_stage2 import k2_label_stability as original
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            protocol = dict(identity='fixture-id')
            atomic_json(run/'PROTOCOL.json',protocol)
            with patch.object(original,'evaluate_frame',dummy_evaluate),patch.object(original,'run',dummy_pipeline):
                execute_k2(run,protocol,None)
                receipt = run/'k2/conditions/fog/same/00000000.json'
                self.assertEqual(resume(receipt,'fixture-id'),[{'fixture':True}])
                evidence = run/'k2/conditions/fog/same/00000000.reproduction.json'
                self.assertEqual(read_json(evidence)['differences']['total_differences'],1)
                before = digest(receipt)
                execute_k2(run,protocol,None)
                self.assertEqual(before,digest(receipt))
                evidence.write_text('{}',encoding='utf-8')
                with self.assertRaisesRegex(ValueError,'Corrupted'):
                    execute_k2(run,protocol,None)

    def test_completed_frame_protects_reproduction_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory); evidence=folder/'0.reproduction.json'
            evidence.write_text('{"accepted":true}',encoding='utf-8')
            receipt=folder/'0.json'
            commit_unit(receipt,'id',[],[evidence])
            self.assertEqual(resume(receipt,'id'),[])
            with self.assertRaises(ValueError):
                resume(receipt,'changed')
            evidence.write_text('{}',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Corrupted'):
                resume(receipt,'id')

    def test_core_source_snapshot_unchanged(self):
        self.assertEqual(read_json(PACKAGE/'source_snapshot.json')['source_hashes'],sources())

    def test_parent_science_must_remain_exact(self):
        parent=dict(identity='old',settings={'threshold':.6},input_hashes={'data':'sha'},source_hashes={'code':'sha'})
        repaired=dict(parent,identity='new',input_hashes={'data':'sha','repair':'sha'},
            k2_replay_repair={'parent_identity':'old'})
        assert_inherited(parent,repaired)
        for key,value in (('settings',{}),('source_hashes',{}),('input_hashes',{})):
            with self.subTest(key=key),self.assertRaises(ValueError):
                assert_inherited(parent,dict(repaired,**{key:value}))

    def test_final_hierarchy_unchanged(self):
        self.assertEqual(final_verdict('PASS','PASS','INCONCLUSIVE','PASS')['H1_SOURCE_SPECIFIC'],'STOP')
        self.assertEqual(final_verdict('PASS','FAIL','PASS','PASS')['H2_TASK_SPECIFIC'],'STOP')

    def test_no_model_environment_import_for_record_tests(self):
        from .contract import environment
        with patch('builtins.__import__', side_effect=AssertionError('GPU import forbidden')):
            with self.assertRaises(AssertionError):
                environment()


if __name__ == '__main__':
    unittest.main()
