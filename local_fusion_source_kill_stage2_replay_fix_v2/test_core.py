"""Only invented trajectories, temporary receipts and function adapters."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from local_fusion_source_kill_stage2.common import (
    KEEP, PACKAGE, WEATHERS, atomic_json, commit_unit, digest, object_hash, read_json, resume, sources)
from .reproduction import compare_trajectory, original_compare
from .contract import verify_source_gate


def fixture():
    keep=[13,0,0,0,0,-1,.9784563779830933,0.,2]
    names=['single:0','single:0','query:1','query:1','query:0','query:0']
    actions=[dict(cls_source=n,reg_source=n,group='E',allowed=False,key=keep[:8]+[1]) for n in names]
    selected=dict(target_index=11,cls_source=KEEP,reg_source=KEEP,changed=False,expansion=None,
                  focal_after=dict(score=keep[6],iou=0.,detected=False))
    return dict(records=[selected], current_state_audit=[dict(target_index=11,keep_key=keep,
        selected_key=keep[:],selected_group='KEEP',shortlisted={'E':6},actions=actions)])


def swapped():
    historical=fixture(); actual=deepcopy(historical)
    actions=actual['current_state_audit'][0]['actions']
    actions[2],actions[5]=actions[5],actions[2]
    return actual,historical


def dummy_evaluate(protocol, runtime, dataset, batch, weather, index, mode, positives):
    records,audits,saved=batch
    if saved is None or records != saved['records'] or audits != saved['current_state_audit']:
        raise ValueError('REPRODUCTION_FAIL: K2 original A_ONLY current-state trajectory differs')
    return positives


def evaluate_frame(*args):
    raise AssertionError('Original dummy evaluator must be replaced in namespace')


def dummy_run(run):
    protocol=read_json(run/'PROTOCOL.json')
    receipt=run/'k2/conditions/fog/same/00000608.json'
    saved=resume(receipt,protocol['identity'])
    if saved is not None:
        return saved
    actual,historical=swapped()
    rows=evaluate_frame(protocol,None,None,
        (actual['records'],actual['current_state_audit'],historical),'fog',608,'same',[{'synthetic':True}])
    commit_unit(receipt,protocol['identity'],rows)
    return rows


class NarrowException(unittest.TestCase):
    def test_exact_original_path(self):
        self.assertTrue(compare_trajectory(fixture(),fixture())['accepted'])

    def test_reported_E_permutation_accepted_with_proof_and_raw_differences(self):
        actual,historical=swapped(); before=deepcopy(actual)
        self.assertFalse(original_compare(actual,historical)['accepted'])
        result=compare_trajectory(actual,historical)
        self.assertTrue(result['accepted'])
        self.assertEqual(result['differences']['total_differences'],4)
        self.assertEqual(result['permitted_unselectable_E_permutations'][0]['candidate_count'],6)
        self.assertEqual(result['permitted_unselectable_E_permutations'][0]['target_index'],11)
        self.assertEqual(actual,before)

    def test_existing_numeric_noise_policy_unchanged(self):
        actual,historical=swapped()
        audit=actual['current_state_audit'][0]
        audit['keep_key'][6]+=1.1920928955078125e-7
        audit['selected_key'][6]=audit['keep_key'][6]
        for row in audit['actions']:
            row['key'][6]=audit['keep_key'][6]
        self.assertTrue(compare_trajectory(actual,historical)['accepted'])

    def test_A_B_or_allowed_candidate_blocks_exception(self):
        for group,allowed in (('A',True),('B',False),('E',True)):
            with self.subTest(group=group,allowed=allowed):
                actual,historical=swapped()
                # Both trajectories share the extra candidate, so candidate
                # names are still the only positional rejection.
                for tree in (actual,historical):
                    tree['current_state_audit'][0]['actions'][0].update(group=group,allowed=allowed)
                self.assertFalse(compare_trajectory(actual,historical)['accepted'])

    def test_all_E_required_in_both_trajectories(self):
        actual,historical=swapped()
        historical['current_state_audit'][0]['actions'][0]['group']='B'
        self.assertFalse(compare_trajectory(actual,historical)['accepted'])

    def test_source_count_or_identity_changes_rejected(self):
        actual,historical=swapped()
        actual['current_state_audit'][0]['actions'][2]['cls_source']='query:9'
        self.assertFalse(compare_trajectory(actual,historical)['accepted'])
        actual,historical=swapped()
        actual['current_state_audit'][0]['actions'].pop()
        self.assertFalse(compare_trajectory(actual,historical)['accepted'])

    def test_selected_source_change_rejected(self):
        actual,historical=swapped()
        actual['records'][0]['cls_source']='query:0'
        self.assertFalse(compare_trajectory(actual,historical)['accepted'])

    def test_selected_modify_in_both_rejected(self):
        actual,historical=swapped()
        for tree in (actual,historical):
            tree['current_state_audit'][0]['selected_group']='A'
            tree['records'][0].update(cls_source='query:0',reg_source='query:0',changed=True,expansion=1.)
        self.assertFalse(compare_trajectory(actual,historical)['accepted'])

    def test_each_E_must_equal_keep_in_outcome8_exactly(self):
        actual,historical=swapped()
        for tree in (actual,historical):
            tree['current_state_audit'][0]['actions'][0]['key'][6]+=1e-7
        self.assertFalse(compare_trajectory(actual,historical)['accepted'])

    def test_homogeneous_actions_must_not_change_discrete_key(self):
        actual,historical=swapped()
        actual['current_state_audit'][0]['actions'][2]['key'][0]+=1
        self.assertFalse(compare_trajectory(actual,historical)['accepted'])

    def test_existing_numeric_limit_remains(self):
        actual,historical=swapped()
        actual['records'][0]['focal_after']['score']+=.001
        self.assertFalse(compare_trajectory(actual,historical)['accepted'])

    def test_ROI_or_other_metadata_change_rejected(self):
        actual,historical=swapped()
        actual['records'][0]['expansion']=1.5
        self.assertFalse(compare_trajectory(actual,historical)['accepted'])

    def test_tiny_ranking_flip_remains_failure(self):
        historical=fixture(); actual=deepcopy(historical)
        historical['current_state_audit'][0]['actions'][0]['key'][6]+=.0000001
        actual['current_state_audit'][0]['actions'][0]['key'][6]-=.0000001
        self.assertFalse(compare_trajectory(actual,historical)['accepted'])

    def test_original_worker_save_resume_and_exception_proof(self):
        from .driver import execute_k2
        from local_fusion_source_kill_stage2 import k2_label_stability as original
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory); protocol={'identity':'synthetic'}
            atomic_json(run/'PROTOCOL.json',protocol)
            with patch.object(original,'evaluate_frame',dummy_evaluate),patch.object(original,'run',dummy_run):
                execute_k2(run,protocol,None)
                evidence=run/'k2/conditions/fog/same/00000608.reproduction.json'
                self.assertTrue(read_json(evidence)['permitted_unselectable_E_permutations'])
                receipt=run/'k2/conditions/fog/same/00000608.json'
                before=digest(receipt)
                execute_k2(run,protocol,None)
                self.assertEqual(before,digest(receipt))
                evidence.write_text('{}',encoding='utf-8')
                with self.assertRaisesRegex(ValueError,'Corrupted'):
                    execute_k2(run,protocol,None)

    def test_original_core_source_snapshot_unchanged(self):
        self.assertEqual(read_json(PACKAGE/'source_snapshot.json')['source_hashes'],sources())


def source_fixture(folder):
    previous=folder/'previous'; source=previous/'source_reproduction'; history=folder/'history'
    old=dict(identity='old',oracle_run=str(history))
    ap=dict(ap30=.8,ap50=.7,ap70=.6); shared=dict(ap30=.7,ap50=.6,ap70=.5)
    saved={w:dict(results={'Shared':shared,'Oracle-Same':ap,'Oracle-Task':ap}) for w in WEATHERS}
    atomic_json(history/'death_test_results.json',dict(conditions=saved))
    conf=object_hash(dict(purpose='Source-only repair AP gate',root='old'))
    identity=object_hash(dict(root='old',null=conf))
    artifacts=[]; conditions={}
    for weather in WEATHERS:
        for mode in ('same','task'):
            path=source/'conditions'/weather/'Source/0'/(mode+'.json')
            raw=path.with_suffix('.jsonl'); raw.parent.mkdir(parents=True,exist_ok=True)
            raw.write_text('{}\n',encoding='utf-8')
            payload=dict(weather=weather,family='Source',seed=0,mode=mode,AP=ap,Shared=shared)
            commit_unit(path,identity,payload,[raw])
            artifacts.extend([path,raw]); conditions[f'{weather}/Source/0/{mode}']=dict(AP=ap,Shared=shared)
    gate=source/'SOURCE_REPRODUCTION_GATE.json'
    atomic_json(gate,dict(status='complete',root_identity='old',tolerance=1e-6,conditions=conditions))
    artifacts.append(gate)
    atomic_json(source/'manifest.json',dict(status='complete',identity='old'))
    commit_unit(source/'result.json','old',{'status':'complete'},artifacts)
    return previous,old


class SourceGate(unittest.TestCase):
    def test_all_eight_source_conditions_and_raw_hashes_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            previous,old=source_fixture(Path(directory))
            files,conditions=verify_source_gate(previous,old)
            self.assertEqual(len(conditions),8)
            self.assertEqual(len(files),19)

    def test_corrupted_raw_source_frames_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            previous,old=source_fixture(Path(directory))
            raw=previous/'source_reproduction/conditions/fog/Source/0/same.jsonl'
            raw.write_text('corrupted\n',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Corrupted'):
                verify_source_gate(previous,old)

    def test_unchanged_absolute_AP_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            previous,old=source_fixture(Path(directory))
            history=Path(old['oracle_run'])/'death_test_results.json'
            saved=read_json(history)
            saved['conditions']['fog']['results']['Oracle-Same']['ap70']+=2e-6
            atomic_json(history,saved)
            with self.assertRaisesRegex(RuntimeError,'REPRODUCTION_FAIL'):
                verify_source_gate(previous,old)


if __name__=='__main__':
    unittest.main()
