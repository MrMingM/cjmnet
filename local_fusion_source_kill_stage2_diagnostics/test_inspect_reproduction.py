"""Fabricated saved failure records; no actual scientific data or forwards."""
from copy import deepcopy
import unittest
from .differences import exact_differences
from .inspect_reproduction import permutation_summary, summarize


def fixture():
    key = [1, 0, 0, 1, 0, 0, .4, .7, 2]
    return dict(records=[], current_state_audit=[dict(target_index=3, keep_key=key,
        selected_key=key, selected_group='KEEP', actions=[dict(cls_source='single:1',
            reg_source='single:1', key=key[:6]+[.4000001, .7, 0], group='B', allowed=False)])])


def evidence(actual, historical, rejected=()):
    return dict(accepted=False, differences=exact_differences(actual, historical),
        rejected_differences=list(rejected), full9_order_changes=[], a_only_legal_order_changes=[],
        actual_trajectory=actual, historical_trajectory=historical)


class Inspection(unittest.TestCase):
    def test_pure_permutation_keeps_selected_record_and_source_keys(self):
        historical=fixture()
        historical['current_state_audit'][0]['actions'].append(dict(
            cls_source='query:0',reg_source='query:0',key=[1,0,0,1,0,0,.3,.7,1],group='B',allowed=False))
        actual=deepcopy(historical)
        actual['current_state_audit'][0]['actions'].reverse()
        record=evidence(actual,historical); before=deepcopy(record)
        result=permutation_summary(record)
        detail=result['changed_targets'][0]
        self.assertEqual(result['selected_record_noncontinuous_difference_count'],0)
        self.assertEqual(detail['aligned_noncontinuous_difference_count'],0)
        self.assertEqual(detail['aligned_full9_order_change_count'],0)
        self.assertEqual(record,before)
        self.assertFalse(result['recorded_accepted'])

    def test_permutation_can_hide_source_specific_discrete_change(self):
        historical=fixture()
        historical['current_state_audit'][0]['actions'].append(dict(
            cls_source='query:0',reg_source='query:0',key=[2,0,0,1,1,0,.8,.7,1],group='A',allowed=True))
        actual=deepcopy(historical)
        aa=actual['current_state_audit'][0]['actions']
        # Swap only names, retaining the outcome at each list position.
        aa[0]['cls_source'],aa[1]['cls_source']=aa[1]['cls_source'],aa[0]['cls_source']
        aa[0]['reg_source'],aa[1]['reg_source']=aa[1]['reg_source'],aa[0]['reg_source']
        detail=permutation_summary(evidence(actual,historical))['changed_targets'][0]
        self.assertGreater(detail['aligned_noncontinuous_difference_count'],0)
        self.assertNotIn('aligned_full9_order_change_count',detail)

    def test_duplicate_pair_not_arbitrarily_matched(self):
        historical=fixture()
        actions=historical['current_state_audit'][0]['actions']
        actions.append(deepcopy(actions[0]))
        actions.append(dict(cls_source='query:0',reg_source='query:0',key=[1,0,0,1,0,0,.3,.7,1],group='B',allowed=False))
        actual=deepcopy(historical)
        actual['current_state_audit'][0]['actions'].reverse()
        detail=permutation_summary(evidence(actual,historical))['changed_targets'][0]
        self.assertEqual(detail['status'],'AMBIGUOUS_DUPLICATE_SOURCE_PAIRS')

    def test_candidate_added_not_called_permutation(self):
        historical=fixture(); actual=deepcopy(historical)
        actual['current_state_audit'][0]['actions'][0]['cls_source']='single:9'
        detail=permutation_summary(evidence(actual,historical))['changed_targets'][0]
        self.assertEqual(detail['status'],'SOURCE_PAIR_MULTISET_CHANGED')

    def test_ineligible_pair_flip_shown_separately(self):
        historical=fixture(); actual=deepcopy(historical)
        actual['current_state_audit'][0]['actions'][0]['key'][6]=.3999999
        result=summarize(evidence(actual,historical))
        self.assertEqual(result['diagnostic_full9_order_change_count'],1)
        self.assertEqual(result['diagnostic_legal_order_change_count'],0)
        detail=result['first_full9_order_changes'][0]
        self.assertEqual(detail['target_index'],3)
        self.assertFalse(detail['current_actions'][1]['allowed'])
        self.assertFalse(result['recorded_accepted'])

    def test_skipped_order_not_reported_as_no_flip(self):
        historical=fixture(); actual=deepcopy(historical)
        actual['current_state_audit'][0]['actions'][0]['group']='A'
        diff=exact_differences(actual,historical)['first_differences']
        result=summarize(evidence(actual,historical,diff))
        self.assertIsNone(result['recorded_full9_order_change_count'])
        self.assertFalse(result['diagnostic_order_check_executed'])
        self.assertIn('NONCONTINUOUS_FIELDS_CHANGED',result['observed_reasons'])

    def test_large_continuous_change_and_order_both_shown(self):
        historical=fixture(); actual=deepcopy(historical)
        actual['current_state_audit'][0]['actions'][0]['key'][6]=.3
        diff=exact_differences(actual,historical)['first_differences']
        result=summarize(evidence(actual,historical,diff))
        self.assertIsNone(result['recorded_legal_order_change_count'])
        self.assertTrue(result['diagnostic_order_check_executed'])
        self.assertIn('CONTINUOUS_VALUE_FAILED_LEGACY_CHECK',result['observed_reasons'])
        self.assertIn('FULL9_ORDER_CHANGED',result['observed_reasons'])

    def test_input_unchanged(self):
        historical=fixture(); actual=deepcopy(historical)
        record=evidence(actual,historical); before=deepcopy(record)
        summarize(record)
        self.assertEqual(record,before)

    def test_incomplete_saved_differences_rejected(self):
        record=evidence(fixture(),fixture())
        record['differences']['total_differences']=1
        with self.assertRaisesRegex(ValueError,'incomplete'):
            summarize(record)


if __name__=='__main__':
    unittest.main()
