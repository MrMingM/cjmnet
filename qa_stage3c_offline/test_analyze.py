import copy
import unittest
from .analyze import overlap, outcome, anchor_features, inspect_frame, quantiles


def action(family, detected, **extra):
    return dict(family=family,focal_detected=detected,lost_gt=[],new_fp_count=0,**extra)


class AnalysisTests(unittest.TestCase):
    def test_overlap_deduplicates_and_applies_safety(self):
        a=action('score',True); b=action('geometry',True); b['new_fp_count']=1
        self.assertEqual(overlap([a,a,b]),dict(score=True,geometry=True,weights=False))
        self.assertEqual(overlap([a,a,b],True),dict(score=True,geometry=False,weights=False))
        self.assertEqual(outcome(b),'harmful_detected')

    def test_anchor_contrast_wraps_yaw_and_handles_missing(self):
        base=dict(score=.1,logit=-2.,decoded_box=[0,0,0,4,2,1,3.13])
        peer=dict(score=.5,logit=0.,decoded_box=[3,4,0,4,2,1,-3.13])
        f=anchor_features(base,peer,peer)
        self.assertEqual(f['peer_full_center_distance'],5.)
        self.assertLess(f['peer_full_abs_yaw_difference'],.03)
        self.assertIsNone(anchor_features(None,peer,peer))
        self.assertEqual(quantiles([None,float('nan')])['n'],0)

    def test_unresolved_and_same_peer_joint_scale(self):
        proposal=dict(score=.1,logit=-2.,decoded_box=[0,0,0,4,2,1,0])
        detail=dict(failure_stage='score_filtered',qualifying_score_margin=-.1,
            stages=dict(decoded=dict(best_iou_candidate_id=4,highest_qualifying_score_candidate_id=4,best_iou=.9)),
            watched_anchors={'4':proposal},qualifying_nms_suppressions=[])
        def row(family,scales):
            return dict(action(family,False),name=family+str(scales),peer=1,scales=scales,alpha=1.,radius=1.,
                        targets={'0':copy.deepcopy(detail)},retained_gt_drop_over_005=[],mask_stats={})
        locals_=[row('score',[]),row('geometry',[])]+[row('weights',s) for s in ([0],[1],[0,1])]
        f=dict(focal_target=0,weather='snow',sample_index=1,cohort='candidate',
               sampling=dict(scene=0,distance_bin='20-40',valid_peers=[1]),
               baseline=dict(targets={'0':detail}),local_actions=locals_,
               global_actions=[row('score',[]),row('geometry',[])],roi_stats={'1.0':{}})
        events,c,u=inspect_frame(f)
        self.assertEqual(len(events),5)
        self.assertIsNotNone(u)
        self.assertIsNone(u['critical_anchor_output_roi']['1.0']['4'])
        locals_[-1]['focal_detected']=True
        _,c,u=inspect_frame(f)
        self.assertIsNone(u)
        self.assertEqual(len(c['joint_scale_cases']),1)
        self.assertTrue(c['modes']['zero_cost']['by_peer']['1']['weights'])


if __name__=='__main__':
    unittest.main()
