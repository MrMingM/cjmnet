"""CPU bookkeeping tests and Torch operator invariants (required on server)."""
import unittest
from types import SimpleNamespace
import numpy as np
from .common import roi_mask, grid_xy, sample_rows, choose_action, action_grid, global_key

try:
    import torch
except ImportError:
    torch = None


class BookkeepingTests(unittest.TestCase):
    def test_report_control_frame_and_tamper_guard(self):
        import json
        import tempfile
        from pathlib import Path
        from .report import summarize
        from gspr_evidence.stage3_runtime import sha
        spec = dict(sample_index=7,target_index=0,cohort='control',valid_peers=[1])
        template = dict(focal_detected=True,lost_gt=[],new_fp_count=0,net_matched_gt=0,
                        recovered_candidates=[],retained_gt_drop_over_005=[],lost_within10m_gt=[],lost_beyond10m_gt=[])
        local, global_ = [], {}
        pairs = []
        for a in action_grid():
            key = global_key(a['family'],a['scales'],a['alpha'])
            name = key+str(a['radius'])
            local.append(dict(template,**a,peer=1,name=name))
            global_[key] = dict(template,**a,peer=1,name=key)
            pairs.append(dict(a,peer=1,local=name,global_action=key))
        decision = {f'{s}/{f}/{p}':dict(action='KEEP_FULL',**{k:template[k] for k in
                    ('focal_detected','lost_gt','new_fp_count','net_matched_gt','recovered_candidates')})
                    for s in ('local','global') for f in ('all','score','geometry','weights')
                    for p in ('zero_cost','recovery_first')}
        frame = dict(sampling=spec,sample_index=7,local_actions=local,global_actions=list(global_.values()),
                     paired_actions=pairs,frame_decisions=decision,baseline=dict(matched_gt=[0]))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'frames').mkdir()
            path = root/'frames/7.json'; path.write_text(json.dumps(frame),encoding='utf8')
            (root/'protocol.json').write_text(json.dumps(dict(stage='3C',smoke=2,plan_sha256='fixed',
                  sampled_rows=[spec],selected_candidate_frames=[7])),encoding='utf8')
            (root/'summary.json').write_text(json.dumps(dict(complete=True,smoke=2,plan_sha256='fixed',
                  frame_sha256={'7':sha(path)})),encoding='utf8')
            result = summarize(root)
            self.assertEqual(result['decisions']['control/local/all/zero_cost']['kept_full'],1)
            self.assertEqual(sum(c['action_evaluations'] for c in result['control_action_harm'].values()),16)
            path.write_text('{}',encoding='utf8')
            with self.assertRaisesRegex(ValueError,'Frame data changed'):
                summarize(root)

    def test_rotated_rectangle(self):
        c = np.array([[-2,-1],[2,-1],[2,1],[-2,1]])
        a = np.pi/4
        rotation = np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
        points = np.array([[0,0],[1.9,.9],[2.1,0],[0,1.1]]) @ rotation.T
        mask, fallback = roi_mask(points, c @ rotation.T, 1)
        np.testing.assert_array_equal(mask, [True,True,False,False])
        self.assertFalse(fallback)

    def test_expansion_and_nearest_fallback(self):
        xy = grid_xy((6,8), [-4,-3,-1,4,3,1])
        c = np.array([[-1,-1],[1,-1],[1,1],[-1,1]])
        m1,_ = roi_mask(xy,c,1); m2,_ = roi_mask(xy,c,1.5)
        self.assertTrue(np.all(m2[m1]))
        tiny = c*.01 + [.6,.6]
        m,fallback = roi_mask(xy,tiny,1)
        self.assertTrue(fallback)
        self.assertEqual(m.sum(),1)
        self.assertEqual(np.argmin(((xy-[.6,.6])**2).sum(-1)), np.flatnonzero(m)[0])

    def test_sampling_unique_reproducible_and_balanced(self):
        rows = [dict(sample_index=i,target_index=j,global_recoverable=bool(i%2),
                     scene=i%3,distance_bin=str(i%4)) for i in range(20) for j in (0,1)]
        a = sample_rows(rows,10,123,'snow',excluded=[0])
        b = sample_rows(list(reversed(rows)),10,123,'snow',excluded=[0])
        self.assertEqual(a,b)
        self.assertEqual(len({r['sample_index'] for r in a}),10)
        self.assertNotIn(0,[r['sample_index'] for r in a])
        self.assertEqual(sum(r['global_recoverable'] for r in a),5)
        with self.assertRaises(ValueError):
            sample_rows(rows,21,123,'snow')

    def test_frame_choice_never_unions_actions(self):
        base = dict(name='KEEP_FULL',recovered_candidates=[],lost_gt=[],new_fp_count=0,net_matched_gt=0)
        a = dict(base,name='A',recovered_candidates=[1],net_matched_gt=1)
        b = dict(base,name='B',recovered_candidates=[2,3],lost_gt=[9],net_matched_gt=1)
        self.assertEqual(choose_action([a,b],base)['name'],'B')
        self.assertEqual(choose_action([a,b],base,safe=True)['name'],'A')
        unsafe = dict(a,new_fp_count=1)
        self.assertEqual(choose_action([unsafe,b],base,safe=True)['name'],'KEEP_FULL')
        self.assertEqual(choose_action([dict(base,name='noop')],base)['name'],'KEEP_FULL')

    def test_grid_has_matched_global_controls(self):
        actions = list(action_grid())
        self.assertEqual(len(actions),16)
        self.assertEqual(len({global_key(r['family'],r['scales'],r['alpha']) for r in actions}),8)


@unittest.skipIf(torch is None, 'Torch unavailable locally; server launcher requires Torch')
class OperatorTests(unittest.TestCase):
    def setUp(self):
        from .operators import WeightCache
        torch.manual_seed(91)
        self.levels = [torch.randn(3,2,4,6) for _ in range(3)]
        base = SimpleNamespace(backbone=SimpleNamespace(deblocks=[torch.nn.Identity() for _ in range(3)]),
                               cls_head=torch.nn.Conv2d(6,2,1),reg_head=torch.nn.Conv2d(6,14,1))
        self.model = SimpleNamespace(engine=SimpleNamespace(base=base))
        self.cache = WeightCache(self.model,dict(levels=self.levels))

    def test_output_only_changes_selected_cells_and_head(self):
        from .operators import replace_output
        full = self.cache.decode(self.cache.original)
        peer = {k: v+7 for k,v in full.items()}
        mask = np.zeros((4,6),bool); mask[1,2] = True
        for family,key,other in [('score','psm','rm'),('geometry','rm','psm')]:
            out = replace_output(full,peer,mask,family)
            expected = full[key].clone(); expected[:,:,1,2] += 7
            torch.testing.assert_close(out[key],expected,rtol=0,atol=0)
            self.assertIs(out[other],full[other])
            torch.testing.assert_close(peer[key]-full[key],torch.full_like(peer[key],7))

    def test_zero_alpha_and_empty_roi_are_identity(self):
        before = [x.clone() for x in self.levels]
        expected = self.cache.decode(self.cache.original)
        for pred in (self.cache.predict(1,[0,1],0),
                     self.cache.predict(1,[0,1],1,[np.zeros((4,6),bool)]*3)):
            for k in expected:
                torch.testing.assert_close(pred[k],expected[k],atol=0,rtol=0)
        for a,b in zip(before,self.levels):
            torch.testing.assert_close(a,b,atol=0,rtol=0)

    def test_full_roi_matches_historical_query_operator(self):
        from gspr_evidence.stage3_diagnostic_runtime import weight_prediction
        for scale in (0,1):
            old,_ = weight_prediction(self.model,dict(levels=self.levels),'peer_query',peer=1,scale=scale)
            now = self.cache.predict(1,[scale],1,[np.ones((4,6),bool)]*3)
            for k in old:
                torch.testing.assert_close(now[k],old[k],rtol=0,atol=0)

    def test_half_strength_interpolates_feature_outputs(self):
        # Linear fake decoder makes the independently expected mixed output exact up to rounding.
        base = self.cache.decode(self.cache.original)
        other = self.cache.predict(1,[0,1],1)
        half = self.cache.predict(1,[0,1],.5)
        for k in half:
            torch.testing.assert_close(half[k],.5*(base[k]+other[k]),atol=1e-6,rtol=1e-6)

    def test_partial_roi_preserves_other_locations(self):
        base = self.cache.decode(self.cache.original)
        mask = np.zeros((4,6),bool); mask[1,2] = True
        local = self.cache.predict(1,[0,1],1,[mask]*3)
        global_ = self.cache.predict(1,[0,1],1)
        for k in base:
            m = torch.as_tensor(mask)[None,None]
            torch.testing.assert_close(local[k],torch.where(m,global_[k],base[k]),atol=0,rtol=0)


if __name__ == '__main__':
    unittest.main()
