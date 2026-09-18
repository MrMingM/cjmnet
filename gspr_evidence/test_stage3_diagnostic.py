"""Meaningful CPU tests; original-runtime tests run on the server when available."""
import json
import tempfile
import unittest
from pathlib import Path
from .stage3_diagnostic_analysis import suppression_kind, compare_targets, subset_edges


def detail(score=.4, matched=True, failure='detected'):
    return dict(matched=matched, failure_stage=failure,
        stages=dict(decoded=dict(highest_qualifying_score=score)),
        watched_anchors={'4': dict(score=score, logit=score*2, gt_iou=.8)})


class DiagnosticTest(unittest.TestCase):
    def test_lineage_audit_distinguishes_new_modules_from_historical_drift(self):
        from .audit_stage3_lineage import audit, digest
        from .stage3_analysis import write_json
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'source'; aroot = Path(temp)/'A'
            folder = root/'gspr_evidence'; folder.mkdir(parents=True)
            runtime = folder/'stage3_runtime.py'; runtime.write_text('historical\n')
            manifest = {}; write_json(folder/'stage3_sources.json', manifest)
            old = {'gspr_evidence/stage3_runtime.py': digest(runtime, source=True)}
            result = dict(smoke=0, protocols={}, weathers={})
            for weather in ('fog', 'rain', 'snow'):
                dest = aroot/weather; dest.mkdir(parents=True)
                protocol = dict(stage='3A', development_only=True, test_data_used=False,
                    audited_sources=manifest, implementation_sha256=old)
                write_json(dest/'protocol.json', protocol)
                write_json(dest/'summary.json', dict(complete=True))
                (dest/'targets.jsonl').write_text('{}\n')
                result['protocols'][weather] = protocol
                result['weathers'][weather] = dict(targets_sha256=digest(dest/'targets.jsonl'))
            write_json(aroot/'stage3a_results.json', result)
            (folder/'stage3b_diagnostic.py').write_text('new diagnostic\n')
            self.assertTrue(audit(root, aroot)['passed'])
            runtime.write_text('different\n')
            changed = audit(root, aroot)
            self.assertFalse(changed['passed'])
            self.assertTrue(any('implementation mismatch' in e for e in changed['errors']))
            self.assertEqual(runtime.read_text(), 'different\n')  # never auto-restores
            runtime.write_text('historical\n')
            (aroot/'fog/targets.jsonl').write_text('altered\n')
            self.assertTrue(any('target log changed' in e for e in audit(root, aroot)['errors']))

    def test_other_vehicle_is_not_called_background(self):
        e = dict(suppressor=dict(gt_iou=.2), suppressor_best_gt=2, suppressor_best_gt_iou=.85)
        self.assertEqual(suppression_kind(e, 0), 'qualifies_other_gt')
        e.update(suppressor_best_gt=0, suppressor_best_gt_iou=.65)
        self.assertEqual(suppression_kind(e, 0), 'target_nearest_but_iou_below70')

    def test_paired_source_edges_no_multi_source_confounds(self):
        bs = [dict(family='subset', subset=s, targets={'0': detail(score, hit, failure)}) for s, score, hit, failure in
              [([0], .1, False, 'score_filtered'), ([0, 1], .8, True, 'detected'),
               ([0, 1, 2], .1, False, 'score_filtered')]]
        edges = subset_edges(bs)
        self.assertEqual(len(edges), 2)
        self.assertTrue(edges[1]['targets']['0']['lost'])
        self.assertAlmostEqual(edges[1]['targets']['0']['anchor_changes']['4']['score_delta'], -.7)
        self.assertEqual(edges[1]['added_source'], 2)

    def test_no_qualified_box_is_not_zero_score(self):
        a, b = detail(), detail()
        b['stages']['decoded']['highest_qualifying_score'] = None
        self.assertIsNone(compare_targets({'0': a}, {'0': b})['0']['highest_qualifying_score_delta'])

    def test_report_does_not_union_incompatible_frame_choices(self):
        from .stage3_diagnostic_report import analyze
        from .stage3_analysis import write_json
        from .stage3_runtime import sha
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            branches = [dict(name=str(i), family='subset', targets={'0': {}, '1': {}},
                recovered_candidates=[i], lost_baseline_gt=[], new_fp_count=0, fp_count_delta=0) for i in (0, 1)]
            frame = dict(sample_index=10, candidate_targets=[0, 1], branches=branches,
                target_metadata={str(j): dict(scene=0, distance=15, failure_stage='score_filtered') for j in (0, 1)},
                baseline_targets={str(j): dict(qualifying_nms_suppressions=[]) for j in (0, 1)},
                source_addition_edges=[])
            path = folder/'diagnostics.jsonl'; path.write_text(json.dumps(frame)+'\n', encoding='utf-8')
            write_json(folder/'summary.json', dict(complete=True, diagnostic_schema=1, diagnostics_sha256=sha(path)))
            write_json(folder/'protocol.json', dict(selected_candidate_frames=[10], candidate_targets=2))
            r = analyze(folder)['families']['subset']
            self.assertEqual(r['target_wise_opportunity'], 2)
            self.assertEqual(r['frame_wise_recovered'], 1)
            self.assertEqual(r['zero_observed_cost_frame_recovered'], 1)
            path.write_text(json.dumps(frame)+'\n'+json.dumps(frame)+'\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'changed'):
                analyze(folder)


class WeightInterventionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest('Torch unavailable locally; server launcher runs these tests')
        cls.torch = torch

    def test_identity_query_value_and_scale_interventions(self):
        from types import SimpleNamespace
        from .stage3_diagnostic_runtime import weight_prediction
        from gspr_communication.masked_attfuse import fuse
        t = self.torch
        levels = [t.tensor([1., 3.]).reshape(2, 1, 1, 1), t.tensor([2., 4.]).reshape(2, 1, 1, 1)]
        base = SimpleNamespace(backbone=SimpleNamespace(deblocks=[t.nn.Identity(), t.nn.Identity()]),
                               cls_head=t.nn.Identity(), reg_head=t.nn.Identity())
        model = SimpleNamespace(engine=SimpleNamespace(base=base)); encoded = dict(levels=levels)
        identity, _ = weight_prediction(model, encoded, 'identity')
        expected = t.cat([fuse(x, t.ones(2, 1, 1, 1)) for x in levels], 1)
        t.testing.assert_close(identity['psm'], expected)
        uniform, _ = weight_prediction(model, encoded, 'uniform')
        t.testing.assert_close(uniform['psm'], t.tensor([2., 3.]).reshape(1, 2, 1, 1))
        query, _ = weight_prediction(model, encoded, 'peer_query', peer=1, scale=0)
        t.testing.assert_close(query['psm'][:, 1], identity['psm'][:, 1])
        self.assertGreater(float(query['psm'][0, 0]), float(identity['psm'][0, 0]))
        zero, _ = weight_prediction(model, encoded, 'zero_ego_value')
        weights = (levels[0][:1]*levels[0]).softmax(0)
        t.testing.assert_close(zero['psm'][:, :1], weights[1:]*levels[0][1:])


if __name__ == '__main__':
    unittest.main()
