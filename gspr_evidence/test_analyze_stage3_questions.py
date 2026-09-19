import unittest
from .analyze_stage3_questions import choose, costs, pareto, pattern, overlap


class OfflineQuestionsTest(unittest.TestCase):
    def setUp(self):
        self.full = dict(name='full', matched_gt=[0, 1], frame_fp=2,
                         recovered_candidates=[], lost_baseline_gt=[], new_fp_count=0, fp_count_delta=0)

    def action(self, **kwargs):
        return dict(self.full, **kwargs)

    def test_keep_full_beats_harmful_action(self):
        harmful = self.action(name='harm', matched_gt=[2], recovered_candidates=[2],
                             lost_baseline_gt=[0, 1], new_fp_count=1)
        for policy in ('zero_cost', 'net_gt_minus_newfp'):
            selected, c = choose([harmful], self.full, policy)
            self.assertEqual(selected['name'], 'KEEP_FULL')
            self.assertEqual(c['net_matched_gt'], 0)

    def test_other_gt_recovery_is_counted(self):
        good = self.action(name='recover', matched_gt=[0, 2, 3], recovered_candidates=[2], lost_baseline_gt=[1])
        c = costs(good, self.full)
        self.assertEqual(c['other_gt_gained'], 1)
        self.assertEqual(c['net_matched_gt'], 1)
        self.assertNotEqual(c['net_matched_gt'], c['recovered']-c['lost'])

    def test_zero_cost_can_recover_and_equal_action_prefers_full(self):
        neutral = self.action(name='neutral')
        self.assertEqual(choose([neutral], self.full, 'zero_cost')[0]['name'], 'KEEP_FULL')
        good = self.action(name='good', matched_gt=[0, 1, 2], recovered_candidates=[2])
        self.assertEqual(choose([good], self.full, 'zero_cost')[0]['name'], 'good')
        self.assertEqual([r['action'] for r in pareto([good], self.full)], ['good'])

    def test_false_lost_count_rejected(self):
        with self.assertRaises(ValueError):
            costs(self.action(matched_gt=[]), self.full)

    def test_intersection_is_not_same_peer_intersection(self):
        score, geometry = {1}, {2}
        self.assertEqual(pattern(bool(score), bool(geometry)), 'both')
        self.assertFalse(score & geometry)
        self.assertEqual([pattern(p in score, p in geometry) for p in (1, 2)], ['score_only', 'geometry_only'])

    def test_polygon_overlap(self):
        try:
            import shapely
        except ImportError:
            self.skipTest('Shapely unavailable locally; run this check in opencood environment')
        a = [[0, 0], [2, 0], [2, 2], [0, 2]]
        b = [[1, 0], [3, 0], [3, 2], [1, 2]]
        self.assertEqual(overlap(a, a), 1)
        self.assertAlmostEqual(overlap(a, b), 1/3, places=7)

    def test_streaming_frame_analysis_and_input_integrity(self):
        import json
        import tempfile
        from pathlib import Path
        from .analyze_stage3_questions import analyze_weather, sha
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)/'snow'; folder.mkdir()
            out = Path(temp)/'out'; out.mkdir()
            def target(hit):
                return dict(matched=hit, matched_candidate_id=4 if hit else None,
                    matched_proposal=None, watched_anchors={}, qualifying_nms_suppressions=[],
                    stages=dict(decoded=dict(highest_qualifying_score_candidate_id=None)))
            full = self.action(family='subset', subset=[0, 1], targets={'2': target(False)})
            def branch(name, family, hit):
                return self.action(name=name, family=family, peer=1, targets={'2': target(hit)},
                    matched_gt=[0, 1, 2] if hit else [0, 1], recovered_candidates=[2] if hit else [])
            frame = dict(sample_index=5, candidate_targets=[2],
                branches=[full, branch('peer', 'peer_alone', True),
                          branch('score', 'peer_score_full_geometry', True),
                          branch('geom', 'full_score_peer_geometry', False)],
                target_metadata={'2': dict(scene=1, distance=30, failure_stage='score_filtered')},
                baseline_targets={'2': target(False)})
            path = folder/'diagnostics.jsonl'; path.write_text(json.dumps(frame)+'\n')
            (folder/'summary.json').write_text(json.dumps(dict(complete=True, diagnostic_schema=1, diagnostics_sha256=sha(path))))
            (folder/'protocol.json').write_text(json.dumps(dict(stage='3B', smoke=0,
                development_only=True, test_data_used=False, selected_candidate_frames=[5], candidate_targets=1,
                postprocessing=dict(score_threshold=.2, nms_threshold=.15))))
            r = analyze_weather(folder, out)
            self.assertEqual(r['q1']['all']['target_score_only'], 1)
            self.assertEqual(r['q3']['peer_score_full_geometry/zero_cost']['recovered'], 1)
            self.assertTrue((out/'box_changes.jsonl').read_text())
            path.write_text('{}\n')
            with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
                analyze_weather(folder, out)


if __name__ == '__main__':
    unittest.main()
