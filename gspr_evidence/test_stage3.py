"""CPU bookkeeping tests plus original-postprocessor integration when installed.

No surrogate postprocessor or polygon implementation is used. Missing optional
OpenCOOD dependencies skip integration locally; server real-model smoke is required.
"""
import unittest
import numpy as np
from .stage3_analysis import (STAGES, FAILURES, traced_nms, target_summary,
    greedy_assignment, stratify, unique_rows, enumerate_subsets, choose_frame_subset)


class BookkeepingTest(unittest.TestCase):
    def test_candidate_id_and_paired_score(self):
        ids = {s: np.array([2, 0]) for s in STAGES}
        out = target_summary([.2, .99, .7], [.9, .1, .8], ids, [0, -1], 0)
        self.assertEqual(out['matched_candidate_id'], 2)
        for stage in out['stages'].values():
            self.assertEqual(stage['best_iou_candidate_id'], 0)
            self.assertEqual(stage['score_at_best_iou'], .2)
            self.assertEqual(stage['highest_qualifying_score_candidate_id'], 2)
            self.assertEqual(stage['highest_qualifying_score'], .7)

    def test_each_disappearance_stage(self):
        for first_empty, failure in enumerate(FAILURES[:6]):
            ids = {s: ([1] if i < first_empty else []) for i, s in enumerate(STAGES)}
            result = target_summary([.9, .3], [.1, .7], ids, [], 0)
            self.assertEqual(result['failure_stage'], failure)

    def test_qualifying_boundary(self):
        ids = {s: [0, 1] for s in STAGES}
        r = target_summary([.99, .1], [.699999, .7], ids, [-1, 0], 0)
        self.assertEqual(r['stages']['decoded']['qualifying_count'], 1)
        self.assertEqual(r['stages']['decoded']['highest_qualifying_score'], .1)

    def test_float32_iou_threshold_matches_stage2_python_float(self):
        below = np.float32(.7)
        above = np.nextafter(below, np.float32(1))
        assignment = greedy_assignment([.9, .8], np.array([[below], [above]]))
        self.assertEqual(assignment.tolist(), [-1, 0])
        r = target_summary([.9, .8], np.array([below, above]),
                           {s: [0, 1] for s in STAGES}, assignment, 0)
        self.assertEqual(r['stages']['decoded']['qualifying_count'], 1)
        self.assertEqual(r['stages']['decoded']['highest_qualifying_score_candidate_id'], 1)

    def test_first_actual_nms_suppressor(self):
        overlap = np.array([[1, .8, .1, .8], [.8, 1, .9, .8],
                            [.1, .9, 1, .9], [.8, .8, .9, 1]])
        pick, top, removed = traced_nms([.9, .8, .7, .6], lambda i, js: overlap[i, js], .5)
        self.assertEqual(pick.tolist(), [0, 2])
        self.assertEqual(removed[1][0], 0)
        self.assertEqual(removed[3][0], 0)  # Not the later, higher-overlap box 2.
        self.assertNotIn(2, removed)  # Removed box 1 must never suppress box 2.

    def test_nms_strict_threshold_ties_and_topk(self):
        score = np.ones(1002)
        pick, top, removed = traced_nms(score, lambda i, js: np.full(len(js), .5), .5)
        np.testing.assert_array_equal(pick, score.argsort()[::-1][:1000])
        np.testing.assert_array_equal(top, pick)
        self.assertEqual(removed, {})
        self.assertEqual(len(set(range(1002))-set(top)), 2)

    def test_empty_nms(self):
        pick, top, removed = traced_nms([], lambda i, js: None, .5)
        self.assertEqual(len(pick), 0)
        self.assertEqual(removed, {})

    def test_matching_competition(self):
        # Only final box qualifies for GT0, but has larger IoU with GT1.
        assignment = greedy_assignment([.8], [[.75, .9]])
        self.assertEqual(assignment.tolist(), [1])
        r = target_summary([.8], [.75], {s: [0] for s in STAGES}, assignment, 0)
        self.assertEqual(r['failure_stage'], 'matching_competition')
        self.assertEqual(r['matching_competitors'], [{'candidate_id': 0, 'assigned_gt': 1}])

    def test_stable_greedy_and_empty_gt(self):
        self.assertEqual(greedy_assignment([.5, .5], [[.8], [.9]]).tolist(), [0, -1])
        self.assertEqual(greedy_assignment([.5], np.zeros((1, 0))).tolist(), [-1])

    def test_scene_denominator_and_frame_units(self):
        rows = [dict(sample_index=f, scene=s, distance=d, any_peer_alone_detected=v,
                     source_valid_full_miss=m) for f, s, d, v, m in
                [(1, 'A', 10, True, True), (1, 'A', 10, True, True),
                 (2, 'A', 30, True, False), (3, 'A', 30, False, False),
                 (4, 'B', 90, True, False)]]
        r = stratify(rows)
        self.assertEqual(r['total']['source_valid_occurrences'], 4)
        self.assertEqual(r['total']['full_miss_occurrences'], 2)
        self.assertEqual(r['total']['unique_frames'], 3)
        self.assertEqual(r['total']['failure_unique_frames'], 1)
        self.assertAlmostEqual(r['by_scene']['A']['failure_rate'], 2/3)
        self.assertEqual(r['by_scene']['B']['failure_rate'], 0)

    def test_duplicate_occurrences_rejected(self):
        with self.assertRaises(ValueError):
            unique_rows([dict(sample_index=1, target_index=2)]*2)

    def test_subset_space(self):
        for n in range(1, 6):
            subsets = enumerate_subsets(n)
            self.assertEqual(len(subsets), 2**(n-1))
            self.assertEqual(len(set(subsets)), len(subsets))
            self.assertTrue(all(s[0] == 0 for s in subsets))
            self.assertIn(tuple(range(n)), subsets)
        with self.assertRaises(ValueError):
            enumerate_subsets(6)

    def test_frame_choice_cannot_union_targets(self):
        rows = [dict(subset=s, recovered_candidates=c, lost_baseline_gt=l, new_fp_count=f,
                     fp_count_delta=delta) for s, c, l, f, delta in
                [([0, 1], [10], [], 1, -4), ([0, 2], [11], [], 0, 0),
                 ([0, 1, 2], [], [], 0, 0)]]
        r = choose_frame_subset(rows)
        self.assertEqual(r['target_wise_recoverable'], [10, 11])
        self.assertEqual(r['chosen']['subset'], [0, 2])
        self.assertEqual(len(r['chosen']['recovered_candidates']), 1)
        self.assertEqual(r['pareto_subsets'], [[0, 2]])

    def test_new_fp_does_not_equal_net_delta(self):
        # One subset FP continues, another is new; one old FP disappeared.
        correspondence = greedy_assignment([.9, .8], [[.9, .0], [.0, .0]])
        self.assertEqual(int((correspondence == -1).sum()), 1)
        self.assertEqual(len(correspondence)-2, 0)

    def test_source_snapshot(self):
        from .stage3_runtime import validate_sources
        self.assertIn('opencood/utils/box_utils.py', validate_sources())


class OriginalPostprocessorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            from opencood.utils import box_utils
            from opencood.data_utils.post_processor.voxel_postprocessor import VoxelPostprocessor
        except (ImportError, OSError) as exc:
            raise unittest.SkipTest('Original OpenCOOD runtime unavailable: '+str(exc))
        cls.torch, cls.bu, cls.pp_type = torch, box_utils, VoxelPostprocessor

    def run_case(self, mode):
        from types import SimpleNamespace
        from .stage3_trace import trace_branch, assert_final_equal
        t = self.torch
        anchors = t.tensor([[0, 0, -1, 1.5, 2, 4, 0], [.1, 0, -1, 1.5, 2, 4, 0],
                            [12, 0, -1, 1.5, 10, 4, 0], [20, 0, 4, 1.5, 2, 4, 0],
                            [500, 0, -1, 1.5, 2, 4, 0], [30, 0, -1, 1.5, 2, 4, 0]], dtype=t.float32)
        psm = t.logit(t.tensor([.9, .8, .7, .6, .9, .5])).reshape(1, 1, 1, 6)
        if mode == 'no_score':
            psm.fill_(-10)
        elif mode == 'no_geometry':
            anchors[:, 2] = 4
        pp = self.pp_type.__new__(self.pp_type)
        pp.params = dict(order='hwl', target_args=dict(score_threshold=.5), nms_thresh=.15)
        gt = self.bu.boxes_to_corners_3d(anchors[:1], order='hwl')
        batch = {'ego': dict(anchor_box=anchors.reshape(1, 6, 1, 7), transformation_matrix=t.eye(4))}
        prediction = dict(psm=psm, rm=t.zeros((1, 7, 1, 6)))
        def post_process(data, output):
            boxes, scores = pp.post_process(data, output)
            return boxes, scores, gt
        ds = SimpleNamespace(post_processor=pp, post_process=post_process)
        trace = trace_branch(ds, batch, prediction)
        reference, score, _ = ds.post_process(batch, {'ego': prediction})
        if mode == 'normal':
            self.assertEqual(trace['ids']['score'].tolist(), [0, 1, 2, 3, 4])
            self.assertEqual(trace['ids']['geometry'].tolist(), [0, 1, 4])
            self.assertEqual(trace['ids']['range'].tolist(), [0])
            self.assertEqual(trace['suppressors'][1][0], 0)
            with self.assertRaises(AssertionError):
                assert_final_equal(reference, score+.1, reference, score)
        elif mode == 'no_score':
            self.assertIsNone(reference)
        else:
            self.assertIsNotNone(reference)
            self.assertEqual(len(reference), 0)

    def test_original_postprocess_final_parity(self):
        self.run_case('normal')

    def test_original_none_semantics(self):
        self.run_case('no_score')

    def test_original_empty_semantics(self):
        self.run_case('no_geometry')


class LineageAndReportTest(unittest.TestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path
        from . import stage3_runtime as sr
        from .stage3_analysis import write_json
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.stage1, self.stage2 = self.root/'stage1', self.root/'stage2'
        result = dict(stage1_root=str(self.stage1), peer_validity={})
        self.write_rows(self.stage1/'clean/targets.jsonl',
                        [dict(sample_index=i, target_index=0, ego_detected=True) for i in range(3)])
        for weather in ('fog', 'rain', 'snow'):
            stats, rows = [], []
            for i in range(3):
                valid, matched = i != 2, i == 1
                stats.append(dict(sample_index=i, target_index=0, scene=i//2, distance=10+i*20,
                                  ego_detected=False, full_detected=matched))
                rows.append(dict(sample_index=i, target_index=0, scene=i//2,
                    peers=[dict(peer_index=1, matched=valid)], ego=dict(matched=False), full=dict(matched=matched),
                    any_peer_alone_detected=valid, detected_peer_indices=[1] if valid else [],
                    source_valid_full_miss=valid and not matched))
            self.write_rows(self.stage1/weather/'targets.jsonl', stats)
            self.write_rows(self.stage2/'peer'/weather/'peer_targets.jsonl', rows)
            shared = dict(development_only=True, test_data_used=False, weather=weather,
                          sample_indices=[0, 1, 2], frontend_sha256='checkpoint', frontend_config_sha256='config')
            write_json(self.stage1/weather/'protocol.json', dict(shared, data_root=sr.VALIDATION,
                smoke=0, collector_sha256=sr.sha(sr.ROOT/'qa_observation_diagnostic/collect.py')))
            write_json(self.stage2/'peer'/weather/'protocol.json', dict(shared, stage1_root=str(self.stage1),
                candidate_targets=3, source_sha256=sr.sha(sr.ROOT/'qa_evidence_validity/peer_audit.py')))
            result['peer_validity'][weather] = dict(task_observed=dict(
                source_valid_full_miss_n=1, any_peer_alone_detected_n=2))
        write_json(self.stage2/'evidence_validity_results.json', result)

    @staticmethod
    def write_rows(path, rows):
        import json
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')

    def test_actual_flags_determine_candidates_and_invalid_flag_aborts(self):
        from .stage3_runtime import load_lineage
        from .stage3_analysis import read_rows
        lineage = load_lineage(self.stage2, 'fog')
        self.assertEqual([k for k, r in lineage['rows'].items() if r['source_valid_full_miss']], [(0, 0)])
        path = self.stage2/'peer/fog/peer_targets.jsonl'
        rows = read_rows(path)
        rows[0]['source_valid_full_miss'] = False
        self.write_rows(path, rows)
        with self.assertRaisesRegex(ValueError, 'flags'):
            load_lineage(self.stage2, 'fog')

    def test_offline_and_complete_report(self):
        from unittest.mock import patch
        from pathlib import Path
        from .stage3_analysis import write_json, read_json
        from .stage3_report import offline, summarize
        out = self.root/'output'
        with patch('gspr_evidence.stage3_report.safe_output', side_effect=Path):
            offline(self.stage2, out/'stage2_5')
            off = read_json(out/'stage2_5/scene_distance_report.json')
            self.assertEqual(off['weathers']['fog']['total']['failure_rate'], .5)
            for weather in ('fog', 'rain', 'snow'):
                self.write_rows(out/weather/'targets.jsonl', [dict(sample_index=0, target_index=0,
                    scene=0, failure_stage='score_filtered')])
                write_json(out/weather/'protocol.json', dict(stage='3A', development_only=True,
                    test_data_used=False, smoke=0, stage2_root=str(self.stage2), candidate_targets=1,
                    replay_limit='fixture', selected_candidate_frames=[0]))
                write_json(out/weather/'summary.json', dict(complete=True, smoke=0, occurrences=1,
                    frames=1, failure_counts=dict(score_filtered=1)))
            summarize(out, '3A')
            result = read_json(out/'stage3a_results.json')
            self.assertEqual(result['weathers']['fog']['fractions']['score_filtered'], 1)
            self.assertTrue((out/'stage3a_report.md').is_file())
            summary_path = out/'fog/summary.json'
            summary = read_json(summary_path)
            summary['complete'] = False
            write_json(summary_path, summary)
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                summarize(out, '3A')

    def test_frame_report_keeps_losses_and_new_fp(self):
        from unittest.mock import patch
        from pathlib import Path
        from .stage3_analysis import write_json, read_json
        from .stage3_report import offline, summarize
        out = self.root/'output_b'
        subsets = [dict(subset=[0], recovered_candidates=[0], lost_baseline_gt=[1],
                        new_fp_count=1, fp_count_delta=0),
                   dict(subset=[0, 1], recovered_candidates=[], lost_baseline_gt=[],
                        new_fp_count=0, fp_count_delta=0)]
        decision = dict(chosen=subsets[0], pareto_subsets=[[0], [0, 1]], target_wise_recoverable=[0],
                        warning='Target-wise union is NOT a realizable frame output or AP.')
        with patch('gspr_evidence.stage3_report.safe_output', side_effect=Path):
            offline(self.stage2, out/'stage2_5')
            for weather in ('fog', 'rain', 'snow'):
                self.write_rows(out/weather/'frames.jsonl', [dict(sample_index=0,
                    candidate_targets=[0], subsets=subsets, frame_oracle=decision)])
                write_json(out/weather/'protocol.json', dict(stage='3B', development_only=True,
                    test_data_used=False, smoke=0, stage2_root=str(self.stage2), candidate_targets=1,
                    replay_limit='fixture', selected_candidate_frames=[0]))
                write_json(out/weather/'summary.json', dict(complete=True, smoke=0, occurrences=1,
                    frames=1, target_wise_recoverable=1, frame_wise_recovered=1,
                    frame_wise_lost_baseline_gt=1, frame_wise_new_fp=1, frame_wise_fp_count_delta=0))
            summarize(out, '3B')
            r = read_json(out/'stage3b_results.json')['weathers']['fog']
            self.assertEqual(r['frame_wise_lost_baseline_gt'], 1)
            self.assertEqual(r['frame_wise_new_fp'], 1)
            self.assertEqual(r['frame_wise_fp_count_delta'], 0)


if __name__ == '__main__':
    unittest.main()
