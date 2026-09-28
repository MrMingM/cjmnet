import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .candidate_hypothesis_replay import (
    MODELS, _competing_pairs, _model_vector, _source_geometry_features,
    cross_fit, discrimination, load_extraction,
    paired_intervention_weather,
)
from .candidate_intervention_probe import INTERVENTION_NAMES, PROXY_NAMES


def example(sample, candidate_id, quality, score, center=0.):
    box = np.asarray([[center, 0], [center + 1, 0],
                      [center + 1, 1], [center, 1]], dtype=np.float32)
    return {
        'key': ('snow', sample, candidate_id),
        'weather': 'snow', 'sample_index': sample, 'candidate_id': candidate_id,
        'quality': quality, 'score': score, 'source_count': 2,
        'within_range': True, 'corners': box,
        'candidate': np.asarray([score, center], dtype=np.float32),
        'proxy': np.asarray([score * .5 if name == 'ego_score' else .8
                             for name in PROXY_NAMES], dtype=np.float32),
        'source_geometry': np.asarray([.2, .3, .1, .8], dtype=np.float32),
        'passive': np.asarray([.2, score - .1], dtype=np.float32),
        'distortion_geometry': np.asarray([.02], dtype=np.float32),
        'intervention': np.asarray([.3, 1. - score], dtype=np.float32),
        'simple_agreement': score * .8,
    }


class CandidateHypothesisReplayTests(unittest.TestCase):
    def test_feature_sets_keep_candidate_and_add_only_visible_inputs(self):
        row = example(1, 1, .8, .1)
        self.assertEqual(len(_model_vector(row, 'candidate_only')), 2)
        self.assertEqual(len(_model_vector(row, 'candidate_plus_simple_agreement')), 3)
        self.assertEqual(len(_model_vector(row, 'candidate_plus_both')), 21)
        copy = dict(row, quality=.2)
        for name in MODELS:
            np.testing.assert_array_equal(_model_vector(row, name), _model_vector(copy, name))

    def test_source_geometry_trajectory_uses_no_gt_label(self):
        row = {'score': .7, 'source_count': 2,
               'source_scores_same_anchor': [.1, .2],
               'source_box_center_shift_to_fused': [.3, .5],
               'source_box_pairwise_iou_mean': .6, 'max_gt_iou': .8}
        geometry, interaction = _source_geometry_features(row)
        np.testing.assert_allclose(geometry, [.4, .5, .1, .6], atol=1e-6)
        np.testing.assert_allclose(interaction, [.2], atol=1e-6)
        row['max_gt_iou'] = .1
        other, other_interaction = _source_geometry_features(row)
        np.testing.assert_array_equal(geometry, other)
        np.testing.assert_array_equal(interaction, other_interaction)

    def test_cross_fit_scores_every_candidate_on_unseen_scenes(self):
        rows = []
        for scene in range(10):
            rows.append(example(scene, 1, .8, .7 + scene * .001))
            rows.append(example(scene, 2, .2, .1 + scene * .001))
            middle = example(scene, 3, .6, .4)
            middle['within_range'] = False
            rows.append(middle)
        scene_map = {str(scene): scene for scene in range(10)}
        result = cross_fit(rows, scene_map, 5, 7)
        self.assertEqual(result['trainable_candidates'], 20)
        self.assertEqual(result['all_scored_candidates'], 30)
        self.assertEqual({scene for fold in result['test_scenes_by_fold'] for scene in fold},
                         {str(scene) for scene in range(10)})
        for row in rows:
            self.assertTrue(all(0 <= row['oof_scores'][name] <= 1 for name in MODELS))

    def test_competition_requires_quality_difference_and_nms_overlap(self):
        good = example(1, 1, .8, .1, center=0.)
        bad = example(1, 2, .2, .7, center=.2)
        distant = example(1, 3, .2, .8, center=4.)
        self.assertEqual(len(_competing_pairs([good, bad, distant], .15)), 1)
        bad['within_range'] = False
        self.assertEqual(_competing_pairs([good, bad, distant], .15), [])

    def test_extraction_joins_frame_targets_and_fixed_anchor_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            indices = list(range(10))
            metadata = {
                'audit_pool': 'top_256', 'stage0_only': False,
                'candidate_feature_cache': True, 'ablation_arm': 'F',
                'validation_indices': indices,
                'validation_scene_map': {str(i): i for i in indices},
                'conditions': {weather: {'ablation_frames': indices,
                                         'original_score_threshold': .2,
                                         'nms_iou_threshold': .15,
                                         'pools': {'F': {
                                             'top_256': {'total_candidates': 20},
                                             'original': {'total_candidates': 10},
                                         }}}
                               for weather in ('clean', 'fog', 'rain', 'snow')},
            }
            (root / 'candidate_audit.json').write_text(json.dumps(metadata), encoding='utf-8')
            for weather in metadata['conditions']:
                folder = root / weather
                folder.mkdir()
                rows = []
                targets = []
                for sample in indices:
                    filename = f'F_features_{sample}.npz'
                    np.savez_compressed(folder / filename,
                                        candidate_id=np.asarray([2, 1]),
                                        feature=np.asarray([[9., 8.], [1., 2.]],
                                                           dtype=np.float32))
                    targets.append({'weather': weather, 'sample_index': sample,
                                    'gt_bev_corners': [[[0, 0], [1, 0],
                                                        [1, 1], [0, 1]]]})
                    for cid, score, quality in ((1, .1, .8), (2, .7, .2)):
                        rows.append({
                            'weather': weather, 'arm': 'F', 'sample_index': sample,
                            'candidate_id': cid, 'within_range': True,
                            'score': score, 'max_gt_iou': quality,
                            'source_count': 2, 'source_scores_same_anchor': [.1, .2],
                            'source_box_iou_same_anchor': [.7, .8],
                            'source_box_center_shift_to_fused': [.1, .2],
                            'source_box_pairwise_iou_mean': .75,
                            'score_source': 1, 'geometry_source': 1,
                            'source_proxy_disagreement': False,
                            'fused_bev_corners': [[0, 0], [1, 0], [1, 1], [0, 1]],
                            'fused_feature_cache': filename,
                            'fused_logit': float(np.log(score / (1 - score))),
                            'fused_regression_deltas': [0.] * 7,
                            'fused_decoded_box': [0., 0., 0., 1., 1., 1., 0.],
                            'leave_one_source_out': {'1': {
                                'logit_drop': .1, 'score_drop': .01,
                                'box_iou_to_full': .9, 'center_shift': .1,
                                'size_shift_l1': .1, 'yaw_shift_abs': .1,
                                'gt_quality_change': 0.,
                            }},
                        })
                (folder / 'candidate_rows.jsonl').write_text(
                    ''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
                (folder / 'frame_targets.jsonl').write_text(
                    ''.join(json.dumps(row) + '\n' for row in targets), encoding='utf-8')
            loaded, frames, items, _ = load_extraction(root, 'F')
            self.assertEqual(loaded['validation_scene_map']['0'], 0)
            self.assertEqual(len(items), 80)
            np.testing.assert_array_equal(items[0]['candidate'][-2:], [1., 2.])
            self.assertEqual(len(frames['snow']['targets']), 10)

    def test_weather_intervention_pair_keeps_same_anchor_and_source_count(self):
        clean = example(1, 1, .8, .1)
        snow = example(1, 1, .8, .09)
        clean['weather'], snow['weather'] = 'clean', 'snow'
        clean['intervention'] = np.zeros(len(INTERVENTION_NAMES), dtype=np.float32)
        snow['intervention'] = np.ones(len(INTERVENTION_NAMES), dtype=np.float32) * .1
        clean['source_count'] = snow['source_count'] = 2
        result = paired_intervention_weather([clean, snow], {'1': 'scene_1'}, 5, 7)
        self.assertEqual(result['snow']['matched_equal_source_count'], 1)
        self.assertEqual(result['snow']['groups_by_weather_quality']['low_good']
                         ['matched_candidates'], 1)
        patterns = result['snow']['groups_by_weather_quality']['all'][
            'cross_weather_asynchrony']['patterns']
        self.assertEqual(patterns['score_drop_geometry_stable']['count'], 0)
        snow['score'] = .03
        result = paired_intervention_weather([clean, snow], {'1': 'scene_1'}, 5, 7)
        patterns = result['snow']['groups_by_weather_quality']['all'][
            'cross_weather_asynchrony']['patterns']
        self.assertEqual(patterns['score_drop_geometry_stable']['count'], 1)
        snow['source_count'] = 3
        result = paired_intervention_weather([clean, snow], {'1': 'scene_1'}, 5, 7)
        self.assertEqual(result['snow']['source_count_mismatch_excluded'], 1)

    def test_discrimination_records_missing_class_instead_of_crashing(self):
        row = example(1, 1, .8, .1)
        row['oof_scores'] = {name: .5 for name in MODELS}
        result = discrimination([row])
        self.assertEqual(result['snow']['low']['status'], 'insufficient_classes')
        self.assertEqual(result['snow']['low']['good'], 1)
        self.assertEqual(result['snow']['high']['status'], 'insufficient_classes')

    def test_scene_bootstrap_compares_out_of_fold_scores_within_score_band(self):
        rows = []
        for scene in range(6):
            for cid, quality, score in ((1, .8, .1), (2, .2, .12),
                                        (3, .8, .6), (4, .2, .7)):
                row = example(scene, cid, quality, score)
                row['oof_scores'] = {name: (.9 if quality >= .7 else .1)
                                     for name in MODELS}
                rows.append(row)
        result = discrimination(rows, {str(i): i for i in range(6)}, 5, 7)
        low = result['snow']['low']
        self.assertEqual(low['status'], 'available')
        comparison = low['paired_scene_bootstrap_auc'][
            'candidate_plus_both_minus_candidate_plus_source']
        self.assertEqual(comparison['valid_repetitions'], 5)
        self.assertEqual(comparison['auc_difference'], 0)


if __name__ == '__main__':
    unittest.main()
