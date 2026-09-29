"""Synthetic contract tests; no detector, dataset or research run is executed."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .candidate_ranker_pilot import (
    FEATURE_GROUPS, MODELS, _feature_parts, _investment_gate, _layout, _load,
    _model_mask, _normalized, _parse_seeds,
)


def _row(weather='clean', sample=7, quality=.8):
    return {
        'weather': weather, 'arm': 'F', 'sample_index': sample,
        'candidate_id': 1, 'score': .1, 'max_gt_iou': quality,
        'source_count': 2, 'within_range': True,
        'source_scores_same_anchor': [.1, .2],
        'source_box_iou_same_anchor': [.7, .8],
        'source_box_center_shift_to_fused': [.1, .2],
        'source_box_pairwise_iou_mean': .75,
        'score_source': 1, 'geometry_source': 1,
        'source_proxy_disagreement': False,
        'fused_bev_corners': [[0, 0], [1, 0], [1, 1], [0, 1]],
        'fused_feature_cache': f'F_features_{sample}.npz',
        'fused_logit': float(np.log(.1 / .9)),
        'fused_regression_deltas': [0.] * 7,
        'fused_decoded_box': [0., 0., 0., 1., 1., 1., 0.],
    }


class CandidateRankerPilotTests(unittest.TestCase):
    def test_feature_groups_are_inference_only_and_model_masks_are_nested(self):
        row = _row()
        parts, agreement = _feature_parts(row, np.asarray([2., 3.]))
        changed, changed_agreement = _feature_parts(
            dict(row, max_gt_iou=.1), np.asarray([2., 3.]))
        self.assertEqual(agreement, changed_agreement)
        for name in FEATURE_GROUPS:
            np.testing.assert_array_equal(parts[name], changed[name])
        layout = _layout(parts)
        active = [_model_mask(layout, name) for name in MODELS]
        self.assertEqual([int(mask.sum()) for mask in active],
                         [18, 19, 33, 42])
        for first, second in zip(active, active[1:]):
            self.assertTrue(np.all(first <= second))

    def test_loader_keeps_train_and_validation_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split in ('train', 'validation'):
                folder = root / split
                folder.mkdir()
                conditions = {}
                for weather in ('clean', 'fog', 'rain', 'snow'):
                    weather_dir = folder / weather
                    weather_dir.mkdir()
                    (weather_dir / 'candidate_rows.jsonl').write_text(
                        ''.join(json.dumps(_row(weather, sample)) + '\n'
                                for sample in (7, 8)), encoding='utf-8')
                    (weather_dir / 'frame_targets.jsonl').write_text(
                        ''.join(json.dumps({
                            'weather': weather, 'sample_index': sample,
                            'gt_bev_corners': [_row(weather, sample)[
                                'fused_bev_corners']],
                        }) + '\n' for sample in (7, 8)), encoding='utf-8')
                    for sample in (7, 8):
                        np.savez_compressed(weather_dir / f'F_features_{sample}.npz',
                                            candidate_id=np.asarray([1]),
                                            feature=np.asarray([[2., 3.]],
                                                               dtype=np.float32))
                    conditions[weather] = {
                        'original_score_threshold': .2,
                        'nms_iou_threshold': .15,
                        'pools': {'F': {
                            'top_256': {'total_candidates': 2},
                            'original': {'total_candidates': 0},
                        }},
                    }
                    if split == 'validation':
                        conditions[weather]['swap_ap'] = {
                            'F_fusion_F_detector': {'ap30': 0., 'ap50': 0.,
                                                    'ap70': 0.}}
                meta = {
                    'audit_pool': 'top_256', 'stage0_only': False,
                    'candidate_feature_cache': True,
                    'conditions': conditions,
                }
                if split == 'train':
                    meta.update(split='train', sample_indices=[7, 8],
                                scene_map={'7': 0, '8': 1})
                else:
                    meta.update(validation_indices=[7, 8],
                                validation_scene_map={'7': 0, '8': 1})
                (folder / 'candidate_audit.json').write_text(
                    json.dumps(meta), encoding='utf-8')
            train = _load(root / 'train', 'train')
            validation = _load(root / 'validation', 'validation')
            self.assertEqual(train['matrix'].shape, (8, 42))
            self.assertEqual(validation['matrix'].shape, (8, 42))
            with self.assertRaisesRegex(ValueError, 'Training input'):
                _load(root / 'validation', 'train')
            with self.assertRaisesRegex(ValueError, 'training extraction'):
                _load(root / 'train', 'validation')

    def test_scaler_and_seed_contract(self):
        np.testing.assert_array_equal(
            _normalized(np.asarray([[2., -100.]], dtype=np.float32),
                        np.asarray([0., 0.]), np.asarray([1., 1.])),
            [[2., -10.]])
        self.assertEqual(_parse_seeds('2,3'), [2, 3])
        with self.assertRaisesRegex(ValueError, 'unique'):
            _parse_seeds('2,2')

    def test_gate_checks_original_and_every_scene_exclusion(self):
        def metric(value):
            return {'ap_frame_order': {'ap70': value},
                    'leave_one_scene_out_ap70_frame': {'0': value, '7': value}}
        report = {
            'protocol': {'seeds': [1, 2]},
            'baselines': {weather: {'scorepass': metric(.5)}
                          for weather in ('clean', 'fog', 'rain', 'snow')},
            'seeds': {},
        }
        for seed in (1, 2):
            report['seeds'][str(seed)] = {}
            for weather in ('clean', 'fog', 'rain', 'snow'):
                proposed = .5 if weather == 'clean' else .52
                control = .5 if weather == 'clean' else .51
                report['seeds'][str(seed)][weather] = {'methods': {
                    **{name: metric(control) for name in MODELS[:-1]},
                    'candidate_plus_distortion': metric(proposed),
                }}
        self.assertTrue(_investment_gate(report)['pilot_gate_pass'])
        for weather in ('fog', 'rain'):
            report['seeds']['2'][weather]['methods'][
                'candidate_plus_distortion']['leave_one_scene_out_ap70_frame']['7'] = .49
        self.assertFalse(_investment_gate(report)['pilot_gate_pass'])


if __name__ == '__main__':
    unittest.main()
