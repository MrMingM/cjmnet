"""CPU checks for the historical evaluation contract; no dataset/GPU needed."""
import copy
import unittest
from pathlib import Path
from .benchmark import ROOTS, test_hypes


class BenchmarkTests(unittest.TestCase):
    def test_fixed_historical_roots_and_no_double_weather(self):
        original = dict(validate_dir='validation', weather_augmentation={'mode': 'mixed_physics_weather'},
                        data_augment=[{'NAME': 'random_world_flip'}], postprocess={'threshold': .2})
        snapshot = copy.deepcopy(original)
        for weather in ROOTS:
            configured = test_hypes(original, weather)
            self.assertEqual(configured['validate_dir'], ROOTS[weather])
            self.assertNotIn('weather_augmentation', configured)
            self.assertEqual(configured['data_augment'], [])
            self.assertEqual(configured['postprocess'], original['postprocess'])
        self.assertEqual(original, snapshot)
        self.assertTrue(ROOTS['clean'].endswith('/opv2v_official_data_dumping/test'))
        for weather in ('fog', 'rain', 'snow'):
            self.assertEqual(ROOTS[weather], f'/data/cjm/datasets/opv2v-w/{weather}/test')

    def test_historical_roots_equal_original_evaluation_script(self):
        # Parse constants without importing original GPU/OpenCOOD dependencies.
        import ast
        tree = ast.parse((Path(__file__).resolve().parents[1]/'evaluate_gspr_all_weather.py').read_text())
        old = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'CONDITIONS' for t in node.targets))
        self.assertEqual(ROOTS, old)

    def test_saved_training_options_unchanged_and_no_fog_resolver(self):
        import sys
        import tempfile
        import types
        import yaml
        from unittest.mock import patch
        from .benchmark import load_config
        options = dict(seed=13, weather_augmentation={'physics_fog': {'lookup_dir': '/missing/training/tables'}})
        fake = types.ModuleType('opencood.hypes_yaml.yaml_utils')
        fake.load_yaml = lambda _: dict(fusion={'core_method': 'IntermediateFusionDataset'}, validate_dir='validation')
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp)/'experiment.yaml'
            config.write_text(yaml.safe_dump(options))
            with patch.dict(sys.modules, {'opencood.hypes_yaml.yaml_utils': fake}), patch.object(Path, 'is_dir', return_value=True):
                loaded, hypes = load_config(config, 'frontend.yaml', 'fog')
        self.assertEqual(loaded, options)
        self.assertNotIn('weather_augmentation', hypes)


if __name__ == '__main__':
    unittest.main()
