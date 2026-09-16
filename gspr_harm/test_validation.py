import json
from pathlib import Path
import tempfile
import unittest
from .validation import check_scene_split, frozen_epsilon, validation_indices


class ValidationTests(unittest.TestCase):
    def test_all_indices_no_stride_no_calibration_exclusion(self):
        self.assertEqual(validation_indices(list(range(17))), list(range(17)))
        with self.assertRaises(ValueError):
            validation_indices([])

    def test_scene_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            train, val = Path(tmp)/'train', Path(tmp)/'val'
            (train/'a').mkdir(parents=True)
            (val/'b').mkdir(parents=True)
            self.assertEqual(check_scene_split(train, val), ['b'])
            (val/'a').mkdir()
            with self.assertRaises(ValueError):
                check_scene_split(train, val)

    def test_reuses_train_epsilon_and_checks_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'fog').mkdir()
            audit = {'purpose': 'TRAIN-scene diagnostic', 'args': {}, 'checkpoint_sha256': 'cp',
                     'frontend_config_sha256': 'cfg', 'communication': {'variant': 'a0b0'}, 'seed': 7}
            documents = {'audit.json': audit, 'summary.json': {'status': 'complete', 'frozen_model_unchanged': True},
                'fog/data_protocol.json': {'weather': {'mode': 'physics_fog'}},
                'fog/epsilon.json': {'epsilon': 0.000123, 'indices': [0, 10]}}
            for file, data in documents.items():
                (root/file).write_text(json.dumps(data))
            args = (root, 'fog', 'cp', 'cfg', {'variant': 'a0b0'}, 7, {'mode': 'physics_fog'})
            self.assertEqual(frozen_epsilon(*args)[0], .000123)
            with self.assertRaises(ValueError):
                frozen_epsilon(root, 'fog', 'wrong_cp', *args[3:])
            audit['purpose'] = 'FULL VALIDATION'
            (root/'audit.json').write_text(json.dumps(audit))
            with self.assertRaises(ValueError):
                frozen_epsilon(*args)


if __name__ == '__main__':
    unittest.main()
