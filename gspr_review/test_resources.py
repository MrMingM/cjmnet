import tempfile
import unittest
from pathlib import Path
from .resources import resolve_weather, DEFAULT_FOG_TABLES


class ResourceTests(unittest.TestCase):
    def test_relative_default_and_no_input_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/DEFAULT_FOG_TABLES
            path.mkdir(parents=True)
            (path/'integral_alpha_0.005.pickle').touch()
            config = {'mode': 'mixed_physics_weather'}
            result = resolve_weather(config, directory, {})
            self.assertEqual(result['physics_fog']['lookup_dir'], str(path.resolve()))
            self.assertNotIn('physics_fog', config)

    def test_override_and_missing_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            config = {'mode': 'physics_fog', 'physics_fog': {'lookup_dir': 'missing'}}
            with self.assertRaisesRegex(FileNotFoundError, 'FOG_LOOKUP_DIR'):
                resolve_weather(config, directory, {})
            (path/'integral_alpha_0.02.pickle').touch()
            self.assertEqual(resolve_weather(config, directory, {'FOG_LOOKUP_DIR': directory})['physics_fog']['lookup_dir'], str(path.resolve()))

    def test_range_and_nonfog(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory)/'alpha_0.06.pickle').touch()
            with self.assertRaisesRegex(ValueError, 'No fog table alpha'):
                resolve_weather({'mode': 'physics_fog'}, directory, {'FOG_LOOKUP_DIR': directory})
        self.assertEqual(resolve_weather({'mode': 'none'}, 'missing', {}), {'mode': 'none'})
