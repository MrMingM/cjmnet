"""Run on the server: parsing, pairing selectors and real 7z selective extraction."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from v2xr_subset.prepare import (archive_inventory, choose_variant, classify,
    parse_listing, safe_member, select_stamps, unpack_selected, verify_sources)


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not sys.platform.startswith('linux'):
            raise RuntimeError('Run checks on the research server')

    def test_snapshot(self):
        self.assertTrue(verify_sources())

    def test_negative_agent_prefix_is_allowed_but_traversal_is_rejected(self):
        self.assertEqual(safe_member('-1/000001.yaml').name, '000001.yaml')
        for name in ('../../a.pcd', '/a.pcd', 'C:\\a.pcd', 'test/a.pcd', 'a\nb.pcd'):
            with self.assertRaises(ValueError):
                safe_member(name)

    def test_weather_classification_excludes_radar_and_camera(self):
        self.assertEqual(classify('200/000001.pcd'), ('000001', 'clean'))
        self.assertEqual(classify('000001_fog_0.060.pcd'), ('000001', 'fog_0.060'))
        self.assertEqual(classify('000001_snow_2.5_2.0.pcd'), ('000001', 'snow_2.5_2.0'))
        for name in ('000001_radar.pcd', '000001_camera0.png', 'data_protocol.yaml', '000001_foggy.pcd'):
            self.assertIsNone(classify(name))

    def test_multiple_weather_strengths_require_explicit_selection(self):
        variants = {'clean', 'yaml', 'fog_0.060', 'fog_0.030', 'snow'}
        with self.assertRaises(ValueError):
            choose_variant('fog', variants, None)
        self.assertEqual(choose_variant('fog', variants, 'fog_0.030'), 'fog_0.030')
        self.assertEqual(choose_variant('snow', variants, None), 'snow')

    def test_frame_cap_spans_the_scene(self):
        values = [f'{i:06}' for i in range(10)]
        self.assertEqual(select_stamps(values, 3), ['000000', '000004', '000009'])
        self.assertEqual(select_stamps(values, 0), values)

    def test_listing_never_treats_archive_header_as_a_member(self):
        text = '7-Zip\nPath = /source/a.7z\nType = 7z\n\n----------\nPath = -1/000001.yaml\nSize = 3\nAttributes = A\nCRC = 12345678\n\n'
        self.assertEqual(len(parse_listing(text)), 1)

    def test_real_7z_extracts_selected_members_and_flattens_agent_prefix(self):
        seven = shutil.which('7zz') or shutil.which('7z') or shutil.which('7za')
        if not seven:
            self.fail('Install/provide 7zz/7z/7za before preparing V2X-R')
        with tempfile.TemporaryDirectory(prefix='v2xr_contract_') as tmp:
            root = Path(tmp)
            source = root / 'source' / '-1'
            source.mkdir(parents=True)
            desired = ['000001.yaml', '000001.pcd', '000001_fog_0.060.pcd', '000001_snow_2.5_2.0.pcd']
            for name in [*desired, '000001_radar.pcd', '000001_camera0.png']:
                (source / name).write_bytes(('synthetic contract file: ' + name).encode())
            archive = root / '-1.7z'
            result = subprocess.run([seven, 'a', str(archive), './-1'], cwd=source.parent,
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            self.assertEqual(result.returncode, 0, result.stdout.decode(errors='replace'))
            inventory = archive_inventory(seven, archive)
            chosen = list(inventory['frames']['000001'].values())
            output = root / 'selected'
            unpack_selected(seven, archive, chosen, output)
            names = {p.name for p in output.iterdir()} - {'selected.txt', '7z.log'}
            self.assertEqual(names, set(desired))
            for name in desired:
                self.assertEqual((output / name).read_bytes(), (source / name).read_bytes())


if __name__ == '__main__':
    unittest.main()
