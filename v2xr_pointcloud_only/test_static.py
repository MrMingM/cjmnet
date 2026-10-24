"""Standard-library fixtures only; no real data, model, Torch or GPU imports."""
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import zlib

import slim_data as slim

ROOT = Path(__file__).resolve().parent.parent


class SlimTests(unittest.TestCase):
    def test_inventory_dimensions_and_adapter(self):
        # Keep all temporary paths in the explicitly writable workspace.
        with tempfile.TemporaryDirectory(prefix='slim_fixture_', dir=str(ROOT)) as tmp:
            directory = Path(tmp)
            base, output = directory / 'source', directory / 'slim'
            chunk = b'IHDR' + struct.pack('>IIBBBBB', 1600, 900, 8, 2, 0, 0, 0)
            png = b'\x89PNG\r\n\x1a\n' + struct.pack('>I', 13) + chunk + struct.pack('>I', zlib.crc32(chunk))
            for name in slim.DATASETS:
                folder = base / name
                folder.mkdir(parents=True)
                (folder / 'frame.pcd').write_bytes(b'synthetic non-scientific text fixture')
                (folder / 'frame.png').write_bytes(png)
            before = {str(p): p.read_bytes() for p in base.rglob('*') if p.is_file()}
            with contextlib.redirect_stdout(io.StringIO()):
                slim.prepare(base, output, copy=True)
                slim.verify(output)
            metadata = slim.load(output)
            self.assertEqual(len(metadata['files']), 3)
            self.assertEqual(len(metadata['images']), 3)
            self.assertFalse(list(output.rglob('*.png')))
            self.assertEqual(before, {str(p): p.read_bytes() for p in base.rglob('*') if p.is_file()})
            spec = importlib.util.spec_from_file_location('_camera_fixture', ROOT / 'gspr_repe/camera_metadata.py')
            camera = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(camera)
            camera.tools = lambda: slim
            fake_base = types.SimpleNamespace(load_camera_data=lambda *a, **k: (_ for _ in ()).throw(
                AssertionError('Pixel decoder must never be called')))
            fake_package = types.ModuleType('opencood.data_utils.datasets.opv2v')
            fake_package.basedataset = fake_base
            args = types.SimpleNamespace(train_root=output / slim.DATASETS[0],
                                         paired_root=output / slim.DATASETS[1],
                                         view_root=output / slim.DATASETS[2])
            with patch.dict(os.environ, {'POINTCLOUD_ONLY_ROOT': str(output)}), patch.dict(
                    sys.modules, {'opencood.data_utils.datasets.opv2v': fake_package}):
                path = output / slim.DATASETS[0] / 'frame.png'
                self.assertTrue(camera.removed_image(path, metadata['images'][str(path.relative_to(output)).replace('\\', '/')]))
                self.assertEqual(camera.remap_original(base / slim.DATASETS[0] / 'frame.pcd'),
                                 output / slim.DATASETS[0] / 'frame.pcd')
                original = fake_base.load_camera_data
                evidence, undo = camera.install(args, require_equivalence=False)
                self.assertEqual(evidence['mode'], 'exact_png_dimensions_no_pixels')
                self.assertEqual(fake_base.load_camera_data([str(path)])[0].size, (1600, 900))
                with self.assertRaises(ValueError):
                    fake_base.load_camera_data([str(path.with_name('unknown.png'))])
                undo()
                self.assertIs(fake_base.load_camera_data, original)
                with self.assertRaises(FileNotFoundError):
                    camera.install(args)
            (output / slim.DATASETS[0] / 'frame.pcd').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                slim.verify(output)

    def test_bad_png_header_rejected(self):
        with tempfile.TemporaryDirectory(prefix='png_header_fixture_', dir=str(ROOT)) as tmp:
            path = Path(tmp) / 'invalid.png'
            path.write_bytes(b'not a PNG')
            with self.assertRaises(ValueError):
                slim.png_size(path)


if __name__ == '__main__':
    unittest.main()
