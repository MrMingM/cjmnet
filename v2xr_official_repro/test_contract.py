"""Server-only standard-library tests for dangerous input/selection boundaries."""
from pathlib import Path
import tempfile
import unittest

from common import link, no_test, server_only
from observe import normalize_state
from prepare import safe_member, scene_inventory
from run import config_difference, select_file


class ContractTests(unittest.TestCase):
    def setUp(self):
        server_only()

    def test_archive_traversal(self):
        for name in ('../x.pcd', '/a.pcd', 'a\\b.pcd', 'a\nb.pcd', 'C:x.pcd'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                safe_member(name)

    def test_checkpoint_ambiguity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('first.pth', 'second.pth'):
                (root / name).write_bytes(b'weight')
            with self.assertRaises(ValueError):
                select_file(root, None, 'checkpoint')
            self.assertEqual(select_file(root, str(root / 'first.pth'), 'checkpoint'), root / 'first.pth')

    def test_test_alias_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'test'
            target.mkdir()
            alias = Path(tmp) / 'innocent'
            alias.symlink_to(target, target_is_directory=True)
            with self.assertRaises(ValueError):
                no_test(alias)

    def test_existing_link_never_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, b, view = root / 'a', root / 'b', root / 'view'
            a.write_bytes(b'a')
            b.write_bytes(b'b')
            link(a, view)
            with self.assertRaises(FileExistsError):
                link(b, view)
            self.assertEqual(view.read_bytes(), b'a')

    def test_config_behavioral_difference(self):
        self.assertEqual(config_difference({'fov': True}, {'fov': False}), ['fov'])
        self.assertEqual(config_difference({'postprocess': {'nms_thresh': .15}},
                                         {'postprocess': {'nms_thresh': .5}}), ['postprocess.nms_thresh'])

    def test_nested_checkpoint_rejected(self):
        with self.assertRaises(ValueError):
            normalize_state({'state_dict': {'weight': 1}})

    def test_agent_timestamp_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for agent, stamp in (('1', '000060'), ('2', '000061')):
                folder = root / 'scene' / agent
                folder.mkdir(parents=True)
                (folder / (stamp + '.yaml')).write_text('vehicles: {}')
            with self.assertRaisesRegex(ValueError, 'timestamps differ'):
                scene_inventory(root)


if __name__ == '__main__':
    unittest.main()
