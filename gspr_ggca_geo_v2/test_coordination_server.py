"""Server-only coordinator checks; helpers are our own short-lived processes."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import coordinator as c
from a1_lane import A1Lane, process
from auditlib import DEFAULTS, atomic_json


class CoordinationTests(unittest.TestCase):
    def test_commands_preserve_old_A1_paths_and_separate_test_gpu(self):
        args=types.SimpleNamespace(**DEFAULTS);args.test_gpu='1'
        old=dict(old_source='/home/cjm/OpenCOOD-main/cjmnet/gspr_ggca',
                 old_root='/home/cjm/datasets/logs/gspr-ggca-v3')
        with patch.object(c,'plan',return_value=old):
            command=c.command(args,'train',Path(old['old_root'])/'A1','A1',old=True)
            self.assertEqual(command[2],old['old_source']+'/main.py')
            self.assertEqual(command[command.index('--experiment-root')+1],old['old_root'])
            self.assertEqual(command[command.index('--output-root')+1],old['old_root']+'/A1')
            self.assertNotIn('/usr/bin/env',command)
            new=c.command(args,'train',Path(args.output_root)/'smoke/A2','A2',smoke=True)
            self.assertEqual(new[:2],['/usr/bin/env','ROCR_VISIBLE_DEVICES=1'])

    def test_adoption_uses_pid_start_time_and_exact_command_without_respawn(self):
        with tempfile.TemporaryDirectory() as tmp:
            command=[sys.executable,'-c','import time; time.sleep(30)']
            child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,start_new_session=True)
            try:
                obj=object.__new__(A1Lane);obj.command=command;obj.folder=Path(tmp)
                row=dict(pid=child.pid,start=process(child.pid)['start'],status='running')
                self.assertTrue(obj.alive(row))
                self.assertFalse(obj.alive(dict(row,start='invalid')))
                atomic_json(obj.folder/'state.json',row)
                with patch.object(obj,'finished',return_value=False),patch('a1_lane.subprocess.Popen') as spawn:
                    obj.ensure();spawn.assert_not_called()
                obj.command=command+['different']
                self.assertFalse(obj.alive(row))
            finally:
                child.terminate();child.wait(timeout=10)

    def test_free_extra_card_is_selected(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,ROCR_VISIBLE_DEVICES='2',GGCA_TEST_GPU='1'):
            with patch.object(c.subprocess,'run',return_value=types.SimpleNamespace(stdout='vram Total Used Memory (MiB): 2')):
                self.assertEqual(c.test_device(Path(tmp)),'1')

    def test_busy_extra_card_uses_sequential_formal_card(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,ROCR_VISIBLE_DEVICES='2',GGCA_TEST_GPU='1'):
            with patch.object(c.subprocess,'run',return_value=types.SimpleNamespace(stdout='vram Total Used Memory (MiB): 16000')):
                self.assertEqual(c.test_device(Path(tmp)),'2')

    def test_busy_test_card_never_gets_claimed(self):
        with patch.object(c.subprocess,'run',return_value=types.SimpleNamespace(stdout='vram Total Used Memory (MiB): 16000')), patch.object(c.time,'sleep'):
            with self.assertRaisesRegex(RuntimeError,'occupied'):
                c.require_free_test_device('1')


if __name__=='__main__':
    if sys.platform!='linux':raise RuntimeError('Run coordinator tests on Haiguang')
    unittest.main()
