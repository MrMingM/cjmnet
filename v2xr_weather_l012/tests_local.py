"""Local-safe synthetic tests: standard library only; no Torch, OpenCOOD or real data."""
import ast
from collections import OrderedDict
import copy
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch

from auditlib import CONFIG, HERE, ROOT, atomic_json, read_json, sha256
from data import pair_key, paired_inventory
from gates import l0, l1


def official_retrieve():
    path = ROOT / 'v2xr_official_repro/upstream/opencood/data_utils/datasets/opv2v/basedataset.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BaseDataset')
    function = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'retrieve_base_data')
    return function, path


class Points:
    def __getitem__(self, index):
        return self


class LocalTests(unittest.TestCase):
    def test_actual_read_guard_missing_fog_hash_and_io_type(self):
        import data
        with tempfile.TemporaryDirectory(prefix='weather_read_', dir=os.environ.get('WEATHER_TEST_TMP')) as temporary:
            root = Path(temporary)
            cav = root / 'scene' / '1'
            cav.mkdir(parents=True)
            clean, fog = cav / '000001.pcd', cav / '000001_fog_0.060.pcd'
            clean.write_bytes(b'clean fixture')
            fog.write_bytes(b'fog fixture')
            manifest = dict(scenes=[dict(scene='scene', ego='1', agents=['1'], timestamps=['000001'])],
                files=[dict(relative=p.relative_to(root).as_posix(), sha256=sha256(p)) for p in (clean, fog)])
            calls = []
            pcd = ModuleType('opencood.utils.pcd_utils')
            def original_reader(path, **kw):
                self.assertIsInstance(path, str)
                calls.append(path)
                return Points(), 0
            pcd.pcd_to_np = original_reader
            fake_utils = ModuleType('opencood.utils')
            fake_utils.pcd_utils = pcd
            fake_numpy = SimpleNamespace(random=SimpleNamespace(get_state=lambda: [123]))
            class Fixture:
                def __init__(self):
                    self.params = dict(root_dir=str(root))
                    self.scenario_database = {0: OrderedDict([('1', OrderedDict([('000001', {}), ('ego', True)]))])}
                    self.max_cav, self.proj_first = 7, True
                    self.train, self.train_sim, self.sim_weather = True, True, '_fog_0.060'
                    self.data_augmentor = SimpleNamespace(forward=lambda x: x)
                def __len__(self):
                    return 1
                def retrieve_base_data(self, *a, **kw):
                    return None
                def get_item_single_car(self, *a, **kw):
                    return None
                def __getitem__(self, index):
                    pcd.pcd_to_np(str(fog), get_weather=True)
                    return dict(ego=dict(object_ids=['original_gt'], object_bbx_mask=SimpleNamespace(sum=lambda: 1),
                        processed_lidar={}, label_dict={}))
            modules = {'numpy': fake_numpy, 'opencood': ModuleType('opencood'), 'opencood.utils': fake_utils,
                       'opencood.utils.pcd_utils': pcd}
            with patch.dict(sys.modules, modules), patch.object(data, 'tensor_digest', lambda x: 'synthetic'), \
                 patch.object(sys.modules[__name__], 'points_in_boxes_cpu', lambda *a: None, create=True):
                observed = data.AuditedDataset(Fixture(), manifest)
                item = observed[0]
                self.assertEqual(item['_audit']['final_gt_ids'], ['original_gt'])
                self.assertEqual(item['_audit']['reads'][0]['condition'], 'fog')
                fog.unlink()
                with self.assertRaisesRegex(ValueError, 'Missing/empty'):
                    observed[0]
                self.assertEqual(calls, [str(fog)])  # Missing Fog never reaches native IO or Clean fallback.
                fog.write_bytes(b'changed fog')
                with self.assertRaisesRegex(ValueError, 'bytes changed'):
                    observed[0]

    def test_pairs(self):
        manifest = dict(scenes=[dict(scene='scene', agents=['1', '2'], timestamps=['000001', '000002'])],
            files=[dict(relative=f'scene/{c}/{t}{suffix}.pcd', sha256='a'*64)
                for c in ('1', '2') for t in ('000001', '000002') for suffix in ('', '_fog_0.060')])
        pairs = paired_inventory(manifest, Path(tempfile.gettempdir()) / 'weather_fixture')
        self.assertEqual((len(pairs['frames']), len(pairs['pairs'])), (2, 4))
        self.assertEqual(pair_key('scene/2/000001_fog_0.060.pcd'), (('scene', '2', '000001'), 'fog'))
        bad = copy.deepcopy(manifest)
        bad['files'].pop()
        with self.assertRaises(ValueError):
            paired_inventory(bad, Path(tempfile.gettempdir()) / 'weather_fixture')
        for name in ('000001_snow_2.5_2.0.pcd', '000001_radar.pcd'):
            with self.assertRaises(ValueError):
                pair_key('scene/1/' + name)

    def test_execute_original_sampling_and_rng(self):
        # Execute the actual frozen method AST with synthetic IO. No imitation of its sampler.
        function, path = official_retrieve()
        reads = []
        def pcd(file, **kw):
            key, condition = pair_key(file) if kw.get('get_weather') else (None, 'radar')
            if key:
                reads.append((key, condition))
                return Points(), 0
            return Points()
        namespace = dict(OrderedDict=OrderedDict, random=random,
            pcd_utils=SimpleNamespace(pcd_to_np=pcd), load_camera_data=lambda p: [Points()])
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
        scene = OrderedDict((cav, {'000001': dict(lidar=f'scene/{cav}/000001.pcd',
            radar=f'scene/{cav}/000001_radar.pcd', camera0='synthetic'), 'ego': cav == '1'}) for cav in ('1', '2'))
        dummy = SimpleNamespace(len_record=[1], scenario_database={0: scene}, train_sim=True, train=True,
            sim_weather='_fog_0.060', return_timestamp_key=lambda *a: '000001',
            calc_dist_to_ego=lambda *a: scene['1'], time_delay_calculation=lambda *a: 0,
            reform_param=lambda *a: {})
        random.seed(20261017)
        state = random.getstate()
        for _ in range(1000):
            namespace['retrieve_base_data'](dummy, 0)
        first = reads.copy()
        fog = sum(c == 'fog' for _, c in first)
        self.assertTrue(.38 < fog / len(first) < .48)
        self.assertTrue(any(first[i][1] != first[i+1][1] for i in range(0, len(first), 2)))
        reads.clear()
        random.setstate(state)
        for _ in range(1000):
            namespace['retrieve_base_data'](dummy, 0)
        self.assertEqual(reads, first)
        reads.clear()
        dummy.train = False
        namespace['retrieve_base_data'](dummy, 0)
        self.assertTrue(all(c == 'clean' for _, c in reads))

    def test_gate_boundaries(self):
        self.assertFalse(l0(.4921260304638023)['passed'])
        self.assertTrue(l0(.4921260304638024)['passed'])
        self.assertTrue(l0(.4921260304638025)['passed'])
        c = dict(clean=.7, fog=.5, snow=.3)
        self.assertTrue(l1(c, dict(clean=.69, fog=.515, snow=.3))['passed'])
        cases = [dict(clean=.689999, fog=.515, snow=.3), dict(clean=.7, fog=.514999, snow=.3),
                 dict(clean=.699, fog=.515, snow=.299)]
        for d in cases:
            gate = l1(c, d)
            self.assertFalse(gate['passed'])
            self.assertTrue(gate['continue_pipeline'])
        self.assertFalse(l0(.48)['continue_pipeline'])

    def test_stage_commit_resume_and_tamper(self):
        from orchestrator import Pipeline
        with tempfile.TemporaryDirectory(prefix='weather_stage_', dir=os.environ.get('WEATHER_TEST_TMP')) as temporary:
            root = Path(temporary)
            (root / 'logs').mkdir()
            args = SimpleNamespace(output_root=str(root))
            run = dict(identity='synthetic')
            calls = []
            def callback():
                calls.append(1)
                atomic_json(root / 'output/value.json', {'value': 1})
            first = Pipeline(args, run)
            first.stage('unit', root / 'output', callback=callback)
            second = Pipeline(args, run)
            second.stage('unit', root / 'output', callback=callback)
            self.assertEqual(len(calls), 1)
            # Simulated interrupted later stage: no completion marker, prior unit stays committed.
            def broken():
                raise RuntimeError('synthetic interruption')
            with self.assertRaises(RuntimeError):
                second.stage('later', root / 'later', callback=broken)
            self.assertTrue((root / 'stages/later/failure.json').exists())
            third = Pipeline(args, run)
            third.stage('unit', root / 'output', callback=callback)
            third.stage('later', root / 'later', callback=lambda: atomic_json(root / 'later/result.json', {'ok': True}))
            self.assertEqual(len(calls), 1)
            atomic_json(root / 'output/value.json', {'value': 2})
            with self.assertRaises(ValueError):
                Pipeline(args, run).stage('unit', root / 'output', callback=callback)
            with self.assertRaises(ValueError):
                Pipeline(args, {'identity': 'different'}).stage('unit', root / 'output', callback=callback)

    def test_full_state_commit_integrity(self):
        from checkpoint import committed
        with tempfile.TemporaryDirectory(prefix='weather_epoch_', dir=os.environ.get('WEATHER_TEST_TMP')) as temporary:
            root = Path(temporary)
            row = dict(status='complete', epoch=1, contract='abc')
            for key in ('checkpoint', 'resume_state', 'clean_summary', 'trace'):
                file = root / (key + '.fixture')
                file.write_bytes(b'synthetic')
                row[key], row[key + '_sha256'] = str(file), sha256(file)
            atomic_json(root / 'epochs/01/complete.json', row)
            self.assertEqual(len(committed(root, 3, 'abc')), 1)
            with self.assertRaises(ValueError):
                committed(root, 3, 'wrong')
            atomic_json(root / 'epochs/03/complete.json', dict(row, epoch=3))
            with self.assertRaises(ValueError):
                committed(root, 3, 'abc')

    def test_serial_whole_pipeline_and_scientific_stops(self):
        import orchestrator
        import results
        for stop_l0 in (False, True):
            with tempfile.TemporaryDirectory(prefix='weather_workflow_', dir=os.environ.get('WEATHER_TEST_TMP')) as temporary:
                root = Path(temporary)
                (root / 'reports').mkdir()
                calls = []
                c = dict(clean=.7, fog=.48 if stop_l0 else .5, snow=.3)
                d = dict(clean=.699, fog=.51, snow=.299)
                summaries = {v: dict(results={w: dict(AP={'O': {'ap70': ap}}) for w, ap in aps.items()})
                             for v, aps in (('C', c), ('D', d))}
                class FakePipeline:
                    def __init__(self, *args):
                        pass
                    def stage(self, name, output, cmd=None, callback=None):
                        calls.append(name)
                        if name in ('C_evaluate', 'D_evaluate'):
                            atomic_json(Path(output) / 'summary.json', summaries[name[0]])
                        if callback:
                            callback()
                args = SimpleNamespace(output_root=str(root), mode='all')
                with patch.object(orchestrator, 'Pipeline', FakePipeline), \
                     patch.object(orchestrator, 'command', lambda *a, **kw: ['synthetic']), \
                     patch.object(orchestrator, 'checkpoint', lambda *a, **kw: 'fixture.pth'), \
                     patch.object(orchestrator, 'diagnosis', lambda *a: None), \
                     patch.object(results, 'result', lambda root, variant: summaries[variant]), \
                     patch.object(results, 'pairing', lambda root: {'synthetic': True}), \
                     patch.object(results, 'final', lambda *a, **kw: calls.append('report')):
                    code = orchestrator.workflow(args, dict(identity='synthetic'))
                if stop_l0:
                    self.assertEqual(code, 20)
                    self.assertNotIn('D_train', calls)
                    self.assertNotIn('C_BN_adapt', calls)
                    self.assertEqual(read_json(root / 'STOP.json')['status'], 'L0_GATE_FAILED')
                else:
                    self.assertEqual(code, 0)
                    self.assertEqual(read_json(root / 'reports/L1_GATE.json')['status'], 'L1_GATE_FAILED')
                    self.assertLess(calls.index('C_train'), calls.index('D_train'))
                    self.assertLess(calls.index('D_evaluate'), calls.index('C_BN_adapt'))
                    self.assertLess(calls.index('D_BN_evaluate'), calls.index('C_FiLM_adapt'))
                    self.assertIn('D_FiLM_evaluate', calls)
                    self.assertEqual(calls[-2:], ['final', 'report'])

    def test_gt_and_evaluation_call_chain_static(self):
        source = (HERE / 'data.py').read_text(encoding='utf-8')
        self.assertIn("item = self.dataset[index]", source)
        self.assertNotIn('generate_label(', source)
        self.assertNotIn('points_in_boxes_cpu(', source)
        evaluation = (HERE / 'evaluation.py').read_text(encoding='utf-8')
        self.assertIn('inherited.evaluate(', evaluation)
        frozen = (ROOT / 'v2xr_protocol_audit/evaluate.py').read_text(encoding='utf-8')
        self.assertIn("identity(stat['O']) != identity(original)", frozen)
        self.assertIn('eval_utils_opv2v.calculate_ap(', frozen)
        self.assertEqual(CONFIG['film']['epochs'], 3)
        self.assertEqual(CONFIG['bn']['frames'], 8083)

    def test_shell_contract_and_quoted_paths(self):
        shell = (HERE / 'run_all.sh').read_bytes()
        self.assertNotIn(b'\r', shell)
        self.assertNotIn(b'pipefail', shell)
        self.assertNotIn(b'[[', shell)
        self.assertIn(b'unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES', shell)
        self.assertIn(b'exec "$@"', shell)
        self.assertIn(b'"$SCRIPT_DIR/orchestrator.py"', shell)
        if sys.platform == 'linux':
            subprocess.run(['sh', '-n', str(HERE / 'run_all.sh')], check=True)
            with tempfile.TemporaryDirectory(prefix='weather_shell_') as temporary:
                subprocess.run(['sh', str(HERE / 'test_shell.sh'), temporary, str(HERE / 'run_all.sh')], check=True)


def run():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(LocalTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    return result.testsRun


if __name__ == '__main__':
    run()
