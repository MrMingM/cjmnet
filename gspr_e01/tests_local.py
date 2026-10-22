"""Synthetic tests only. Never imports Torch, OpenCOOD, a checkpoint or point cloud."""
import ast
import copy
import json
from pathlib import Path
import random
import pickle
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch
from adapters import pair_key, retrieve_tree, select_path, training_tree
from e01_common import HERE, PROJECT, SETTINGS, atomic, finish, frozen_identity, gpu_env, identity, read, receipt, safe, sha
from e1_data import compare_epoch
from orchestrator import Controller, run_branches
from reports import comparisons, delta, e1_report, protocol_match, screen, synthesis, target_changes, validate_summary


class Synthetic(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='.e01_synthetic_', dir=PROJECT)
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_real_native_AST_location_and_rng_call(self):
        source = (PROJECT / 'v2xr_official_repro/upstream/opencood/data_utils/datasets/opv2v/basedataset.py').read_text(encoding='utf-8')
        tree = retrieve_tree(source)
        self.assertEqual(sum(isinstance(n, ast.Call) and ast.unparse(n.func) == 'random.randint' for n in ast.walk(tree)), 1)
        self.assertEqual(sum(isinstance(n, ast.Call) and ast.unparse(n.func) == '_e01_select_path' for n in ast.walk(tree)), 1)

    def test_actual_training_AST_only_registered_edits(self):
        original = (PROJECT / 'v2xr_weather_l012/training.py').read_text(encoding='utf-8')
        result, edits = training_tree(original)
        compile(result, 'synthetic_compilation_of_frozen_training', 'exec')
        self.assertEqual(edits, dict(epochs=3, subset=2, weather=1))
        self.assertIn('scheduler.step(epoch)', ast.unparse(result))
        self.assertIn('optimizer.step()', ast.unparse(result))
        self.assertIn('loss_fn(prediction, batch[\'ego\'][\'label_dict\'])', ast.unparse(result))

    def test_native_AST_change_is_rejected(self):
        source = (PROJECT / 'v2xr_official_repro/upstream/opencood/data_utils/datasets/opv2v/basedataset.py').read_text(encoding='utf-8')
        with self.assertRaises(ValueError):
            retrieve_tree(source.replace('random.randint(0, 6)', 'random.randint(0, 7)'))

    def test_training_AST_change_is_rejected(self):
        source = (PROJECT / 'v2xr_weather_l012/training.py').read_text(encoding='utf-8')
        with self.assertRaises(ValueError):
            training_tree(source.replace('dict(clean=0, fog=0)', 'dict(clean=1, fog=0)'))

    def test_recipe_counts_exact(self):
        for recipe in ('T1', 'T2'):
            values = [pair_key(select_path('/train/s/c/0001.pcd', n, recipe))[1] for n in range(7)]
            self.assertEqual([values.count(w) for w in ('clean', 'fog', 'snow')], SETTINGS['recipes'][recipe])

    def test_paths_use_exact_suffixes(self):
        for path, condition in (('100.pcd', 'clean'), ('100_fog_0.060.pcd', 'fog'), ('100_snow_2.5_2.0.pcd', 'snow')):
            self.assertEqual(pair_key('/train/scene/cav/'+path), (('scene', 'cav', '100'), condition))

    def test_wrong_suffix_and_radar_rejected(self):
        for name in ('100_snow_3.pcd', '100_radar.pcd', 'x_y.pcd'):
            with self.assertRaises(ValueError):
                pair_key('/train/scene/cav/'+name)

    def test_sampling_does_not_advance_global_rng(self):
        old = random.getstate()
        select_path('/train/s/c/001.pcd', 2, 'T2')
        self.assertEqual(old, random.getstate())

    def test_per_CAV_draws_not_frame_wide(self):
        source = "class BaseDataset:\n    def retrieve_base_data(self, paths):\n        out=[]\n        for lidar_file in paths:\n            if self.train_sim and self.train:\n                aug = random.randint(0, 6)\n                if aug <= 2:\n                    lidar_file = lidar_file.replace('.pcd', self.sim_weather + '.pcd')\n            out.append(lidar_file)\n        return out\n"
        draws = iter((0, 3, 6))
        namespace = dict(random=types.SimpleNamespace(randint=lambda a, b: next(draws)), _e01_select_path=select_path)
        exec(compile(retrieve_tree(source), 'synthetic', 'exec'), namespace)
        obj = types.SimpleNamespace(train_sim=True, train=True, _e01_recipe='T1')
        result = namespace['retrieve_base_data'](obj, ['/train/s/a/1.pcd', '/train/s/b/1.pcd', '/train/s/c/1.pcd'])
        self.assertEqual([pair_key(p)[1] for p in result], ['fog', 'snow', 'clean'])
        obj.train = False
        self.assertEqual(namespace['retrieve_base_data'](obj, ['/train/s/a/1.pcd']), ['/train/s/a/1.pcd'])

    def test_identity_independent_of_dict_order(self):
        self.assertEqual(identity(dict(a=1, b=2)), identity(dict(b=2, a=1)))

    def test_frozen_training_contract_hash_keeps_its_original_serialization(self):
        value = dict(recipe='T1', nested=dict(a=1, b=2))
        expected = __import__('hashlib').sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        self.assertEqual(frozen_identity(value), expected)
        self.assertNotEqual(frozen_identity(value), identity(value))

    def test_immutable_rejects_changes(self):
        path = self.root / 'request.json'
        atomic(path, dict(a=1), immutable=True)
        with self.assertRaises(ValueError):
            atomic(path, dict(a=2), immutable=True)

    def test_partial_directory_not_complete(self):
        atomic(self.root / 'partial.json', dict(a=1))
        self.assertIsNone(receipt(self.root, 'id'))

    def test_receipt_checks_nested_epoch_markers(self):
        path = self.root / 'epochs/01/complete.json'
        atomic(path, dict(status='complete'))
        finish(self.root, 'id')
        self.assertTrue(receipt(self.root, 'id'))
        atomic(path, dict(status='failed'))
        with self.assertRaises(ValueError):
            receipt(self.root, 'id')

    def test_resume_source_identity_rejected(self):
        atomic(self.root / 'a.json', {})
        finish(self.root, 'id')
        with self.assertRaises(ValueError):
            receipt(self.root, 'changed')

    def test_hash_tamper_rejected(self):
        path = self.root / 'prediction.json'
        atomic(path, dict(boxes=[]))
        finish(self.root, 'id')
        atomic(path, dict(boxes=[1]))
        with self.assertRaises(ValueError):
            receipt(self.root, 'id')

    def test_locks_not_committed(self):
        (self.root / '.worker.lock').write_text('1')
        atomic(self.root / 'result.json', {})
        finish(self.root, 'id')
        (self.root / '.worker.lock').write_text('2')
        self.assertTrue(receipt(self.root, 'id'))

    def test_test_partition_refused(self):
        with self.assertRaises(ValueError):
            safe(self.root / 'test/cloud.pcd')

    def traces(self):
        row = dict(index=0, frame_key=['scene', '1'], reads=[], gt_calls=[], final_gt_ids=[], native_point_filter=[],
            configured_world_augmentation_calls=0, final_gt_count=0, rng_before='a', rng_after='b',
            input_sha256='input', worker=dict(id=0, seed=1), recipe='T1',
            requested_frames_including_empty_GT_retries=[0], zero_point_GT_boxes=0,
            read_CAV_count_including_retries=0, empty_GT_frames_resampled=0,
            retained_CAV_count=1, retained_CAV_order=['ego'])
        a, b = self.root / 'C.jsonl', self.root / 'D.jsonl'
        for p in (a, b):
            p.write_text(json.dumps([row])+'\n', encoding='utf-8')
        return a, b, row

    def test_pairing_requires_all_fields_and_counts(self):
        a, b, _ = self.traces()
        self.assertTrue(compare_epoch(a, b, 1)['exact'])
        with self.assertRaises(ValueError):
            compare_epoch(a, b, 8)

    def test_worker_rng_mismatch_blocks_pairing(self):
        a, b, row = self.traces()
        row['worker']['seed'] = 2
        b.write_text(json.dumps([row])+'\n', encoding='utf-8')
        with self.assertRaises(ValueError):
            compare_epoch(a, b, 1)

    def test_tensor_mismatch_blocks_pairing(self):
        a, b, row = self.traces()
        row['input_sha256'] = 'different'
        b.write_text(json.dumps([row])+'\n', encoding='utf-8')
        with self.assertRaises(ValueError):
            compare_epoch(a, b, 1)

    def values(self, offset=0):
        return {w: dict(AP={p: {m: .5+offset for m in ('ap30', 'ap50', 'ap70')} for p in ('O', 'O*', 'F')}) for w in ('clean', 'fog', 'snow')}

    def test_differences_use_pp_full_precision(self):
        self.assertAlmostEqual(delta(self.values(.0123456789), self.values())['fog']['F']['ap70'], 1.23456789)

    def test_gain_change_formula(self):
        values = {t+'-'+v: self.values(x) for t, v, x in
            (('T0', 'C', 0), ('T0', 'D', .01), ('T1', 'C', .02), ('T1', 'D', .04), ('T2', 'C', .03), ('T2', 'D', .035))}
        result = comparisons(values)
        self.assertAlmostEqual(result['gain_change_pp']['T1']['snow']['F']['ap70'], 1)
        self.assertAlmostEqual(result['gain_change_pp']['T2']['snow']['F']['ap70'], -.5)

    def test_screening_fixed_and_independent(self):
        values = {t+'-'+v: self.values(.03 if t == 'T2' and v == 'D' else 0) for t in ('T0', 'T1', 'T2') for v in ('C', 'D')}
        result = screen(values)
        self.assertFalse(result['recipes']['T1']['all_passed'])
        self.assertTrue(result['recipes']['T2']['all_passed'])

    def test_protocol_no_silent_missing_fields(self):
        with self.assertRaises(ValueError):
            protocol_match({}, {})

    def test_protocol_checks_weather_frames_and_seed(self):
        keys = ('official_source_hashes', 'official_config_sha256', 'input_manifest_identity', 'frame_keys',
            'protocols', 'inference_seed', 'PYTHONHASHSEED', 'loader_workers', 'versions', 'HIP', 'torch_backend', 'frozen_d1c_gt_identity')
        a = {k: k for k in keys}
        self.assertTrue(protocol_match(a, a))
        b = dict(a, inference_seed=0)
        with self.assertRaises(ValueError):
            protocol_match(a, b)

    def test_target_changes_are_from_match_ids(self):
        for folder, ids in (('A', ['x', 'z']), ('B', ['x', 'y'])):
            p = self.root / folder
            p.mkdir()
            lines = []
            for w in ('clean', 'fog', 'snow'):
                lines.append(dict(condition=w, index=0, frame_key=['s', '1'], fixed_gt_count=3,
                    protocols=dict(F=dict(matched_pairs={t: [dict(object_id=i) for i in ids] for t in ('0.3', '0.5', '0.7')}))))
            (p / 'per_frame_metrics.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in lines), encoding='utf-8')
        result = target_changes(self.root / 'A', self.root / 'B', 1)['snow']['0.7']
        self.assertEqual((result['newly_detected'], result['lost'], result['net_TP']), (1, 1, 0))

    def test_serial_E0_failure_still_runs_E1(self):
        observed = []
        def fail():
            raise RuntimeError('synthetic E0 error')
        errors = run_branches('serial', fail, lambda: observed.append('E1 done'))
        self.assertEqual(observed, ['E1 done'])
        self.assertIn('E0', errors)

    def test_parallel_waits_for_E1_after_E0_failure(self):
        barrier = threading.Barrier(2)
        observed = []
        def fail():
            barrier.wait(timeout=5)
            raise RuntimeError('synthetic E0 error')
        def e1():
            barrier.wait(timeout=5)
            observed.append('E1 done')
        errors = run_branches('parallel', fail, e1)
        self.assertEqual(observed, ['E1 done'])
        self.assertIn('E0', errors)

    def test_T2_runs_after_scientific_T1_result(self):
        cli = types.SimpleNamespace(output_root=self.root)
        controller = Controller(cli)
        planned = []
        def stage(name, command, output, gpu):
            planned.append(name)
            if name.endswith('_train'):
                model = Path(command[command.index('--model-root')+1])
                atomic(model / 'selected.json', dict(checkpoint='synthetic.pth'))
        controller.stage = stage
        controller.cli.train_root = self.root / 'train'
        controller.cli.original_root = self.root / 'original'
        controller.models(self.root / 'E1', '2', smoke=False)
        self.assertEqual([x for x in planned if x.endswith('_train')], ['T1_C_train', 'T1_D_train', 'T2_C_train', 'T2_D_train'])

    def test_smoke_names_and_output_isolation(self):
        source = (HERE / 'orchestrator.py').read_text(encoding='utf-8')
        self.assertIn("self.root / 'smoke/E1'", source)
        self.assertIn("self.root / 'E1'", source)
        self.assertNotIn('if AP70', source)

    def test_ambiguous_GPU_aliases_rejected(self):
        with self.assertRaises(ValueError):
            gpu_env('02')
        env = gpu_env('2')
        self.assertEqual(env['ROCR_VISIBLE_DEVICES'], '2')
        self.assertNotIn('HIP_VISIBLE_DEVICES', env)
        self.assertNotIn('CUDA_VISIBLE_DEVICES', env)

    def test_clean_eval_lock_does_not_conflict_with_waiting_trainer(self):
        from worker import worker_lock_path
        cli = types.SimpleNamespace(stage='train', model_root=self.root,
            destination=self.root / 'epochs/01/clean_ap', checkpoint=self.root / 'synthetic.txt')
        trainer_lock = worker_lock_path(cli)
        cli.stage = 'evaluate'
        evaluation_lock = worker_lock_path(cli)
        self.assertNotEqual(trainer_lock, evaluation_lock)
        self.assertNotIn(cli.destination, evaluation_lock.parents)

    def test_geometry_observer_accepts_frozen_smoke_subset(self):
        from e1_targets import native_dataset
        native = types.SimpleNamespace(get_item_single_car=lambda: None)
        subset = types.SimpleNamespace(dataset=native, indices=range(12))
        self.assertIs(native_dataset(native), native)
        self.assertIs(native_dataset(subset), native)
        subset.indices = [2, 1]
        with self.assertRaises(ValueError):
            native_dataset(subset)

    def synthetic_restore(self, drop_optimizer=False):
        # A stdlib-only state machine exercises the wrapper's argument order and exact checks.
        # This stub is not PyTorch and never reads a real .pth.
        from e1_runtime import install_restore_check
        # Import before patch.dict restores sys.modules, avoiding re-importing NumPy extensions.
        import numpy as np
        self.assertTrue(hasattr(np, 'ndarray'))
        model, optimizer, scheduler, criterion = dict(w=1.0), dict(momentum=.3), dict(epoch=0), dict(counter=2)
        def state(m, o, s, c, epoch, contract):
            return copy.deepcopy(dict(model=m, optimizer=o, scheduler=s, criterion=c,
                rng=random.getstate(), epoch=epoch, contract=contract))
        def load(path, **kw):
            with Path(path).open('rb') as stream:
                return pickle.load(stream)
        def save(path, value):
            with Path(path).open('wb') as stream:
                pickle.dump(value, stream)
        def restore(path, m, o, s, c, contract, epoch):
            value = load(path)
            if value['epoch'] != epoch or value['contract'] != contract:
                raise ValueError('Synthetic restore arguments reversed')
            for dest, key in ((m, 'model'), (o, 'optimizer'), (s, 'scheduler'), (c, 'criterion')):
                dest.clear()
                dest.update(value[key])
            if drop_optimizer:
                o.clear()
            random.setstate(value['rng'])
        fake = types.SimpleNamespace(Tensor=type('NeverInstantiatedTensorStub', (), {}), load=load)
        module = types.SimpleNamespace(state=state, save=save, restore=restore)
        with patch.dict(sys.modules, torch=fake):
            install_restore_check(module, self.root, smoke=True)
            value = module.state(model, optimizer, scheduler, criterion, 0, 'synthetic_contract')
            module.save(self.root / 'synthetic_full_state.bin', value)
        return read(self.root / 'restore_audit/00.json')

    def test_actual_restore_wrapper_keeps_optimizer_scheduler_rng_arguments(self):
        row = self.synthetic_restore()
        self.assertTrue(row['exact_model_Adam_scheduler_criterion_RNG'])

    def test_weights_only_stub_is_rejected_by_restore_wrapper(self):
        with self.assertRaises(ValueError):
            self.synthetic_restore(drop_optimizer=True)

    def smoke_fixture(self):
        # Artificial files only; none is a real checkpoint, inference cache or scientific result.
        from reports import WEATHERS, PROTOCOLS, METRICS
        root = self.root / 'fixture'
        for t in ('T1', 'T2'):
            for v in ('C', 'D'):
                model = root / t / v
                model.mkdir(parents=True)
                (model / 'config.yaml').write_text('synthetic: true\n', encoding='utf-8')
                contract = dict(variant=v, epochs=20, smoke_only=True,
                    settings=dict(e01=dict(recipe=t)), config_sha256=sha(model / 'config.yaml'))
                contract['identity'] = frozen_identity(contract)
                atomic(model / 'contract.json', contract)
                rows = []
                for i in (1, 2):
                    directory = model / 'epochs' / f'{i:02d}'
                    directory.mkdir(parents=True)
                    artifacts = {}
                    for name in ('checkpoint', 'resume_state', 'trace'):
                        path = directory / (name+'.txt')
                        path.write_text('synthetic '+name+str(i), encoding='utf-8')
                        artifacts[name], artifacts[name+'_sha256'] = str(path), sha(path)
                    path = directory / 'clean_summary.json'
                    atomic(path, dict(status='complete', conditions=['clean'], frame_count=12,
                        checkpoint_sha256=artifacts['checkpoint_sha256'], results=dict(clean=dict(AP=dict(O=dict(ap70=.5))))))
                    artifacts.update(clean_summary=str(path), clean_summary_sha256=sha(path))
                    row = dict(**artifacts, epoch=i, contract=contract['identity'], AP70=.5,
                        sampling_counts=dict(clean=8, fog=8, snow=4), weather_fractions=dict(clean=.4, fog=.4, snow=.2),
                        gt_total=10, official_resampling_attempts=0, train_updates=4)
                    atomic(directory / 'complete.json', row)
                    rows.append(row)
                    if v == 'D':
                        peer = read(root / t / 'C/epochs' / f'{i:02d}/complete.json')
                        atomic(model / 'pairing' / f'{i:02d}.json', dict(exact=True, consumed_frames=32,
                            D_trace_sha256=row['trace_sha256'], C_trace_sha256=peer['trace_sha256']))
                atomic(model / 'smoke_selected.json', dict(epoch=1, checkpoint=rows[0]['checkpoint'],
                    checkpoint_sha256=rows[0]['checkpoint_sha256'], eligible_for_formal_selection=False))
                atomic(model / 'initialization_audit.json', dict(exact=True))
                for i in (0, 1, 2):
                    atomic(model / 'restore_audit' / f'{i:02d}.json', dict(exact_model_Adam_scheduler_criterion_RNG=True))
                destination = model / 'final_evaluation'
                protocol_keys = ('official_source_hashes', 'official_config_sha256', 'input_manifest_identity',
                    'frame_keys', 'protocols', 'inference_seed', 'PYTHONHASHSEED', 'loader_workers', 'versions',
                    'HIP', 'torch_backend', 'frozen_d1c_gt_identity')
                atomic(destination / 'protocol.json', {k: 'synthetic' for k in protocol_keys})
                gt, predictions, frames = [], [], []
                for w in WEATHERS:
                    for i in range(12):
                        path = destination / 'predictions' / w / f'{i:06d}.json'
                        atomic(path, dict(condition=w, index=i, frame_key=['s', str(i)]))
                        predictions.append(dict(condition=w, index=i, path=str(path), sha256=sha(path)))
                        gt.append(dict(condition=w, index=i))
                        frames.append(dict(condition=w, index=i, frame_key=['s', str(i)], fixed_gt_count=1,
                            protocols={p: dict(matched_pairs={str(th): [dict(object_id='a')] for th in (.3, .5, .7)}) for p in PROTOCOLS}))
                atomic(destination / 'gt_manifest.json', dict(frames=gt))
                atomic(destination / 'prediction_manifest.json', dict(files=predictions))
                (destination / 'per_frame_metrics.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in frames), encoding='utf-8')
                result = dict(status='complete', frame_count=12, full_validation=False, conditions=list(WEATHERS),
                    checkpoint_sha256=rows[0]['checkpoint_sha256'],
                    results={w: dict(AP={p: {m: .5 for m in METRICS} for p in PROTOCOLS}) for w in WEATHERS},
                    output_hashes={n: sha(destination / n) for n in ('gt_manifest.json', 'prediction_manifest.json', 'protocol.json', 'per_frame_metrics.jsonl')})
                atomic(destination / 'summary.json', result)
                atomic(model / 'target_analysis.json', dict(status='complete', synthetic=True))
        return root

    def test_complete_smoke_report_is_not_a_scientific_result(self):
        root = self.smoke_fixture()
        e1_report(root, self.root / 'unused_original', smoke=True)
        report = read(root / 'E1_RESULTS.json')
        self.assertFalse(report['eligible_for_scientific_conclusions'])
        self.assertEqual(len(report['results']), 4)
        self.assertIn('Smoke', (root / 'E1_RESULTS.md').read_text(encoding='utf-8'))
        self.assertEqual(read(root / 'E1_MATCHED_COMPARISON.json')['status'], 'smoke_only')

    def test_missing_last_epoch_cannot_produce_report(self):
        root = self.smoke_fixture()
        (root / 'T2/D/epochs/02/complete.json').unlink()
        with self.assertRaises(FileNotFoundError):
            e1_report(root, self.root / 'unused_original', smoke=True)
        self.assertFalse((root / 'E1_RESULTS.json').exists())

    def test_selection_ties_must_choose_earliest(self):
        root = self.smoke_fixture()
        path = root / 'T1/C/smoke_selected.json'
        row = read(path)
        row['epoch'] = 2
        atomic(path, row)
        with self.assertRaises(ValueError):
            e1_report(root, self.root / 'unused_original', smoke=True)

    def test_incomplete_cache_rejected_even_with_updated_manifest_hash(self):
        root = self.smoke_fixture()
        destination = root / 'T1/C/final_evaluation'
        path = destination / 'prediction_manifest.json'
        row = read(path)
        row['files'].pop()
        atomic(path, row)
        summary = read(destination / 'summary.json')
        summary['output_hashes']['prediction_manifest.json'] = sha(path)
        atomic(destination / 'summary.json', summary)
        with self.assertRaises(ValueError):
            validate_summary(destination / 'summary.json', 12)

    def test_synthesis_rejects_smoke(self):
        atomic(self.root / 'E0/E0_RESULTS.json', dict(status='complete'))
        atomic(self.root / 'E1/E1_RESULTS.json', dict(status='complete', eligible_for_scientific_conclusions=False))
        with self.assertRaises(ValueError):
            synthesis(self.root)
        self.assertFalse((self.root / 'reports/E01_FINAL_CONCLUSIONS.md').exists())

    def test_final_report_serializes_complete_synthetic_evidence(self):
        values = {t+'-'+v: self.values() for t in ('T0', 'T1', 'T2') for v in ('C', 'D')}
        atomic(self.root / 'E0/E0_RESULTS.json', dict(status='complete'))
        atomic(self.root / 'E1/E1_RESULTS.json', dict(status='complete', eligible_for_scientific_conclusions=True, results=values))
        atomic(self.root / 'E0/E0_INTERVENTIONS.json', dict(delta_from_R0_pp={a: {w: dict(F=dict(ap70=0))
            for w in ('clean', 'fog', 'snow')} for a in ('R1', 'R2', 'R3')}))
        atomic(self.root / 'E0/E0_POINT_DEGRADATION.json', dict(aggregate=dict(snow=dict(
            zero_retained_point_targets=0, fixed_targets=1, missing_clean_cells=0, clean_occupied_cells=1))))
        atomic(self.root / 'E0/E0_TARGET_ANALYSIS.json', dict(weather=dict(snow=dict(counts={'0.7': dict(recovered=0, lost=0, D_tp=1, C_tp=1)}))))
        atomic(self.root / 'E1/E1_WEATHER_TRADEOFF.json', screen(values))
        atomic(self.root / 'E1/E1_TRAINING_AUDIT.json', dict(status='complete', synthetic=True))
        atomic(self.root / 'E1/E1_TARGET_ANALYSIS.json', dict(status='complete', synthetic=True))
        synthesis(self.root)
        text = (self.root / 'reports/E01_FINAL_CONCLUSIONS.md').read_text(encoding='utf-8')
        self.assertIn('AP30/AP50', text)
        self.assertIn('尚不足以确定', text)
        self.assertFalse(read(self.root / 'reports/E01_EVIDENCE.json')['E2_E3_implemented'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
