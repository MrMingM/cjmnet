"""Local-safe synthetic checks. Never import Torch/OpenCOOD or read real datasets/weights."""
import ast
import copy
from contextlib import nullcontext
import json
import os
import re
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from v2xr_author_fog_eval import checkpoint as cp
from v2xr_author_fog_eval import comparison as cmp
from v2xr_author_fog_eval.core import (CHECKPOINT_SHA, FROZEN_GT, HERE, ROOT, bind,
    completed, identity, read, reference_source, sha256, verify_sources, write)
from v2xr_author_fog_eval.main import parser, workflow


def original_functions(path, names, namespace):
    """Compile only named original functions; discard imports that require server libraries."""
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in selected} != set(names):
        raise ValueError('Frozen function interface changed')
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


class Tensor:
    def __init__(self, value, dtype='float32'):
        self.data = np.asarray(value, dtype=dtype)
        self.shape, self.dtype = self.data.shape, self.data.dtype

    def is_floating_point(self):
        return self.dtype.kind == 'f'

    def is_complex(self):
        return self.dtype.kind == 'c'

    def detach(self):
        return self

    def cpu(self):
        return self

    def requires_grad_(self, value):
        self.requires_grad = value


class Model:
    def __init__(self, state, corrupt=False):
        self.state = state
        self.strict_calls = []
        self.corrupt = corrupt

    def state_dict(self):
        return self.state

    def load_state_dict(self, state, strict):
        self.strict_calls.append(strict)
        self.state = copy.deepcopy(state)
        if self.corrupt:
            self.state['head'].data.flat[0] += 1

    def eval(self):
        self.evaluation = True

    def parameters(self):
        return self.state.values()


FAKE_TORCH = SimpleNamespace(no_grad=nullcontext,
    isfinite=lambda t: np.isfinite(t.data), equal=lambda a, b: np.array_equal(a.data, b.data),
    is_tensor=lambda t: isinstance(t, Tensor))


def temporary():
    # Windows sandbox cannot atomically rename files in its redirected system temp.
    # Keep synthetic files in a verified workspace child; Linux uses exported /data TMPDIR.
    if sys.platform == 'win32':
        result = tempfile.TemporaryDirectory(dir=ROOT, prefix='.author_eval_synthetic_')
        target = Path(result.name).resolve()
        if target.parent != ROOT.resolve() or not target.name.startswith('.author_eval_synthetic_'):
            raise ValueError('Invalid synthetic temporary directory')
        return result
    return tempfile.TemporaryDirectory(prefix='author_eval_synthetic_')


class Loading(unittest.TestCase):
    def setUp(self):
        self.state = dict(head=Tensor([[.1, .2]]), count=Tensor(3, 'int64'))

    def test_safe_read_never_requests_unsafe_pickle(self):
        seen = []
        fake = SimpleNamespace(is_tensor=FAKE_TORCH.is_tensor,
            load=lambda path, **kwargs: (seen.append(kwargs) or self.state))
        cp.safe_read(fake, 'synthetic.pth')
        self.assertEqual(seen, [dict(map_location='cpu', weights_only=True)])

    def test_wrapper_and_uniform_prefix_are_naming_only(self):
        state, audit = cp.unpack(dict(state_dict={'module.' + k: v for k, v in self.state.items()}, epoch=1), FAKE_TORCH.is_tensor)
        self.assertEqual(state, self.state)
        self.assertTrue(audit['module_prefix_removed'])
        self.assertEqual(audit['unconsumed_wrapper_fields'], ['epoch'])

    def test_unknown_wrapper_mixed_prefix_and_nonstandard_names_fail(self):
        for raw in ({'model': self.state}, {'module.head': Tensor(1), 'count': Tensor(1)},
                    {'state_dict': self.state, 'model_state_dict': self.state}):
            with self.assertRaises(ValueError):
                cp.unpack(raw, FAKE_TORCH.is_tensor)

    def test_strict_complete_loading_and_every_tensor_equality(self):
        model = Model(copy.deepcopy(self.state))
        audit = {}
        cp.strict_load(model, self.state, FAKE_TORCH, audit)
        self.assertEqual(model.strict_calls, [True])
        self.assertTrue(audit['all_tensors_loaded_exactly'])
        self.assertTrue(model.evaluation)
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))

    def test_missing_head_shape_dtype_nan_and_corruption_stop(self):
        for state in (dict(count=self.state['count']), dict(self.state, head=Tensor([1])),
                      dict(self.state, head=Tensor([[.1, .2]], 'float64')),
                      dict(self.state, head=Tensor([[float('nan'), .2]]))):
            with self.assertRaises(ValueError):
                cp.strict_load(Model(copy.deepcopy(self.state)), state, FAKE_TORCH, {})
        with self.assertRaises(ValueError):
            cp.strict_load(Model(copy.deepcopy(self.state), corrupt=True), self.state, FAKE_TORCH, {})


class Protocols(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Toy identity-IoU matrices isolate AP and one-to-one bookkeeping. No real geometry,
        # Torch, checkpoint or dataset is loaded. Server tests cover actual polygons/tensors.
        common = SimpleNamespace(torch_tensor_to_numpy=lambda t: t,
            convert_format=lambda boxes: [int(x[0, 0]) for x in boxes],
            compute_iou=lambda box, targets: np.array([float(box == t) for t in targets]))
        cls.api = original_functions(ROOT / 'v2xr_official_repro/upstream/opencood/utils/eval_utils_opv2v.py',
            ('voc_ap', 'caluclate_tp_fp', 'calculate_ap'), dict(np=np, common_utils=common))

    def stat(self):
        return {.7: dict(tp=[], fp=[], score=[], gt=0)}

    def boxes(self, *ids):
        return np.array([np.full((8, 3), i) for i in ids]).reshape(-1, 8, 3)

    def test_three_protocols_share_predictions_none_denominators(self):
        result = {}
        for protocol in ('O', 'O*', 'F'):
            stat = self.stat()
            self.api['caluclate_tp_fp'](self.boxes(1), np.array([.9]), self.boxes(1, 2) if protocol == 'F' else self.boxes(1), stat, .7)
            if protocol != 'O':
                self.api['caluclate_tp_fp'](None, None, self.boxes(3), stat, .7)
            result[protocol] = self.api['calculate_ap'](copy.deepcopy(stat), .7, False)[0]
        self.assertEqual(result, {'O': 1., 'O*': .5, 'F': 1/3})

    def test_empty_is_counted_and_duplicate_predictions_cannot_reuse_gt(self):
        stat = self.stat()
        self.api['caluclate_tp_fp'](self.boxes(), np.array([]), self.boxes(1), stat, .7)
        self.assertEqual(stat[.7]['gt'], 1)
        self.api['caluclate_tp_fp'](self.boxes(1, 1), np.array([.4, .9]), self.boxes(1), stat, .7)
        self.assertEqual(stat[.7]['tp'], [1, 0])
        self.assertEqual(stat[.7]['fp'], [0, 1])

    def test_no_global_sort_is_preserved(self):
        stat = {.7: dict(tp=[1, 0], fp=[0, 1], score=[.1, .9], gt=1)}
        self.assertEqual(self.api['calculate_ap'](copy.deepcopy(stat), .7, False)[0], 1.)
        self.assertEqual(self.api['calculate_ap'](copy.deepcopy(stat), .7, True)[0], .5)


class IdentityAndResume(unittest.TestCase):
    def cache_reader(self):
        namespace = dict(identity=identity, read_json=read, sha256=sha256, atomic_json=write, re=re, np=np)
        original_functions(ROOT / 'v2xr_protocol_audit/support.py',
                           ('check_identity', 'load_prediction', 'compare_gt'), namespace)
        tree = ast.parse((ROOT / 'v2xr_protocol_audit/predict.py').read_text(encoding='utf-8'))
        generate = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'generate')
        # Exercise the unmodified original cache branch only. Remove server imports and stop
        # before the GPU path. No function from the inference branch is executed locally.
        body = []
        for node in generate.body:
            if isinstance(node, ast.If) and 'args.stage' in ast.unparse(node.test):
                break
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                body.append(node)
        generate.body = body + [ast.Return(value=ast.Name(id='manifest', ctx=ast.Load()))]
        ast.fix_missing_locations(generate)
        exec(compile(ast.Module(body=[generate], type_ignores=[]), 'frozen_cache_prefix', 'exec'), namespace)
        return namespace['generate']

    def test_original_partial_cache_adopts_verified_orphan_and_detects_committed_tampering(self):
        generate = self.cache_reader()
        with temporary() as directory:
            root = Path(directory)
            cache = root / 'predictions/fog'
            cache.mkdir(parents=True)
            contract = dict(identity='synthetic', frame_keys=[['scene', 'time']])
            evidence = dict(frames=[dict(conditions={'fog': dict(final_gt=[])})])
            row = dict(contract_identity='synthetic', condition='fog', index=0,
                       frame_key=['scene', 'time'], official_gt=[])
            row['record_identity'] = identity(row)
            frame = cache / '000000.json'
            write(frame, row)
            digest = sha256(frame)
            manifest = generate('fog', root, SimpleNamespace(stage='all'), evidence, contract)
            self.assertEqual(manifest['entries']['0']['sha256'], digest)
            self.assertEqual(sha256(frame), digest)
            row['extra'] = 'tampered after commit'
            row['record_identity'] = identity({k: v for k, v in row.items() if k != 'record_identity'})
            write(frame, row)
            with self.assertRaisesRegex(ValueError, 'Committed prediction bytes changed'):
                generate('fog', root, SimpleNamespace(stage='all'), evidence, contract)

    def test_original_complete_cache_reuses_only_when_every_frame_and_hash_exist(self):
        generate = self.cache_reader()
        with temporary() as directory:
            root = Path(directory)
            cache = root / 'predictions/fog'
            cache.mkdir(parents=True)
            contract = dict(identity='synthetic', frame_keys=[['scene', 'time']])
            evidence = dict(frames=[dict(conditions={'fog': dict(final_gt=[])})])
            row = dict(contract_identity='synthetic', condition='fog', index=0,
                       frame_key=['scene', 'time'], official_gt=[])
            row['record_identity'] = identity(row)
            write(cache / '000000.json', row)
            manifest = generate('fog', root, SimpleNamespace(stage='all'), evidence, contract)
            manifest.update(status='complete', official_output_sha256={})
            write(cache / 'manifest.json', manifest)
            result = generate('fog', root, SimpleNamespace(stage='all'), evidence, contract)
            self.assertEqual(result['status'], 'complete')
            (cache / '000000.json').unlink()
            with self.assertRaisesRegex(ValueError, 'missing'):
                generate('fog', root, SimpleNamespace(stage='all'), evidence, contract)

    def test_same_request_reuses_changed_request_refused(self):
        with temporary() as directory:
            p = Path(directory) / 'identity.json'
            bind(p, dict(checkpoint='x', gpu='2'))
            bind(p, dict(checkpoint='x', gpu='2'))
            with self.assertRaises(ValueError):
                bind(p, dict(checkpoint='y', gpu='2'))

    def test_completion_needs_hashes_and_rejects_tampering(self):
        with temporary() as directory:
            root = Path(directory)
            write(root / 'frame.json', dict(synthetic=True))
            marker = root / 'summary.json'
            write(marker, dict(status='complete', output_hashes={'frame.json': sha256(root / 'frame.json')}))
            completed(marker)
            write(root / 'frame.json', dict(synthetic=False))
            with self.assertRaises(ValueError):
                completed(marker)
            write(marker, dict(status='complete'))
            with self.assertRaises(ValueError):
                completed(marker)

    def test_original_cache_identity_rejects_smoke_and_cross_model_reuse(self):
        namespace = dict(identity=identity, read_json=read)
        original_functions(ROOT / 'v2xr_protocol_audit/support.py', ('check_identity', 'load_prediction'), namespace)
        with temporary() as directory:
            p = Path(directory) / '000000.json'
            row = dict(contract_identity='smoke', condition='fog', index=0, frame_key=['scene', 'time'])
            row['record_identity'] = identity(row)
            write(p, row)
            namespace['load_prediction'](p, {'identity': 'smoke'}, 'fog', 0, ['scene', 'time'])
            with self.assertRaises(ValueError):
                namespace['load_prediction'](p, {'identity': 'formal'}, 'fog', 0, ['scene', 'time'])
            row['condition'] = 'snow'
            write(p, row)
            with self.assertRaises(ValueError):
                namespace['load_prediction'](p, {'identity': 'smoke'}, 'snow', 0, ['scene', 'time'])


class Comparisons(unittest.TestCase):
    def setUp(self):
        self.reference = read(HERE / 'reference/FINAL_RESULTS.json')

    def test_original_full_precision_archive_and_definitions(self):
        source = reference_source(ROOT / 'logs/train/FINAL_RESULTS.json')
        self.assertEqual(source['sha256'], sha256(HERE / 'reference/FINAL_RESULTS.json'))
        for name in ('C', 'D'):
            cmp.validate_summary(self.reference['results'][name], name)
        delta = cmp.differences(self.reference['results']['D'], self.reference['results']['C'])
        self.assertEqual(delta, self.reference['matched_deltas_pp']['D-C'])

    def test_partial_wrong_gt_and_test_like_summaries_refused(self):
        for key, value in (('frame_count', 12), ('full_validation', False), ('frozen_d1c_gt_identity', 'other')):
            summary = copy.deepcopy(self.reference['results']['C'])
            summary[key] = value
            with self.assertRaises(ValueError):
                cmp.validate_summary(summary, 'C')

    def test_fog_one_to_one_counts_are_from_matching_not_ap(self):
        c = cmp.fog_counts(self.reference['results']['C'])
        self.assertEqual((c['F_tp70'], c['F_fn70'], c['predictions']), (3031, 3193, 3879))
        d = cmp.fog_counts(self.reference['results']['D'])
        self.assertEqual((d['F_tp70'], d['F_fn70']), (3212, 3012))

    def test_missing_raw_records_cannot_invent_official_tp_from_fixed_tp(self):
        counts = cmp.fog_protocol_counts(self.reference['results']['C'], None)
        self.assertEqual(counts['F']['tp'], 3031)
        self.assertEqual(counts['O']['status'], 'unavailable')
        self.assertEqual(counts['O*']['status'], 'unavailable')
        self.assertNotIn('tp', counts['O'])

    def test_paired_gain_distribution_distinguishes_different_frames_and_target_ids(self):
        left, right = {}, {}
        def row(index, ids):
            return dict(frame_key=['scene', str(index)], removed_ids=['removed'], prediction_count=len(ids),
                protocols={p: dict(matched_pairs={'0.7': [dict(object_id=k) for k in ids]},
                    thresholds={'0.7': dict(tp=len(ids), fp=0, fn=2-len(ids))}) for p in ('O', 'O*', 'F')})
        for i in range(829):
            left[('fog', i)] = row(i, ['removed'] if i == 0 else [])
            right[('fog', i)] = row(i, ['retained'] if i == 1 else [])
        rows, dist = cmp.frame_gains(left, right, 'fog')
        self.assertEqual((dist['benefited_frames'], dist['degraded_frames'], dist['equal_tp_frames']), (1, 1, 827))
        self.assertEqual(dist['net_tp70'], 0)
        self.assertEqual((dist['newly_detected_fixed_targets'], dist['no_longer_detected_fixed_targets']), (1, 1))
        self.assertEqual(rows[0]['newly_detected_removed_ids'], ['removed'])
        self.assertEqual(rows[1]['no_longer_detected_fixed_ids'], ['retained'])

    def test_protocol_decomposition_and_reports_use_all_27_deltas(self):
        author = copy.deepcopy(self.reference['results']['C'])
        author.update(model_variant=cmp.AUTHOR, checkpoint_sha256=CHECKPOINT_SHA)
        models = dict(self.reference['results'], **{cmp.AUTHOR: author})
        effects = cmp.effects(models)
        for conditions in effects.values():
            for metrics in conditions.values():
                for e in metrics.values():
                    self.assertAlmostEqual(e['O_delta_pp'], e['F_delta_pp'] + e['None_skip_component_pp'] + e['GT_filter_component_pp'], places=12)
        self.assertEqual(len(cmp.differences(models['D'], author)), 3)

    def test_report_entry_missing_raw_data_still_reports_aggregate_and_missing_sources(self):
        # All fabricated author values stay in a temporary synthetic test directory.
        with temporary() as directory:
            root = Path(directory)
            (root / 'author_evaluation').mkdir()
            (root / 'frozen').mkdir()
            write(root / 'frozen/gt_manifest.json', {'frames': []})
            write(root / 'author_evaluation/evidence.json', {'synthetic': True})
            write(root / 'author_evaluation/protocol.json', {})
            summary = copy.deepcopy(self.reference['results']['C'])
            summary.update(model_variant=cmp.AUTHOR, checkpoint_sha256=CHECKPOINT_SHA)
            summary['output_hashes'] = {n: sha256(root / 'author_evaluation' / n)
                                        for n in ('evidence.json', 'protocol.json')}
            write(root / 'author_evaluation/summary.json', summary)
            write(root / 'preflight.json', dict(
                cd_reference=dict(path=str(HERE / 'reference/FINAL_RESULTS.json'), sha256=sha256(HERE / 'reference/FINAL_RESULTS.json')),
                protocols={}, d1c={'gt_manifest.json': sha256(root / 'frozen/gt_manifest.json')}))
            request = dict(identity='synthetic', arguments=dict(d1c_root=str(root / 'frozen'), cd_root=str(root / 'missing')))
            with patch.object(cmp, 'authenticated_frames', return_value=(None, dict(status='unavailable'))):
                data = cmp.report(root, request)
            self.assertEqual(len(data['deltas_pp']), 3)
            self.assertEqual(data['per_frame_gain_distribution']['D-C']['fog']['status'], 'unavailable')
            self.assertTrue((root / 'reports/CONCLUSIONS.md').exists())
            completed(root / 'reports/completion.json')


class EntryAndStatic(unittest.TestCase):
    def test_official_yaml_adapter_allows_paths_and_refuses_modality_changes(self):
        import yaml
        from v2xr_author_fog_eval.pipeline import check_config
        functions = original_functions(ROOT / 'v2xr_official_repro/run.py', ('config_difference',), {})
        original = dict(b=dict(d1a_summary=dict(actual=dict(max_cav=7))))
        template = ROOT / 'v2xr_official_repro/upstream/opencood/hypes_yaml/V2X-R/LiDAR/V2XR_attfuse.yaml'
        config = yaml.safe_load(template.read_text(encoding='utf-8'))
        with temporary() as directory:
            root = Path(directory)
            config['root_dir'] = '/synthetic/train'
            config['validate_dir'] = '/synthetic/validate'
            path = root / 'config.yaml'
            path.write_text(yaml.safe_dump(config), encoding='utf-8')
            with patch.dict(sys.modules, {'run': SimpleNamespace(**functions)}):
                result = check_config(SimpleNamespace(model_dir=str(root)), original)
                self.assertEqual(set(result['differences']), {'root_dir', 'validate_dir'})
                config['model']['args']['use_modality'] = 'processed_radar'
                path.write_text(yaml.safe_dump(config), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'beyond root paths'):
                    check_config(SimpleNamespace(model_dir=str(root)), original)

    def test_modes_and_order_have_no_training(self):
        self.assertEqual(workflow('all'), ('preflight', 'checkpoint-audit', 'smoke', 'author-evaluation', 'report'))
        self.assertEqual(workflow('smoke')[-1], 'smoke')
        args = parser().parse_args(['--gpu', '2', '--resume', '1'])
        self.assertEqual((args.gpu, args.resume), ('2', 1))
        with self.assertRaises(SystemExit):
            with patch('sys.stderr'):
                parser().parse_args(['--gpu', '2', '--mode', 'train'])

    def test_frozen_interfaces_and_no_training_calls(self):
        source = ast.parse((ROOT / 'v2xr_train_attfuse/evaluation.py').read_text(encoding='utf-8'))
        evaluate = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'evaluate')
        self.assertEqual([a.arg for a in evaluate.args.args], ['args', 'own', 'checkpoint', 'destination', 'conditions', 'limit'])
        calls = [n.func.id for n in ast.walk(evaluate) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
        self.assertNotIn('selection', calls)
        self.assertNotIn('weather_checkpoint_gate', calls)
        for path in HERE.glob('*.py'):
            if path.name == 'tests.py':
                continue
            tree = ast.parse(path.read_text(encoding='utf-8'))
            bad = [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and n.func.attr in ('backward', 'step', 'train', 'inference_mode', 'setup_optimizer')]
            self.assertEqual(bad, [], str(path))
        self.assertNotIn(b'\r', (HERE / 'run_server.sh').read_bytes())

    def test_controller_stops_after_failed_stage_without_claiming_completion(self):
        from v2xr_author_fog_eval import main as entry
        with temporary() as directory:
            root = Path(directory)
            checkpoint = root / 'synthetic.pth'
            checkpoint.write_bytes(b'not a real weight; synthetic controller test')
            defaults = dict(output_root=str(root), checkpoint=str(checkpoint), cd_results=str(root / 'reference.json'))
            args = SimpleNamespace(mode='all', resume=0, gpu='2', **defaults)
            seen = []
            def stage(root, name, fd):
                seen.append(name)
                if name == 'smoke':
                    raise RuntimeError('synthetic interruption')
            with patch.object(entry, 'parser', return_value=SimpleNamespace(parse_args=lambda: args)), \
                 patch.object(entry, 'check_paths', return_value=root), \
                 patch.object(entry, 'DEFAULTS', defaults), \
                 patch.object(entry, 'verify_sources', return_value={}), \
                 patch.object(entry, 'reference_source', return_value={}), \
                 patch.object(entry, 'lock', return_value=nullcontext(0)), \
                 patch.object(entry, 'run_stage', side_effect=stage), \
                 patch.dict(os.environ, {'TMPDIR': str(root / 'tmp')}):
                # Match the normal launcher's allowed top-level files.
                checkpoint.unlink()
                checkpoint = root / 'logs/synthetic.pth'
                checkpoint.parent.mkdir()
                checkpoint.write_bytes(b'synthetic')
                args.checkpoint = str(checkpoint)
                with self.assertRaisesRegex(RuntimeError, 'synthetic interruption'):
                    entry.main()
            self.assertEqual(seen, ['preflight', 'checkpoint-audit', 'smoke'])
            self.assertEqual(read(root / 'completion.json')['status'], 'failed')
            self.assertFalse(read(root / 'completion.json')['full_evaluation'])

    def test_no_heavy_library_was_imported(self):
        self.assertFalse(any(k == 'torch' or k.startswith('torch.') or k == 'opencood' or k.startswith('opencood.') for k in sys.modules))


def run_tests():
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    if not result.wasSuccessful():
        raise ValueError('Local-safe synthetic checks failed')


if __name__ == '__main__':
    run_tests()
