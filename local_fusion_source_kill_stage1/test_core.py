"""Pure synthetic tests: no Torch/OpenCOOD imports, data, checkpoints or experiments.

Extract ORIGINAL Oracle functions by AST for tests that would otherwise import
the entire GPU stack. This executes the original function bodies, not a second
implementation of their semantics. Production uses the ordinary original module.
"""
import ast
from copy import deepcopy
import itertools
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from .common import (ROOT, KEEP, FORBIDDEN, atomic_json, digest, guard_tree,
                     load_condition, object_hash, safe_path)
from .k0_group_margin import (audit_group, aggregate, groups_from_frame, old_risk_reference,
                              primary, risk_gate)
from .k1_0_oracle_decomposition import reproduction_gate
from .k1_0_report import retention, summarize
from .oracle_adapter import KeyPolicy, eligible, partition, run_restricted


def temporary_directory():
    # Windows sandbox supports atomic rename inside the writable workspace.
    directory = ROOT / '.tmp'
    directory.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(dir=directory, prefix='stage1_unit_')


def original_functions(names, namespace=None):
    tree = ast.parse((ROOT / 'local_fusion_task_source_oracle/oracle.py').read_text(encoding='utf-8'))
    selected = ast.Module(body=[node for node in tree.body
                               if isinstance(node, ast.FunctionDef) and node.name in names], type_ignores=[])
    env = {'KEEP': KEEP, **(namespace or {})}
    exec(compile(selected, 'ORIGINAL oracle.py functions', 'exec'), env)
    return env


def outcome(tp=2, score=.6, iou=.8, **changes):
    return dict(baseline_tp=2, action_tp=tp, lost=max(2 - tp, 0), recovered=max(tp - 2, 0),
                new_fp=0, action_fp=1, focal_detected_after=True,
                focal_score_after=score, focal_iou_after=iou, **changes)


def frame(frame_id=1, proposal=10):
    return dict(frame=frame_id, split='validation', weather='fog', scene=1,
                source_names=[KEEP, 'single:0', 'query:0'], proposals={'anchor_ids': [proposal]},
                labels=[dict(proposal_position=0, target_index=0, background=False,
                             outcomes={task: [outcome(), outcome(score=.7), outcome()]
                                       for task in ('cls', 'reg')})])


class K0Tests(unittest.TestCase):
    def test_exact_tie(self):
        row = audit_group([outcome(), outcome()], [KEEP, 'single:0'])
        self.assertEqual(row['category'], 'EXACT_OUTCOME_TIE')
        self.assertEqual(row['top_tie_count'], 2)

    def test_continuous(self):
        self.assertEqual(audit_group([outcome(), outcome(score=.7)], [KEEP, 's'])['category'],
                         'CONTINUOUS_ONLY_MARGIN')

    def test_discrete(self):
        self.assertEqual(audit_group([outcome(), outcome(tp=3)], [KEEP, 's'])['category'], 'DISCRETE_MARGIN')

    def test_keep_only(self):
        row = audit_group([outcome()], [KEEP])
        self.assertTrue(row['keep_only'])
        self.assertIsNone(row['runner_up_key'])
        self.assertEqual(row['top_tie_count'], 1)
        self.assertEqual(row['category'], 'EXACT_OUTCOME_TIE')

    def test_multiple_top_ties(self):
        row = audit_group([outcome(), outcome(tp=3), outcome(tp=3)], [KEEP, 'b', 'a'])
        self.assertEqual(row['top_equivalent_actions'], ['a', 'b'])
        self.assertEqual(row['category'], 'EXACT_OUTCOME_TIE')
        self.assertTrue(row['has_discrete_beneficial_action'])

    def test_keep_simplicity_is_not_margin(self):
        row = audit_group([outcome(), outcome()], [KEEP, 's'])
        self.assertEqual(row['top_key'][-1], 1)
        self.assertEqual(row['runner_up_key'][-1], 0)
        self.assertEqual(row['category'], 'EXACT_OUTCOME_TIE')

    def test_benefit_and_harm_can_coexist(self):
        row = audit_group([outcome(), outcome(tp=3), outcome(tp=1)], [KEEP, 'a', 'b'])
        self.assertTrue(row['has_discrete_beneficial_action'])
        self.assertTrue(row['harmful_discrete_exists'])
        self.assertFalse(row['continuous_only_benefit'])

    def test_continuous_benefit_excludes_discrete_benefit(self):
        self.assertTrue(audit_group([outcome(), outcome(score=.9)], [KEEP, 'a'])['continuous_only_benefit'])
        self.assertFalse(audit_group([outcome(), outcome(tp=3), outcome(score=.9)],
                                     [KEEP, 'a', 'b'])['continuous_only_benefit'])

    def test_order_invariance(self):
        actions = [('a', outcome(tp=3)), ('b', outcome(tp=3)), ('c', outcome(tp=1))]
        rows = [audit_group([outcome()] + [x[1] for x in perm], [KEEP] + [x[0] for x in perm])
                for perm in itertools.permutations(actions)]
        for row in rows[1:]:
            self.assertEqual(row, rows[0])

    def test_original_key_order(self):
        row = outcome()
        row.update(lost=1, new_fp=2, recovered=3, action_fp=4)
        key = audit_group([row], [KEEP])['top_key']
        self.assertEqual(key, (2, -1, -2, 1, 3, -4, .6, .8, 1))

    def test_group_identity_frame_proposal_task_focal(self):
        rows = list(groups_from_frame(frame())) + list(groups_from_frame(frame(2))) + list(groups_from_frame(frame(1, 11)))
        self.assertEqual(len({row['identity'] for row in rows}), 6)
        original = frame()
        duplicate = deepcopy(original['labels'][0])
        duplicate['target_index'] = 1
        original['labels'].append(duplicate)
        self.assertEqual(len({row['identity'] for row in groups_from_frame(original)}), 4)

    def test_duplicate_group_rejected(self):
        value = frame()
        value['labels'].append(deepcopy(value['labels'][0]))
        with self.assertRaises(ValueError):
            list(groups_from_frame(value))

    def test_primary_scope(self):
        for split, stratum in itertools.product(('train', 'validation'), ('associated', 'background', 'all')):
            self.assertEqual(primary(dict(split=split, stratum=stratum)),
                             split == 'validation' and stratum == 'associated')

    def test_scene_coverage(self):
        rows = list(groups_from_frame(frame()))
        for row in rows:
            row['category'] = 'DISCRETE_MARGIN'
            row['has_discrete_beneficial_action'] = True
        rows[1]['scene'] = 2
        value = aggregate(rows)
        self.assertEqual(value['discrete_margin_scene_coverage']['scene_count'], 2)
        self.assertEqual(value['beneficial_set_scene_coverage']['max_single_scene_fraction'], .5)

    def test_original_risk_schema_and_stop(self):
        reference = old_risk_reference((ROOT / 'logs/sectionB/all/LABEL_AUDIT.md').read_text(encoding='utf-8'))
        self.assertEqual(len(reference), 48)
        self.assertEqual(reference['validation/snow/cls/associated'], ['0.86%', '19.03%', '80.11%'])
        row = aggregate(list(groups_from_frame(frame())))
        with self.assertRaisesRegex(RuntimeError, 'STOP Stage 1'):
            risk_gate({'validation/fog/cls/associated': row}, reference)


class FakeTensor:
    def __init__(self, value):
        self.value = value
    def clone(self):
        return FakeTensor(deepcopy(self.value))
    def detach(self):
        return self
    def cpu(self):
        return self
    def numpy(self):
        return np.array(self.value)
    def __len__(self):
        return len(self.value)


def synthetic_oracle():
    gt = FakeTensor([[0], [1]])
    dataset = SimpleNamespace(post_process=lambda batch, pred: (pred['ego'], None, gt))
    def state(post, target, threshold):
        value = post[0]['psm'].value
        return dict(matched={0, 1} if value >= 10 else {0}, fp=[],
                    focal=dict(detected=True, score=value / 100, iou=.8))
    def compose(current, pool, mask, cls, reg):
        return {'psm': pool[cls]['psm'] if cls != KEEP else current['psm'], 'rm': current['rm']}
    def shortlist(current, fixed, gt, masks, names, expansions, mode, threshold, k):
        rows = [dict(cls_source=name, reg_source=name if mode == 'same' else KEEP,
                     mask=True, expansion=1., mask_stats={}, proxy=(1,)) for name in names if name != KEEP]
        return rows, dict(shortlisted_actions=len(rows), all_proxy_actions=len(rows))
    namespace = dict(count_new_false_positives=lambda a, b, t: 0,
                     detection_outcome=lambda *args: ({0}, []), _state=state,
                     masks_for_frame=lambda *args: (True, None, {}), _proxy_cache=lambda ds, batch, p: p,
                     _shortlist_actions=shortlist, compose_prediction=compose)
    env = original_functions({'run_oracle', '_action_key', '_target_order'}, namespace)
    module = SimpleNamespace(**env)
    shared = {'psm': FakeTensor(0), 'rm': FakeTensor(0)}
    pool = {KEEP: shared, 'a': {'psm': FakeTensor(10)}, 'b': {'psm': FakeTensor(1)}}
    args = (dataset, {'ego': {'anchor_box': None}}, None, [], shared, pool, {}, None, (1., 1.5), 'task', .7, 6)
    return module, args, shared


class K10Tests(unittest.TestCase):
    def test_partition_exclusive_exhaustive(self):
        keep = (2, 0, 0, 1, 0, -1, .5, .8, 2)
        for field in range(9):
            changed = list(keep)
            changed[field] += 1
            expected = 'A' if field < 6 else 'B' if field < 8 else 'E'
            self.assertEqual(partition(changed, keep), expected)

    def test_keep_uses_current_state(self):
        module, _, _ = synthetic_oracle()
        state = dict(matched={4, 5}, fp=[], focal=dict(detected=True, score=.7, iou=.9))
        policy = KeyPolicy(module._action_key, 'B_ONLY')
        key = policy(state, state, KEEP, KEEP)
        self.assertEqual(tuple(policy.targets[0]['keep_key']), key)
        self.assertEqual(key[0], 2)
        self.assertEqual(key[-1], 2)

    def test_keep_always_legal(self):
        for variant in ('FULL', 'A_ONLY', 'B_ONLY'):
            self.assertTrue(eligible(variant, 'KEEP'))
        self.assertFalse(eligible('A_ONLY', 'B'))
        self.assertFalse(eligible('A_ONLY', 'E'))
        self.assertFalse(eligible('B_ONLY', 'A'))
        self.assertFalse(eligible('B_ONLY', 'E'))

    def test_full_executes_original_body_same_outputs(self):
        module, args, shared = synthetic_oracle()
        expected = module.run_oracle(*args)
        actual = run_restricted(module, 'FULL', *args)
        self.assertEqual(expected[0]['psm'].value, actual[0]['psm'].value)
        self.assertEqual(expected[2], actual[2])
        self.assertEqual(shared['psm'].value, 0)

    def test_independent_paths_and_current_state_partition(self):
        module, args, shared = synthetic_oracle()
        results = {v: run_restricted(module, v, *args) for v in ('FULL', 'A_ONLY', 'B_ONLY')}
        self.assertEqual(results['FULL'][0]['psm'].value, 10)
        self.assertEqual(results['A_ONLY'][0]['psm'].value, 10)
        self.assertEqual(results['B_ONLY'][0]['psm'].value, 1)
        for result in results.values():
            self.assertEqual(result[3][0]['keep_key'][6], 0)
        self.assertEqual(results['FULL'][3][0]['actions'][0]['group'], 'A')
        self.assertEqual(results['FULL'][3][1]['actions'][0]['group'], 'E')
        self.assertEqual(results['B_ONLY'][3][1]['actions'][1]['group'], 'E')
        self.assertEqual(shared['psm'].value, 0)

    def test_same_task_and_source_count(self):
        for mode in ('same', 'task'):
            for count in (0, 1, 4):
                module, args, _ = synthetic_oracle()
                args = list(args)
                args[5] = {KEEP: args[4], **{f'single:{i}': {'psm': FakeTensor(10 + i)} for i in range(count)}}
                args[9] = mode
                for variant in ('FULL', 'A_ONLY', 'B_ONLY'):
                    result = run_restricted(module, variant, *args)
                    self.assertEqual(len(result[2]), 2)

    def test_real_shortlist_task_exceeds_six(self):
        cu = SimpleNamespace(convert_format=lambda a: list(a), compute_iou=lambda a, b: [.8] * len(b))
        env = original_functions({'_shortlist_actions', '_actions', '_proxy_tuple', '_roi_anchor_ids'}, dict(np=np, cu=cu))
        cache = dict(scores=np.array([.8]), corners=np.zeros((1, 8, 3)), anchor_num=1)
        names = (KEEP,) + tuple(f'single:{i}' for i in range(4))
        fixed = {name: cache for name in names[1:]}
        masks = {1.: (np.ones((1, 1), dtype=bool), {}), 1.5: (np.ones((1, 1), dtype=bool), {})}
        same, _ = env['_shortlist_actions'](cache, fixed, np.zeros((8, 3)), masks, names, (1., 1.5), 'same', .7, 6)
        task, _ = env['_shortlist_actions'](cache, fixed, np.zeros((8, 3)), masks, names, (1., 1.5), 'task', .7, 6)
        self.assertEqual(len(same), 6)
        self.assertGreater(len(task), 6)
        self.assertLessEqual(len(task), 18)

    def test_adapter_leaves_original_globals_unchanged(self):
        module, args, _ = synthetic_oracle()
        key = module.run_oracle.__globals__['_action_key']
        shortlist = module.run_oracle.__globals__['_shortlist_actions']
        run_restricted(module, 'A_ONLY', *args)
        self.assertIs(module.run_oracle.__globals__['_action_key'], key)
        self.assertIs(module.run_oracle.__globals__['_shortlist_actions'], shortlist)

    def test_no_s0_groups_or_extra_gt_in_adapter(self):
        import inspect
        from . import oracle_adapter
        text = inspect.getsource(oracle_adapter)
        tree = ast.parse(text)
        # Adapter has only key comparisons and binds original run function.
        self.assertNotIn('torch', text)
        self.assertNotIn('s0_run', text)
        self.assertNotIn('target_index]', text)
        imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        self.assertEqual(imports, ['collections', 'types', 'common'])

    def test_full_reproduction_strict(self):
        saved = {key: .8 for key in ('ap30', 'ap50', 'ap70')}
        reproduction_gate(saved, saved, 'synthetic')
        with self.assertRaisesRegex(RuntimeError, 'STOP Stage 1'):
            reproduction_gate({**saved, 'ap70': .800002}, saved, 'synthetic')
        with self.assertRaises(RuntimeError):
            reproduction_gate({**saved, 'ap70': float('nan')}, saved, 'synthetic')

    def test_ratios(self):
        self.assertEqual(retention(.499, .4)['status'], 'UNSTABLE_RATIO')
        self.assertAlmostEqual(retention(8, 6)['value'], .75)

    def test_description_keeps_major_mechanism_visible(self):
        from .k1_0_report import describe
        description, focus = describe({'Full_Task-Shared_pp': 8,
                                       'AOnly_Task-Shared_pp': 2, 'BOnly_Task-Shared_pp': 6})
        self.assertIn('ranking-only', description)
        self.assertIn('N1', focus)


class IOTests(unittest.TestCase):
    def test_test_paths_and_descendants_rejected(self):
        for path in FORBIDDEN:
            for value in (path, path + '/scene/frame.pcd'):
                with self.assertRaises(ValueError):
                    safe_path(value)

    def test_symlink_alias_rejected(self):
        with temporary_directory() as directory:
            root = Path(directory)
            target = root / 'test'
            target.mkdir()
            alias = root / 'validate'
            try:
                alias.symlink_to(target, target_is_directory=True)
            except OSError:
                # Windows may deny symlink creation. Emulate the OS-resolved alias;
                # the same test uses an actual symlink on the POSIX server.
                original_resolve = Path.resolve
                def resolve(path, *args, **kwargs):
                    return target if path == alias else original_resolve(path, *args, **kwargs)
                with patch.object(Path, 'resolve', resolve), self.assertRaises(ValueError):
                    safe_path(alias, dataset=True)
            else:
                with self.assertRaises(ValueError):
                    safe_path(alias, dataset=True)

    def test_nested_symlink_guard(self):
        with temporary_directory() as directory:
            root = Path(directory)
            validate = root / 'validate'
            validate.mkdir()
            target = root / 'test'
            target.mkdir()
            try:
                (validate / 'scene').symlink_to(target, target_is_directory=True)
            except OSError:
                alias = validate / 'scene'
                alias.mkdir()
                original_resolve = Path.resolve
                def resolve(path, *args, **kwargs):
                    return target if path == alias else original_resolve(path, *args, **kwargs)
                with patch.object(Path, 'resolve', resolve), self.assertRaises(ValueError):
                    guard_tree(validate)
            else:
                with self.assertRaises(ValueError):
                    guard_tree(validate)

    def test_atomic_condition_resume_rejects_corruption(self):
        with temporary_directory() as directory:
            run = Path(directory)
            atomic_json(run / 'protocol.json', {'identity': 'abc'})
            payload = {'identity': 'abc', 'status': 'complete', 'AP': .8}
            self.assertIsNone(load_condition(run, 'fog_same_FULL'))
            atomic_json(run / 'conditions/fog_same_FULL.json', {'payload': payload, 'sha256': object_hash(payload)})
            self.assertEqual(load_condition(run, 'fog_same_FULL'), payload)
            payload['AP'] = .9
            atomic_json(run / 'conditions/fog_same_FULL.json', {'payload': payload, 'sha256': 'wrong'})
            with self.assertRaises(ValueError):
                load_condition(run, 'fog_same_FULL')

    def test_full_gate_precedes_all_restricted_runs(self):
        from . import k1_0_oracle_decomposition as module
        calls = []
        protocol = {'references': {'oracle_run': '/synthetic'}, 'validation_indices': [0]}
        def fake_condition(run, p, runtime, weather, mode, variant, old):
            calls.append(variant)
            if weather == 'snow' and mode == 'task':
                raise RuntimeError('reproduction fails')
            return {}
        from contextlib import contextmanager
        @contextmanager
        def fake_component(*args):
            yield protocol
        with patch.object(module, 'contract', return_value=protocol), patch.object(
                module, 'component_status', return_value={'status': 'pending'}), patch.object(
                module, 'component', fake_component), patch.object(module, 'make_runtime'), patch.object(
                module, 'read_json', return_value={}), patch.object(module, 'condition', fake_condition):
            with self.assertRaises(RuntimeError):
                module.build(Path('/synthetic'))
        self.assertEqual(calls, ['FULL'] * 8)

    def test_no_local_experiment_imports(self):
        import sys
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('local_fusion_task_source_oracle.oracle', sys.modules)

    def test_reports_all_ap_and_nonadditive_gains(self):
        from .common import WEATHERS
        from .k1_0_report import render
        conditions = {}
        for weather in WEATHERS:
            for variant, gain in (('FULL', .08), ('A_ONLY', .06), ('B_ONLY', .04)):
                for mode in ('same', 'task'):
                    ap = {m: .6 + gain for m in ('ap30', 'ap50', 'ap70')}
                    shared = {m: .6 for m in ap}
                    counts = {key: 0 for key in ('targets', 'changed', 'task_separated', 'recovered', 'lost', 'new_fp')}
                    counts.update({'shortlisted_' + x: 0 for x in ('A', 'B', 'E')})
                    counts.update({'selected_' + x: 0 for x in ('A', 'B', 'E', 'KEEP')})
                    conditions[f'{weather}_{mode}_{variant}'] = dict(
                        AP=ap, Shared=shared, counts=counts,
                        availability_rates=dict(A=0, B=0, E=0), shortlisted_group_rates=dict(A=0, B=0, E=0))
        result = summarize(conditions)
        delta = result['deltas_pp']['adverse mean']
        self.assertAlmostEqual(delta['Full_Task-Shared_pp'], 8)
        self.assertAlmostEqual(delta['AOnly_Task-Shared_pp'], 6)
        self.assertAlmostEqual(delta['BOnly_Task-Shared_pp'], 4)
        text = render(result)
        self.assertIn('AP30', text)
        self.assertIn('UNSTABLE_RATIO', text)  # Task-Same is zero.
        self.assertIn('restricted-policy attribution', text)
        self.assertFalse(result['next_stage_auto_start'])

    def test_k0_report_all_primary_rows(self):
        from .common import WEATHERS
        from .k0_report import render
        groups = list(groups_from_frame(frame()))
        row = aggregate(groups)
        conditions = {f'{split}/{weather}/{task}/{stratum}': row
                      for split in ('train', 'validation') for weather in WEATHERS
                      for task in ('cls', 'reg') for stratum in ('associated', 'background', 'all')}
        result = dict(conditions=conditions, signal='K0_SIGNAL_MIXED', primary_adverse_macro={task: row for task in ('cls', 'reg')})
        text = render(result)
        self.assertIn('ACTION-LEVEL CONTEXT', text)
        self.assertIn('adverse mean', text)
        self.assertIn('Scene', text)


if __name__ == '__main__':
    unittest.main()
