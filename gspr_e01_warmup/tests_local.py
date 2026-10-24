"""Stdlib-only synthetic checks. No Torch, real model, point cloud or experiment."""
import ast
import copy
from contextlib import contextmanager
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch, Mock

import warm_common as common
from warm_adapter import pause_tree, adapted_run
from warm_runtime import audit_warmup, publish_warmup, activate
from controller import Controller, existing_e0, bind_inputs

BASE = None


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n',
                    encoding='utf-8', newline='\n')


@contextmanager
def changed(path, value):
    """Restore only one synthetic fixture file, including on assertion failure."""
    original = path.read_bytes()
    try:
        write(path, value)
        yield
    finally:
        path.write_bytes(original)


@contextmanager
def sandbox():
    with tempfile.TemporaryDirectory(prefix='.gw_synthetic_', dir=common.PROJECT) as folder:
        root = Path(folder).resolve()
        if root.parent != common.PROJECT.resolve():
            raise ValueError('Synthetic fixture escaped workspace')
        yield root


TOY = '''
def run(rows, events, interrupt=None):
    epochs = 20
    completed = len(rows)
    if completed == epochs:
        events.append('finish')
        return
    try:
        for epoch in range(completed, epochs):
            for operation in ('train_full', 'validation_loss', 'clean_829', 'scheduler', 'full_state'):
                events.append((epoch + 1, operation))
                if interrupt == (epoch + 1, operation):
                    raise RuntimeError('synthetic interruption')
            rows.append(epoch + 1)
            events.append((epoch + 1, 'complete'))
            events.append((epoch + 1, 'history_progress'))
            print('EPOCH COMMITTED', epoch + 1)
        events.append('finish')
    finally:
        events.append('writer_closed')
'''


def toy(limit):
    tree = pause_tree(ast.parse(TOY))
    namespace = dict(_formal_warmup_limit=limit)
    exec(compile(tree, '<synthetic-commit-loop>', 'exec'), namespace)
    return namespace['run']


class CommitBoundaries(unittest.TestCase):
    def invoke(self, limit, rows=None, interrupt=None):
        rows = [] if rows is None else list(rows)
        events = []
        with patch('sys.stdout', io.StringIO()):
            toy(limit)(rows, events, interrupt)
        return rows, events

    def test_two_full_epochs_are_counted_once(self):
        rows, events = self.invoke(2)
        self.assertEqual(rows, [1, 2])
        self.assertEqual(events[-2:], [(2, 'history_progress'), 'writer_closed'])
        self.assertNotIn('finish', events)
        for epoch in (1, 2):
            self.assertEqual([x[1] for x in events if isinstance(x, tuple) and x[0] == epoch],
                ['train_full', 'validation_loss', 'clean_829', 'scheduler', 'full_state', 'complete', 'history_progress'])

    def test_continuation_runs_only_epochs_three_to_twenty(self):
        rows, events = self.invoke(None, [1, 2])
        self.assertEqual(rows, list(range(1, 21)))
        self.assertEqual(sum(isinstance(x, tuple) and x[1] == 'train_full' for x in events), 18)
        self.assertEqual(events[0], (3, 'train_full'))
        self.assertEqual(events[-2:], ['finish', 'writer_closed'])

    def test_already_committed_gate_performs_no_updates(self):
        rows, events = self.invoke(2, [1, 2])
        self.assertEqual(rows, [1, 2])
        self.assertEqual(events, [])

    def test_resume_after_one_epoch_finishes_only_second(self):
        rows, events = self.invoke(2, [1])
        self.assertEqual(rows, [1, 2])
        self.assertEqual(events[0], (2, 'train_full'))

    def test_uncommitted_epoch_is_not_counted(self):
        rows, events = [], []
        with patch('sys.stdout', io.StringIO()), self.assertRaisesRegex(RuntimeError, 'interruption'):
            toy(2)(rows, events, (2, 'clean_829'))
        self.assertEqual(rows, [1])
        self.assertEqual(events[-1], 'writer_closed')

    def test_no_limit_is_original_twenty(self):
        rows, events = self.invoke(None)
        self.assertEqual(len(rows), 20)
        self.assertIn('finish', events)

    def test_unrecognized_commit_boundary_refused(self):
        with self.assertRaisesRegex(ValueError, 'boundary'):
            pause_tree(ast.parse(TOY.replace('EPOCH COMMITTED', 'uncommitted')))

    def test_unrecognized_epoch_counter_refused(self):
        with self.assertRaisesRegex(ValueError, 'count'):
            pause_tree(ast.parse(TOY.replace('completed = len(rows)', 'completed = 0')))

    def test_original_ast_unchanged_after_removing_two_guards(self):
        from adapters import training_tree
        source = (common.PROJECT / 'v2xr_weather_l012/training.py').read_text(encoding='utf-8')
        original, counts = training_tree(source)
        result = pause_tree(original)
        found = []
        class Remove(ast.NodeTransformer):
            def visit_If(self, node):
                if '_formal_warmup_limit' in ast.unparse(node.test):
                    found.append(node)
                    return None
                return self.generic_visit(node)
        stripped = Remove().visit(result)
        self.assertEqual(len(found), 2)
        self.assertEqual(counts, dict(epochs=3, subset=2, weather=1))
        self.assertEqual(ast.dump(stripped, include_attributes=False), ast.dump(original, include_attributes=False))

    def test_adapter_identity_does_not_change_between_stages(self):
        module = types.SimpleNamespace(__file__=str(common.PROJECT / 'v2xr_weather_l012/training.py'))
        first, a = adapted_run(module, 2)
        final, b = adapted_run(module, None)
        self.assertEqual(a, b)
        self.assertEqual(first.__globals__['_formal_warmup_limit'], 2)
        self.assertIsNone(final.__globals__['_formal_warmup_limit'])


def fixture(root, variant):
    """Plain text synthetic state files; no model is created or loaded."""
    model = root / 'T1' / variant
    model.mkdir(parents=True)
    (model / 'config.yaml').write_text('synthetic: true\n', encoding='utf-8')
    contract = dict(smoke_only=False, epochs=20, variant=variant, frozen_gt='synthetic-frozen-GT',
        config_sha256=common.sha(model / 'config.yaml'),
        settings=dict(e01=dict(recipe='T1', settings=dict(formal_startup=common.POLICY))))
    contract['identity'] = BASE.frozen_identity(contract)
    write(model / 'contract.json', contract)
    write(model / 'selection_policy.json', dict(synthetic=True, count=20))
    write(model / 'adapter.json', dict(synthetic=True, extra_return_guards=2))
    for epoch in range(3):
        state = model / 'resume_states' / f'{epoch:02d}.pth'
        state.parent.mkdir(parents=True, exist_ok=True)
        state.write_text(f'SYNTHETIC FULL STATE {epoch}\n', encoding='utf-8')
        write(model / 'restore_audit' / f'{epoch:02d}.json', dict(status='complete', epoch=epoch,
            contract=contract['identity'], source_sha256=common.sha(state),
            exact_model_Adam_scheduler_criterion_RNG=True))
    write(model / 'resume_states/00.json', dict(sha256=common.sha(model / 'resume_states/00.pth')))
    for epoch in (1, 2):
        directory = model / 'epochs' / f'{epoch:02d}'
        output = directory / 'clean_ap_synthetic'
        output.mkdir(parents=True)
        weight, trace = directory / 'checkpoint.pth', directory / 'sampling.jsonl'
        weight.write_text('SYNTHETIC WEIGHT\n', encoding='utf-8')
        trace.write_text('SYNTHETIC TRACE\n', encoding='utf-8')
        frames, records = [], []
        for index in range(829):
            path = output / 'predictions/clean' / f'{index:06d}.json'
            write(path, dict(synthetic=True, index=index))
            records.append(dict(condition='clean', index=index, path=str(path), sha256=common.sha(path)))
            frames.append(dict(condition='clean', index=index))
        write(output / 'prediction_manifest.json', dict(files=records))
        write(output / 'gt_manifest.json', dict(frames=frames))
        for name in ('contract.json', 'protocol.json', 'predictions/clean/manifest.json',
                     'stat_clean_O.json', 'stat_clean_Ostar.json', 'stat_clean_F.json'):
            write(output / name, dict(synthetic=True))
        for name in ('per_frame_metrics.jsonl', 'removed_gt_metrics.jsonl', 'RESULTS.md'):
            (output / name).write_text('SYNTHETIC ONLY\n', encoding='utf-8')
        ap = {p: dict(ap30=0.2, ap50=0.1, ap70=0.0) for p in ('O', 'O*', 'F')}
        summary = dict(status='complete', conditions=['clean'], frame_count=829, full_validation=True,
            checkpoint_sha256=common.sha(weight), frozen_d1c_gt_identity='synthetic-frozen-GT',
            results=dict(clean=dict(counts=dict(fixed_gt=6224, official_gt=6224), AP=ap)))
        names = ('contract.json', 'protocol.json', 'prediction_manifest.json', 'gt_manifest.json',
            'predictions/clean/manifest.json', 'per_frame_metrics.jsonl', 'removed_gt_metrics.jsonl',
            'RESULTS.md', 'stat_clean_O.json', 'stat_clean_Ostar.json', 'stat_clean_F.json')
        summary['output_hashes'] = {name: common.sha(output / name) for name in names}
        write(output / 'summary.json', summary)
        row = dict(status='complete', epoch=epoch, contract=contract['identity'], train_updates=1010,
            consumed_input_frames=8080, accepted_train_frames=8083, dropped_frames=3, smoke_only=False,
            AP70=0.0, clean_AP=ap)
        for key, path in dict(checkpoint=weight, trace=trace, clean_summary=output / 'summary.json',
                              resume_state=model / 'resume_states' / f'{epoch:02d}.pth').items():
            row[key], row[key + '_sha256'] = str(path), common.sha(path)
        write(directory / 'complete.json', row)
        if variant == 'D':
            write(model / 'pairing' / f'{epoch:02d}.json', dict(exact=True, consumed_frames=8080,
                D_trace_sha256=row['trace_sha256'],
                C_trace_sha256=common.read(model.parent / 'C/epochs' / f'{epoch:02d}/complete.json')['trace_sha256']))
    if variant == 'D':
        write(model / 'initialization_audit.json', dict(exact=True,
            C_initial_sha256=common.sha(model.parent / 'C/resume_states/00.pth')))
    return model


class FullDataGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scope = sandbox()
        cls.root = cls.scope.__enter__()
        try:
            cls.c, cls.d = fixture(cls.root, 'C'), fixture(cls.root, 'D')
        except BaseException:
            cls.scope.__exit__(*sys.exc_info())
            raise

    @classmethod
    def tearDownClass(cls):
        cls.scope.__exit__(None, None, None)

    def test_full_c_and_d_gate_accept_zero_ap(self):
        for variant, root in (('C', self.c), ('D', self.d)):
            result = audit_warmup(root, 'T1', variant)
            self.assertEqual(result['status'], 'formal_warmup_complete')
            self.assertEqual(result['completed_gate_epochs'], [1, 2])
            self.assertTrue(result['no_BN_changes'])
            self.assertGreater(len(result['files']), 1658)

    def test_short_smoke_refused(self):
        path = self.c / 'epochs/02/complete.json'
        row = common.read(path)
        row.update(smoke_only=True, train_updates=4, consumed_input_frames=32)
        with changed(path, row), self.assertRaisesRegex(ValueError, 'full-data'):
            audit_warmup(self.c, 'T1', 'C')

    def test_wrong_full_update_count_refused(self):
        path = self.c / 'epochs/01/complete.json'
        row = common.read(path)
        row['train_updates'] = 1009
        with changed(path, row), self.assertRaisesRegex(ValueError, 'full-data'):
            audit_warmup(self.c, 'T1', 'C')

    def test_exact_restore_proof_required(self):
        path = self.c / 'restore_audit/02.json'
        row = common.read(path)
        row['exact_model_Adam_scheduler_criterion_RNG'] = False
        with changed(path, row), self.assertRaisesRegex(ValueError, 'restore'):
            audit_warmup(self.c, 'T1', 'C')

    def test_weight_file_cannot_replace_full_state(self):
        path = self.c / 'resume_states/02.pth'
        raw = path.read_bytes()
        try:
            path.write_bytes(b'SYNTHETIC WEIGHTS ONLY')
            with self.assertRaisesRegex(ValueError, 'artifact changed'):
                audit_warmup(self.c, 'T1', 'C')
        finally:
            path.write_bytes(raw)

    def test_missing_restore_file_refused(self):
        path = self.c / 'restore_audit/01.json'
        raw = path.read_bytes()
        try:
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                audit_warmup(self.c, 'T1', 'C')
        finally:
            path.write_bytes(raw)

    def test_tampered_individual_prediction_refused(self):
        path = self.c / 'epochs/01/clean_ap_synthetic/predictions/clean/000828.json'
        with changed(path, dict(synthetic='corruption')), self.assertRaisesRegex(ValueError, 'artifact changed'):
            audit_warmup(self.c, 'T1', 'C')

    def test_truncated_829_frame_manifest_refused_even_with_rebound_hashes(self):
        output = self.c / 'epochs/01/clean_ap_synthetic'
        manifest = common.read(output / 'prediction_manifest.json')
        manifest['files'].pop()
        summary = common.read(output / 'summary.json')
        row = common.read(output.parent / 'complete.json')
        with changed(output / 'prediction_manifest.json', manifest):
            summary['output_hashes']['prediction_manifest.json'] = common.sha(output / 'prediction_manifest.json')
            with changed(output / 'summary.json', summary):
                row['clean_summary_sha256'] = common.sha(output / 'summary.json')
                with changed(output.parent / 'complete.json', row), self.assertRaisesRegex(ValueError, 'prediction cache'):
                    audit_warmup(self.c, 'T1', 'C')

    def test_gt_totals_not_forged_to_accept(self):
        output = self.c / 'epochs/01/clean_ap_synthetic'
        summary = common.read(output / 'summary.json')
        summary['results']['clean']['counts']['official_gt'] = 6200
        row = common.read(output.parent / 'complete.json')
        with changed(output / 'summary.json', summary):
            row['clean_summary_sha256'] = common.sha(output / 'summary.json')
            with changed(output.parent / 'complete.json', row), self.assertRaisesRegex(ValueError, 'GT identity'):
                audit_warmup(self.c, 'T1', 'C')

    def test_wrong_recipe_or_variant_refused(self):
        for recipe, variant in (('T2', 'C'), ('T1', 'D')):
            with self.subTest(recipe=recipe, variant=variant), self.assertRaisesRegex(ValueError, 'contract'):
                audit_warmup(self.c, recipe, variant)

    def test_contract_not_rebound_implicitly(self):
        path = self.c / 'contract.json'
        row = common.read(path)
        row['epochs'] = 22
        with changed(path, row), self.assertRaisesRegex(ValueError, 'contract'):
            audit_warmup(self.c, 'T1', 'C')

    def test_config_bytes_must_match_original(self):
        path = self.c / 'config.yaml'
        raw = path.read_bytes()
        try:
            path.write_bytes(b'synthetic: changed\n')
            with self.assertRaisesRegex(ValueError, 'contract'):
                audit_warmup(self.c, 'T1', 'C')
        finally:
            path.write_bytes(raw)

    def test_d_requires_exact_c_d_data_pairing(self):
        path = self.d / 'pairing/02.json'
        row = common.read(path)
        row['exact'] = False
        with changed(path, row), self.assertRaisesRegex(ValueError, 'pairing'):
            audit_warmup(self.d, 'T1', 'D')

    def test_d_requires_shared_initial_state_proof(self):
        path = self.d / 'initialization_audit.json'
        row = common.read(path)
        row['C_initial_sha256'] = 'changed'
        with changed(path, row), self.assertRaisesRegex(ValueError, 'initial'):
            audit_warmup(self.d, 'T1', 'D')

    def test_gate_survives_mutable_history_and_later_epochs(self):
        before = audit_warmup(self.c, 'T1', 'C')
        path = publish_warmup(self.c, 'T1', 'C')
        digest = common.sha(path)
        write(self.c / 'history.json', dict(synthetic=True, epochs=20))
        write(self.c / 'progress.json', dict(completed_epoch=20))
        write(self.c / 'epochs/20/complete.json', dict(synthetic=True))
        after = audit_warmup(self.c, 'T1', 'C')
        self.assertEqual(before, after)
        publish_warmup(self.c, 'T1', 'C')
        self.assertEqual(common.sha(path), digest)
        self.assertNotIn(str(self.c / 'history.json'), after['files'])


class RuntimeBridge(unittest.TestCase):
    def setup_runtime(self, count=2):
        selection = Mock(return_value='original selected')
        checkpoint = types.SimpleNamespace(committed=Mock(return_value=[{}] * count))
        evaluation = types.SimpleNamespace(selection=selection)
        install = Mock(return_value=('args', 'audit', 'training', checkpoint, evaluation))
        restore = Mock()
        module = types.SimpleNamespace(install=install, install_restore_check=restore)
        return module, evaluation, checkpoint, selection, restore

    def test_warmup_preserves_formal_flag_and_defers_only_final_selection(self):
        module, evaluation, checkpoint, selection, restore = self.setup_runtime()
        cli = types.SimpleNamespace(stage='warmup', smoke=False)
        with patch.dict(sys.modules, e1_runtime=module):
            active = activate(cli)
            active.install(cli)
            with patch('warm_runtime.read', return_value=dict(identity='synthetic')):
                self.assertIsNone(evaluation.selection(common.PROJECT))
            selection.assert_not_called()
            checkpoint.committed.assert_called_once_with(common.PROJECT, 20, 'synthetic')
            active.install_restore_check('checkpoint', 'root', False)
            restore.assert_called_once_with('checkpoint', 'root', True)
            self.assertIs(cli.smoke, False)

    def test_warmup_calls_original_selection_if_twenty_are_complete(self):
        module, evaluation, _, selection, _ = self.setup_runtime(20)
        cli = types.SimpleNamespace(stage='warmup', smoke=False)
        with patch.dict(sys.modules, e1_runtime=module):
            activate(cli).install(cli)
            with patch('warm_runtime.read', return_value=dict(identity='synthetic')):
                self.assertEqual(evaluation.selection(common.PROJECT), 'original selected')
            selection.assert_called_once()

    def test_continuation_preserves_original_selection_and_restore_wrapper(self):
        module, evaluation, _, selection, restore = self.setup_runtime()
        install = module.install
        with patch.dict(sys.modules, e1_runtime=module):
            activate(types.SimpleNamespace(stage='train'))
            self.assertIs(module.install, install)
            self.assertIs(evaluation.selection, selection)
            self.assertIs(module.install_restore_check, restore)

    def test_clean_eval_uses_new_worker_full_validation(self):
        module, _, _, _, _ = self.setup_runtime()
        cli = types.SimpleNamespace(stage='train', original_root=Path('original'), recipe='T1', variant='C')
        args = types.SimpleNamespace(output_root=Path('model'), train_root=Path('data'))
        result = dict(status='complete', checkpoint_sha256='synthetic-sha', conditions=['clean'],
                      frame_count=829, full_validation=True)
        with patch.dict(sys.modules, e1_runtime=module), patch.object(BASE, 'run') as run:
            active = activate(cli)
            with patch('warm_runtime.read', return_value=result), patch('warm_runtime.sha', return_value='synthetic-sha'):
                self.assertEqual(active.clean_ap(cli, args, Path('checkpoint'), Path('destination')), result)
            command = list(map(str, run.call_args.args[0]))
            self.assertEqual(command[2], str(common.HERE / 'worker.py'))
            self.assertIn('--clean-only', command)
            self.assertIn('--original-root', command)
            self.assertNotIn('--smoke', command)

    def test_short_clean_validation_cannot_be_promoted(self):
        module, _, _, _, _ = self.setup_runtime()
        cli = types.SimpleNamespace(stage='train', original_root=Path('original'), recipe='T1', variant='C')
        args = types.SimpleNamespace(output_root=Path('model'), train_root=Path('data'))
        result = dict(status='complete', checkpoint_sha256='synthetic', conditions=['clean'],
                      frame_count=12, full_validation=False)
        with patch.dict(sys.modules, e1_runtime=module), patch.object(BASE, 'run'):
            active = activate(cli)
            with patch('warm_runtime.read', return_value=result), patch('warm_runtime.sha', return_value='synthetic'):
                with self.assertRaisesRegex(ValueError, 'incomplete'):
                    active.clean_ap(cli, args, Path('checkpoint'), Path('destination'))


class ControllerFlow(unittest.TestCase):
    def cli(self, root):
        return types.SimpleNamespace(output_root=root, train_root=root / 'data', original_root=root / 'original',
            e0_existing_root=root / 'oldE0', gpu='2')

    def test_all_four_full_gates_precede_all_continuations(self):
        with sandbox() as root:
            control = Controller(self.cli(root), BASE, dict(identity='synthetic'))
            write(root / 'input_identity.json', dict(synthetic=True))
            write(root / 'execution_identity.json', dict(synthetic=True))
            seen = []
            def stage(name, command, artifacts):
                seen.append(name)
                if name.startswith('gate_'):
                    model = Path(command[command.index('--model-root') + 1])
                    write(model / 'formal_warmup/receipt.json', dict(status='formal_warmup_complete', files={}))
                if name.endswith('_train'):
                    gate = common.read(root / 'formal_warmup_gate.json')
                    self.assertEqual(len(gate['models']), 4)
                    model = Path(command[command.index('--model-root') + 1])
                    write(model / 'selected.json', dict(checkpoint=str(model / 'selected_synthetic.pth')))
                if name == 'synthesis':
                    (root / 'reports').mkdir()
                    (root / 'reports/E01_FINAL_CONCLUSIONS.md').write_text('SYNTHETIC', encoding='utf-8')
            control.stage = stage
            with patch('sys.stdout', io.StringIO()):
                control.execute()
            self.assertEqual(seen[:6], ['E0_import', 'E1_self_test', 'gate_T1_C', 'gate_T1_D', 'gate_T2_C', 'gate_T2_D'])
            self.assertEqual(len([n for n in seen if n.endswith('_train')]), 4)
            gate = common.read(root / 'formal_warmup_gate.json')
            self.assertEqual(gate['formal_epochs_already_counted'] + gate['remaining_epochs'], list(range(1, 21)))
            self.assertTrue(common.read(root / 'completion.json')['scientific_completion'])
            self.assertNotIn('E0_train', seen)

    def test_completed_stage_is_reused_after_hash_verification(self):
        with sandbox() as root:
            control = Controller(self.cli(root), BASE, dict(identity='synthetic'))
            artifact = root / 'artifact.json'
            def complete(*args, **kwargs):
                write(artifact, dict(synthetic=True))
            with patch.object(BASE, 'run', side_effect=complete) as run, patch('sys.stdout', io.StringIO()):
                for _ in range(2):
                    control.stage('synthetic', ['never_execute'], lambda: {str(artifact): common.sha(artifact)})
            self.assertEqual(run.call_count, 1)

    def test_corrupt_completed_stage_fails_before_worker_launch(self):
        with sandbox() as root:
            control = Controller(self.cli(root), BASE, dict(identity='synthetic'))
            artifact = root / 'artifact.json'
            write(artifact, dict(synthetic=True))
            with patch.object(BASE, 'run'):
                control.stage('synthetic', ['never_execute'], lambda: {str(artifact): common.sha(artifact)})
            write(artifact, dict(synthetic='corrupt'))
            with patch.object(BASE, 'run') as run, self.assertRaisesRegex(ValueError, 'changed'):
                control.stage('synthetic', ['never_execute'], lambda: {})
            run.assert_not_called()

    def test_existing_output_does_not_mean_stage_complete(self):
        with sandbox() as root:
            control = Controller(self.cli(root), BASE, dict(identity='synthetic'))
            write(root / '.stages/synthetic/artifacts.json', dict(status='complete', files={}))
            self.assertIs(control.verify('synthetic'), False)

    def test_stage_receipt_bound_to_whole_run_identity(self):
        with sandbox() as root:
            control = Controller(self.cli(root), BASE, dict(identity='synthetic-A'))
            artifact = root / 'artifact.json'
            write(artifact, dict(synthetic=True))
            with patch.object(BASE, 'run'):
                control.stage('synthetic', ['never_execute'], lambda: {str(artifact): common.sha(artifact)})
            other = Controller(self.cli(root), BASE, dict(identity='synthetic-B'))
            with self.assertRaisesRegex(ValueError, 'request differs'):
                other.verify('synthetic')

    def test_input_identity_rebound_is_refused(self):
        with sandbox() as root:
            row = dict(status='complete', train_frames=8083, validation_frames=829)
            keys = ('sources', 'train_manifest_sha256', 'train_manifest_identity', 'train_data_acceptance',
                'three_weather_training_identity', 'input_identity', 'frozen_GT_identity', 'T0_reference')
            row.update({k: 'synthetic-' + k for k in keys})
            path = root / 'preflight/preflight.json'
            write(path, row)
            bind_inputs(BASE, root)
            bind_inputs(BASE, root)
            row['three_weather_training_identity'] = 'changed'
            write(path, row)
            with self.assertRaisesRegex(ValueError, 'Immutable'):
                bind_inputs(BASE, root)

    def test_legacy_e0_auto_discovery_and_verifier_are_required(self):
        with sandbox() as root:
            legacy = root / 'legacy'
            cli = self.cli(root / 'new')
            cli.original_root, cli.train_root = root / 'original', root / 'data'
            cli.legacy_run_root, cli.e0_existing_root = legacy, None
            original_sources = dict(synthetic='original frozen sources')
            request = dict(source=original_sources, output=str(legacy), original=str(cli.original_root),
                           train=str(cli.train_root), E0_existing=None)
            request['identity'] = BASE.identity(request)
            write(legacy / 'request.json', request)
            verifier = Mock(return_value=(legacy / 'E0', legacy / 'E0/stages/final'))
            with patch('controller.base_sources', return_value=original_sources), patch('controller.frozen_worker',
                    return_value=types.SimpleNamespace(verified_e0_source=verifier)):
                result, provenance = existing_e0(cli, BASE)
                self.assertEqual(result, legacy / 'E0')
                verifier.assert_called_once_with(legacy / 'E0', cli.original_root)
                self.assertEqual(provenance['request_sha256'], common.sha(legacy / 'request.json'))
                verifier.side_effect = ValueError('E0 incomplete')
                with self.assertRaisesRegex(ValueError, 'incomplete'):
                    existing_e0(cli, BASE)


class SourceContract(unittest.TestCase):
    def test_old_seventeen_files_and_transitive_dependencies_are_unchanged(self):
        own = common.own_sources()
        original = common.base_sources()
        self.assertEqual(len(common.read(common.HERE / 'PINS.json')['files']), 17)
        self.assertEqual(len(original['dependencies']), 445)
        self.assertIn('policy.json', own)

    def test_all_new_python_is_310_grammar_utf8_lf(self):
        for path in common.HERE.iterdir():
            if path.is_file() and path.suffix in ('.py', '.json', '.sh', '.md'):
                raw = path.read_bytes()
                raw.decode('utf-8')
                self.assertNotIn(b'\r', raw, str(path))
                if path.suffix == '.py':
                    ast.parse(raw, filename=str(path), feature_version=(3, 10))

    def test_runtime_entry_refuses_local_scientific_execution(self):
        with patch.object(sys, 'platform', 'win32'), self.assertRaisesRegex(RuntimeError, 'server-only'):
            BASE.server_only()
        with patch.object(sys, 'platform', 'win32'), self.assertRaisesRegex(RuntimeError, 'Server execution only'):
            with common.short_temp():
                self.fail('Local IPC directory allocated')

    def test_request_hash_binds_gpu_environment_source_data_and_policy(self):
        value = dict(gpu='2', source='synthetic', data='synthetic', threads={'OMP': None}, policy=copy.deepcopy(common.POLICY))
        identity = BASE.identity(value)
        for key, change in (('gpu', '3'), ('source', 'changed'), ('data', 'changed'),
                            ('threads', {'OMP': '8'}), ('policy', {'formal_epochs': 22})):
            candidate = dict(value, **{key: change})
            self.assertNotEqual(BASE.identity(candidate), identity)

    def test_new_code_contains_no_bn_nms_or_parameter_mutation(self):
        for name in ('warm_common.py', 'warm_adapter.py', 'warm_runtime.py', 'controller.py', 'worker.py'):
            tree = ast.parse((common.HERE / name).read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    called = ast.unparse(node.func)
                    self.assertFalse(called.endswith(('.train', '.eval', '.backward', '.step', '.load_state_dict'))
                        and called != 'runtime.train', called)
                    self.assertNotIn(called, ('torch.load', 'torch.save'))


class IPCAndSignals(unittest.TestCase):
    def test_short_private_env_restored_even_on_failure(self):
        directory = '/data/cjm/datasets/tmp/gw-synthetic'
        before = {key: os.environ.get(key) for key in ('TMPDIR', 'TEMP', 'TMP')}
        fake = Mock()
        fake.__enter__ = Mock(return_value=directory)
        fake.__exit__ = Mock(return_value=False)
        with patch.object(sys, 'platform', 'linux'), patch('warm_common.Path.mkdir'), \
                patch('warm_common.tempfile.TemporaryDirectory', return_value=fake) as allocate:
            with self.assertRaisesRegex(RuntimeError, 'synthetic'):
                with common.short_temp():
                    self.assertEqual({os.environ[key] for key in before}, {directory})
                    raise RuntimeError('synthetic failure')
            self.assertEqual(allocate.call_args.kwargs['prefix'], 'gw-')
            fake.__exit__.assert_called_once()
        self.assertEqual({key: os.environ.get(key) for key in before}, before)

    def test_socket_probe_performs_actual_kind_of_bind_with_mock_socket(self):
        directory = '/data/cjm/datasets/tmp/gw-synthetic'
        listener = Mock(address=directory + '/pymp-123/listener-123')
        factory = Mock()
        factory.return_value.__enter__ = Mock(return_value=listener)
        factory.return_value.__exit__ = Mock(return_value=False)
        environment = {key: directory for key in ('TMPDIR', 'TEMP', 'TMP')}
        with patch.dict(os.environ, environment), patch('warm_common.tempfile.gettempdir', return_value=directory), \
                patch('multiprocessing.connection.Listener', factory):
            result = common.ipc_probe()
        factory.assert_called_once_with(family='AF_UNIX')
        self.assertEqual(result['status'], 'complete')
        self.assertLessEqual(result['socket_path_bytes'], 107)

    def test_cached_tmp_mismatch_fails_before_bind(self):
        with patch.dict(os.environ, dict(TMPDIR='/long', TEMP='/long', TMP='/long')), \
                patch('warm_common.tempfile.gettempdir', return_value='/different'), \
                patch('multiprocessing.connection.Listener') as factory:
            with self.assertRaisesRegex(ValueError, 'cache differs'):
                common.ipc_probe()
            factory.assert_not_called()

    def test_oversized_or_outside_socket_refused(self):
        directory = '/data/cjm/datasets/tmp/gw-synthetic'
        environment = {key: directory for key in ('TMPDIR', 'TEMP', 'TMP')}
        for address in ('/elsewhere/listener', directory + '/' + 'x' * 108):
            factory = Mock()
            factory.return_value.__enter__ = Mock(return_value=Mock(address=address))
            factory.return_value.__exit__ = Mock(return_value=False)
            with self.subTest(address=address), patch.dict(os.environ, environment), \
                    patch('warm_common.tempfile.gettempdir', return_value=directory), \
                    patch('multiprocessing.connection.Listener', factory), self.assertRaisesRegex(ValueError, 'socket'):
                common.ipc_probe()

    def test_bind_failure_stops_before_models(self):
        directory = '/data/cjm/datasets/tmp/gw-synthetic'
        environment = {key: directory for key in ('TMPDIR', 'TEMP', 'TMP')}
        with patch.dict(os.environ, environment), patch('warm_common.tempfile.gettempdir', return_value=directory), \
                patch('multiprocessing.connection.Listener', side_effect=OSError('synthetic bind error')):
            with self.assertRaisesRegex(OSError, 'bind error'):
                common.ipc_probe()

    def fake_signals(self, handlers):
        def register(number, handler):
            handlers[number] = handler
        return types.SimpleNamespace(SIGTERM=15, SIGINT=2, SIGHUP=1, SIG_IGN='ignored',
            getsignal=lambda n: handlers[n], signal=Mock(side_effect=register),
            Signals=lambda n: types.SimpleNamespace(name={15: 'SIGTERM', 2: 'SIGINT', 1: 'SIGHUP'}[n]))

    def test_nohup_inherited_ignored_signals_stay_ignored(self):
        handlers = {15: 'original', 2: 'ignored', 1: 'ignored'}
        signal_api = self.fake_signals(handlers)
        with common.signal_handlers(signal_api):
            self.assertEqual(handlers[1], 'ignored')
            self.assertEqual(handlers[2], 'ignored')
            self.assertTrue(callable(handlers[15]))
        self.assertEqual(handlers, {15: 'original', 2: 'ignored', 1: 'ignored'})
        self.assertEqual([row.args[0] for row in signal_api.signal.call_args_list], [15, 15])

    def test_repeated_signals_do_not_interrupt_child_cleanup(self):
        handlers = {15: 'original', 2: 'original', 1: 'original'}
        signal_api = self.fake_signals(handlers)
        event = Mock()
        with patch('warm_common.bridge', return_value=types.SimpleNamespace(STOP=event)):
            with common.signal_handlers(signal_api):
                callback = handlers[15]
                with self.assertRaisesRegex(KeyboardInterrupt, 'SIGTERM'):
                    callback(15, None)
                self.assertIsNone(callback(15, None))
                event.set.assert_called_once()
        self.assertEqual(handlers, {15: 'original', 2: 'original', 1: 'original'})

    def test_signal_handlers_restored_on_unrelated_exception(self):
        handlers = {15: 'original', 2: 'original', 1: 'original'}
        signal_api = self.fake_signals(handlers)
        with self.assertRaisesRegex(RuntimeError, 'synthetic'):
            with common.signal_handlers(signal_api):
                raise RuntimeError('synthetic')
        self.assertEqual(handlers, {15: 'original', 2: 'original', 1: 'original'})


def main():
    global BASE
    BASE = common.bridge()
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if 'torch' in sys.modules or 'opencood' in sys.modules:
        raise RuntimeError('Synthetic tests imported a forbidden real research runtime')
    if not result.wasSuccessful():
        raise SystemExit(1)
    print('PASS: stdlib synthetic checks only; no model, real weights, point cloud or experiment run')


if __name__ == '__main__':
    main()
