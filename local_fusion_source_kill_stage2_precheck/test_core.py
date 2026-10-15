"""Pure schema/fixture tests; no Torch, detector, GPU or real S0 data."""
from collections import Counter
from copy import deepcopy
import itertools
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from .common import ROOT, KEEP, WEATHERS, atomic_json, digest, scene_summary
from .source_redundancy import analyze_group, action_identity, build_cells, percentile
from .data_adequacy import h1_from_summary, h2_from_summary, separation_proxy
from .oracle_records import analyze_target, summarize_details
from .driver import create_results, write_outputs
from .inputs import load_reports, read_oracle_details, read_s0


def temporary_directory():
    folder = ROOT / '.tmp'
    folder.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(dir=folder, prefix='p0_fixture_')


def outcome(tp=2, score=.6):
    return dict(baseline_tp=2, action_tp=tp, lost=max(0, 2-tp), recovered=max(0, tp-2),
                new_fp=0, action_fp=0, focal_detected_after=True,
                focal_score_after=score, focal_iou_after=.8)


def fixture_data():
    from local_fusion_source_kill_stage1.k0_group_margin import audit_group, aggregate
    k0 = {'conditions': {}}
    raw, pairs = [], []
    names = [KEEP, 'single:0', 'query:0', 'single:1', 'query:1']
    for split, weather, stratum in itertools.product(('train', 'validation'), WEATHERS, ('associated', 'background')):
        per_task = {}
        for task in ('cls', 'reg'):
            values = [outcome(), outcome(3), outcome(3), outcome(3 if task == 'cls' else 1), outcome(1)]
            if stratum == 'background':
                values = [outcome() for _ in names]
            old = dict(audit_group(values, names), scene=0)
            k0['conditions'][f'{split}/{weather}/{task}/{stratum}'] = aggregate([old])
            row = dict(analyze_group(values, names), split=split, weather=weather, task=task,
                       stratum=stratum, frame=0, scene_id=split + ':0',
                       physical_id=[split, 0, 10, None if stratum == 'background' else 0],
                       train_partition='probe_fit')
            raw.append(row)
            per_task[task] = row
        pairs.append({**{k: per_task['cls'][k] for k in ('split', 'weather', 'stratum', 'scene_id')},
                      **separation_proxy(per_task['cls'], per_task['reg'])})
    k10 = {'conditions': {}, 'reproduction_gate': {'status': 'complete',
            'absolute_ap_tolerance': 1e-6, 'checks': {}}}
    for weather, variant, mode in itertools.product(WEATHERS, ('FULL', 'A_ONLY'), ('same', 'task')):
        k10['conditions'][f'{weather}_{mode}_{variant}'] = {'counts': dict(
            recovered=1 if mode == 'same' else 2, selected_A=1, task_separated=mode == 'task')}
        k10['reproduction_gate']['checks'][f'{weather}_{mode}'] = {'Shared': dict(ap30=0, ap50=0, ap70=0),
                                                                'Full': dict(ap30=0, ap50=0, ap70=0)}
    return k0, k10, raw, pairs


def target_fixture(mode='task', task_tp=4, cls='single:0', reg='query:1', variant='FULL'):
    keep = (2, 0, 0, 1, 0, 0, .6, .8, 2)
    same = (3, 0, 0, 1, 1, 0, .7, .8, 1)
    task = (task_tp, 0, 0, 1, task_tp-2, 0, .9, .8, 0)
    actions = [dict(cls_source='single:0', reg_source='single:0', group='A', key=list(same), allowed=True)]
    if mode == 'task':
        actions.append(dict(cls_source=cls, reg_source=reg, group='A', key=list(task), allowed=True))
    else:
        task, cls, reg = same, 'single:0', 'single:0'
    audit = dict(target_index=0, keep_key=list(keep), actions=actions, selected_key=list(task),
                 selected_group='A', shortlisted={'A': len(actions)})
    record = dict(target_index=0, cls_source=cls, reg_source=reg, changed=True,
                  task_separated=mode == 'task', matched_after=task_tp if mode == 'task' else 3, fp_after=0)
    return record, audit


class RedundancyTests(unittest.TestCase):
    def test_same_source_two_families(self):
        row = analyze_group([outcome(), outcome(3), outcome(3)], [KEEP, 'single:2', 'query:2'])
        self.assertEqual(row['unique_source_count'], 1)
        self.assertEqual(row['equivalent_best_action_count'], 2)
        self.assertTrue(row['same_source_cross_family'])
        self.assertFalse(row['cross_source_redundancy'])

    def test_cross_source(self):
        row = analyze_group([outcome(), outcome(3), outcome(3)], [KEEP, 'single:1', 'query:3'])
        self.assertEqual(row['unique_source_count'], 2)
        self.assertTrue(row['MODIFY_STABLE_SOURCE_AMBIGUOUS'])
        self.assertFalse(row['same_source_cross_family'])

    def test_family_and_source_overlap(self):
        row = analyze_group([outcome(), outcome(3), outcome(3), outcome(3)],
                            [KEEP, 'single:1', 'query:1', 'query:2'])
        self.assertTrue(row['same_source_cross_family'])
        self.assertTrue(row['cross_source_redundancy'])

    def test_continuous_changes_not_beneficial(self):
        row = analyze_group([outcome(), outcome(score=.9)], [KEEP, 'single:0'])
        self.assertFalse(row['beneficial'])
        self.assertEqual(row['equivalent_best_actions'], [])

    def test_score_cannot_split_discrete_best(self):
        row = analyze_group([outcome(), outcome(3, .1), outcome(3, .9)], [KEEP, 'single:0', 'query:1'])
        self.assertEqual(row['unique_source_count'], 2)

    def test_original_lexicographic_order(self):
        keep, improved = outcome(), outcome()
        keep['action_fp'] = 1
        row = analyze_group([keep, improved], [KEEP, 'single:0'])
        self.assertTrue(row['beneficial'])

    def test_keep_only_empty_best(self):
        row = analyze_group([outcome()], [KEEP])
        self.assertFalse(row['beneficial'])

    def test_permutation_invariance(self):
        actions = [('single:0', outcome(3)), ('query:0', outcome(3)), ('single:1', outcome(3))]
        expected = None
        for perm in itertools.permutations(actions):
            row = analyze_group([outcome()] + [a[1] for a in perm], [KEEP] + [a[0] for a in perm])
            expected = expected or row
            self.assertEqual(row, expected)

    def test_source_identity_canonical(self):
        self.assertEqual(action_identity('query:12'), ('query', 12))
        for name in ('query:01', 'foo:1', 'single:-1', 'single:1:2'):
            with self.assertRaises(ValueError):
                action_identity(name)

    def test_quantile(self):
        self.assertEqual(percentile([1, 2, 3], .9), 2.8)
        self.assertIsNone(percentile([], .9))

    def test_pooled_scenes_and_dedup(self):
        _, _, raw, _ = fixture_data()
        cells = build_cells(raw)
        row = cells['train+validation/adverse pooled/cls/associated']
        self.assertEqual(row['beneficial_discrete_groups'], 6)
        self.assertEqual(row['scene_coverage']['scene_count'], 2)
        self.assertEqual(row['unique_physical_positive_groups'], 2)
        self.assertEqual(row['scene_coverage']['max_scene_share'], .5)
        self.assertEqual(cells['train/adverse pooled/cls/background']['beneficial_discrete_groups'], 0)

    def test_equal_weight_not_pooled_fraction(self):
        _, _, raw, _ = fixture_data()
        extra = deepcopy(next(r for r in raw if (r['split'], r['weather'], r['task'], r['stratum']) ==
                              ('validation', 'fog', 'cls', 'associated')))
        extra.update(beneficial=False, physical_id=['validation', 1, 10, 0])
        raw += [deepcopy(extra) for _ in range(9)]
        cells = build_cells(raw)
        macro = cells['validation/adverse equal-weight/cls/associated']
        self.assertAlmostEqual(macro['positive_fraction'], (.1+1+1)/3)
        self.assertNotAlmostEqual(macro['positive_fraction'], cells['validation/adverse pooled/cls/associated']['positive_fraction'])


class AdequacyTests(unittest.TestCase):
    def test_exact_summary_counts(self):
        k0, _, _, _ = fixture_data()
        cells = h1_from_summary(k0)
        self.assertEqual(cells['train/adverse pooled/reg/associated']['beneficial_discrete_groups'], 3)
        self.assertEqual(cells['train+validation/adverse pooled/reg/associated']['scene_coverage']['scene_count'], 2)
        self.assertIsNone(cells['validation/clean/reg/associated']['unique_source_beneficial_groups'])

    def test_train_h2_unknown_not_zero(self):
        _, k10, _, _ = fixture_data()
        cells = h2_from_summary(k10, 'FULL')
        self.assertIsNone(cells['train/adverse pooled']['task_extra_recovered_targets'])
        self.assertIsNone(cells['train+validation/adverse pooled']['task_extra_recovered_targets'])
        self.assertEqual(cells['train+validation/adverse pooled']['observed_validation_extra_recovered'], 3)
        self.assertIsNone(cells['validation/adverse pooled']['true_task_improvement_targets'])
        self.assertEqual(cells['validation/adverse pooled']['joint_same_positive_target_groups'], 3)

    def test_proxy_is_not_positive(self):
        a = analyze_group([outcome(), outcome(3)], [KEEP, 'single:0'])
        b = analyze_group([outcome(), outcome(3)], [KEEP, 'query:1'])
        row = separation_proxy(a, b)
        self.assertTrue(row['proxy_disjoint_best_cav_sets'])
        self.assertFalse(row['is_true_oracle_positive'])

    def test_proxy_requires_both_beneficial(self):
        a = analyze_group([outcome(), outcome(3)], [KEEP, 'single:0'])
        b = analyze_group([outcome(), outcome(score=.9)], [KEEP, 'query:1'])
        self.assertFalse(separation_proxy(a, b)['proxy_disjoint_best_cav_sets'])

    def test_h1_tasks_not_summed_into_joint(self):
        k0, k10, raw, pairs = fixture_data()
        _, b, result = create_results(k0, k10, raw, pairs)
        self.assertIsNone(b['joint_same_s0_candidate_level']['positive_n'])
        self.assertFalse(result['recommendations']['frozen'])
        self.assertTrue(result['k3_frozen'])


class OracleRecordTests(unittest.TestCase):
    def test_true_same_state_discrete_improvement(self):
        record, audit = target_fixture()
        row = analyze_target(record, audit, 'task', 'FULL')
        self.assertTrue(row['true_task_improvement'])
        self.assertTrue(row['cross_cav_selected'])

    def test_score_only_task_advantage_not_positive(self):
        record, audit = target_fixture(task_tp=3)
        row = analyze_target(record, audit, 'task', 'FULL')
        self.assertFalse(row['true_task_improvement'])

    def test_same_cav_cross_family_not_cross_cav(self):
        record, audit = target_fixture(reg='query:0')
        row = analyze_target(record, audit, 'task', 'FULL')
        self.assertTrue(row['same_cav_cross_family_selected'])
        self.assertFalse(row['cross_cav_selected'])

    def test_keep_source_separation(self):
        record, audit = target_fixture(reg=KEEP)
        row = analyze_target(record, audit, 'task', 'FULL')
        self.assertTrue(row['keep_vs_source_separated'])
        self.assertFalse(row['cross_cav_selected'])

    def test_same_true_joint_positive(self):
        record, audit = target_fixture(mode='same')
        row = analyze_target(record, audit, 'same', 'A_ONLY')
        self.assertTrue(row['selected_discrete_positive'])
        self.assertFalse(row['true_task_improvement'])

    def test_corrupt_partition_rejected(self):
        record, audit = target_fixture()
        audit['actions'][0]['group'] = 'B'
        with self.assertRaises(ValueError):
            analyze_target(record, audit, 'task', 'FULL')

    def test_current_keep_and_winner_order_rejected(self):
        record, audit = target_fixture()
        audit['keep_key'][-1] = 1
        with self.assertRaises(ValueError):
            analyze_target(record, audit, 'task', 'FULL')
        record, audit = target_fixture()
        audit['selected_key'] = audit['actions'][0]['key']
        with self.assertRaises(ValueError):
            analyze_target(record, audit, 'task', 'FULL')

    def test_true_train_pooled_unavailable(self):
        record, audit = target_fixture()
        row = dict(analyze_target(record, audit, 'task', 'FULL'), split='validation',
                   weather='fog', frame=1, scene_id='validation:0', mode='task')
        result = summarize_details([row])
        self.assertEqual(result['validation/adverse pooled']['true_task_improvement_targets'], 1)
        self.assertIsNone(result['train+validation/adverse pooled']['true_task_improvement_targets'])


class SafetyAndSmokeTests(unittest.TestCase):
    def test_s0_fixture_io_with_cpu_deserializer(self):
        k0, _, _, _ = fixture_data()
        names = [KEEP, 'single:0', 'query:0', 'single:1', 'query:1']
        with temporary_directory() as directory:
            run = Path(directory)
            s0 = dict(train_indices=[0], validation_indices=[0], train_scenes=[0], validation_scenes=[0])
            manifest = {'entries': {}}
            for split, weather in itertools.product(('train', 'validation'), WEATHERS):
                labels = []
                for position in (0, 1):
                    outcomes = {}
                    for task in ('cls', 'reg'):
                        values = [outcome(), outcome(3), outcome(3), outcome(3 if task == 'cls' else 1), outcome(1)]
                        if position == 1:
                            values = [outcome() for _ in names]
                        outcomes[task] = values
                    labels.append(dict(proposal_position=position, target_index=0 if position == 0 else None,
                                       background=position == 1, outcomes=outcomes))
                frame = dict(split=split, weather=weather, frame=0, scene=0, source_names=names,
                             proposals={'anchor_ids': [10, 20]}, labels=labels)
                name = f'cache/{split}/{weather}/00000000.pt'
                atomic_json(run / name, frame)  # Fake serialization, not a Torch tensor/cache.
                manifest['entries'][f'S0-frame/{split}/{weather}/0'] = {
                    'status': 'complete', 'artifacts': {name: digest(run / name)}}
            calls = []
            def deserialize(path, **kwargs):
                calls.append(kwargs)
                return json.loads(Path(path).read_text(encoding='utf-8'))
            with patch('local_fusion_source_kill_stage2_precheck.inputs.full_inputs',
                       return_value=({}, run, s0, manifest)), patch.dict('os.environ', {}), patch.dict(
                       sys.modules, {'torch': SimpleNamespace(load=deserialize)}):
                rows, pairs, mapping, _ = read_s0(run, {}, k0)
            self.assertEqual(len(rows), 32)
            self.assertEqual(len(pairs), 16)
            self.assertEqual(mapping['validation', 0], 'validation:0')
            self.assertTrue(all(call == {'map_location': 'cpu', 'weights_only': True} for call in calls))

    def test_saved_oracle_jsonl_schema_and_hash(self):
        with temporary_directory() as directory:
            folder = Path(directory)
            k10 = {'conditions': {}}
            for weather, mode, variant in itertools.product(WEATHERS, ('same', 'task'), ('FULL', 'A_ONLY')):
                record, audit = target_fixture(mode=mode)
                name = f'{weather}_{mode}_{variant}'
                relative = 'conditions/' + name + '_frames.jsonl'
                path = folder / relative
                path.parent.mkdir(exist_ok=True)
                path.write_text(json.dumps(dict(sample_index=0, records=[record], current_state_audit=[audit])) + '\n', encoding='utf-8')
                result = analyze_target(record, audit, mode, variant)
                k10['conditions'][name] = dict(frame_artifact=relative, frame_sha256=digest(path), frames=1,
                    counts=dict(targets=1, changed=1, task_separated=mode == 'task', selected_A=1,
                                step_recovered=result['step_recovered'], step_lost=0, step_new_fp=0))
            rows = read_oracle_details(folder, k10, {('validation', 0): 'validation:0'}, {'validation_indices': [0]})
            self.assertEqual(len(rows), 16)
            self.assertEqual(sum(r['true_task_improvement'] for r in rows), 8)
            first = next(iter(k10['conditions'].values()))
            first['frame_sha256'] = 'corrupt'
            with self.assertRaises(ValueError):
                read_oracle_details(folder, k10, {('validation', 0): 'validation:0'}, {'validation_indices': [0]})

    def test_forbidden_roots(self):
        from .common import safe_path
        for root in ('/data/scd/datasets/opv2v_official_data_dumping/test',
                     '/data/cjm/datasets/opv2v-w/fog/test', '/data/cjm/datasets/opv2v-w/rain/test',
                     '/data/cjm/datasets/opv2v-w/snow/test'):
            with self.assertRaises(ValueError):
                safe_path(root)

    def test_alias_resolves_before_read(self):
        from .common import safe_path
        root = ROOT / '.tmp'
        alias, forbidden = root / 'alias', root / 'test'
        original = Path.resolve
        def resolve(path, *args, **kwargs):
            return forbidden if path == alias else original(path, *args, **kwargs)
        with patch.object(Path, 'resolve', resolve), self.assertRaises(ValueError):
            safe_path(alias)

    def test_smoke(self):
        smoke()

    def test_no_research_imports(self):
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('local_fusion_task_source_oracle.oracle', sys.modules)


def smoke():
    k0, k10, raw, pairs = fixture_data()
    details = []
    for weather, mode, variant in itertools.product(WEATHERS, ('same', 'task'), ('FULL', 'A_ONLY')):
        record, audit = target_fixture(mode=mode, variant=variant)
        details.append(dict(analyze_target(record, audit, mode, variant), split='validation', weather=weather,
                            frame=0, scene_id='validation:0', mode=mode, variant=variant))
    with temporary_directory() as folder:
        folder = Path(folder)
        for arguments in ((k0, k10), (k0, k10, raw, pairs, details)):
            p0a, p0b, final = create_results(*arguments)
            write_outputs(folder, p0a, p0b, final)
            assert len(list(folder.glob('*.md'))) == 4
            assert json.loads((folder / 'P0_STAGE2_PRECHECK_FINAL.json').read_text(encoding='utf-8'))['k3_frozen']
            assert all((folder / name).read_text(encoding='utf-8').find('一句话结论') >= 0
                       for name in ('P0_SOURCE_REDUNDANCY.md', 'P0_DATA_ADEQUACY.md', 'P0_STAGE2_PRECHECK_FINAL.md'))
            assert final['P0_B']['H2']['FULL']['train/adverse pooled']['true_task_improvement_targets'] is None
    print('Synthetic CPU smoke complete (no research data/cache/model)', flush=True)


if __name__ == '__main__':
    if '--smoke' in sys.argv:
        smoke()
    else:
        unittest.main()
