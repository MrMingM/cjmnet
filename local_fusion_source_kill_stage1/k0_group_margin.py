"""K0 prime: read original S0 outcomes on CPU; never run a detector."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from statistics import mean, median

from local_fusion_action_utility_audit.counterfactual import action_keys
from .common import (KEEP, ROOT, WEATHERS, ADVERSE, atomic_json, atomic_text,
                     component, component_status, contract, digest, external_run,
                     read_json, settings)

MARGIN_NAMES = ('DISCRETE_MARGIN', 'CONTINUOUS_ONLY_MARGIN', 'EXACT_OUTCOME_TIE')


def audit_group(outcomes, names):
    keys = action_keys(outcomes, names)
    if any(len(key) != 9 or not all(math.isfinite(x) for x in key) for key in keys):
        raise ValueError('Invalid original action key')
    order = sorted(range(len(keys)), key=keys.__getitem__, reverse=True)
    top = keys[order[0]]
    runner = keys[order[1]] if len(order) > 1 else None
    category = ('EXACT_OUTCOME_TIE' if runner is None or top[:8] == runner[:8] else
                'CONTINUOUS_ONLY_MARGIN' if top[:6] == runner[:6] else 'DISCRETE_MARGIN')
    keep = keys[0]
    beneficial = any(key[:6] > keep[:6] for key in keys[1:])
    continuous = not beneficial and any(
        key[:6] == keep[:6] and key[:8] > keep[:8] for key in keys[1:])
    # Explicitly retain the complete top-equivalent set, never a unique source claim.
    equivalent = sorted(name for name, key in zip(names, keys) if key[:8] == top[:8])
    risk = Counter()
    for outcome in outcomes[1:]:
        delta = outcome['action_tp'] - outcome['baseline_tp']
        risk['tp_increase' if delta > 0 else 'tp_decrease' if delta < 0 else 'tp_unchanged'] += 1
    return dict(category=category, top_tie_count=len(equivalent), top_equivalent_actions=equivalent,
                top_key=top, runner_up_key=runner, keep_only=len(keys) == 1,
                has_discrete_beneficial_action=beneficial, continuous_only_benefit=continuous,
                harmful_discrete_exists=any(key[:6] < keep[:6] for key in keys[1:]),
                action_level=dict(risk))


def groups_from_frame(frame):
    """Keep focal GT association in identity, including duplicate proposal associations."""
    if frame['split'] not in ('train', 'validation') or frame['weather'] not in WEATHERS:
        raise ValueError('Unexpected S0 scope')
    names = frame['source_names']
    if not names or names[0] != KEEP or len(set(names)) != len(names):
        raise ValueError('S0 source identity invalid')
    # Original S0 exhausts both families for every source, with KEEP first.
    singles = {x.split(':', 1)[1] for x in names[1:] if x.startswith('single:')}
    queries = {x.split(':', 1)[1] for x in names[1:] if x.startswith('query:')}
    if singles != queries or len(names) != 1 + 2 * len(singles):
        raise ValueError('Incomplete S0 KEEP/single/query pool')
    seen = set()
    for label in frame['labels']:
        focal = label['target_index']
        if label['background'] != (focal is None):
            raise ValueError('S0 focal/background schema mismatch')
        position = label['proposal_position']
        anchor = frame['proposals']['anchor_ids'][position]
        for task in ('cls', 'reg'):
            identity = (frame['split'], frame['weather'], frame['frame'], anchor, task, focal)
            if identity in seen:
                raise ValueError('Duplicate S0 group')
            seen.add(identity)
            result = audit_group(label['outcomes'][task], names)
            yield dict(identity=identity, split=frame['split'], weather=frame['weather'],
                       frame=frame['frame'], proposal=anchor, task=task, focal=focal,
                       scene=frame['scene'], stratum='background' if focal is None else 'associated',
                       **result)


def primary(row):
    return row['split'] == 'validation' and row['stratum'] == 'associated'


def aggregate(rows):
    rows = list(rows)
    if not rows:
        raise ValueError('Empty K0 cell; report cannot silently treat missing groups as zero')
    n = len(rows)
    counts = Counter(row['category'] for row in rows)
    result = {'group_count': n}
    for key, category in zip(('discrete_margin_rate', 'continuous_only_margin_rate',
                              'exact_outcome_tie_rate'), MARGIN_NAMES):
        result[key] = counts[category] / n
    ties = [row['top_tie_count'] for row in rows]
    result.update(top_equivalent_multi_action_rate=sum(x > 1 for x in ties) / n,
                  mean_top_tie_count=mean(ties), median_top_tie_count=median(ties),
                  keep_only_count=sum(row['keep_only'] for row in rows),
                  top_tie_histogram=dict(Counter(ties)))
    for dest, field in (('discrete_beneficial_set_rate', 'has_discrete_beneficial_action'),
                        ('continuous_only_benefit_rate', 'continuous_only_benefit'),
                        ('harmful_discrete_exists_rate', 'harmful_discrete_exists')):
        result[dest] = sum(row[field] for row in rows) / n
    scene_counts = Counter(str(row['scene']) for row in rows)
    result['scenes'] = dict(scene_counts)
    result['single_scene_max_fraction'] = max(scene_counts.values()) / n
    for name, selected in (
            ('discrete_margin', [r for r in rows if r['category'] == 'DISCRETE_MARGIN']),
            ('beneficial_set', [r for r in rows if r['has_discrete_beneficial_action']])):
        scenes = Counter(str(row['scene']) for row in selected)
        result[name + '_scene_coverage'] = dict(scene_count=len(scenes),
            group_count=len(selected), counts=dict(scenes),
            max_single_scene_fraction=max(scenes.values()) / len(selected) if selected else None)
    risk = Counter()
    for row in rows:
        risk.update(row['action_level'])
    total = sum(risk.values())
    result['action_level_net_tp_count_change'] = dict(
        unit='non-KEEP action; ACTION-LEVEL CONTEXT', action_count=total,
        **{key: risk[key] for key in ('tp_increase', 'tp_decrease', 'tp_unchanged')},
        **{key + '_rate': risk[key] / total if total else None
           for key in ('tp_increase', 'tp_decrease', 'tp_unchanged')},
        ratio_decrease_increase=risk['tp_decrease'] / risk['tp_increase'] if risk['tp_increase'] else None)
    return result


def old_risk_reference(text):
    reference = {}
    for line in text.splitlines():
        cells = [x.strip() for x in line.strip().strip('|').split('|')]
        if len(cells) > 6 and cells[0] in ('train', 'validation') and cells[4] == 'label_group':
            values = cells[5].split('/')
            if len(values) != 3 or any(not x.endswith('%') for x in values):
                raise ValueError('Unsupported old label audit risk schema')
            reference['/'.join(cells[:4])] = values
    if len(reference) != 48:
        raise ValueError('Expected original 2 splits x 4 weather x 2 task x 3 strata risk rows')
    return reference


def risk_gate(conditions, reference):
    checked = {}
    for key, row in conditions.items():
        risk = row['action_level_net_tp_count_change']
        observed = [f"{risk[name + '_rate']:.2%}" for name in ('tp_increase', 'tp_decrease', 'tp_unchanged')]
        if observed != reference[key]:
            raise RuntimeError(f'STOP Stage 1: old ACTION-LEVEL CONTEXT mismatch {key}: '
                               f'{observed} vs {reference[key]}; inspect schema')
        checked[key] = {'observed': observed, 'saved': reference[key], 'passed': True}
    return checked


def build(run):
    if component_status(run, 'K0_PRIME')['status'] == 'complete':
        contract(run)
        print('K0 prime complete; reuse hash-verified report', flush=True)
        return
    artifacts = ['K0_GROUP_MARGIN.json', 'K0_GROUP_MARGIN.md', 'k0_group_records.jsonl']
    with component(run, 'K0_PRIME', artifacts) as protocol:
        import torch
        s0_run = Path(protocol['references']['s0_run'])
        s0 = read_json(s0_run / 'protocol.json')
        manifest = read_json(s0_run / 'manifest.json')
        cells = defaultdict(list)
        record_path = run / 'k0_group_records.jsonl'
        temporary = record_path.with_suffix('.jsonl.tmp')
        with temporary.open('w', encoding='utf-8', newline='\n') as stream:
            for split in ('train', 'validation'):
                for weather in WEATHERS:
                    for index in s0[split + '_indices']:
                        name = f'cache/{split}/{weather}/{index:08d}.pt'
                        entry = manifest['entries'][f'S0-frame/{split}/{weather}/{index}']
                        if entry['status'] != 'complete' or digest(s0_run / name) != entry['artifacts'][name]:
                            raise ValueError('S0 immutable cache differs: ' + name)
                        frame = torch.load(s0_run / name, map_location='cpu', weights_only=True)
                        if (frame['frame'], frame['split'], frame['weather']) != (index, split, weather):
                            raise ValueError('S0 frame metadata differs from protocol')
                        if frame['scene'] not in s0[split + '_scenes']:
                            raise ValueError('S0 scene outside recorded split')
                        for row in groups_from_frame(frame):
                            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                            # Only scalar group statistics are kept in memory.
                            compact = {k: v for k, v in row.items() if k not in (
                                'top_key', 'runner_up_key', 'top_equivalent_actions', 'identity')}
                            for stratum in (row['stratum'], 'all'):
                                cells['/'.join((split, weather, row['task'], stratum))].append(compact)
                    print(f'K0 read immutable {split}/{weather} complete', flush=True)
        temporary.replace(record_path)
        conditions = {key: aggregate(rows) for key, rows in cells.items()}
        reference = old_risk_reference((ROOT / 'logs/sectionB/all/LABEL_AUDIT.md').read_text(encoding='utf-8'))
        checks = risk_gate(conditions, reference)
        macros = {}
        for task in ('cls', 'reg'):
            selected = [conditions[f'validation/{weather}/{task}/associated'] for weather in ADVERSE]
            macros[task] = {key: mean(row[key] for row in selected)
                            for key in selected[0] if key.endswith('_rate') or key == 'mean_top_tie_count'}
        rates = [macros[task]['discrete_margin_rate'] for task in ('cls', 'reg')]
        spec = settings()
        signal_name = ('K0_SIGNAL_STRONG' if min(rates) >= spec['k0_signal_strong_rate'] else
                       'K0_SIGNAL_WEAK' if max(rates) < spec['k0_signal_weak_rate'] else 'K0_SIGNAL_MIXED')
        result = dict(conditions=conditions, primary_adverse_macro=macros, signal=signal_name,
                      signal_is_descriptive_only=True, action_risk_reproduction=checks,
                      source='original S0 frame labels.outcomes; no regenerated counterfactuals',
                      k3_frozen=True, next_stage_auto_start=False)
        from .k0_report import render
        atomic_json(run / artifacts[0], result)
        atomic_text(run / artifacts[1], render(result))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--run', required=True)
    args = cli.parse_args()
    build(external_run(args.run))


if __name__ == '__main__':
    main()
