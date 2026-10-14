"""Server GPU diagnostic: exact section 33, independent restricted trajectories."""
import argparse
from collections import Counter
import json
from pathlib import Path

from .common import (ROOT, WEATHERS, atomic_json, atomic_text, component, component_status,
                     contract, digest, external_run, load_condition, read_json, save_condition)
from .oracle_adapter import run_restricted, action_statistics


def reproduction_gate(observed, saved, label):
    differences = {}
    for metric in ('ap30', 'ap50', 'ap70'):
        actual, expected = observed[metric], saved[metric]
        import math
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (actual, expected)):
            raise RuntimeError('STOP Stage 1: nonfinite AP: ' + label)
        difference = abs(actual - expected)
        differences[metric] = difference
        if difference > 1e-6:
            raise RuntimeError(f'STOP Stage 1: FULL reproduction failed {label}.{metric}: '
                               f'{actual} vs {expected}; abs diff {difference}')
    return differences


def make_runtime(protocol):
    from local_fusion_task_source_oracle import oracle as b
    from gspr_evidence import runtime as er
    from local_fusion_v3 import runtime as v3rt
    inputs = protocol['inputs']
    b0_run = Path(inputs['b0_run'])
    b0 = read_json(b0_run / 'protocol.json')
    b._verify_b0_source_contract(b0)
    spec = b.settings(ROOT / 'local_fusion_task_source_oracle/experiment.yaml')
    options, hypes = er.load_config(inputs['v3_config'], inputs['frontend_config'])
    target = b.device()
    model, frontend_digest = er.load_model(hypes, options, inputs['frontend_checkpoint'], target)
    model.requires_grad_(False).eval()
    expected = v3rt.contract(options, inputs['frontend_config'], frontend_digest)
    if expected != b0['v3_contract'] or frontend_digest != b0['frontend_sha256']:
        raise ValueError('Frozen frontend/config/source contract differs')
    start_digest = digest(inputs['v3_checkpoint'])
    if start_digest != b0['v3_checkpoint_sha256']:
        raise ValueError('Frozen v3 checkpoint differs')
    source, checkpoint = v3rt.load(inputs['v3_checkpoint'], expected, target)
    if checkpoint['variant'] != 'residual':
        raise ValueError('Original section 33 requires residual checkpoint')
    arm = b._load_b0_arm(b0_run, 'Shared.pth', source, target, 'shared', start_digest)
    return b, spec, options, hypes, target, model, arm


def evaluate(run, protocol, runtime, weather, mode, variant):
    b, spec, options, hypes, target, model, arm = runtime
    from local_fusion_action_utility_audit.common import assert_loader_paths
    b.seed_all(int(spec['seed']) + 1)
    indices = protocol['validation_indices']
    dataset, loader = b.selected_loader(hypes, options, 'validation', weather, indices)
    assert_loader_paths(dataset)  # Before the first __getitem__ or worker launch.
    stats = b.empty_stats()
    shared_stats = b.empty_stats()
    counts = Counter()
    frames = 0
    name = f'{weather}_{mode}_{variant}'
    record_path = run / 'conditions' / (name + '_frames.jsonl')
    record_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = record_path.with_suffix('.jsonl.tmp')
    lidar_range = hypes['postprocess']['anchor_args']['cav_lidar_range']
    with b.torch.no_grad(), temporary.open('w', encoding='utf-8', newline='\n') as stream:
        for batch in loader:
            index = int(batch['ego']['communication_sample_index'][0])
            if frames >= len(indices) or index != indices[frames]:
                raise ValueError('Original B0 validation loader order changed')
            batch = b.to_device(batch, target)
            context = b.v3rt.context(model, batch['ego'],
                                     'clean' if weather == 'clean' else 'weather', verify=frames == 0)
            shared, _ = arm.predict(model.engine.base, context['levels'])
            pool = b.build_candidate_pool(model.engine.base, context['levels'], shared, spec['candidate_families'])
            proxy = b.build_proxy_pool(dataset, batch, pool)
            shared_post = dataset.post_process(batch, {'ego': shared})
            _, post, records, audit = run_restricted(
                b, variant, dataset, batch, model.engine.base, context['levels'], shared,
                pool, proxy, lidar_range, spec['roi_expansions'], mode,
                spec['target_iou'], spec['shortlist_k'])
            for threshold in stats:
                b.eval_utils.caluclate_tp_fp(*post, stats, threshold)
                b.eval_utils.caluclate_tp_fp(*shared_post, shared_stats, threshold)
            counts.update(action_statistics(records, audit))
            counts.update(b._compare(shared_post, post, spec['target_iou']))
            stream.write(json.dumps(dict(sample_index=index, source_count=len(context['levels'][0]),
                                        candidate_pool=sorted(pool), records=records, current_state_audit=audit),
                                    ensure_ascii=False, allow_nan=False) + '\n')
            frames += 1
            print(f'K1-0 {name} frame {frames}/{len(indices)}', flush=True)
    if frames != len(indices):
        raise RuntimeError('Incomplete Oracle condition')
    temporary.replace(record_path)
    payload = dict(weather=weather, mode=mode, variant=variant, frames=frames,
                   AP=b.ap_values(stats, b.eval_utils),
                   Shared=b.ap_values(shared_stats, b.eval_utils),
                   counts=dict(counts), global_sort=False,
                   frame_artifact=record_path.relative_to(run).as_posix(),
                   frame_sha256=digest(record_path))
    targets = counts['targets']
    shortlisted = sum(counts['shortlisted_' + x] for x in ('A', 'B', 'E'))
    payload['availability_rates'] = {x: counts[x + '_available_targets'] / targets if targets else None
                                     for x in ('A', 'B', 'E')}
    payload['shortlisted_group_rates'] = {x: counts['shortlisted_' + x] / shortlisted if shortlisted else None
                                         for x in ('A', 'B', 'E')}
    return payload


def condition(run, protocol, runtime, weather, mode, variant, old_results):
    name = f'{weather}_{mode}_{variant}'
    payload = load_condition(run, name)
    if payload is None:
        payload = evaluate(run, protocol, runtime, weather, mode, variant)
        # Failed FULL never gets committed as a completed reusable condition.
        verify_condition(run, protocol, payload, old_results)
        save_condition(run, name, payload)
    else:
        verify_condition(run, protocol, payload, old_results)
        print('Reuse complete condition: ' + name, flush=True)
    return payload


def verify_condition(run, protocol, payload, old_results):
    weather, mode, variant = payload['weather'], payload['mode'], payload['variant']
    if payload['frames'] != len(protocol['validation_indices']) or payload['global_sort'] is not False:
        raise ValueError('Condition incomplete or evaluation protocol differs')
    if digest(run / payload['frame_artifact']) != payload['frame_sha256']:
        raise ValueError('Condition frame records corrupted')
    reference = old_results['conditions'][weather]['results']
    reproduction_gate(payload['Shared'], reference['Shared'], f'{weather}/{mode}/{variant}/Shared')
    if variant == 'FULL':
        reproduction_gate(payload['AP'], reference['Oracle-' + mode.title()], f'{weather}/Full-{mode}')


def build(run):
    protocol = contract(run)
    if component_status(run, 'K1_0')['status'] == 'complete':
        print('K1-0 complete; reuse hash-verified report', flush=True)
        return
    artifacts = ['K1_0_ORACLE_DECOMPOSITION.json', 'K1_0_ORACLE_DECOMPOSITION.md', 'FULL_REPRODUCTION_GATE.json']
    with component(run, 'K1_0', artifacts):
        old_results = read_json(Path(protocol['references']['oracle_run']) / 'death_test_results.json')
        runtime = make_runtime(protocol)
        conditions = {}
        # Global gate: ALL FOUR weather FULLs pass BEFORE any A_ONLY/B_ONLY runs.
        for weather in WEATHERS:
            for mode in ('same', 'task'):
                name = f'{weather}_{mode}_FULL'
                conditions[name] = condition(run, protocol, runtime, weather, mode, 'FULL', old_results)
        gate = {'status': 'complete', 'absolute_ap_tolerance': 1e-6, 'checks': {}}
        for name, row in conditions.items():
            ref = old_results['conditions'][row['weather']]['results']
            gate['checks'][name] = {
                'Shared': reproduction_gate(row['Shared'], ref['Shared'], name + '/Shared'),
                'Full': reproduction_gate(row['AP'], ref['Oracle-' + row['mode'].title()], name)}
        atomic_json(run / artifacts[2], gate)
        for variant in ('A_ONLY', 'B_ONLY'):
            for weather in WEATHERS:
                for mode in ('same', 'task'):
                    name = f'{weather}_{mode}_{variant}'
                    conditions[name] = condition(run, protocol, runtime, weather, mode, variant, old_results)
        from .k1_0_report import summarize, render
        result = summarize(conditions)
        result['reproduction_gate'] = gate
        atomic_json(run / artifacts[0], result)
        atomic_text(run / artifacts[1], render(result))
        artifacts.extend(p.relative_to(run).as_posix()
                         for p in sorted((run / 'conditions').iterdir())
                         if p.suffix in ('.json', '.jsonl'))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--run', required=True)
    args = cli.parse_args()
    build(external_run(args.run))


if __name__ == '__main__':
    main()
