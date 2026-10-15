"""Replay P0 A_ONLY states; rerun all eight interventions from each same state."""
from collections import defaultdict
import json
from pathlib import Path
from .common import (ADVERSE, atomic_text, commit_unit, digest,
    external_run, jsonlines, load_contract, read_json, resource_check,
    resume, settings, worker, now, object_hash)
from .preflight import baseline_rows
from .source_adapter import (evaluate_target, make_runtime, masks_at_multiplier, original_args, run_with_hooks)
from .stability import (compare_labels, label_summary, perturb_corners, serialize_label)
from .verdicts import sample_gate


def verify_target(baseline, p0):
    if baseline['winner_key'][:6] != p0['selected_discrete6'] or baseline['discrete_positive'] != p0['selected_discrete_positive']:
        raise ValueError('P0 baseline current-state discrete outcome does not reproduce')
    if p0['mode'] == 'task' and (baseline['task_positive'] != p0['true_task_improvement'] or
                                baseline['same_discrete6'] != p0['best_same_discrete6']):
        raise ValueError('P0 same-state Task comparator does not reproduce')


def evaluate_frame(protocol, runtime, dataset, batch, weather, index, mode, positives):
    b, spec, options, hypes, target, model, arm = runtime
    batch = b.to_device(batch, target)
    context = b.v3rt.context(model, batch['ego'], 'weather', verify=False)
    shared, _ = arm.predict(model.engine.base, context['levels'])
    pool = b.build_candidate_pool(model.engine.base, context['levels'], shared, spec['candidate_families'])
    proxy = b.build_proxy_pool(dataset, batch, pool)
    shared_post = dataset.post_process(batch, {'ego': shared})
    gt = shared_post[2].detach().cpu().numpy()
    lidar_range = hypes['postprocess']['anchor_args']['cav_lidar_range']
    by_target = {r['target_index']: r for r in positives}
    results = []
    reproduced = []
    def focal(target_index, current, original_shortlist, shortlist_info):
        if target_index not in by_target:
            return
        resource_check(protocol)
        p0 = by_target[target_index]
        def evaluation(corners, scale):
            expansions = tuple(float(x)*scale for x in spec['roi_expansions'])
            masks = {}
            for expansion in expansions:
                mask, _, stats = masks_at_multiplier(batch['ego']['anchor_box'], corners, context['levels'], lidar_range, expansion)
                masks[expansion] = (mask, stats)
            return evaluate_target(b, dataset, batch, current, pool, proxy,
                gt[target_index], masks, expansions, mode, target_index, spec['target_iou'], spec['shortlist_k'])
        baseline_evaluation = evaluation(gt[target_index], 1.)
        baseline = label_summary(baseline_evaluation)
        verify_target(baseline, p0)
        reproduced.append(target_index)
        for name, perturbation in settings()['k2']['perturbations'].items():
            corners, scale = perturb_corners(gt[target_index], perturbation)
            changed_evaluation = evaluation(corners, scale)
            changed = label_summary(changed_evaluation, scale)
            results.append(dict(layer=mode, weather=weather, frame=index, target_index=target_index,
                scene_id=p0['scene_id'], variant=name, family=perturbation[0],
                baseline=serialize_label(baseline), perturbed=serialize_label(changed),
                baseline_actions=baseline_evaluation['actions'], perturbed_actions=changed_evaluation['actions'],
                **compare_labels(baseline, changed)))
    args = original_args(b, spec, dataset, batch, model, context['levels'], shared, pool, proxy, lidar_range, mode)
    _, _, records, audits = run_with_hooks(b, 'A_ONLY', args, focal)
    # Reproduce the entire saved trajectory, not only positive labels. This
    # catches any earlier-state divergence before interpreting perturbations.
    saved_condition = read_json(Path(protocol['stage1_run']) / 'K1_0_ORACLE_DECOMPOSITION.json')['conditions'][f'{weather}_{mode}_A_ONLY']
    saved_path = Path(protocol['stage1_run']) / saved_condition['frame_artifact']
    if digest(saved_path) != saved_condition['frame_sha256']:
        raise ValueError('Stage 1 raw trajectory changed')
    saved = next((row for row in jsonlines(saved_path) if row['sample_index'] == index), None)
    if saved is None or records != saved['records'] or audits != saved['current_state_audit']:
        raise ValueError('REPRODUCTION_FAIL: K2 original A_ONLY current-state trajectory differs')
    if set(reproduced) != set(by_target):
        raise ValueError('Missing P0 focal targets in replay')
    return results


def run(run):
    protocol = load_contract(run)
    with worker(run, 'k2', protocol) as folder:
        if resume(folder / 'result.json', protocol['identity']) is not None:
            return
        try:
            targets = baseline_rows(Path(protocol['p0_run']) / 'p0_oracle_target_records.jsonl')
            gates = {layer: sample_gate(rows, layer) for layer, rows in targets.items()}
            if any(r['verdict'] != 'PASS' for r in gates.values()):
                raise ValueError('K2 sample gate INCONCLUSIVE')
            runtime = make_runtime(protocol)
            b, spec, options, hypes, target, model, arm = runtime
            from local_fusion_action_utility_audit.common import assert_loader_paths
            all_rows = []
            for weather in ADVERSE:
                for mode in ('same', 'task'):
                    by_frame = defaultdict(list)
                    for row in targets[mode]:
                        if row['weather'] == weather:
                            by_frame[row['frame']].append(row)
                    # Walk the original full ordered loader even when only a few
                    # frames need GPU work: preserve online-weather RNG realization.
                    b.seed_all(spec['seed']+1)
                    dataset, loader = b.selected_loader(hypes, options, 'validation', weather, protocol['validation_indices'])
                    assert_loader_paths(dataset)
                    seen = []
                    with b.torch.no_grad():
                        for ordinal, batch in enumerate(loader):
                            index = int(batch['ego']['communication_sample_index'][0])
                            if index != protocol['validation_indices'][ordinal]:
                                raise ValueError('K2 loader order differs')
                            seen.append(index)
                            if index not in by_frame:
                                continue
                            path = folder / 'conditions' / weather / mode / f'{index:08d}.json'
                            payload = resume(path, protocol['identity'])
                            if payload is None:
                                resource_check(protocol)
                                started_at=now()
                                payload = evaluate_frame(protocol, runtime, dataset, batch, weather, index, mode, by_frame[index])
                                commit_unit(path, protocol['identity'], payload, metadata=dict(
                                    seed=spec['seed']+1, started_at=started_at, completed_at=now(),
                                    source_hash=object_hash(protocol['source_hashes']), input_hash=object_hash(protocol['input_hashes'])))
                            all_rows.extend(payload)
                            print(f'K2 {weather}/{mode} frame={index} perturbation records={len(payload)}', flush=True)
                    if seen != protocol['validation_indices']:
                        raise ValueError('Incomplete K2 loader')
            expected = 8 * sum(len(rows) for rows in targets.values())
            if len(all_rows) != expected:
                raise ValueError('K2 target/variant completeness mismatch')
            atomic_text(folder / 'k2_flip_records.jsonl', ''.join(json.dumps(r, allow_nan=False) + '\n' for r in all_rows))
            from .reports import k2_report
            k2_report(folder, protocol, all_rows, gates)
            artifacts=[folder/name for name in ('K2_LABEL_STABILITY.json','K2_LABEL_STABILITY.md',
                       'k2_scene_bootstrap.json','k2_flip_records.jsonl')]
            artifacts += list((folder/'conditions').rglob('*.json'))
            commit_unit(folder / 'result.json', protocol['identity'], dict(status='complete'),artifacts)
        except BaseException as exc:
            from .reports import inconclusive_report
            inconclusive_report(folder, 'K2', protocol, exc)
            raise


def main():
    import argparse
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--run', required=True)
    args = cli.parse_args()
    run(external_run(args.run))


if __name__ == '__main__':
    main()
