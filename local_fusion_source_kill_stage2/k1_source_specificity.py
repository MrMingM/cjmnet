"""Original Source reproduction first, then matched null sequential Oracles."""
from collections import Counter
import json
from pathlib import Path
import uuid
from .common import (FAMILIES, KEEP, WEATHERS, atomic_json, commit_unit,
    external_run, jsonlines, load_contract, now, object_hash, read_json, resource_check,
    resume, settings, worker)
from .source_adapter import (action_counts, assert_shortlist_contract, frame_stats, inference_input_hash, make_runtime, original_args,
                            reproduction, run_with_hooks, total_ap)
from .nulls import (MAGNITUDES, assert_fair, candidate_ids, empirical_predictions,
                   magnitude_summary, mix_predictions, pool_magnitudes, rng_for)
from .calibration import prepare, load_pools


def condition_path(folder, weather, family, seed, mode):
    return folder / 'conditions' / weather / family / str(seed) / (mode + '.json')


def evaluate(folder, protocol, runtime, weather, family, seed, mode, null_config, pools):
    b, spec, options, hypes, target, model, arm = runtime
    identity = object_hash(dict(root=protocol['identity'], null=null_config['config_hash']))
    path = condition_path(folder, weather, family, seed, mode)
    saved = resume(path, identity)
    if saved is not None:
        print('Reuse:', path, flush=True)
        return saved
    reference_frames=None
    if family != 'Source':
        gate=read_json(folder/'SOURCE_REPRODUCTION_GATE.json')
        if gate['status']!='complete' or gate['root_identity']!=protocol['identity'] or gate['null_config_hash']!=null_config['config_hash']:
            raise ValueError('Source reproduction gate missing or mismatched')
        source_path=condition_path(folder,weather,'Source',0,mode)
        source_payload=resume(source_path,identity)
        if source_payload is None:
            raise ValueError('Source condition incomplete before Null')
        reference_frames=list(jsonlines(source_path.parent/source_payload['stats_file']))
    resource_check(protocol)
    started_at = now()
    from local_fusion_action_utility_audit.common import assert_loader_paths, frame_scene
    from local_fusion_action_utility_audit.s0_counterfactual import prediction_hash
    b.seed_all(spec['seed']+1)  # identical online weather realization for every family/seed/mode
    dataset, loader = b.selected_loader(hypes, options, 'validation', weather, protocol['validation_indices'])
    assert_loader_paths(dataset)
    counts = Counter()
    selected_actions = Counter()
    frames, shared_frames = [], []
    magnitudes = {k: [] for k in MAGNITUDES}
    source_magnitudes = {k: [] for k in MAGNITUDES}
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = path.with_suffix('.jsonl')
    temp = raw_path.with_name(raw_path.name + '.' + uuid.uuid4().hex + '.tmp')
    with b.torch.no_grad(), temp.open('w', encoding='utf-8', newline='\n') as stream:
        for ordinal, batch in enumerate(loader):
            resource_check(protocol)
            index = int(batch['ego']['communication_sample_index'][0])
            if index != protocol['validation_indices'][ordinal]:
                raise ValueError('Validation frame order differs')
            scene = 'validation:' + str(frame_scene(dataset, index))
            batch = b.to_device(batch, target)
            model_input_hash=inference_input_hash(batch['ego'])
            context = b.v3rt.context(model, batch['ego'], 'clean' if weather == 'clean' else 'weather', verify=ordinal == 0)
            shared, _ = arm.predict(model.engine.base, context['levels'])
            shared_digest=prediction_hash(shared)
            if reference_frames is not None:
                recorded=reference_frames[ordinal]
                if (recorded['model_input_sha256']!=model_input_hash or recorded['shared_prediction_sha256']!=shared_digest or
                    recorded['source_count']!=len(context['levels'][0])):
                    raise ValueError('Source/Null frame input, Shared prediction or CAV count mismatch')
            source = b.build_candidate_pool(model.engine.base, context['levels'], shared, spec['candidate_families'])
            if len(source)-1 != 2*len(context['levels'][0]):
                raise ValueError('Original single/query action count mismatch')
            extra = {}
            if family == 'Source':
                pool = source
            elif family == 'N2':
                pool = mix_predictions(b, model.engine.base, context['levels'], shared, null_config['alpha'],
                                       rng_for(seed, weather, index, family))
            else:
                pool, extra = empirical_predictions(dataset, batch, shared,
                    sorted(n for n in source if n != KEEP), pools, weather, rng_for(seed, weather, index, family),
                    score_only=family == 'score-only')
            assert_fair(source, pool)
            ids = candidate_ids(b, dataset, batch, shared)
            for key, values in pool_magnitudes(b, dataset, batch, shared, pool, ids).items():
                magnitudes[key].extend(values)
            for key, values in pool_magnitudes(b, dataset, batch, shared, source, ids).items():
                source_magnitudes[key].extend(values)
            proxy = b.build_proxy_pool(dataset, batch, pool)
            shared_post = dataset.post_process(batch, {'ego': shared})
            args = original_args(b, spec, dataset, batch, model, context['levels'], shared, pool, proxy,
                hypes['postprocess']['anchor_args']['cav_lidar_range'], mode)
            _, post, records, audit = run_with_hooks(b, 'FULL', args)
            assert_shortlist_contract(records, audit, len(pool), spec['roi_expansions'], mode, spec['shortlist_k'])
            counts.update(action_counts(records, audit))
            for record in records:
                selected_actions[f'{record["cls_source"]}|{record["reg_source"]}|{record["expansion"]}'] += 1
            counts.update(b._compare(shared_post, post, spec['target_iou']))
            frame = dict(sample_index=index, scene_id=scene, stats=frame_stats(b, post))
            baseline = dict(sample_index=index, scene_id=scene, stats=frame_stats(b, shared_post))
            frames.append(frame)
            shared_frames.append(baseline)
            stream.write(json.dumps(dict(frame=frame, shared=baseline, shared_prediction_sha256=shared_digest,
                model_input_sha256=model_input_hash, source_count=len(context['levels'][0]),
                nonkeep_source=len(source)-1, nonkeep_null=len(pool)-1, records=records, audit=audit,
                null_details=extra), allow_nan=False) + '\n')
            print(f'K1 {weather}/{family}/{seed}/{mode} frame {ordinal+1}/{len(loader)}', flush=True)
    if len(frames) != len(protocol['validation_indices']):
        raise ValueError('Incomplete K1 condition')
    actual, baseline_ap = total_ap(b, frames), total_ap(b, shared_frames)
    reference = read_json(Path(protocol['oracle_run']) / 'death_test_results.json')['conditions'][weather]['results']
    reproduction(b, baseline_ap, reference['Shared'], weather+'/Shared')
    if family == 'Source':
        reproduction(b, actual, reference['Oracle-'+mode.title()], weather+'/'+mode)
    temp.replace(raw_path)
    payload = dict(weather=weather, family=family, seed=seed, mode=mode, AP=actual, Shared=baseline_ap,
        frames=len(frames), counts=dict(counts), action_count_fair=True, original_proxy=True,
        selected_actions=dict(selected_actions), shortlisted_actions=sum(counts['shortlisted_'+g] for g in ('A','B','E')),
        original_shortlist=True, full_postprocess=True, stats_file=raw_path.name,
        output_magnitude=magnitude_summary(magnitudes), source_output_magnitude=magnitude_summary(source_magnitudes),
        source_hash=object_hash(protocol['source_hashes']), input_hash=object_hash(protocol['input_hashes']),
        config_hash=identity, started_at=started_at, completed_at=now(), global_sort=False)
    commit_unit(path, identity, payload, (raw_path,))
    return payload


def run(run):
    protocol = load_contract(run)
    with worker(run, 'k1', protocol) as folder:
        if (folder / 'K1_SOURCE_SPECIFICITY.json').exists():
            identity = protocol['identity']
            completed = resume(folder / 'result.json', identity)
            if completed is not None:
                return
        try:
            runtime = make_runtime(protocol)
            null_config = prepare(folder, protocol, runtime)
            pools = load_pools(folder, null_config)
            from .server_smoke import execute as server_smoke
            server_smoke(folder / 'smoke', protocol, runtime)
            conditions = {}
            # All eight source conditions complete and reproduce BEFORE ANY null.
            for weather in WEATHERS:
                for mode in ('same', 'task'):
                    name = f'{weather}/Source/0/{mode}'
                    conditions[name] = evaluate(folder, protocol, runtime, weather, 'Source', 0, mode, null_config, pools)
            atomic_json(folder / 'SOURCE_REPRODUCTION_GATE.json', dict(status='complete', tolerance=1e-6,
                root_identity=protocol['identity'], null_config_hash=null_config['config_hash'],
                conditions={k: v['AP'] for k, v in conditions.items()}))
            for weather in WEATHERS:
                seeds = settings()['seeds'][:1] if weather == 'clean' else settings()['seeds']
                for family in FAMILIES:
                    for seed in seeds:
                        for mode in ('same', 'task'):
                            name = f'{weather}/{family}/{seed}/{mode}'
                            conditions[name] = evaluate(folder, protocol, runtime, weather, family, seed, mode, null_config, pools)
            from .reports import k1_report
            k1_report(folder, protocol, conditions, runtime[0].eval_utils.calculate_ap)
            artifacts = [folder / name for name in ('K1_SOURCE_SPECIFICITY.json', 'K1_SOURCE_SPECIFICITY.md',
                'k1_seed_details.json', 'k1_scene_bootstrap.json', 'SOURCE_REPRODUCTION_GATE.json', 'NULL_CONFIG.json')]
            artifacts += [p for p in (folder/'conditions').rglob('*') if p.suffix in ('.json','.jsonl')]
            artifacts += [folder/'PROTOCOL.json'] + [folder/p for p in null_config['artifacts']]
            commit_unit(folder / 'result.json', protocol['identity'], dict(status='complete'), artifacts)
        except BaseException as exc:
            from .reports import inconclusive_report
            inconclusive_report(folder, 'K1', protocol, exc)
            raise


def main():
    import argparse
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--run', required=True)
    args = cli.parse_args()
    run(external_run(args.run))


if __name__ == '__main__':
    main()
