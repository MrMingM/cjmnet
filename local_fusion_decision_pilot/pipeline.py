"""One-shot small decision pilot on official train/validation scenes only.

The test roots and OPV2V-W are intentionally never opened here. GT is used to
make training labels and to score held-out validation, never to form candidates
or choose actions during evaluation.
"""
import argparse
import json
from pathlib import Path
import shutil
import time

import torch
from torch.utils.data import DataLoader, Subset
import yaml

from gspr_communication.runtime import (
    device, new_output, seed_all, seed_worker, sha256, verify_frozen, write_json,
)
from gspr_evidence import runtime as er
from local_fusion_utility_v2.outcomes import compare_predictions
from local_fusion_utility_v2.fusion import DESCRIPTOR_NAMES
from local_fusion_utility_v2.runtime import prepare_context, raw_feature_bytes
from opencood.tools.train_utils import to_device
from .core import (
    FEATURE_NAMES, DecisionSelector, candidate_features, choose_frames,
    outcome_class, patch_prediction, residual_with_gate_maps, sampled_tiles,
    scores_for_rows, selected_indices, weighted_cross_entropy,
)
from . import core
from local_fusion_v3 import runtime as v3rt


CONDITIONS = {'train': ('clean', 'mixed'),
              'validation': ('clean', 'fog', 'rain', 'snow')}


def pilot_settings(path):
    settings = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    for key in ('seed', 'train_scenes', 'validation_scenes', 'frames_per_scene',
                'tiles_per_frame', 'tile_size', 'selector_epochs', 'selector_hidden'):
        if type(settings.get(key)) is not int or settings[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if not 0 <= settings['low_score_floor'] < settings['high_score_floor'] <= 1:
        raise ValueError('Invalid score strata')
    if not 0 < settings['selector_learning_rate']:
        raise ValueError('Invalid selector learning rate')
    budgets = settings['action_budgets']
    if not budgets or any(type(b) is not int or b < 1 for b in budgets) or len(set(budgets)) != len(budgets):
        raise ValueError('Action budgets must be distinct positive integers')
    scales = settings['changed_scales']
    if tuple(scales) != (0, 1):
        raise ValueError('Pilot is defined for the v3 modified scales [0,1]')
    for key in ('target_iou', 'fp_identity_iou'):
        if not 0 < settings[key] <= 1:
            raise ValueError(f'Invalid {key}')
    return settings


def selected_loader(hypes, options, split, condition, settings):
    # Also covers num_workers=0, where augmentation runs in this process.
    seed_all(options['seed'])
    weather = 'mixed' if condition == 'mixed' else condition
    dataset, _, _ = er.make_loader(hypes, options, train=split == 'train', weather=weather)
    scene_count = settings['train_scenes'] if split == 'train' else settings['validation_scenes']
    # Same frames in every condition. Different official roots keep scenes separate.
    scenes, indices = choose_frames(dataset.len_record,
                                    settings['seed']+(0 if split == 'train' else 1),
                                    scene_count, settings['frames_per_scene'])
    loader = DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False,
                        num_workers=options['workers'], collate_fn=dataset.collate_batch_test,
                        worker_init_fn=seed_worker,
                        generator=torch.Generator().manual_seed(options['seed']))
    return dataset, loader, scenes, indices


def candidate(model, module, batch, branch, settings, seed, verify=False):
    ctx = prepare_context(model, batch['ego'], branch, {'tile_size': settings['tile_size']}, verify=verify)
    fused, gates = residual_with_gate_maps(module, ctx['levels'])
    features, _ = candidate_features(ctx, fused, gates, model.engine.base)
    tiles = sampled_tiles(ctx['activity'], settings, seed) if features.shape[0] else torch.empty(0, dtype=torch.long)
    if len(tiles):
        rows = features[0, :, tiles//ctx['grid'][1], tiles%ctx['grid'][1]].T.contiguous()
        gate_columns = [FEATURE_NAMES.index('gate_s0'), FEATURE_NAMES.index('gate_s1')]
        gate_score = rows[:, gate_columns].mean(1)
        conf = ctx['descriptors'][:, DESCRIPTOR_NAMES.index('peer_cls_max')]
        full = ctx['descriptors'][:, DESCRIPTOR_NAMES.index('full_cls_max')]
        confidence_map = (conf-full).amax(0)
        confidence = confidence_map.flatten()[tiles]
    else:
        rows = features.new_empty((0, len(FEATURE_NAMES)))
        gate_score = rows.new_empty((0,))
        confidence = rows.new_empty((0,))
    return ctx, fused, dict(tiles=tiles.cpu(), features=rows.cpu(),
                            gate=gate_score.cpu(), confidence=confidence.cpu())


def collect(run, model, module, hypes, options, split, condition, settings, target, contract):
    dataset, loader, scenes, indices = selected_loader(hypes, options, split, condition, settings)
    folder = run/'cache'/split/condition
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder/'manifest.json'
    manifest = dict(contract=contract, split=split, condition=condition,
                    scenes=scenes, indices=indices, complete=False)
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding='utf-8'))
        if {k:v for k,v in previous.items() if k != 'complete'} != {k:v for k,v in manifest.items() if k != 'complete'}:
            raise ValueError(f'Cache protocol mismatch: {folder}')
    else:
        write_json(manifest_path, manifest)
    branch = 'clean' if condition == 'clean' else 'weather'
    condition_index = CONDITIONS[split].index(condition)
    start = time.perf_counter()
    with torch.no_grad():
        for position, batch in enumerate(loader):
            index = indices[position]
            observed = int(batch['ego']['communication_sample_index'][0])
            if observed != index:
                raise RuntimeError(f'Dataloader sample/index mismatch: {observed} != {index}')
            path = folder/f'{index:08d}.pt'
            if path.is_file():
                continue
            batch = to_device(batch, target)
            ctx, fused, proposal = candidate(
                model, module, batch, branch, settings,
                settings['seed']+index*11+condition_index, verify=position == 0)
            baseline = dataset.post_process(batch, {'ego':ctx['baseline_prediction']})
            outcomes = []
            for tile in proposal['tiles'].tolist():
                prediction = patch_prediction(model.engine.base, ctx, fused, [tile], settings['changed_scales'])
                action = dataset.post_process(batch, {'ego':prediction})
                result = compare_predictions(baseline, action,
                                             settings['target_iou'], settings['fp_identity_iou'])
                outcomes.append([result['recovered'], result['lost'], result['new_fp']])
            proposal['targets'] = torch.tensor(outcomes, dtype=torch.float32).reshape(-1, 3)
            proposal['sample_index'] = index
            proposal['raw_feature_bytes'] = raw_feature_bytes(ctx)
            temporary = path.with_suffix('.partial')
            torch.save(proposal, temporary)
            temporary.replace(path)
            if (position+1) % 20 == 0 or position+1 == len(indices):
                print(f'{split}/{condition}: {position+1}/{len(indices)}, '
                      f'{(time.perf_counter()-start)/60:.1f} min', flush=True)
    if any(not (folder/f'{i:08d}.pt').is_file() for i in indices):
        raise RuntimeError(f'Incomplete cache: {folder}')
    manifest['complete'] = True
    write_json(manifest_path, manifest)
    return manifest


def records(run, split, condition, contract):
    folder = run/'cache'/split/condition
    manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    if not manifest['complete'] or manifest['contract'] != contract:
        raise ValueError(f'Incomplete or incompatible cache: {folder}')
    for index in manifest['indices']:
        row = torch.load(folder/f'{index:08d}.pt', map_location='cpu', weights_only=True)
        if row['sample_index'] != index or len(row['tiles']) != len(row['features']) or len(row['tiles']) != len(row['targets']):
            raise ValueError(f'Invalid candidate cache: {folder}/{index:08d}.pt')
        yield row


def fit(run, settings, contract):
    by_condition = {condition:list(records(run, 'train', condition, contract))
                    for condition in CONDITIONS['train']}
    train = [r for rows in by_condition.values() for r in rows]
    x = torch.cat([r['features'] for r in train], 0).float()
    y = torch.cat([r['targets'] for r in train], 0).float()
    if not len(x):
        raise RuntimeError('No candidate actions on selected training scenes')
    mean, std = x.mean(0), x.std(0, unbiased=False).clamp_min(1e-4)
    x = ((x-mean)/std).clamp(-20, 20)
    classes = outcome_class(y)
    seed_all(settings['seed'])
    selector = DecisionSelector(x.shape[1], settings['selector_hidden'])
    optimizer = torch.optim.AdamW(selector.parameters(), lr=settings['selector_learning_rate'])
    history = []
    for epoch in range(settings['selector_epochs']):
        selector.train()
        optimizer.zero_grad(set_to_none=True)
        loss, counts = weighted_cross_entropy(selector(x), classes)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    path = run/'selector.pth'
    torch.save(dict(state=selector.state_dict(), mean=mean, std=std,
                    input_dim=x.shape[1], hidden=settings['selector_hidden'],
                    contract=contract), path)
    result = dict(actions=len(x), harmful=int(counts[0]), unchanged=int(counts[1]),
                  beneficial=int(counts[2]), losses=history,
                  by_condition={condition:dict(zip(('harmful', 'unchanged', 'beneficial'),
                      torch.bincount(outcome_class(torch.cat([r['targets'] for r in rows])),
                                     minlength=3).tolist()))
                      for condition, rows in by_condition.items()},
                  decision='inconclusive_if_either_nonzero_class_has_fewer_than_10_examples',
                  adequate_class_support=bool(counts[0] >= 10 and counts[2] >= 10))
    write_json(run/'selector_training.json', result)
    return result


def load_selector(run, contract):
    state = torch.load(run/'selector.pth', map_location='cpu', weights_only=True)
    if state['contract'] != contract or state['input_dim'] != len(FEATURE_NAMES):
        raise ValueError('Selector/source/protocol mismatch')
    selector = DecisionSelector(state['input_dim'], state['hidden'])
    selector.load_state_dict(state['state'])
    return selector.eval(), state['mean'], state['std']


def evaluate_condition(run, model, module, hypes, options, condition, settings, target, contract, selector, mean, std):
    from ceif_audit.scoring import empty_stats, ap_values
    from opencood.utils import eval_utils

    dataset, loader, scenes, indices = selected_loader(hypes, options, 'validation', condition, settings)
    names = ['baseline']
    for budget in settings['action_budgets']:
        names += [f'{method}_top{budget}' for method in ('learned', 'confidence', 'v3_gate')]
    names += [f'learned_keep_top{max(settings["action_budgets"])}']
    stats = {name:empty_stats() for name in names}
    result = {name:dict(recovered=0, lost=0, new_fp=0, actions=0,
                        beneficial_actions=0, harmful_actions=0, unchanged_actions=0)
              for name in names if name != 'baseline'}
    cache = list(records(run, 'validation', condition, contract))
    pool_counts = torch.bincount(
        outcome_class(torch.cat([row['targets'] for row in cache])), minlength=3).tolist()
    payload = 0
    branch = 'clean' if condition == 'clean' else 'weather'
    condition_index = CONDITIONS['validation'].index(condition)
    with torch.no_grad():
        for position, batch in enumerate(loader):
            index = indices[position]
            observed = int(batch['ego']['communication_sample_index'][0])
            if observed != index:
                raise RuntimeError(f'Dataloader sample/index mismatch: {observed} != {index}')
            batch = to_device(batch, target)
            ctx, fused, proposal = candidate(
                model, module, batch, branch, settings,
                settings['seed']+index*11+condition_index, verify=position == 0)
            saved = cache[position]
            if saved['sample_index'] != index or not torch.equal(saved['tiles'], proposal['tiles']) or not torch.allclose(saved['features'], proposal['features'], atol=1e-4, rtol=1e-4):
                raise RuntimeError(f'Validation features changed since labeling: {condition}/{index}')
            payload += raw_feature_bytes(ctx)
            baseline = dataset.post_process(batch, {'ego':ctx['baseline_prediction']})
            for threshold in stats['baseline']:
                eval_utils.caluclate_tp_fp(*baseline, stats['baseline'], threshold)
            scores = scores_for_rows(proposal['features'], proposal['gate'], proposal['confidence'],
                                     selector, mean, std)
            for name in result:
                if name.startswith('learned_keep_'):
                    selected = selected_indices(scores['learned'], max(settings['action_budgets']), True)
                else:
                    method, budget = name.rsplit('_top', 1)
                    selected = selected_indices(scores[method], int(budget))
                targets = saved['targets'][selected]
                result[name]['actions'] += len(selected)
                if len(targets):
                    labels = outcome_class(targets)
                    result[name]['beneficial_actions'] += int((labels == 2).sum())
                    result[name]['harmful_actions'] += int((labels == 0).sum())
                    result[name]['unchanged_actions'] += int((labels == 1).sum())
                tiles = proposal['tiles'][selected].tolist()
                prediction = patch_prediction(model.engine.base, ctx, fused, tiles, settings['changed_scales']) if tiles else ctx['baseline_prediction']
                action = dataset.post_process(batch, {'ego':prediction})
                outcome = compare_predictions(baseline, action,
                                              settings['target_iou'], settings['fp_identity_iou'])
                for key in ('recovered', 'lost', 'new_fp'):
                    result[name][key] += outcome[key]
                for threshold in stats[name]:
                    eval_utils.caluclate_tp_fp(*action, stats[name], threshold)
            if (position+1) % 20 == 0 or position+1 == len(indices):
                print(f'evaluate/{condition}: {position+1}/{len(indices)}', flush=True)
    return dict(split='official_validation_scenario_subset', condition=condition,
                frames=len(indices), scene_ids=scenes, selected_frame_indices=indices,
                mean_raw_full_feature_bytes=payload/len(indices),
                candidate_pool=dict(zip(('harmful', 'unchanged', 'beneficial'), pool_counts)),
                results={name:ap_values(value, eval_utils) for name, value in stats.items()},
                diagnostics=result,
                action_budget_note='Top-K rows per frame; method comparisons use identical K. '
                                   'Individual labels do not equal joint post-NMS results.',
                ap_note='Small development subset only; no independent test claim.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot-config', default='local_fusion_decision_pilot/experiment.yaml')
    parser.add_argument('--v3-config', required=True)
    parser.add_argument('--v3-checkpoint', required=True)
    parser.add_argument('--frontend-config', required=True)
    parser.add_argument('--frontend-checkpoint', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    verify_frozen()
    settings = pilot_settings(args.pilot_config)
    options, hypes = er.load_config(args.v3_config, args.frontend_config)
    v3rt.settings(options)
    target = device()
    seed_all(settings['seed'])
    model, frontend_hash = er.load_model(hypes, options, args.frontend_checkpoint, target)
    model.requires_grad_(False)
    v3_contract = v3rt.contract(options, args.frontend_config, frontend_hash)
    module, state = v3rt.load(args.v3_checkpoint, v3_contract, target)
    if state['variant'] != 'residual':
        raise ValueError('Pilot requires the trained v3 residual checkpoint')
    module.requires_grad_(False)
    source_paths = {'core.py':Path(core.__file__), 'pipeline.py':Path(__file__),
                    'v2_supervision.py':Path(__file__).parents[1]/'local_fusion_utility_v2'/'supervision.py'}
    contract = dict(v3=v3_contract, checkpoint_sha256=sha256(args.v3_checkpoint),
                    pilot_config_sha256=sha256(args.pilot_config),
                    pilot_sources={name:sha256(path) for name, path in source_paths.items()},
                    purpose='train/validation small decision feasibility; no OPV2V-W')
    run = Path(args.run).resolve()
    if args.resume:
        old = json.loads((run/'contract.json').read_text(encoding='utf-8'))
        if old != contract:
            raise ValueError('Resume contract mismatch')
    else:
        run = new_output(run)
        write_json(run/'contract.json', contract)
        (run/'pilot_config.yaml').write_text(yaml.safe_dump(settings, sort_keys=False), encoding='utf-8')
        snapshot = run/'source_snapshot'
        snapshot.mkdir()
        for name, source in source_paths.items():
            shutil.copy2(source, snapshot/name)
    journal = []
    for split, conditions in CONDITIONS.items():
        for condition in conditions:
            started = time.perf_counter()
            collect(run, model, module, hypes, options, split, condition, settings, target, contract)
            journal.append(dict(stage=f'collect_{split}_{condition}', seconds=time.perf_counter()-started))
            write_json(run/'progress.json', journal)
    training = fit(run, settings, contract)
    selector, mean, std = load_selector(run, contract)
    reports = {}
    for condition in CONDITIONS['validation']:
        started = time.perf_counter()
        reports[condition] = evaluate_condition(run, model, module, hypes, options, condition,
                                                settings, target, contract, selector, mean, std)
        write_json(run/f'{condition}_validation.json', reports[condition])
        journal.append(dict(stage=f'evaluate_{condition}', seconds=time.perf_counter()-started))
        write_json(run/'progress.json', journal)
    write_json(run/'decision_pilot_results.json', dict(training=training, validation=reports,
        limitations=['Single v3-residual tile action; no peer-action selection yet',
                     'OPV2V-W is not accessed',
                     'Validation subset is for feasibility, not final AP']))
    verify_frozen()
    print('DECISION PILOT COMPLETE:', run/'decision_pilot_results.json', flush=True)


if __name__ == '__main__':
    main()
