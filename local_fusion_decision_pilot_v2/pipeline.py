"""Small A follow-up: prediction-guided regions and source-specific actions.

Only official train and validation scenes are accessed. OPV2V-W is excluded.
GT is used for action labels and retrospective validation metrics, never for
region/source/action ranking at inference.
"""
import argparse
import json
from pathlib import Path
import shutil
import time

import torch
import yaml

from ceif_audit.scoring import ap_values, empty_stats
from gspr_communication.runtime import (device, new_output, seed_all, sha256,
                                        verify_frozen, write_json)
from gspr_evidence import runtime as er
from local_fusion_decision_pilot.core import (
    DecisionSelector, candidate_features, outcome_class, residual_with_gate_maps,
    weighted_cross_entropy,
)
from local_fusion_decision_pilot.pipeline import selected_loader
from local_fusion_utility_v2.outcomes import compare_predictions
from local_fusion_utility_v2.runtime import prepare_context, raw_feature_bytes
from local_fusion_v3 import runtime as v3rt
from opencood.tools.train_utils import to_device
from opencood.utils import eval_utils
from .core import (ACTION_FEATURE_NAMES, action_features, actions_for_tiles,
                   choose_distinct_rows, prediction_for_choices, select_tiles,
                   simple_scores)
from . import core


CONDITIONS = {'train':('clean','mixed'),
              'validation':('clean','fog','rain','snow')}
METHODS = ('learned','confidence','response_change')


def settings_from(path):
    settings = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    for key in ('seed','train_scenes','validation_scenes','frames_per_scene',
                'tiles_per_frame','tile_size','max_peer_actions_per_tile',
                'selector_epochs','selector_hidden','minimum_class_support'):
        if type(settings.get(key)) is not int or settings[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if not 0 <= settings['low_score_floor'] < settings['high_score_floor'] <= 1:
        raise ValueError('Invalid prediction score strata')
    if not 0 < settings['selector_learning_rate']:
        raise ValueError('Invalid selector learning rate')
    if settings['max_peer_actions_per_tile'] > 2:
        raise ValueError('Pilot provides at most two distinct peer-query actions per tile')
    if tuple(settings['changed_scales']) != (0,1):
        raise ValueError('Pilot actions must modify v3 scales 0 and 1 only')
    budgets = settings['action_budgets']
    if not budgets or any(type(k) is not int or k < 1 or k > settings['tiles_per_frame'] for k in budgets):
        raise ValueError('Action budgets must fit the available distinct tiles')
    if len(set(budgets)) != len(budgets):
        raise ValueError('Action budgets must be distinct')
    for key in ('target_iou','fp_identity_iou'):
        if not 0 < settings[key] <= 1:
            raise ValueError(f'Invalid {key}')
    return settings


def propose(model, module, batch, branch, settings, verify=False):
    context = prepare_context(model,batch['ego'],branch,
                              {'tile_size':settings['tile_size']},verify=verify)
    residual, gates = residual_with_gate_maps(module,context['levels'])
    base_features, _ = candidate_features(context,residual,gates,model.engine.base)
    tiles = select_tiles(context,base_features,settings) if len(context['descriptors']) else []
    actions = actions_for_tiles(context,tiles,settings)
    features, confidence, response, predictions, strata = [],[],[],[],[]
    low, high = settings['low_score_floor'], settings['high_score_floor']
    for action in actions:
        prediction = prediction_for_choices(model.engine.base,context,residual,
                                            [action],settings['changed_scales'])
        features.append(action_features(context,base_features,action,prediction).detach().cpu())
        conf, change = simple_scores(context,action,prediction)
        confidence.append(conf)
        response.append(change)
        tile_activity = float(context['activity'].flatten()[action[0]])
        strata.append(0 if tile_activity >= high else 1 if tile_activity >= low else 2)
        predictions.append(prediction)
    rows = dict(actions=torch.tensor(actions,dtype=torch.long).reshape(-1,2),
                features=torch.stack(features) if features else torch.empty((0,len(ACTION_FEATURE_NAMES))),
                confidence=torch.tensor(confidence,dtype=torch.float32),
                response_change=torch.tensor(response,dtype=torch.float32),
                strata=torch.tensor(strata,dtype=torch.long))
    return context,residual,rows,predictions


def collect(run,model,module,hypes,options,split,condition,settings,target,contract):
    dataset,loader,scenes,indices = selected_loader(hypes,options,split,condition,settings)
    folder = run/'cache'/split/condition
    folder.mkdir(parents=True,exist_ok=True)
    manifest_path = folder/'manifest.json'
    manifest = dict(contract=contract,split=split,condition=condition,
                    scenes=scenes,indices=indices,complete=False)
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding='utf-8'))
        if {k:v for k,v in old.items() if k != 'complete'} != {k:v for k,v in manifest.items() if k != 'complete'}:
            raise ValueError(f'Incompatible resumed cache: {folder}')
    else:
        write_json(manifest_path,manifest)
    branch = 'clean' if condition == 'clean' else 'weather'
    started = time.perf_counter()
    new_actions = 0
    with torch.no_grad():
        for position,batch in enumerate(loader):
            index = indices[position]
            if int(batch['ego']['communication_sample_index'][0]) != index:
                raise RuntimeError('Dataloader/sample index mismatch')
            path = folder/f'{index:08d}.pt'
            if path.is_file():
                continue
            batch = to_device(batch,target)
            context,residual,rows,predictions = propose(model,module,batch,branch,settings,
                                                         verify=position == 0)
            baseline = dataset.post_process(batch,{'ego':context['baseline_prediction']})
            outcomes = []
            for prediction in predictions:
                action = dataset.post_process(batch,{'ego':prediction})
                value = compare_predictions(baseline,action,
                                            settings['target_iou'],settings['fp_identity_iou'])
                outcomes.append([value['recovered'],value['lost'],value['new_fp']])
            rows.update(targets=torch.tensor(outcomes,dtype=torch.float32).reshape(-1,3),
                        sample_index=index,raw_feature_bytes=raw_feature_bytes(context))
            temporary = path.with_suffix('.partial')
            torch.save(rows,temporary)
            temporary.replace(path)
            new_actions += len(outcomes)
            if (position+1) % 20 == 0 or position+1 == len(indices):
                print(f'collect {split}/{condition}: {position+1}/{len(indices)} '
                      f'frames, {new_actions} new actions, '
                      f'{(time.perf_counter()-started)/60:.1f} min',flush=True)
    if any(not (folder/f'{i:08d}.pt').is_file() for i in indices):
        raise RuntimeError(f'Incomplete cache: {folder}')
    manifest['complete'] = True
    write_json(manifest_path,manifest)
    return manifest


def records(run,split,condition,contract):
    folder = run/'cache'/split/condition
    manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    if not manifest['complete'] or manifest['contract'] != contract:
        raise ValueError(f'Incomplete or incompatible cache: {folder}')
    for index in manifest['indices']:
        row = torch.load(folder/f'{index:08d}.pt',map_location='cpu',weights_only=True)
        size = len(row['actions'])
        if row['sample_index'] != index or any(len(row[key]) != size for key in
            ('features','confidence','response_change','strata','targets')):
            raise ValueError(f'Invalid candidate cache: {condition}/{index}')
        yield row


def class_counts(records_):
    labels = outcome_class(torch.cat([row['targets'] for row in records_]))
    return dict(zip(('harmful','unchanged','beneficial'),
                    torch.bincount(labels,minlength=3).tolist()))


def fit(run,settings,contract):
    groups = {condition:list(records(run,'train',condition,contract))
              for condition in CONDITIONS['train']}
    all_rows = [row for rows in groups.values() for row in rows]
    x = torch.cat([row['features'] for row in all_rows]).float()
    y = torch.cat([row['targets'] for row in all_rows]).float()
    if not len(x):
        raise RuntimeError('No sampled source actions on training scenes')
    mean,std = x.mean(0),x.std(0,unbiased=False).clamp_min(1e-4)
    x = ((x-mean)/std).clamp(-20,20)
    classes = outcome_class(y)
    seed_all(settings['seed'])
    selector = DecisionSelector(len(ACTION_FEATURE_NAMES),settings['selector_hidden'])
    optimizer = torch.optim.AdamW(selector.parameters(),lr=settings['selector_learning_rate'])
    history = []
    for _ in range(settings['selector_epochs']):
        selector.train()
        optimizer.zero_grad(set_to_none=True)
        loss,counts = weighted_cross_entropy(selector(x),classes)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    state = dict(model=selector.state_dict(),mean=mean,std=std,contract=contract,
                 input_dim=len(ACTION_FEATURE_NAMES),hidden=settings['selector_hidden'])
    torch.save(state,run/'selector.pth')
    report = dict(actions=len(x),harmful=int(counts[0]),unchanged=int(counts[1]),
                  beneficial=int(counts[2]),by_condition={c:class_counts(r) for c,r in groups.items()},
                  by_action_and_region=pool_breakdown(all_rows),
                  losses=history,minimum_class_support=settings['minimum_class_support'],
                  adequate_class_support=bool(counts[0] >= settings['minimum_class_support'] and
                                              counts[2] >= settings['minimum_class_support']))
    write_json(run/'selector_training.json',report)
    return report


def load_selector(run,contract):
    value = torch.load(run/'selector.pth',map_location='cpu',weights_only=True)
    if value['contract'] != contract or value['input_dim'] != len(ACTION_FEATURE_NAMES):
        raise ValueError('Selector/protocol mismatch')
    selector = DecisionSelector(value['input_dim'],value['hidden'])
    selector.load_state_dict(value['model'])
    return selector.eval(),value['mean'],value['std']


def pool_breakdown(rows):
    counts = {'residual':dict(harmful=0,unchanged=0,beneficial=0),
              'peer_query':dict(harmful=0,unchanged=0,beneficial=0),
              'active':dict(harmful=0,unchanged=0,beneficial=0),
              'low_score':dict(harmful=0,unchanged=0,beneficial=0),
              'background':dict(harmful=0,unchanged=0,beneficial=0)}
    labels = ('harmful','unchanged','beneficial')
    strata = ('active','low_score','background')
    for row in rows:
        classes = outcome_class(row['targets']).tolist()
        for i,cls in enumerate(classes):
            counts['residual' if int(row['actions'][i,1]) == 1 else 'peer_query'][labels[cls]] += 1
            counts[strata[int(row['strata'][i])]][labels[cls]] += 1
    return counts


def evaluate_condition(run,model,module,hypes,options,condition,settings,target,
                       contract,selector,mean,std):
    dataset,loader,scenes,indices = selected_loader(hypes,options,'validation',condition,settings)
    saved_rows = list(records(run,'validation',condition,contract))
    names = ['baseline']+[f'{method}_top{k}' for k in settings['action_budgets']
                          for method in METHODS]
    names.append(f'learned_keep_top{max(settings["action_budgets"])}')
    stats = {name:empty_stats() for name in names}
    outcomes = {name:dict(recovered=0,lost=0,new_fp=0,actions=0,
                          beneficial_actions=0,harmful_actions=0,unchanged_actions=0)
                for name in names if name != 'baseline'}
    payload,branch = 0,('clean' if condition == 'clean' else 'weather')
    with torch.no_grad():
        for position,batch in enumerate(loader):
            index = indices[position]
            if int(batch['ego']['communication_sample_index'][0]) != index:
                raise RuntimeError('Dataloader/sample index mismatch')
            batch = to_device(batch,target)
            context,residual,current,_ = propose(model,module,batch,branch,settings,
                                                   verify=position == 0)
            saved = saved_rows[position]
            if (saved['sample_index'] != index or
                not torch.equal(saved['actions'],current['actions']) or
                not torch.allclose(saved['features'],current['features'],atol=1e-4,rtol=1e-4)):
                raise RuntimeError(f'Validation candidates changed since labeling: {condition}/{index}')
            payload += raw_feature_bytes(context)
            baseline = dataset.post_process(batch,{'ego':context['baseline_prediction']})
            for threshold in stats['baseline']:
                eval_utils.caluclate_tp_fp(*baseline,stats['baseline'],threshold)
            x = ((current['features'].float()-mean)/std).clamp(-20,20)
            probabilities = selector(x).softmax(-1)
            scores = dict(learned=probabilities[:,2]-probabilities[:,0],
                          confidence=current['confidence'],
                          response_change=current['response_change'])
            actions = current['actions'].tolist()
            for name,accumulator in outcomes.items():
                if name.startswith('learned_keep_'):
                    selected = choose_distinct_rows(scores['learned'],actions,
                                                     max(settings['action_budgets']),True)
                else:
                    method,budget = name.rsplit('_top',1)
                    selected = choose_distinct_rows(scores[method],actions,int(budget))
                targets = saved['targets'][selected]
                accumulator['actions'] += len(selected)
                if len(targets):
                    bincount = torch.bincount(outcome_class(targets),minlength=3)
                    for key,count in zip(('harmful_actions','unchanged_actions','beneficial_actions'),bincount):
                        accumulator[key] += int(count)
                chosen = [actions[i] for i in selected]
                prediction = (prediction_for_choices(model.engine.base,context,residual,
                    chosen,settings['changed_scales']) if chosen else context['baseline_prediction'])
                action = dataset.post_process(batch,{'ego':prediction})
                comparison = compare_predictions(baseline,action,
                                                 settings['target_iou'],settings['fp_identity_iou'])
                for key in ('recovered','lost','new_fp'):
                    accumulator[key] += comparison[key]
                for threshold in stats[name]:
                    eval_utils.caluclate_tp_fp(*action,stats[name],threshold)
            if (position+1) % 20 == 0 or position+1 == len(indices):
                print(f'evaluate/{condition}: {position+1}/{len(indices)}',flush=True)
    return dict(frames=len(indices),scene_ids=scenes,selected_frame_indices=indices,
                candidate_pool=class_counts(saved_rows),
                candidate_breakdown=pool_breakdown(saved_rows),
                mean_raw_full_feature_bytes=payload/len(indices),
                results={name:ap_values(value,eval_utils) for name,value in stats.items()},
                diagnostics=outcomes,
                note='Official validation subset only; equal distinct-tile action budgets; '
                     'single-action labels and joint post-NMS outputs both reported.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot-config',default='local_fusion_decision_pilot_v2/experiment.yaml')
    parser.add_argument('--v3-config',required=True)
    parser.add_argument('--v3-checkpoint',required=True)
    parser.add_argument('--frontend-config',required=True)
    parser.add_argument('--frontend-checkpoint',required=True)
    parser.add_argument('--run',required=True)
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    verify_frozen()
    settings = settings_from(args.pilot_config)
    options,hypes = er.load_config(args.v3_config,args.frontend_config)
    v3rt.settings(options)
    target = device()
    seed_all(settings['seed'])
    model,frontend_hash = er.load_model(hypes,options,args.frontend_checkpoint,target)
    model.requires_grad_(False)
    v3_contract = v3rt.contract(options,args.frontend_config,frontend_hash)
    module,state = v3rt.load(args.v3_checkpoint,v3_contract,target)
    if state['variant'] != 'residual':
        raise ValueError('Requires the trained v3 residual checkpoint')
    module.requires_grad_(False)
    project = Path(__file__).parents[1]
    sources = {'core.py':Path(core.__file__), 'pipeline.py':Path(__file__),
               'pilot_v1_core.py':project/'local_fusion_decision_pilot'/'core.py',
               'pilot_v1_pipeline.py':project/'local_fusion_decision_pilot'/'pipeline.py',
               'v2_fusion.py':project/'local_fusion_utility_v2'/'fusion.py'}
    contract = dict(v3=v3_contract,v3_checkpoint_sha256=sha256(args.v3_checkpoint),
                    pilot_config_sha256=sha256(args.pilot_config),
                    sources={name:sha256(path) for name,path in sources.items()},
                    purpose='A source-action follow-up; train/validation only; no OPV2V-W')
    run = Path(args.run).resolve()
    if args.resume:
        old = json.loads((run/'contract.json').read_text(encoding='utf-8'))
        if old != contract:
            raise ValueError('Resume contract differs')
    else:
        run = new_output(run)
        write_json(run/'contract.json',contract)
        (run/'pilot_config.yaml').write_text(yaml.safe_dump(settings,sort_keys=False),encoding='utf-8')
        snapshot = run/'source_snapshot'
        snapshot.mkdir()
        for name,path in sources.items():
            shutil.copy2(path,snapshot/name)
    journal = []
    for split,conditions in CONDITIONS.items():
        for condition in conditions:
            began = time.perf_counter()
            collect(run,model,module,hypes,options,split,condition,settings,target,contract)
            journal.append(dict(stage=f'collect_{split}_{condition}',seconds=time.perf_counter()-began))
            write_json(run/'progress.json',journal)
    training = fit(run,settings,contract)
    selector,mean,std = load_selector(run,contract)
    reports = {}
    for condition in CONDITIONS['validation']:
        began = time.perf_counter()
        reports[condition] = evaluate_condition(run,model,module,hypes,options,condition,
                                                settings,target,contract,selector,mean,std)
        write_json(run/f'{condition}_validation.json',reports[condition])
        journal.append(dict(stage=f'evaluate_{condition}',seconds=time.perf_counter()-began))
        write_json(run/'progress.json',journal)
    write_json(run/'decision_pilot_v2_results.json',dict(training=training,validation=reports,
        limitations=['Only prediction-guided tiles and v3/peer-query actions, not a full fusion method',
                     'No OPV2V-W access or independent test',
                     'Validation AP comes from a small scenario subset']))
    verify_frozen()
    print('A DECISION FOLLOW-UP COMPLETE:',run/'decision_pilot_v2_results.json',flush=True)


if __name__ == '__main__':
    main()
