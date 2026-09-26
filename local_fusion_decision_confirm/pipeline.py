"""Frozen Scheme A check on validation frames absent from the pilot.

No training, threshold tuning, GT-based proposal, or OPV2V-W access occurs here.
Ground truth is used only by the final retrospective evaluation.
"""
import argparse
import bisect
import json
from pathlib import Path
import shutil
import time

import torch
from torch.utils.data import DataLoader, Subset
import yaml

from ceif_audit.scoring import ap_values, empty_stats, merge
from gspr_communication.runtime import (
    device, new_output, seed_all, seed_worker, sha256, verify_frozen, write_json,
)
from gspr_evidence import runtime as er
from local_fusion_decision_pilot_v2 import core as v2core
from local_fusion_decision_pilot_v2.pipeline import load_selector, propose
from local_fusion_utility_v2.outcomes import compare_predictions
from local_fusion_utility_v2.runtime import raw_feature_bytes
from local_fusion_v3 import runtime as v3rt
from opencood.tools.train_utils import to_device
from opencood.utils import eval_utils

from .core import policy_rows
from .sampling import blocked_frames, holdout_frames
from . import core


CONDITIONS = ('clean', 'fog', 'rain', 'snow')


def settings_from(path):
    settings = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    for key in ('seed', 'max_holdout_scenes', 'frames_per_scene'):
        if type(settings.get(key)) is not int or settings[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if type(settings.get('exclusion_radius_frames')) is not int or settings['exclusion_radius_frames'] < 0:
        raise ValueError('exclusion_radius_frames must be a nonnegative integer')
    if settings.get('fixed_budgets') != [1, 4]:
        raise ValueError('This confirmation is pre-registered for top1/top4')
    for key in ('target_iou', 'fp_identity_iou'):
        if not 0 < settings[key] <= 1:
            raise ValueError(f'Invalid {key}')
    return settings


def pilot_manifest(pilot_run, condition, expected_contract):
    path = pilot_run/'cache'/'validation'/condition/'manifest.json'
    value = json.loads(path.read_text(encoding='utf-8'))
    if not value['complete'] or value['contract'] != expected_contract:
        raise ValueError(f'Pilot validation manifest is incomplete/incompatible: {path}')
    return value


def source_contract(pilot_run, v3_contract, v3_checkpoint):
    old = json.loads((pilot_run/'contract.json').read_text(encoding='utf-8'))
    if old['v3'] != v3_contract or old['v3_checkpoint_sha256'] != sha256(v3_checkpoint):
        raise ValueError('V3 model differs from the trained pilot selector')
    project = Path(__file__).parents[1]
    paths = {
        'core.py': project/'local_fusion_decision_pilot_v2'/'core.py',
        'pipeline.py': project/'local_fusion_decision_pilot_v2'/'pipeline.py',
        'pilot_v1_core.py': project/'local_fusion_decision_pilot'/'core.py',
        'pilot_v1_pipeline.py': project/'local_fusion_decision_pilot'/'pipeline.py',
        'v2_fusion.py': project/'local_fusion_utility_v2'/'fusion.py',
    }
    if old['sources'] != {name:sha256(path) for name,path in paths.items()}:
        raise ValueError('Pilot source code changed since selector training')
    if old['pilot_config_sha256'] != sha256(project/'local_fusion_decision_pilot_v2'/'experiment.yaml'):
        raise ValueError('Pilot settings changed since selector training')
    reference = pilot_manifest(pilot_run, 'clean', old)
    for condition in CONDITIONS[1:]:
        row = pilot_manifest(pilot_run, condition, old)
        if row['scenes'] != reference['scenes'] or row['indices'] != reference['indices']:
            raise ValueError('Pilot weather conditions used different validation scenes')
    return old, reference


def synchronized_now():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def stats_for(prediction):
    stats = empty_stats()
    for threshold in stats:
        eval_utils.caluclate_tp_fp(*prediction, stats, threshold)
    return stats


def evaluate_frame(batch, dataset, model, module, selector, mean, std,
                   pilot_settings, confirm_settings, condition):
    before = synchronized_now()
    context, residual, current, _ = propose(
        model, module, batch, 'clean' if condition == 'clean' else 'weather',
        pilot_settings)
    proposed_ms = (synchronized_now()-before)*1000
    before = synchronized_now()
    baseline = dataset.post_process(batch, {'ego':context['baseline_prediction']})
    baseline_ms = (synchronized_now()-before)*1000
    with torch.no_grad():
        features = ((current['features'].float()-mean)/std).clamp(-20,20)
        probabilities = selector(features).softmax(-1)
    scores = {'learned': probabilities[:,2]-probabilities[:,0],
              'confidence': current['confidence']}
    actions = current['actions'].tolist()
    choices = policy_rows(scores, actions, confirm_settings['fixed_budgets'])
    result = {'baseline': dict(stats=stats_for(baseline), timing_ms=baseline_ms,
                               recovered=0, lost=0, new_fp=0, actions=0)}
    for name, selected in choices.items():
        before = synchronized_now()
        chosen = [actions[i] for i in selected]
        if chosen:
            prediction = v2core.prediction_for_choices(
                model.engine.base, context, residual, chosen,
                pilot_settings['changed_scales'])
            output = dataset.post_process(batch, {'ego':prediction})
            inference_ms = (synchronized_now()-before)*1000
        else:
            output, inference_ms = baseline, 0.
        difference = compare_predictions(baseline, output,
            confirm_settings['target_iou'], confirm_settings['fp_identity_iou'])
        result[name] = dict(stats=stats_for(output), timing_ms=inference_ms,
                            actions=len(selected),
                            residual_actions=sum(code == 1 for _,code in chosen),
                            peer_query_actions=sum(code > 1 for _,code in chosen),
                            **difference)
    return dict(policies=result, proposal_ms=proposed_ms,
                candidate_actions=len(actions),
                raw_full_feature_bytes=raw_feature_bytes(context))


def aggregate(frame_rows, scene_ends, pilot_scenes, pilot_indices,
              exclusion_radius, scene_ids, indices):
    names = list(frame_rows[0]['policies'])
    totals = {name:empty_stats() for name in names}
    scene_stats = {scene:{name:empty_stats() for name in names} for scene in scene_ids}
    diagnostics = {name:dict(recovered=0,lost=0,new_fp=0,actions=0,
                             residual_actions=0,peer_query_actions=0,
                             inference_ms=0.) for name in names if name != 'baseline'}
    proposal_ms = 0.
    payload_bytes = 0
    candidate_actions = 0
    blocked = blocked_frames(scene_ends,pilot_scenes,pilot_indices,exclusion_radius)
    for index,row in zip(indices,frame_rows):
        scene = bisect.bisect_right(scene_ends,index)
        if index in blocked or scene not in scene_stats:
            raise AssertionError('Validation frame overlap/temporal-gap drift')
        proposal_ms += row['proposal_ms']
        payload_bytes += row['raw_full_feature_bytes']
        candidate_actions += row['candidate_actions']
        for name in names:
            merge(totals[name], row['policies'][name]['stats'])
            merge(scene_stats[scene][name], row['policies'][name]['stats'])
            if name != 'baseline':
                for key in diagnostics[name]:
                    diagnostics[name][key] += row['policies'][name][
                        'timing_ms' if key == 'inference_ms' else key]
    metrics = {name:ap_values(stats,eval_utils) for name,stats in totals.items()}
    scene_ap70 = {str(scene):{name:ap_values(stats,eval_utils)['ap70']
                              for name,stats in rows.items()}
                  for scene,rows in scene_stats.items()}
    return dict(frames=len(indices),scene_ids=scene_ids,selected_frame_indices=indices,
                pilot_scene_ids=sorted(pilot_scenes),results=metrics,
                overlapping_scene_ids=sorted(set(scene_ids) & set(pilot_scenes)),
                pilot_frame_count=len(pilot_indices),
                exclusion_radius_frames=exclusion_radius,
                diagnostics=diagnostics,scene_ap70=scene_ap70,
                timing=dict(mean_proposal_ms=proposal_ms/len(indices),
                    mean_baseline_postprocess_ms=sum(r['policies']['baseline']['timing_ms']
                        for r in frame_rows)/len(indices),
                    mean_candidate_actions=candidate_actions/len(indices),
                    mean_policy_inference_ms={name:row['inference_ms']/len(indices)
                        for name,row in diagnostics.items()},
                    mean_raw_full_feature_bytes=payload_bytes/len(indices)),
                note='Pilot-frame-excluded official validation subset with a temporal gap; '
                     'scenarios can overlap. Frozen pilot selector and settings; '
                     'confidence_matched_keep uses the same number of actions per frame '
                     'as learned KEEP. Timing excludes data loading and GT metrics.')


def run_condition(run, model, module, selector, mean, std, hypes, options,
                  target, condition, pilot_settings, confirm_settings,
                  pilot_scenes, pilot_indices, expected_scenes, expected_indices):
    seed_all(options['seed'])
    dataset, _, _ = er.make_loader(hypes, options, train=False, weather=condition)
    scene_ids, indices = holdout_frames(dataset.len_record, pilot_scenes,
        pilot_indices, confirm_settings['seed'], confirm_settings['max_holdout_scenes'],
        confirm_settings['frames_per_scene'],confirm_settings['exclusion_radius_frames'])
    if scene_ids != expected_scenes or indices != expected_indices:
        raise ValueError(f'Weather condition {condition} changed scenario/frame selection')
    loader = DataLoader(Subset(dataset,indices),batch_size=1,shuffle=False,
        num_workers=options['workers'],collate_fn=dataset.collate_batch_test,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(options['seed']))
    folder = run/'frames'/condition
    folder.mkdir(parents=True,exist_ok=True)
    began = time.perf_counter()
    with torch.no_grad():
        for position,batch in enumerate(loader):
            index = indices[position]
            if int(batch['ego']['communication_sample_index'][0]) != index:
                raise RuntimeError('Dataloader/sample index mismatch')
            path = folder/f'{index:08d}.pt'
            if not path.is_file():
                batch = to_device(batch,target)
                frame = evaluate_frame(batch,dataset,model,module,selector,mean,std,
                                       pilot_settings,confirm_settings,condition)
                frame['sample_index'] = index
                temporary = path.with_suffix('.partial')
                torch.save(frame,temporary)
                temporary.replace(path)
            if (position+1) % 20 == 0 or position+1 == len(indices):
                print(f'confirm/{condition}: {position+1}/{len(indices)} frames, '
                      f'{(time.perf_counter()-began)/60:.1f} min',flush=True)
                write_json(run/'progress.json',dict(condition=condition,
                    condition_frames_done=position+1,condition_total_frames=len(indices),
                    elapsed_minutes=(time.perf_counter()-began)/60))
    frames = [torch.load(folder/f'{index:08d}.pt',map_location='cpu',weights_only=True)
              for index in indices]
    if any(row['sample_index'] != index for index,row in zip(indices,frames)):
        raise RuntimeError('Resumed frame cache/index mismatch')
    return aggregate(frames,dataset.len_record,set(pilot_scenes),pilot_indices,
                     confirm_settings['exclusion_radius_frames'],scene_ids,indices)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--confirm-config',default='local_fusion_decision_confirm/experiment.yaml')
    parser.add_argument('--pilot-run',required=True)
    parser.add_argument('--v3-config',required=True)
    parser.add_argument('--v3-checkpoint',required=True)
    parser.add_argument('--frontend-config',required=True)
    parser.add_argument('--frontend-checkpoint',required=True)
    parser.add_argument('--run',required=True)
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    verify_frozen()
    confirm_settings = settings_from(args.confirm_config)
    pilot_settings = yaml.safe_load(Path('local_fusion_decision_pilot_v2/experiment.yaml').read_text(encoding='utf-8'))
    for key in ('target_iou', 'fp_identity_iou'):
        if confirm_settings[key] != pilot_settings[key]:
            raise ValueError(f'Confirmation changed pilot {key}')
    options,hypes = er.load_config(args.v3_config,args.frontend_config)
    v3rt.settings(options)
    target = device()
    seed_all(options['seed'])
    model,frontend_hash = er.load_model(hypes,options,args.frontend_checkpoint,target)
    model.requires_grad_(False)
    v3_contract = v3rt.contract(options,args.frontend_config,frontend_hash)
    module,state = v3rt.load(args.v3_checkpoint,v3_contract,target)
    if state['variant'] != 'residual':
        raise ValueError('Requires the trained v3 residual checkpoint')
    module.requires_grad_(False)
    pilot_run = Path(args.pilot_run).resolve()
    old_contract,pilot_validation = source_contract(
        pilot_run,v3_contract,args.v3_checkpoint)
    selector,mean,std = load_selector(pilot_run,old_contract)
    source_paths = {'core.py':Path(core.__file__),'pipeline.py':Path(__file__),
                    'sampling.py':Path(__file__).with_name('sampling.py'),
                    'run_all.sh':Path(__file__).with_name('run_all.sh'),
                    'test_core.py':Path(__file__).with_name('test_core.py'),
                    'test_sampling.py':Path(__file__).with_name('test_sampling.py')}
    contract = dict(pilot_contract=old_contract,
                    pilot_selector_sha256=sha256(pilot_run/'selector.pth'),
                    confirm_config_sha256=sha256(args.confirm_config),
                    sources={name:sha256(path) for name,path in source_paths.items()},
                    purpose='Frozen Scheme A confirmation; pilot-frame-excluded validation only')
    # Read the clean dataset only to construct the one fixed holdout list.
    dataset,_,_ = er.make_loader(hypes,options,train=False,weather='clean')
    scene_ids,indices = holdout_frames(dataset.len_record,pilot_validation['scenes'],
        pilot_validation['indices'],confirm_settings['seed'],
        confirm_settings['max_holdout_scenes'],confirm_settings['frames_per_scene'],
        confirm_settings['exclusion_radius_frames'])
    if len(indices) <= len(pilot_validation['indices']):
        raise ValueError('Confirmation must use more frames than the pilot')
    manifest = dict(pilot_run=str(pilot_run),pilot_scene_ids=pilot_validation['scenes'],
                    pilot_frame_indices=pilot_validation['indices'],
                    scene_ids=scene_ids,indices=indices,
                    overlapping_scene_ids=sorted(set(scene_ids) & set(pilot_validation['scenes'])),
                    exclusion_radius_frames=confirm_settings['exclusion_radius_frames'],
                    selection_uses_gt=False,selector_frozen=True)
    print(f'Holdout preflight: {len(scene_ids)} scenes, {len(indices)} new frames, '
          f'{len(manifest["overlapping_scene_ids"])} pilot scenes shared, '
          f'gap=±{confirm_settings["exclusion_radius_frames"]} indices',flush=True)
    run = Path(args.run).resolve()
    if args.resume:
        if json.loads((run/'contract.json').read_text(encoding='utf-8')) != contract:
            raise ValueError('Resume contract differs')
        if json.loads((run/'holdout_manifest.json').read_text(encoding='utf-8')) != manifest:
            raise ValueError('Holdout manifest differs')
    else:
        run = new_output(run)
        write_json(run/'contract.json',contract)
        snapshot = run/'source_snapshot'
        snapshot.mkdir()
        for name,path in source_paths.items():
            shutil.copy2(path,snapshot/name)
        shutil.copy2(args.confirm_config,run/'confirm_config.yaml')
        write_json(run/'holdout_manifest.json',manifest)
    reports = {}
    for condition in CONDITIONS:
        path = run/f'{condition}_confirmation.json'
        if args.resume and path.is_file():
            reports[condition] = json.loads(path.read_text(encoding='utf-8'))
            continue
        reports[condition] = run_condition(run,model,module,selector,mean,std,
            hypes,options,target,condition,pilot_settings,confirm_settings,
            pilot_validation['scenes'],pilot_validation['indices'],scene_ids,indices)
        write_json(path,reports[condition])
    write_json(run/'confirmation_results.json',dict(
        holdout=manifest,conditions=reports,
        limitations=['Pilot frames and neighboring frames are excluded, but scenarios '
                     'can overlap and the official validation split is not an untouched test set.',
                     'OPV2V-W is not opened.',
                     'No selector retraining or threshold tuning.',
                     'Equal action count does not ensure equal action type or exact compute.']))
    verify_frozen()
    print('A CONFIRMATION COMPLETE:',run/'confirmation_results.json',flush=True)


if __name__ == '__main__':
    main()
