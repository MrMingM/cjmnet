"""Complete-scene validation and fixed OPV2V/OPV2V-W evaluation."""
import argparse
import copy
import json
import time
from pathlib import Path
import torch
from gspr_evidence import runtime as er, benchmark
from gspr_communication.runtime import device, seed_all, new_output, write_json, sha256, verify_frozen
from ceif_audit.scoring import empty_stats, ap_values
from local_fusion_utility_v2.outcomes import compare_predictions, detection_outcome
from local_fusion_utility_v2.runtime import raw_feature_bytes
from . import runtime as rt


def synchronize(target):
    if target.type == 'cuda':
        torch.cuda.synchronize()


def choose(summaries, name, gates):
    """Validation-only model-versus-KEEP choice, same gates for both variants."""
    feasible, improvements = True, []
    for condition, summary in summaries.items():
        base, result = summary['results']['baseline'], summary['results'][name]
        harm = summary['diagnostics_vs_baseline'][name]
        denom = max(summary['baseline_tp'],1)
        feasible &= harm['lost'] <= gates['max_lost_tp_fraction']*denom
        feasible &= harm['new_fp'] <= gates['max_new_fp_per_baseline_tp']*denom
        feasible &= result['ap50'] >= base['ap50']-gates['ap50_tolerance']
        if condition == 'clean':
            feasible &= result['ap70'] >= base['ap70']-gates['clean_ap70_tolerance']
        improvements.append(result['ap70']-base['ap70'])
    gain = sum(improvements)/len(improvements)
    return dict(enabled=bool(feasible and gain > gates['minimum_mean_ap70_gain']),
                feasible=bool(feasible),mean_ap70_gain=gain)


def condition(model, modules, ds, loader, indices, branch, target, output, decisions):
    from opencood.tools.train_utils import to_device
    from opencood.utils import eval_utils
    names = ['baseline', *modules]
    if decisions is not None:
        names += [name+'_deployed' for name in modules]
    stats = {name:empty_stats() for name in names}
    diagnostics = {name:dict(recovered=0,lost=0,new_fp=0) for name in names if name != 'baseline'}
    timing = {name:0. for name in modules}
    changes = {name:0. for name in modules}
    count = payload = baseline_tp = baseline_fp = 0
    with torch.no_grad(), (output/'frames.jsonl').open('w',encoding='utf-8') as stream:
        for batch in loader:
            batch = to_device(batch,target)
            synchronize(target)
            started = time.perf_counter()
            ctx = rt.context(model,batch['ego'],branch,verify=count == 0)
            synchronize(target)
            shared_ms = (time.perf_counter()-started)*1000
            predictions = {'baseline':ctx['baseline_prediction']}
            times = {}
            for name,module in modules.items():
                synchronize(target)
                started = time.perf_counter()
                predictions[name],info = module.predict(model.engine.base,ctx['levels'])
                synchronize(target)
                times[name] = shared_ms+(time.perf_counter()-started)*1000
                changes[name] += float(info['change'])
                if decisions is not None:
                    predictions[name+'_deployed'] = predictions[name] if decisions[name]['enabled'] else predictions['baseline']
            post = {}
            for name,prediction in predictions.items():
                synchronize(target)
                started = time.perf_counter()
                post[name] = ds.post_process(batch,{'ego':prediction})
                synchronize(target)
                if name in times:
                    timing[name] += times[name]+(time.perf_counter()-started)*1000
                boxes,scores,gt = post[name]
                for threshold in stats[name]:
                    eval_utils.caluclate_tp_fp(boxes,scores,gt,stats[name],threshold)
            for name in diagnostics:
                result = compare_predictions(post['baseline'],post[name],.7,.7)
                for key in diagnostics[name]:
                    diagnostics[name][key] += result[key]
            matched,false = detection_outcome(*post['baseline'],.7)
            baseline_tp += len(matched)
            baseline_fp += len(false)
            size = raw_feature_bytes(ctx)
            payload += size
            count += 1
            stream.write(json.dumps(dict(sample_index=int(batch['ego']['communication_sample_index'][0]),
                                         raw_feature_bytes=size))+'\n')
            if count == 1 or count % 20 == 0:
                print(f'{count}/{len(indices)} frames',flush=True)
    if count != len(indices) or count == 0 or not stats['baseline'][.7]['gt']:
        raise RuntimeError('Incomplete or empty evaluation')
    results = {name:ap_values(value,eval_utils) for name,value in stats.items()}
    for name,value in stats.items():
        folder = output/name
        folder.mkdir()
        eval_utils.eval_final_results(copy.deepcopy(value),str(folder),False)
    return dict(frames=count,results=results,diagnostics_vs_baseline=diagnostics,
                baseline_tp=baseline_tp,baseline_fp=baseline_fp,global_sort=False,
                mean_raw_full_feature_bytes=payload/count,
                mean_source_weight_l1={k:v/count for k,v in changes.items()},
                compute_including_context_head_postprocess_ms={k:v/count for k,v in timing.items()},
                timing_scope='Includes frozen encoding, baseline/context construction, module, head, NMS; excludes loader, H2D and metric matching. Shared context charged to each variant.')


def main():
    p = argparse.ArgumentParser()
    for name in ('config','frontend-config','frontend-checkpoint','run'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--phase',choices=('calibration','development','benchmark'),required=True)
    args = p.parse_args()
    verify_frozen()
    run = Path(args.run)
    output = new_output(run/args.phase)
    calibration_path = run/'calibration/calibration.json'
    calibration = None if args.phase == 'calibration' else json.loads(calibration_path.read_text(encoding='utf-8'))
    checkpoint_paths = {name:run/name/'best.pth' for name in ('attention','residual')}
    hashes = {name:sha256(path) for name,path in checkpoint_paths.items()}
    if calibration is not None and hashes != calibration['checkpoint_hashes']:
        raise ValueError('Checkpoints changed after calibration')
    summaries = {}
    target = device()
    for weather in ('clean','fog','rain','snow'):
        if args.phase == 'benchmark':
            options,hypes = benchmark.load_config(args.config,args.frontend_config,weather)
            branch,loader_weather = 'clean','clean'
        else:
            options,hypes = er.load_config(args.config,args.frontend_config)
            branch,loader_weather = ('clean','clean') if weather == 'clean' else ('weather',weather)
        seed_all(options['seed'])
        model,digest = er.load_model(hypes,options,args.frontend_checkpoint,target)
        specification = rt.contract(options,args.frontend_config,digest)
        if calibration is not None and specification != calibration['contract']:
            raise ValueError('Evaluation contract differs from validation calibration')
        modules = {name:rt.load(path,specification,target)[0] for name,path in checkpoint_paths.items()}
        ds,loader,indices = er.make_loader(hypes,options,weather=loader_weather)
        folder = output/weather
        folder.mkdir()
        summaries[weather] = condition(model,modules,ds,loader,indices,branch,target,folder,
                                      None if calibration is None else calibration['methods'])
        summaries[weather].update(root=hypes['validate_dir'],online_weather=args.phase != 'benchmark' and weather != 'clean')
        write_json(folder/'summary.json',summaries[weather])
        del modules,model,ds,loader
    write_json(output/'protocol.json',dict(conditions=summaries,contract=specification,phase=args.phase,
        raw_methods='attention/residual always reported even if validation chooses KEEP',
        selection='validation only; development uses the same split and is not independent'))
    if args.phase == 'calibration':
        write_json(calibration_path,dict(contract=specification,checkpoint_hashes=hashes,test_data_used=False,
            selection_data='full_validation_joint_outputs',gates=options['calibration'],
            methods={name:choose(summaries,name,options['calibration']) for name in checkpoint_paths}))
    verify_frozen()


if __name__ == '__main__':
    main()
