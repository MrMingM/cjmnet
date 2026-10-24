"""CUDA inference wrapper; native official inference, postprocess, AP and matcher unchanged."""
import copy
from pathlib import Path
import json
from auditlib import atomic_json,bound_json,complete,identity,read_json,sha256


def dataset_keys(dataset):
    keys=[]
    for scene in dataset.scenario_database.values():
        ego=next(records for records in scene.values() if records['ego'])
        for stamp,record in ego.items():
            if stamp!='ego':
                keys.append([Path(record['yaml']).parent.parent.name,stamp])
    return keys


def evaluate(args,checkpoint,destination,clean_only=False,smoke=False):
    from cuda_runtime import prepare,input_root
    args.command='evaluate'
    _,evidence=prepare(args)
    import torch
    from torch.utils.data import DataLoader,Subset
    from opencood.hypes_yaml import yaml_utils
    from opencood.data_utils.datasets import build_dataset
    from opencood.tools import train_utils,inference_utils
    from opencood.utils import eval_utils_opv2v
    from support import patched,compare_gt,corner_array,atomic_jsonl
    from predict import live_gt_observer
    # Load reviewed official match trace without importing conflicting generic evaluate module.
    from auditlib import ROOT,load_frozen,sources
    trace=load_frozen('cuda_official_match','evaluate.py',ROOT/'v2xr_protocol_audit').official_match_trace
    from checkpoint import seed
    proof=complete(input_root()/'summary.json')
    keys=proof['frame_keys'][:12 if smoke else 829]
    conditions=['clean'] if clean_only else ['clean','fog','snow']
    root=Path(destination);root.mkdir(parents=True,exist_ok=True)
    contract=dict(status='registered',backend='cuda',checkpoint_sha256=sha256(checkpoint),
        variant=args.variant,conditions=conditions,frame_keys=keys,sources=sources(),
        input_evidence_sha256=sha256(input_root()/'summary.json'),
        environment=evidence['a'],inference_seed=20261016,
        official_config_sha256=sha256(Path(args.runtime_root)/'opencood/hypes_yaml/V2X-R/LiDAR/V2XR_attfuse.yaml'),
        GT_protocols='official O skips None; Ostar counts None; F fixes fresh live Clean GT',global_sort=False)
    contract['identity']=identity(contract)
    bound_json(root/'protocol.json',contract)
    if (root/'summary.json').exists():
        done=complete(root/'summary.json')
        if done['contract_identity']!=contract['identity']:
            raise ValueError('CUDA evaluation source/input/checkpoint changed')
        for row in read_json(root/'prediction_manifest.json')['files']:
            if sha256(row['path'])!=row['sha256']:
                raise ValueError('Completed CUDA prediction cache changed')
        return done
    frames=[]; predictions=[]; results={}
    source=Path(args.runtime_root)/'opencood/hypes_yaml/V2X-R/LiDAR/V2XR_attfuse.yaml'
    for weather in conditions:
        seed(20261016)
        h=yaml_utils.load_yaml(str(source))
        h.update(root_dir=str(Path(args.train_root)/'train'),validate_dir=str(input_root()/'validate'),
            train_sim=False,eval_sim=weather!='clean',sim_weather={'clean':'','fog':'_fog_0.060','snow':'_snow_2.5_2.0'}[weather])
        dataset=build_dataset(h,visualize=True,train=False)
        if dataset_keys(dataset)!=proof['frame_keys'] or dataset.max_cav!=7 or not dataset.proj_first:
            raise ValueError('Official CUDA validation context changed')
        loader=DataLoader(Subset(dataset,range(len(keys))),batch_size=1,num_workers=16,
            collate_fn=dataset.collate_batch_test,shuffle=False,pin_memory=False,drop_last=False)
        model=train_utils.create_model(h).cuda().eval()
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model.load_state_dict(state,strict=True)
        del state
        stats={p:{t:dict(tp=[],fp=[],gt=0,score=[]) for t in (.3,.5,.7)} for p in ('O','O*','F')}
        counted=dict(official_gt=0,fixed_gt=0,none_frames=0,prediction_count=0)
        cache=root/'predictions'/weather;cache.mkdir(parents=True,exist_ok=True)
        marker=cache/'manifest.json'
        journal=read_json(marker) if marker.exists() else dict(contract_identity=contract['identity'],entries={})
        if journal['contract_identity']!=contract['identity']:
            raise ValueError('Partial CUDA prediction cache has incompatible contract')
        if set(journal['entries'])-{str(i) for i in range(len(keys))}:
            raise ValueError('Unexpected prediction cache frame index')
        for index,batch in enumerate(loader):
            expected=evidence['frames'][index]['conditions'][weather]['final_gt']
            fixed=evidence['frames'][index]['conditions']['clean']['final_gt']
            path=cache/(f'{index:06d}.json')
            batch=train_utils.to_device(batch,torch.device('cuda:0'))
            holder={}
            original=dataset.post_processor.generate_gt_bbx
            def gt_observer(data):
                tensor,rows=live_gt_observer(dataset.post_processor,data,original)
                holder['rows']=rows
                return tensor
            with torch.no_grad(),patched(dataset.post_processor,'generate_gt_bbx',gt_observer):
                if path.exists():
                    row=read_json(path)
                    if (row['contract_identity']!=contract['identity'] or row['frame_key']!=keys[index] or
                        row['condition']!=weather or row['record_identity']!=identity({k:v for k,v in row.items() if k!='record_identity'})):
                        raise ValueError('Invalid interrupted prediction cache: '+str(path))
                    if str(index) in journal['entries'] and journal['entries'][str(index)]!=sha256(path):
                        raise ValueError('Committed CUDA prediction bytes changed')
                    dataset.post_processor.generate_gt_bbx(batch)
                    compare_gt(holder['rows'],row['official_gt'],'CUDA cache/live GT')
                    boxes=None if row['prediction_is_none'] else torch.tensor(row['prediction_boxes'],dtype=torch.float32).reshape(-1,8,3)
                    scores=None if row['scores'] is None else torch.tensor(row['scores'],dtype=torch.float32)
                else:
                    if str(index) in journal['entries']:
                        raise ValueError('Committed CUDA prediction missing')
                    boxes,scores,gt=inference_utils.inference_intermediate_fusion(batch,model,dataset)
                    for tensor in (boxes,scores,gt):
                        if tensor is not None and not torch.isfinite(tensor).all():
                            raise ValueError('Nonfinite official CUDA prediction/GT')
                    boxes=boxes.cpu() if boxes is not None else None
                    scores=scores.cpu() if scores is not None else None
                    row=dict(contract_identity=contract['identity'],condition=weather,index=index,frame_key=keys[index],
                        prediction_is_none=boxes is None,prediction_count=0 if boxes is None else len(boxes),
                        prediction_boxes=None if boxes is None else boxes.tolist(),scores=None if scores is None else scores.tolist(),
                        official_gt=holder['rows'])
                    row['record_identity']=identity(row);atomic_json(path,row)
            compare_gt(holder['rows'],expected,'CUDA live GT vs fresh preflight')
            journal['entries'][str(index)]=sha256(path);atomic_json(marker,journal)
            predictions.append(dict(path=str(path),sha256=sha256(path),condition=weather,index=index))
            counted['official_gt']+=len(expected);counted['fixed_gt']+=len(fixed)
            counted['none_frames']+=row['prediction_is_none'];counted['prediction_count']+=row['prediction_count']
            frame=dict(index=index,frame_key=keys[index],condition=weather,protocols={})
            for protocol in stats:
                skipped=protocol=='O' and boxes is None
                gt_rows=fixed if protocol=='F' else expected
                tensor=torch.from_numpy(corner_array(gt_rows))
                record=dict(skipped=skipped,thresholds={},matched_pairs={})
                for threshold in (.3,.5,.7):
                    before=len(stats[protocol][threshold]['tp'])
                    if not skipped:
                        if protocol=='F':
                            pairs,_=trace(boxes,scores,tensor,[x['object_id'] for x in fixed],stats[protocol],threshold)
                            record['matched_pairs'][str(threshold)]=pairs
                        else:
                            eval_utils_opv2v.caluclate_tp_fp(boxes,scores,tensor,stats[protocol],threshold)
                    tp=sum(stats[protocol][threshold]['tp'][before:]);fp=sum(stats[protocol][threshold]['fp'][before:])
                    record['thresholds'][str(threshold)]=dict(tp=tp,fp=fp,fn=0 if skipped else len(gt_rows)-tp)
                frame['protocols'][protocol]=record
            frames.append(frame)
            if index%25==0:
                print('CUDA official inference',args.variant,weather,index,'/',len(keys),flush=True)
        result={p:{f'ap{round(t*100)}':float(eval_utils_opv2v.calculate_ap(copy.deepcopy(stat),t,False)[0])
            for t in (.3,.5,.7)} for p,stat in stats.items()}
        for protocol,stat in stats.items():
            atomic_json(root/('stat_'+weather+'_'+protocol.replace('*','star')+'.json'),stat)
        if weather=='clean' and result['O*']!=result['F']:
            raise ValueError('Clean F/Ostar mismatch')
        results[weather]=dict(AP=result,counts=counted,counted_gt={p:s[.7]['gt'] for p,s in stats.items()})
        del model,loader,dataset
        torch.cuda.empty_cache()
    atomic_jsonl(root/'per_frame_metrics.jsonl',frames)
    atomic_json(root/'prediction_manifest.json',dict(status='complete',files=predictions))
    from models import OBSERVATIONS
    atomic_json(root/'repe_inference.json',dict(status='complete',observations=OBSERVATIONS,
        scope='synchronized actual uncached detector forwards in this invocation; excludes loading/postprocessing'))
    summary=dict(status='complete',backend='cuda',stage='GSPR_REPE_CUDA',main_protocol='O',
        contract_identity=contract['identity'],model_variant=args.variant,checkpoint_sha256=sha256(checkpoint),
        conditions=conditions,frame_count=len(keys),full_validation=len(keys)==829,
        frozen_d1c_gt_identity=proof['gt_identity'],gt_identity=proof['gt_identity'],
        gt_identity_scope='fresh CUDA official GT audit, historical field retained for report compatibility',results=results,
        output_hashes={f.name:sha256(f) for f in root.iterdir() if f.is_file() and f.name!='summary.json' and not f.name.endswith('.tmp')})
    from results import write_evaluation
    write_evaluation(root,summary)
    summary['output_hashes']['RESULTS.md']=sha256(root/'RESULTS.md')
    atomic_json(root/'summary.json',summary)
    return summary
