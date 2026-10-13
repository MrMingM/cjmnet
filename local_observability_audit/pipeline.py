"""Independent Linux-only pipeline; identities checked before any cache reuse."""
import argparse
import copy
import os
from pathlib import Path
import sys
import traceback
import numpy as np
from .common import (PACKAGE, Manifest, atomic_json, read_json, config, digest,
    object_hash, install_paths, run_path, run_lock, safe_path, verify_snapshot,verify_protocol)

def frontend(cfg):
    directory=safe_path(cfg['frontend_dir'])
    config_path=safe_path(directory/'config.yaml'); checkpoint=safe_path(directory/cfg['checkpoint_name'])
    if digest(checkpoint)!=cfg['checkpoint_sha256']:
        raise ValueError('Frozen checkpoint SHA256 mismatch')
    from opencood.hypes_yaml.yaml_utils import load_yaml
    h=load_yaml(str(config_path))
    if h['model']['core_method']!='point_pillar_gspr_attfuse':
        raise ValueError('Unsupported detector; no wrapper/B0 replacement allowed')
    return h,checkpoint,{'config_path':str(config_path),'config_sha256':digest(config_path),
        'checkpoint_path':str(checkpoint),'checkpoint_sha256':digest(checkpoint)}

def new_protocol(cfg,mode,gpu):
    from .data import selection_and_manifest
    sources=verify_snapshot()
    h,checkpoint,front=frontend(cfg)
    print('Hashing selected validation PCD/YAML files (read only)...',flush=True)
    selected,data=selection_and_manifest(cfg,int(h['train_params']['max_cav']),mode)
    identity=object_hash({'config':cfg,'mode':mode,'sources':sources,'frontend':front,'dataset':data,
                          'python':sys.version})
    protocol={'schema':1,'identity':identity,'config':cfg,'mode':mode,'gpu':gpu,'sources':sources,
              'frontend':front,'dataset':data,'selection':selected,
              'source_hash':object_hash(sources),'config_hash':object_hash(cfg),'dataset_hash':object_hash(data),
              'preregistered_decision':cfg['decision'],
              'normalization':'theta = theta_ref + diag(parameter_scales) eta; I_norm = S^T I S',
              'cohort':'GT ROI + fixed GT face assignment; utility from independent true source-subset forward',
              'reference_gspr':'Full selected-frame forward; not assumed subset-independent',
              'matching':'Frozen native postprocess then score-ordered 1:1 BEV IoU >= configured threshold',
              'sampling':'paired weather, hash-ordered scene round-robin, no S4 split',
              'uncertainty':'scene-cluster bootstrap; fixed-alpha leave-one-scene-out ridge, train-fold normalization',
              'structural_limit':'single face rank <=2; two adjacent faces rank <=3 for five free BEV parameters',
              'independence':'No imports from other experimental modules; no S4 artifacts accessed',
              'python':sys.version}
    return protocol,h,checkpoint

def artifact_key(record,weather):
    return weather+'/'+object_hash([record['scene_id'],record['frame_id']])[:24]+'.json'

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default=str(PACKAGE/'experiment.yaml'))
    parser.add_argument('--run',required=True)
    parser.add_argument('--gpu',type=int,required=True)
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--smoke',action='store_true')
    group.add_argument('--full-validation',action='store_true')
    parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args()
    cfg=config(args.config)
    from .device_guard import check
    guard=check(args.gpu,cfg)
    print('GPU preflight passed:',args.gpu,flush=True)
    if args.preflight_only:
        print(guard); return
    install_paths()
    mode='smoke' if args.smoke else 'full-validation' if args.full_validation else 'O0'
    run=run_path(args.run)
    if run.exists() and not (run/'protocol.json').exists():
        # launch.sh may have opened only driver.log / pid before the pipeline.
        if set(p.name for p in run.iterdir())-{'driver.log','pid','.lock'}:
            raise ValueError('Existing non-audit output directory; refusing overwrite')
    run.mkdir(parents=True,exist_ok=True)
    with run_lock(run):
        try:
            protocol,hypes,checkpoint=new_protocol(cfg,mode,args.gpu)
            pp=run/'protocol.json'
            if pp.exists():
                old=read_json(pp)
                verify_protocol(old)
                if old['identity']!=protocol['identity']:
                    raise ValueError('Resume source/config/checkpoint/dataset identity differs')
                protocol=old
            else:
                atomic_json(pp,protocol)
            manifest=Manifest(run,protocol['identity'])
            # Hashing may take time; recheck immediately before any CUDA context.
            guard=check(args.gpu,cfg)
            atomic_json(run/'diagnostics'/'gpu_preflight.json',guard)
            manifest.stage('server-smoke-and-extraction')
            # Reject hardware/library drift on resume as well as source/data drift.
            import torch
            import scipy
            import shapely
            environment={'torch':torch.__version__,'numpy':np.__version__,'scipy':scipy.__version__,
                'shapely':shapely.__version__,'hip':torch.version.hip,'device':torch.cuda.get_device_name(0)}
            ep=run/'environment.json'
            if ep.exists() and read_json(ep)!=environment:
                raise ValueError('Runtime environment changed')
            atomic_json(ep,environment)
            manifest.value['environment_hash']=digest(ep)
            torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed'])
            torch.backends.cudnn.benchmark=False
            torch.backends.cudnn.deterministic=True
            torch.use_deterministic_algorithms(True)
            from attfuse_gspr.point_pillar_gspr_attfuse import PointPillarGsprAttfuse
            model=PointPillarGsprAttfuse(hypes['model']['args'])
            state=torch.load(checkpoint,map_location='cpu',weights_only=True)
            model.load_state_dict(state.get('state_dict',state),strict=True)
            model.eval().to('cuda:0')
            for p in model.parameters():
                p.requires_grad_(False)
            from .data import build_dataset,get_frame
            from .subset_forward import all_predictions
            from .extract import source_rows,contributions
            from opencood.tools.train_utils import to_device
            frames=[]
            for weather in cfg['weathers']:
                ds=build_dataset(hypes,cfg['data'][weather])
                for pos,record in enumerate(protocol['selection']):
                    key=artifact_key(record,weather)
                    existing=manifest.get('source_observability/'+key)
                    predictions=manifest.get('subset_predictions/'+key)
                    diagnostic=manifest.get('diagnostics/frames/'+key)
                    if existing is not None and predictions is not None and diagnostic is not None:
                        frames.append(existing); continue
                    print(f"EXTRACT {weather} {pos+1}/{len(protocol['selection'])} {record['scene_id']}/{record['frame_id']}",flush=True)
                    batch,metadata=get_frame(ds,record,cfg['seed'])
                    # GPU contains detector inputs and evaluation metadata; only processed_lidar/record_len reach model.
                    batch=to_device(batch,torch.device('cuda:0'))
                    with torch.inference_mode():
                        preds,rel,diag=all_predictions(model,batch,ds.post_processor,cfg,
                            verify=pos<cfg['verify_subset_frames_per_weather'] or mode=='smoke')
                    sources,boxes,ids=source_rows(batch,metadata,rel,hypes,record,weather,cfg)
                    contrib=contributions(preds,boxes,ids,len(metadata),cfg)
                    value={'record':record,'weather':weather,'sources':sources,'contributions':contrib}
                    manifest.put('subset_predictions/'+key,preds)
                    manifest.put('diagnostics/frames/'+key,diag)
                    manifest.put('source_observability/'+key,value)
                    frames.append(value)
                    del batch,rel,preds
                del ds
            # End GPU work before potentially long descriptive CPU statistics.
            del model
            torch.cuda.empty_cache()
            manifest.stage('analysis')
            from .evaluate import evaluate
            result=evaluate(frames,cfg,mode)
            result['subset_equivalence']={}
            for key in manifest.value['artifacts']:
                if key.startswith('diagnostics/frames/'):
                    d=manifest.get(key)
                    if d['cache_equivalence'] is not None:
                        result['subset_equivalence'][key]=d['cache_equivalence']
            # Check sources/checkpoint/data again before declaring complete.
            from .data import verify_files
            if object_hash(verify_snapshot())!=protocol['source_hash'] or digest(checkpoint)!=protocol['frontend']['checkpoint_sha256']:
                raise ValueError('Source/checkpoint changed during run')
            if digest(protocol['frontend']['config_path'])!=protocol['frontend']['config_sha256']:
                raise ValueError('Frontend config changed during run')
            verify_files(protocol['dataset'])
            from .report import write_reports
            manifest.value['final_hashes']=write_reports(run,result,protocol)
            manifest.value['decision']=result['decision']
            manifest.stage('complete')
            print('FINAL',result['decision'],flush=True)
        except Exception as exc:
            atomic_json(run/'diagnostics'/'failure.json',{'type':type(exc).__name__,'message':str(exc),
                'weather':locals().get('weather'),'record':locals().get('record'),
                'traceback':traceback.format_exc(),'policy':'No frame removed or silently substituted. Resume only identical contract.'})
            if 'manifest' in locals():
                manifest.stage('failed')
            raise

if __name__=='__main__':
    main()
