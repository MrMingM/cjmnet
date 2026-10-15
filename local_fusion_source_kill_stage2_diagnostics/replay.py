"""Remote single-frame diagnostic. Existing Stage 2 files remain untouched."""
import argparse
import json
from datetime import datetime,timezone
import os
from pathlib import Path
from types import FunctionType
import traceback

from local_fusion_source_kill_stage2.common import (atomic_json,atomic_text,digest,
    external_run,jsonlines,load_contract,read_json,resource_check,resume,safe_path)
from local_fusion_source_kill_stage2.preflight import baseline_rows
from local_fusion_source_kill_stage2.source_adapter import (inference_input_hash,
    make_runtime,original_args,run_with_hooks)
from .differences import exact_differences,render_summary


def select_frame(run,protocol,weather=None,mode=None,frame=None):
    targets=baseline_rows(Path(protocol['p0_run'])/'p0_oracle_target_records.jsonl')
    for w in ((weather,) if weather else ('fog','rain','snow')):
        for m in ((mode,) if mode else ('same','task')):
            available={r['frame'] for r in targets[m] if r['weather']==w}
            choices=[index for index in protocol['validation_indices'] if index in available]
            if frame is not None:
                choices=[index for index in choices if index==frame]
            for index in choices:
                if frame is None:
                    path=run/'k2'/'conditions'/w/m/f'{index:08d}.json'
                    if resume(path,protocol['identity']) is not None:
                        continue
                positive=[r for r in targets[m] if r['weather']==w and r['frame']==index]
                return w,m,index,positive
    raise ValueError('No pending primary K2 frame found; set --weather/--mode/--frame explicitly')


def saved_trajectory(protocol,weather,mode,index):
    one=Path(protocol['stage1_run'])
    payload=read_json(one/'K1_0_ORACLE_DECOMPOSITION.json')['conditions'][f'{weather}_{mode}_A_ONLY']
    path=safe_path(one/payload['frame_artifact'])
    if one not in path.parents or digest(path)!=payload['frame_sha256']:
        raise ValueError('Stage 1 frame file escaped run or original hash differs')
    frame=next((r for r in jsonlines(path) if r['sample_index']==index),None)
    if frame is None:
        raise ValueError('Selected frame missing from Stage 1 saved trajectory')
    return frame,path,payload['frame_sha256']


def trajectory(records,audits):
    return dict(records=records,current_state_audit=audits)


def run_diagnostic(run,weather=None,mode=None,frame=None,probe_interleaved=True):
    run=external_run(run)
    protocol=load_contract(run)
    resource_check(protocol)
    weather,mode,index,positives=select_frame(run,protocol,weather,mode,frame)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    folder=safe_path(run/'k2_diagnostics'/f'{stamp}_{os.getpid()}')
    folder.mkdir(parents=True,exist_ok=False)
    saved,saved_path,saved_hash=saved_trajectory(protocol,weather,mode,index)
    expected=trajectory(saved['records'],saved['current_state_audit'])
    atomic_json(folder/'stage1_expected.json',expected)
    report=dict(scope='single-frame troubleshooting; no scientific verdict',
        root_identity=protocol['identity'],original_stage2_sources_unchanged=True,
        weather=weather,mode=mode,frame=index,baseline_positive_targets=[r['target_index'] for r in positives],
        saved_stage1_file=str(saved_path),saved_stage1_file_sha256=saved_hash,
        diagnostic_source_hashes={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},
        environment={k:os.environ.get(k) for k in ('ROCR_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES')},
        comparisons={},conclusion='诊断尚未完成。')
    atomic_json(folder/'DIAGNOSIS.json',report)
    print('DIAGNOSTIC_OUTPUT='+str(folder),flush=True)
    try:
        runtime=make_runtime(protocol)
        b,spec,options,hypes,target,model,arm=runtime
        report['environment'].update(torch=b.torch.__version__,
            hip=b.torch.version.hip,device=str(target),device_name=b.torch.cuda.get_device_name(0))
        from local_fusion_action_utility_audit.common import assert_loader_paths
        from local_fusion_action_utility_audit.s0_counterfactual import prediction_hash
        from local_fusion_source_kill_stage1.oracle_adapter import run_restricted
        from local_fusion_source_kill_stage1.common import guard_tree
        for root in protocol['dataset_roots'].values():
            guard_tree(root)
        b.seed_all(spec['seed']+1)
        dataset,loader=b.selected_loader(hypes,options,'validation',weather,protocol['validation_indices'])
        assert_loader_paths(dataset)
        with b.torch.no_grad():
            found=False
            for ordinal,batch in enumerate(loader):
                resource_check(protocol)
                current=int(batch['ego']['communication_sample_index'][0])
                if current!=protocol['validation_indices'][ordinal]:
                    raise ValueError('Original validation loader order differs')
                if current!=index:
                    continue
                found=True
                batch=b.to_device(batch,target)
                report['model_input_sha256']=inference_input_hash(batch['ego'])
                # Match the failing K2 frame's forward setup exactly. The
                # original Stage 1 adapter is then tested on those SAME inputs.
                context=b.v3rt.context(model,batch['ego'],'weather',verify=False)
                report['context_verify']=False
                shared,_=arm.predict(model.engine.base,context['levels'])
                report['shared_prediction_sha256']=prediction_hash(shared)
                report['source_count']=len(context['levels'][0])
                if report['source_count']!=saved['source_count']:
                    raise ValueError('Valid CAV count differs from Stage 1')
                pool=b.build_candidate_pool(model.engine.base,context['levels'],shared,spec['candidate_families'])
                if sorted(pool)!=saved['candidate_pool']:
                    raise ValueError('Candidate pool names differ from Stage 1')
                proxy=b.build_proxy_pool(dataset,batch,pool)
                args=original_args(b,spec,dataset,batch,model,context['levels'],shared,pool,proxy,
                                  hypes['postprocess']['anchor_args']['cav_lidar_range'],mode)
                # Re-run the EXACT Stage 1 adapter first, without any K2
                # callbacks or perturbation evaluations between greedy steps.
                _,_,records,audits=run_restricted(b,'A_ONLY',*args)
                control=trajectory(records,audits)
                atomic_json(folder/'original_stage1_adapter.json',control)
                report['comparisons']['original_stage1_adapter_vs_saved']=exact_differences(control,expected)
                resource_check(protocol)
                _,_,records,audits=run_restricted(b,'A_ONLY',*args)
                repeated=trajectory(records,audits)
                atomic_json(folder/'original_stage1_adapter_repeated.json',repeated)
                report['comparisons']['original_stage1_repeat_same_inputs']=exact_differences(repeated,control)
                resource_check(protocol)
                _,_,records,audits=run_with_hooks(b,'A_ONLY',args)
                pure=trajectory(records,audits)
                atomic_json(folder/'stage2_without_perturbations.json',pure)
                report['comparisons']['stage2_without_perturbations_vs_saved']=exact_differences(pure,expected)
                report['comparisons']['stage2_vs_stage1_adapters_same_inputs']=exact_differences(pure,control)
                # Existing S0 cache records the exact Shared snapshot hash.
                # Read one frame on CPU only after checking its manifest hash.
                s0=Path(protocol['s0_run'])
                cache_name=f'cache/validation/{weather}/{index:08d}.pt'
                entry=read_json(s0/'manifest.json')['entries'][f'S0-frame/validation/{weather}/{index}']
                cache=safe_path(s0/cache_name)
                if s0 not in cache.parents or entry['status']!='complete' or digest(cache)!=entry['artifacts'][cache_name]:
                    raise ValueError('Original S0 frame cache incomplete or changed')
                s0_frame=b.torch.load(cache,map_location='cpu',weights_only=True)
                report['s0_shared_reference']=dict(source_file=str(cache),original_sha256=s0_frame['prediction_hash'],
                    replay_sha256=report['shared_prediction_sha256'],
                    exact_shared_hash_equal=s0_frame['prediction_hash']==report['shared_prediction_sha256'])
                # Only probe the existing K2 interleaving when both baseline
                # paths match. This avoids repeating eight perturbations when
                # the original Stage 1 adapter itself already fails replay.
                if probe_interleaved and all(report['comparisons'][key]['exact_equal'] for key in (
                    'original_stage1_adapter_vs_saved','original_stage1_repeat_same_inputs','stage2_without_perturbations_vs_saved')):
                    resource_check(protocol)
                    from local_fusion_source_kill_stage2 import k2_label_stability as k2
                    def traced(*a,**kw):
                        output=run_with_hooks(*a,**kw)
                        actual=trajectory(output[2],output[3])
                        atomic_json(folder/'stage2_with_perturbations.json',actual)
                        report['comparisons']['stage2_with_perturbations_vs_saved']=exact_differences(actual,expected)
                        return output
                    original=k2.evaluate_frame
                    instrumented=FunctionType(original.__code__,{**original.__globals__,'run_with_hooks':traced},
                                              original.__name__,original.__defaults__,original.__closure__)
                    try:
                        instrumented(protocol,runtime,dataset,batch,weather,index,mode,positives)
                    except Exception:
                        report['interleaved_error']=traceback.format_exc()
                break
            if not found:
                raise ValueError('Selected primary frame not reached')
        comparisons=report['comparisons']
        if not comparisons['original_stage1_adapter_vs_saved']['exact_equal']:
            report['conclusion']='原 Stage 1 adapter 在当前环境下也无法逐字段复现旧记录；差异发生在 K2 扰动之前。'
        elif not comparisons['original_stage1_repeat_same_inputs']['exact_equal']:
            report['conclusion']='同一输入重复执行原 Stage 1 adapter 也出现差异，需要核对 postprocess 重复执行的一致性。'
        elif not comparisons['stage2_without_perturbations_vs_saved']['exact_equal']:
            report['conclusion']='原 Stage 1 adapter 能复现；Stage 2 基础 hook 重放路径出现差异，需要核查 adapter 与重复 evaluation。'
        elif report.get('interleaved_error') or not comparisons.get('stage2_with_perturbations_vs_saved',{'exact_equal':True})['exact_equal']:
            report['conclusion']='无扰动重放能复现；加入原 K2 回调后失败，需要检查回调及额外 postprocess 的影响。'
        else:
            report['conclusion']='该帧的诊断路径未重现错误；需对原失败 frame/GPU/环境继续核对，未放宽 gate。'
        load_contract(run)
        report['status']='diagnostic_complete'
    except BaseException:
        report['status']='diagnostic_error'
        report['error']=traceback.format_exc()
        report['conclusion']='诊断入口未完成；具体错误已保存，原 K1/K2 文件未修改。'
        raise
    finally:
        atomic_json(folder/'DIAGNOSIS.json',report)
        atomic_text(folder/'DIAGNOSIS.md',render_summary(report))
        print('REPORT='+str(folder/'DIAGNOSIS.md'),flush=True)
        for label,comparison in report['comparisons'].items():
            print(label+': exact_equal='+str(comparison['exact_equal'])+
                  '; differences='+str(comparison['total_differences']),flush=True)
            if comparison['first_differences']:
                print('FIRST_DIFF '+json.dumps(comparison['first_differences'][0],ensure_ascii=False,default=str)[:1500],flush=True)
    print(report['conclusion'],flush=True)
    return report


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--run',required=True)
    cli.add_argument('--weather',choices=('fog','rain','snow'))
    cli.add_argument('--mode',choices=('same','task'))
    cli.add_argument('--frame',type=int)
    cli.add_argument('--baseline-only',action='store_true')
    args=cli.parse_args()
    run_diagnostic(args.run,args.weather,args.mode,args.frame,not args.baseline_only)


if __name__=='__main__':
    main()
