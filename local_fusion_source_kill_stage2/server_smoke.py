"""Two real Source frames on SERVER only; no null, scientific AP or verdict."""
import argparse
from .common import (atomic_json, commit_unit, external_run,
                     resource_check, resume)
from .source_adapter import (assert_shortlist_contract, make_runtime, original_args,
                            run_with_hooks)


def execute(folder, protocol, runtime):
    path=folder/'smoke.json'
    if resume(path,protocol['identity']) is not None:
        return
    resource_check(protocol)
    b,spec,options,hypes,target,model,arm=runtime
    from local_fusion_action_utility_audit.common import assert_loader_paths
    b.seed_all(spec['seed']+1)
    indices=protocol['validation_indices'][:2]
    dataset,loader=b.selected_loader(hypes,options,'validation','fog',indices)
    assert_loader_paths(dataset)
    rows=[]
    with b.torch.no_grad():
        for ordinal,batch in enumerate(loader):
            resource_check(protocol)
            index=int(batch['ego']['communication_sample_index'][0])
            if index != indices[ordinal]:
                raise ValueError('Smoke frame order differs')
            batch=b.to_device(batch,target)
            context=b.v3rt.context(model,batch['ego'],'weather',verify=ordinal==0)
            shared,_=arm.predict(model.engine.base,context['levels'])
            pool=b.build_candidate_pool(model.engine.base,context['levels'],shared,spec['candidate_families'])
            proxy=b.build_proxy_pool(dataset,batch,pool)
            for mode in ('same','task'):
                args=original_args(b,spec,dataset,batch,model,context['levels'],shared,pool,proxy,
                    hypes['postprocess']['anchor_args']['cav_lidar_range'],mode)
                _,post,records,audits=run_with_hooks(b,'FULL',args)
                assert_shortlist_contract(records,audits,len(pool),spec['roi_expansions'],mode,spec['shortlist_k'])
                rows.append(dict(frame=index,mode=mode,targets=len(records),detections=0 if post[0] is None else len(post[0]),
                    changed=sum(r['changed'] for r in records),nonkeep_candidates=len(pool)-1))
    if len(rows)!=2*len(indices):
        raise ValueError('Server smoke incomplete')
    payload=dict(status='complete',scope='smoke only',scientific_result=False,AP_computed=False,
        null_run=False,source_reproduction_claim=False,rows=rows,
        chain=['load','source candidate generation','GT proxy','original shortlist','actual postprocess/NMS',
               'greedy','atomic save','resume','smoke report'])
    commit_unit(path,protocol['identity'],payload)
    if resume(path,protocol['identity']) != payload:
        raise ValueError('Server smoke resume failed')
    atomic_json(folder/'SMOKE_REPORT.json',payload)
    print('SERVER SOURCE SMOKE COMPLETE; no scientific AP, no null run',flush=True)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--run',required=True)
    args=cli.parse_args()
    run=external_run(args.run)
    from .preflight import server_preflight
    protocol=server_preflight(run)
    execute(run/'smoke',protocol,make_runtime(protocol))


if __name__=='__main__':
    main()
