"""Read-only reconstruction of the real A1 full checkpoint on the server."""
from pathlib import Path
import torch
from auditlib import atomic_json, complete, read_json, sha256


def equal(a,b):
    import numpy as np
    if isinstance(a,torch.Tensor):
        if not torch.equal(a.detach().cpu(),b.detach().cpu()):raise AssertionError('Tensor mismatch')
    elif isinstance(a,np.ndarray):
        if not np.array_equal(a,b):raise AssertionError('Numpy state mismatch')
    elif isinstance(a,dict):
        if a.keys()!=b.keys():raise AssertionError('State keys mismatch')
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        if len(a)!=len(b):raise AssertionError('State length mismatch')
        for x,y in zip(a,b):equal(x,y)
    elif a!=b:raise AssertionError('State value mismatch')


def run(args):
    from lineage import plan
    from checkpoint import committed, restore, rng
    from models import create, criterion
    from opencood.hypes_yaml import yaml_utils
    from opencood.tools import train_utils
    root=Path(args.experiment_root)
    p=plan(root);folder=Path(p['old_root'])/'A1'
    contract=read_json(folder/'contract.json')
    rows=committed(folder,20,contract['identity'])
    stop=read_json(Path(p['boundary_record']))
    engineering=getattr(args,'engineering_active_A1',False)
    if not engineering:
        assert stop['status']=='stopped_after_complete_epoch' and len(rows)>=stop['completed_epoch']
    path=Path(rows[-1]['resume_state']);digest=sha256(path)
    saved=torch.load(path,map_location='cpu',weights_only=False)
    h=yaml_utils.load_yaml(str(folder/'config.yaml'))
    model=create(h,'A1').cuda();loss=criterion(h,'A1').cuda()
    optimizer=train_utils.setup_optimizer(h,model)
    scheduler=train_utils.setup_lr_schedular(h,optimizer,n_iter_per_epoch=1010)
    restore(path,model,optimizer,scheduler,loss,contract['identity'],len(rows))
    equal(saved['model'],model.state_dict());equal(saved['optimizer'],optimizer.state_dict())
    equal(saved['scheduler'],scheduler.state_dict());equal(saved['criterion'],loss.state_dict())
    equal(saved['accumulators'],{k:getattr(loss,k) for k in saved['accumulators']})
    equal(saved['rng'],rng())
    assert sha256(path)==digest
    record=dict(status='complete',checkpoint=str(path),checkpoint_sha256=digest,
        completed_epoch=len(rows),next_epoch=len(rows)+1,model_exact=True,optimizer_exact=True,
        engineering_read_only_while_original_trainer_runs=engineering,
        scheduler_exact=True,criterion_and_accumulators_exact=True,all_rng_exact=True,
        optimization_steps_on_preserved_A1=0,original_contract_sha256=sha256(folder/'contract.json'))
    atomic_json(Path(args.output_root)/'A1_STATE_RESTORE_TEST.json',record)
    del model,loss,optimizer,scheduler,saved
    torch.cuda.empty_cache()
    return record
