"""Recover only evaluation and diagnosis from completed training checkpoints."""
from pathlib import Path
import sys
import traceback

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from v2xr_fusion_interaction.common.audit import (OUT,DEVICES,atomic_json,device,idle,identity,
                                                lr,read_json,settings,sha256,source_files)
from common import lock


def recover(group):
    import torch
    from opencood.hypes_yaml import yaml_utils
    from checkpoint import seed
    from v2xr_fusion_interaction.common.model import build_model
    from v2xr_fusion_interaction.common.training import frozen_hash
    from v2xr_fusion_interaction.common.evaluation import baseline,final_evaluation

    root=OUT/group
    contract=read_json(OUT/'CONTRACT.json')
    run_id=identity(contract)
    old=contract['source']
    current=source_files()
    allowed={'common/evaluation.py','recover_evaluation.py','compare_paired.py'}
    if {k for k in old.keys()|current.keys() if old.get(k)!=current.get(k)}-allowed:
        raise ValueError('Unexpected training or evaluation source change')
    for name,digest in contract['context_sources'].items():
        if sha256(HERE.parent/'v2xr_fusion_pot'/name)!=digest:
            raise ValueError(f'Historical helper changed: {name}')
    if sha256(lr.WEIGHT)!=contract['p0_sha256']:
        raise ValueError('P0 weight changed')
    rows=[]
    for epoch in range(1,21):
        row=read_json(root/'epochs'/f'{epoch:02d}'/'complete.json')
        if row['contract']!=run_id or row['epoch']!=epoch or row['train_updates']!=1010:
            raise ValueError(f'Uncommitted epoch {epoch}')
        rows.append(row)
    best=min(rows,key=lambda row:(-row['clean_AP']['O']['ap70'],row['epoch']))
    selected=read_json(root/'selected.json')
    if (selected['epoch']!=best['epoch'] or selected['checkpoint']!=best['checkpoint']
            or selected['sha256']!=best['checkpoint_sha256'] or sha256(selected['checkpoint'])!=selected['sha256']):
        raise ValueError('Selected checkpoint does not match the committed Clean winner')
    idle([group])
    mapped=device(group)
    atomic_json(root/'RECOVERY_PROVENANCE.json',dict(contract=run_id,selected=selected,
        diagnostic_source_sha256=current['common/evaluation.py'],
        source_repair='Read target ID mapping returned by object_support directly'))
    seed(20261016)
    model=build_model(yaml_utils.load_yaml(str(lr.CONFIG)),settings(group))
    model.load_state_dict(torch.load(selected['checkpoint'],map_location='cpu',weights_only=True),strict=True)
    if frozen_hash(model)!=read_json(root/'SMOKE.json')['frozen_sha256']:
        raise ValueError('Frozen model changed')
    print('RECOVER FROM',group,selected,mapped,flush=True)
    final_evaluation(root,model,group,run_id,mapped,selected,
                     max(row['peak_train_gpu_allocated_bytes'] for row in rows))
    if group=='F2':
        del model
        torch.cuda.empty_cache()
        baseline(run_id,mapped)


if __name__=='__main__':
    group=sys.argv[1]
    if group not in DEVICES:
        raise ValueError(group)
    root=OUT/group
    with lock(root/'.lock'):
        try:
            atomic_json(root/'STATUS.json',dict(status='running',stage='recover_evaluation'))
            recover(group)
            atomic_json(root/'STATUS.json',dict(status='complete',stage='recover_evaluation'))
            print('RECOVERY COMPLETE',group,flush=True)
        except BaseException:
            failure=dict(status='failed',stage='recover_evaluation',traceback=traceback.format_exc())
            atomic_json(root/'RECOVERY_FAILURE.json',failure)
            atomic_json(root/'STATUS.json',failure)
            raise
