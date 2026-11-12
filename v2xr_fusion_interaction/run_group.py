"""Assigned-device worker used by the one POSIX launcher."""
from pathlib import Path
import sys
import traceback

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from v2xr_fusion_interaction.common.audit import (DEVICES,OUT,atomic_json,audit,device,idle,identity,
                                                lr,read_json,settings,sha256,source_files)
from common import lock


def work(group,stage):
    from checkpoint import seed
    from v2xr_fusion_interaction.tests.checks import unit
    from v2xr_fusion_interaction.common.training import formal,smoke
    from v2xr_fusion_interaction.common.evaluation import baseline,final_evaluation

    root=OUT/group
    cfg=settings(group)
    contract=read_json(OUT/'CONTRACT.json')
    if contract['source']!=source_files() or contract['configuration'][group]!=cfg:
        raise ValueError('Source/config changed after audit')
    if sha256(lr.WEIGHT)!=contract['p0_sha256']:
        raise ValueError('Frozen weight changed after audit')
    run_id=identity(contract)
    idle([group])
    mapped=device(group)
    lr.bind(root/'PREFLIGHT.json',dict(status='complete',contract=run_id,device=mapped,
                                      p0_sha256=contract['p0_sha256'],settings=cfg))
    print('DEVICE VERIFIED',group,mapped,flush=True)
    if stage=='smoke':
        seed(20261017)
        checks=unit(cfg)
        lr.bind(root/'UNIT.json',dict(contract=run_id,checks=checks))
        print('UNIT PASSED',group,checks,flush=True)
        report=smoke(root,run_id,cfg)
        print('SMOKE PASSED',group,report['trainable_parameters'],flush=True)
        return
    gate=read_json(OUT/'SMOKE_GATE.json')
    if gate['contract']!=run_id or any(sha256(OUT/g/'SMOKE.json')!=h for g,h in gate['smoke_sha256'].items()):
        raise ValueError('All-group smoke gate changed')
    test=read_json(root/'SMOKE.json')
    model,selected,peak=formal(root,run_id,cfg,test['frozen_sha256'])
    final_evaluation(root,model,group,run_id,mapped,selected,peak)
    if group=='F2':
        del model
        baseline(run_id,mapped)


def gate():
    tests={group:read_json(OUT/group/'SMOKE.json') for group in DEVICES}
    contract=identity(read_json(OUT/'CONTRACT.json'))
    def matching(field):
        return len({r[field] for r in tests.values()})==1
    if (any(r['status']!='complete' or r['contract']!=contract for r in tests.values())
            or not all(matching(field) for field in ('initial_interaction_sha256','frozen_sha256','trainable_parameters'))
            or len({tuple(step['input_sha256'] for step in r['updates']) for r in tests.values()})!=1):
        raise ValueError('Smoke data order or shared initialization differs across groups')
    record=dict(status='complete',contract=contract,matched_smoke_input_order=True,
                smoke_sha256={group:sha256(OUT/group/'SMOKE.json') for group in DEVICES})
    lr.bind(OUT/'SMOKE_GATE.json',record)
    print('THREE SMOKES PASSED; SHARED INITIALIZATION AND INPUT ORDER VERIFIED',flush=True)


if __name__=='__main__':
    if sys.argv[1:]==['audit']:
        audit()
    elif sys.argv[1:]==['gate']:
        gate()
    else:
        group,stage=sys.argv[1:]
        if group not in DEVICES or stage not in ('smoke','full'):
            raise ValueError('Unexpected worker mode')
        root=OUT/group
        with lock(root/'.lock'):
            try:
                atomic_json(root/'STATUS.json',dict(status='running',stage=stage))
                work(group,stage)
                atomic_json(root/'STATUS.json',dict(status='complete',stage=stage))
            except BaseException:
                atomic_json(root/'FAILURE.json',dict(status='failed',stage=stage,traceback=traceback.format_exc()))
                atomic_json(root/'STATUS.json',dict(status='failed',stage=stage))
                raise
