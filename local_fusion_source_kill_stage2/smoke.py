"""Pure synthetic end-to-end check; no scientific AP or real frame loading."""
from pathlib import Path
import tempfile
import numpy as np
from .common import ROOT, KEEP, commit_unit, resume, object_hash, atomic_json
from .fixtures import Dataset, Tensor, oracle_fixture
from .source_adapter import run_with_hooks, evaluate_target
from .stability import compare_labels, label_summary
from .verdicts import final_verdict


def smoke(folder):
    b, dataset = oracle_fixture(), Dataset()
    shared = dict(psm=Tensor(.1), rm=Tensor(.1))
    pool = {KEEP:shared, 'single:0':dict(psm=Tensor(.8),rm=Tensor(.8)),
            'query:0':dict(psm=Tensor(.7),rm=Tensor(.7))}
    proxy = {name:b._proxy_cache(dataset,{},p) for name,p in pool.items() if name != KEEP}
    observed = []
    args = (dataset,{'ego':{'anchor_box':None}},None,[np.zeros((1,1,1,8))],shared,pool,proxy,None,(1.,1.5),'task',.7,6)
    _, post, records, audits = run_with_hooks(b,'FULL',args,
        lambda target,current,rows,info: observed.append((target,len(rows))))
    if len(records) != 2 or not observed or dataset.calls < 2:
        raise AssertionError('Fixture chain incomplete')
    before = evaluate_target(b,dataset,{},shared,pool,proxy,np.ones((8,3)),
        {x:(np.ones((1,8),dtype=bool),{}) for x in (1.,1.5)},(1.,1.5),'same',0)
    baseline = label_summary(before)
    metrics = compare_labels(baseline,baseline)
    identity = object_hash({'synthetic':True})
    path = folder/'unit.json'
    payload = dict(scope='synthetic fixture only', frames=2, observed=observed,
                   records=records, audits=audits, metrics=metrics)
    commit_unit(path,identity,payload)
    saved = resume(path,identity)
    if saved['metrics'] != metrics:
        raise AssertionError('Atomic save/resume differs')
    verdict = final_verdict('PASS','FAIL','PASS','PASS')
    if verdict['decision'] != 'TASK SEPARATION STOP':
        raise AssertionError('Report state machine differs')
    atomic_json(folder/'SMOKE.json',dict(status='complete',scientific_result=False,
        chain=['fixture load','generate candidates','original proxy','original shortlist',
               'fixture postprocess/NMS boundary','original greedy','save','resume','report'],
        verdict_fixture=verdict))
    return {'status':'complete','scope':'synthetic only; no real data, GPU, model or scientific AP'}


def main():
    root = ROOT/'.tmp'
    root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root, prefix='stage2_smoke_') as value:
        print(smoke(Path(value)))


if __name__ == '__main__':
    main()
