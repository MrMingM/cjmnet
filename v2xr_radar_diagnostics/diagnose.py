"""Standalone read-only mechanism audit for frozen F2 and F3 checkpoints."""
import os
from pathlib import Path
import re
import socket
import subprocess
import sys

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent/'v2xr_radar_ra_rb'
sys.path.insert(0,str(SOURCE))
import run as experiment
import torch

from cached import summarize_f3
from checkpoint import seed
from common import atomic_json, identity, lock, read_json, sha256
from feature_probe import instrument, merge_features, preflight
from pair_audit import inspect, logs
from opencood.hypes_yaml import yaml_utils
from opencood.tools import train_utils

HISTORY = experiment.OUT
OUT = HISTORY.parent/'v2xr-radar-diagnostics-v1'
WEATHERS = ('clean','fog')
LIMIT = 120


def audit():
    card = os.environ['ROCR_VISIBLE_DEVICES']
    if socket.gethostname().split('.')[0]!='admin1' or card=='1':
        raise ValueError('Use a separate free ROCm card on admin1')
    info = subprocess.check_output(['rocm-smi','--showmeminfo','vram'],text=True)
    match = re.search(rf'HCU\[{int(card)}\].*?vram Total Used Memory \(MiB\):\s*(\d+)',info)
    if not match or int(match[1])>256:
        raise ValueError(f'Physical card {card} is occupied')
    if torch.cuda.device_count()!=1 or not torch.version.hip:
        raise RuntimeError('Exactly one visible ROCm card required')
    contract = read_json(HISTORY/'CONTRACT.json')
    if any(sha256(SOURCE / p)!=h for p,h in contract['source'].items()):
        raise ValueError('Formal code changed')
    if any(sha256(SOURCE.parent / p)!=h for p,h in contract['frozen'].items()):
        raise ValueError('Frozen dependency changed')
    lr0 = read_json(experiment.LR0/'CONTRACT.json')
    if experiment.source_contract()!=lr0['sources'] or sha256(HISTORY/'DATA.json')!=contract['data']:
        raise ValueError('Historical runtime or data identity changed')
    return contract


def load(group, epoch):
    folder = HISTORY/group
    config = folder/'config.yaml'
    info = read_json(folder/'CONTRACT.json')
    row = read_json(folder/'epochs'/f'{epoch:02d}/complete.json')
    weights = Path(row['checkpoint']).resolve(strict=True)
    digest = sha256(weights)
    if (digest!=row['checkpoint_sha256'] or sha256(config)!=info['base_config']
            or weights.parent!=folder/'epochs'/f'{epoch:02d}'):
        raise ValueError('Checkpoint or configuration differs')
    if group=='F2':
        chosen = read_json(folder/'selected.json')
        if (chosen['epoch'],chosen['sha256'])!=(epoch,digest):
            raise ValueError('F2 epoch 16 is not the original selected model')
    else:
        cached = read_json(HISTORY/'mechanism_review_20261111/F3/full/DIAGNOSTIC.json')['config']
        if (cached['epoch'],cached['sha256'])!=(epoch,digest):
            raise ValueError('F3 checkpoint differs from archived intervention')
    seed(20261016)
    model = experiment.make_model(yaml_utils.load_yaml(str(config)), info['settings'])
    model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=False),strict=True)
    return model.eval(), config, digest


def validate_gt(path, reference):
    expected = [r for r in read_json(reference)['frames'] if r['condition'] in WEATHERS and r['index']<LIMIT]
    actual = read_json(path/'gt_manifest.json')['frames']
    fields = ('condition','index','frame_key','fixed_ids','fixed_gt_identity')
    if len(actual)!=len(expected) or any(any(a[k]!=b[k] for k in fields) for a,b in zip(actual,expected)):
        raise ValueError('Fixed ground truth differs')


def evaluate_mode(model, config, digest, mode, contract, reference):
    path = OUT/mode
    baseline = HISTORY/'F2/final_evaluation'
    seed(20261016)
    with instrument(model,path,mode,baseline):
        report = experiment.engine.evaluate(model,None,config,path,WEATHERS,LIMIT,
            identity([contract,digest,mode,LIMIT]),digest)
    validate_gt(path,reference)
    features = merge_features(path,WEATHERS,LIMIT)
    result = dict(AP={w:v['AP'] for w,v in report['results'].items()},
        counts={w:v['matches']['F']['0.7'] for w,v in report['results'].items()}, features=features,
        evidence=str(path))
    atomic_json(path/'RESULT.json',result)
    print('COMPLETE',mode,flush=True)
    return result


def local_targets(a,b):
    old, new = (x['features']['fog']['targets'] for x in (a,b))
    if old.keys()!=new.keys():
        raise ValueError('Different fixed damaged target groups')
    summary = {}
    for key,selected in dict(all=list(old), only_0_1_neighbors=[
        k for k,v in old.items() if v['max_neighbors'] is not None and v['max_neighbors']<=1],
        no_pillar=[k for k,v in old.items() if v['max_neighbors'] is None]).items():
        pairs = [(old[k]['iou'],new[k]['iou']) for k in selected]
        common = [y-x for x,y in pairs if x is not None and y is not None]
        summary[key] = dict(targets=len(pairs), d0_tp=sum(x is not None for x,y in pairs),
            mode_tp=sum(y is not None for x,y in pairs),
            recovered=sum(x is None and y is not None for x,y in pairs),
            lost=sum(x is not None and y is None for x,y in pairs),
            common=len(common), mean_iou_change=sum(common)/len(common) if common else None)
    return summary


def main():
    parent = audit()
    reference = experiment.LR0/'final_evaluation/gt_manifest.json'
    if sha256(reference)!=parent['fixed_gt']:
        raise ValueError('Original fixed GT bytes changed')
    code = {p.name:sha256(p) for p in HERE.iterdir() if p.suffix in ('.py','.sh')}
    contract = identity([parent,code,os.environ['ROCR_VISIBLE_DEVICES']])
    f2, config, digest = load('F2',16)
    data,_ = experiment.build(config,condition='clean')
    fixture = data.native.collate_batch_test([data[0]])
    ego = train_utils.to_device(fixture['ego'],torch.device('cuda:0'))
    gate = preflight(f2,ego)
    atomic_json(OUT/'INTERVENTION_CHECK.json',gate)
    print('PASS identity, empty Radar, and multiplier checks',flush=True)
    del ego, fixture
    torch.cuda.empty_cache()

    # D0 supplies the per-pillar observation required in item 1.
    baseline = evaluate_mode(f2,config,digest,'D0',contract,reference)

    f3,f3_config,f3_digest = load('F3',10)
    paired = dict(sample=inspect(experiment,f3,f3_config), history=logs(HISTORY/'F3',10), weights=f3_digest)
    atomic_json(OUT/'PAIR_AUDIT.json',paired)
    print('COMPLETE paired training audit',flush=True)
    del f3
    torch.cuda.empty_cache()

    cached = summarize_f3(HISTORY)
    atomic_json(OUT/'F3_CACHED_COUNTS.json',cached)
    print('COMPLETE cached F3 analysis',flush=True)

    changes = {}
    for mode in ('D1','D2'):
        result = evaluate_mode(f2,config,digest,mode,contract,reference)
        changes[mode] = dict(AP=result['AP'],counts=result['counts'],
                             target_changes=local_targets(baseline,result))
    summary = dict(status='complete', checkpoint=digest, audit=gate, paired=paired,
        cached_f3=cached, D0=dict(AP=baseline['AP'],counts=baseline['counts'],evidence=baseline['evidence']),
        interventions=changes, conditions=WEATHERS, frames_per_weather=LIMIT,
        policy='Fixed prefix selected before outcomes. Only R-B multiplicative support is changed.')
    atomic_json(OUT/'SUMMARY.json',summary)
    print('COMPLETE ALL',flush=True)


if __name__=='__main__':
    with lock(OUT/'.lock'):
        try:
            main()
        except Exception:
            import traceback
            atomic_json(OUT/'FAILURE.json',dict(traceback=traceback.format_exc()))
            raise

