"""Frozen paths and weights for read-only diagnostics."""
import hashlib
import json
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
REPO=HERE.parent
DATA=Path('/data/cjm/datasets')
HISTORY=DATA/'logs/v2xr-radar-ra-rb-v1'
F0=HISTORY/'F0'
LR0=DATA/'logs/v2xr-lr0-v3'
ARCHIVE=HISTORY/'mechanism_review_20261111/F3'
OUT=DATA/'logs/v2xr-radar-evidence-v1'
WEATHER=('clean','fog','snow')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(temp,path)


def sha(path):
    result=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(4194304),b''):
            result.update(block)
    return result.hexdigest()


def bind(path,value):
    if path.exists() and read(path)!=value:
        raise ValueError(f'Identity differs at {path}')
    if not path.exists():
        write(path,value)


def setup():
    sys.path.insert(0,str(REPO/'v2xr_radar_ra_rb'))
    import run
    from common import sha256
    parent=read(HISTORY/'CONTRACT.json')
    for name,digest in parent['source'].items():
        if sha256(REPO/'v2xr_radar_ra_rb'/name)!=digest:
            raise ValueError(f'Frozen source changed: {name}')
    if sha256(LR0/'final_evaluation/gt_manifest.json')!=parent['fixed_gt']:
        raise ValueError('Frozen Clean GT changed')
    return run


def load(run,group):
    import torch
    from checkpoint import seed
    from opencood.hypes_yaml import yaml_utils
    folder=HISTORY/group
    if group=='F0':
        row=read(folder/'selected.json')
        if row['epoch']!=20:
            raise ValueError('Expected F0 epoch 20')
        expected=row['sha256']
    else:
        row=read(folder/'epochs/10/complete.json')
        expected=read(ARCHIVE/'full/DIAGNOSTIC.json')['config']['sha256']
    weight=Path(row['checkpoint'])
    if sha(weight)!=expected:
        raise ValueError('Weight digest differs')
    config=folder/'config.yaml'
    contract=read(folder/'CONTRACT.json')
    if sha(config)!=contract['base_config']:
        raise ValueError('Configuration differs')
    seed(20261016)
    model=run.make_model(yaml_utils.load_yaml(str(config)),contract['settings']).eval()
    model.load_state_dict(torch.load(weight,map_location='cpu',weights_only=False),strict=True)
    return model,config,expected


def frozen_clean(index,key):
    record=read(LR0/'final_evaluation/predictions/clean'/f'{index:06d}.json')
    if record['frame_key']!=list(key):
        raise ValueError('Fixed Clean frame identity differs')
    return record['official_gt']

