import json
import os
from pathlib import Path

DATA=Path('/data/cjm/datasets')
OUT=DATA/'logs/v2xr-radar-occupancy-v1'
HISTORY=DATA/'logs/v2xr-radar-ra-rb-v1'
F0=HISTORY/'F0'
F3=HISTORY/'mechanism_review_20261111/F3'
LR0=DATA/'logs/v2xr-lr0-v3'
WEATHERS=('clean','fog','snow')


def read(path):
    return json.loads(path.read_text())


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    os.replace(temp,path)

