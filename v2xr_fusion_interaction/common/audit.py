"""Pin input evidence and verify physical HCU assignments."""
import ctypes
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys

HERE = Path(__file__).resolve().parents[1]
POT = HERE.parent / 'v2xr_fusion_pot'
sys.path[:0] = [str(POT), str(POT/'parallel')]
import run as lr
from common import atomic_json, identity, read_json, sha256

ROOT = Path('/data/new_disk/cjm_logs')
OUT = ROOT / 'v2xr-fusion-interaction-v1'
DEVICES = {'F2': (2, '0000:0f:00.0', '7137d9dd68923121'),
           'F3': (3, '0000:16:00.0', '7137d9b2a09238c1'),
           'F4': (4, '0000:1d:00.0', '7137d9b2a09a2061')}
WEATHERS = ('clean','fog','snow')


def settings(group):
    import yaml
    data = yaml.safe_load((HERE/'configs'/f'{group.lower()}.yaml').read_text())
    expected = dict(mode=group, gpu=DEVICES[group][0], rank=16, beta=.1, single_support=.25)
    if data != expected:
        raise ValueError(f'Unexpected frozen settings for {group}: {data}')
    return data


def idle(groups):
    if socket.gethostname().split('.')[0] != 'admin1':
        raise RuntimeError('Expected HaiGuang admin1')
    cards=json.loads(subprocess.check_output(
        ['rocm-smi','--showbus','--showuniqueid','--showuse','--showmeminfo','vram','--json'],text=True))
    pids=subprocess.check_output(['rocm-smi','--showpids'],text=True).lower()
    for group in groups:
        card,bus,uid=DEVICES[group]
        row=cards[f'card{card}']
        if row['PCI Bus'].lower()!=bus or row['Unique ID'].lower()!=uid:
            raise RuntimeError(f'{group} physical card identity differs')
        if (bus in pids or re.search(rf"hcu index:.*'{card}'",pids)
                or float(row['HCU use (%)'])>0 or int(row['vram Total Used Memory (MiB)'])>16):
            raise RuntimeError(f'{group} physical GPU {card} is occupied: {row}')


def device(group):
    import torch
    card,bus,uid=DEVICES[group]
    if (os.environ.get('ROCR_VISIBLE_DEVICES')!=str(card)
            or any(k in os.environ for k in ('HIP_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES'))
            or not torch.version.hip or not torch.cuda.is_available() or torch.cuda.device_count()!=1):
        raise RuntimeError('Unexpected HIP visibility')
    paths=[row.split()[-1] for row in Path('/proc/self/maps').read_text().splitlines()
           if '/libamdhip64.so' in row]
    hip=ctypes.CDLL(paths[0] if paths else 'libamdhip64.so')
    actual=ctypes.create_string_buffer(32)
    if hip.hipDeviceGetPCIBusId(actual,len(actual),0) or actual.value.decode().lower()!=bus:
        raise RuntimeError(f'HIP logical zero does not map to {bus}')
    probe=torch.zeros(64*1024*1024,device='cuda:0',dtype=torch.uint8)
    torch.cuda.synchronize()
    pids=subprocess.check_output(['rocm-smi','--showpids'],text=True).lower()
    match=re.search(rf'(?m)^pid: {os.getpid()}\s*\n(.*?)(?=\npid:|\Z)',pids,re.S)
    if match is None or bus not in match.group(1):
        raise RuntimeError('rocm-smi did not confirm this process on the assigned card')
    del probe
    return dict(physical_gpu=card,logical_gpu='cuda:0',pci_bus=bus,unique_id=uid,
                torch=str(torch.__version__),hip=torch.version.hip,gpu=torch.cuda.get_device_name(0))


def source_files():
    return {p.relative_to(HERE).as_posix():sha256(p) for p in sorted(HERE.rglob('*'))
            if p.is_file() and p.suffix in ('.py','.yaml','.sh')}


def audit():
    from p0_failure import select_targets
    from runtime import build
    from v2xr_lidar_radar_baseline.run import source_contract
    from inputs import TRAIN_MANIFEST,VAL_MANIFEST
    from diagnostics import verify_gt
    if OUT.resolve()!=OUT or not OUT.is_dir() or not os.access(OUT,os.W_OK):
        raise PermissionError(f'Expected writable real output directory {OUT}')
    if shutil.disk_usage(OUT).free<40*2**30:
        raise RuntimeError('Less than 40 GiB available on the experiment disk')
    idle(DEVICES)
    configuration={g:settings(g) for g in DEVICES}
    frozen=read_json(lr.LR0/'CONTRACT.json')
    if source_contract()!=frozen['sources'] or sha256(lr.CONFIG)!=frozen['config_sha256']:
        raise ValueError('Frozen LR0 source or config changed')
    if sha256(lr.WEIGHT)!='0eb0ec8bd6a6e0bc368c7cc8aacd49a33cfe014413463f8b314fdbf76c6defd2':
        raise ValueError('P0 checkpoint SHA256 differs')
    receipt=read_json(lr.LR0/'DATA.json')
    if sha256(lr.LR0/'DATA.json')!=frozen['data_sha256']:
        raise ValueError('LR0 data receipt changed')
    for split,path,n in (('train',TRAIN_MANIFEST,8083),('validate',VAL_MANIFEST,829)):
        row=receipt['splits'][split]
        if row['frames']!=n or sha256(path)!=row['input_manifest_sha256']:
            raise ValueError(f'{split} input identity differs')
    training,cfg=build(lr.CONFIG,split='train')
    if len(training)!=8083 or cfg['train_params']['batch_size']!=8:
        raise ValueError('LR0 training loader contract differs')
    for weather in WEATHERS:
        validation,_=build(lr.CONFIG,condition=weather)
        if len(validation)!=829:
            raise ValueError('Validation length differs')
    cohort=select_targets()
    old=lr.OUT/'p0-frozen-failure-v1/COHORT.json'
    if json.loads(json.dumps(cohort))!=read_json(old):
        raise ValueError('Historical lost-target IDs differ')
    verify_gt(lr.OUT/'ablations/P0/same_weights_gpu4')
    files=source_files()
    if any(b'\r' in (HERE/p).read_bytes() for p in files):
        raise ValueError('Expected LF in source/config files')
    contract=dict(source=files,configuration=configuration,lr0=frozen,
        p0_sha256=sha256(lr.WEIGHT),cohort_sha256=sha256(old),
        context_sources={name:sha256(POT/name) for name in ('diagnostics.py','parallel/p0_failure.py',
                                                          'parallel/geometry_warnings.py')},
        gpu={g:list(row) for g,row in DEVICES.items()})
    lr.bind(OUT/'CONTRACT.json',contract)
    lr.bind(OUT/'COHORT.json',read_json(old))
    print('AUDIT PASSED',identity(contract),flush=True)
