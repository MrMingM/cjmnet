"""Load the pinned LR0 runtime and the previous diagnostic helpers."""
import ctypes
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys

HERE = Path(__file__).resolve().parents[1]
POT = HERE.parent / 'v2xr_fusion_pot'
sys.path[:0] = [str(POT), str(POT / 'parallel')]
import run as pot
from common import atomic_json, identity, read_json, sha256
from p0_failure import select_targets

OUT = pot.DATA / 'logs/v2xr-fusion-diagnostics-v1'
HISTORY = pot.OUT / 'p0-frozen-failure-v1'
WEATHERS = ('clean', 'fog', 'snow')
DEVICES = {'D1': (0, '0000:07:00.0', '7100fa88668e40a1'),
           'D2': (2, '0000:0f:00.0', '7137d9dd68923121'),
           'D3': (4, '0000:1d:00.0', '7137d9b2a09a2061')}


def idle(groups):
    if socket.gethostname().split('.')[0] != 'admin1':
        raise RuntimeError('This entry requires the verified HaiGuang server')
    cards = json.loads(subprocess.check_output(
        ['rocm-smi','--showbus','--showuniqueid','--showuse','--showmeminfo','vram','--json'], text=True))
    pids = subprocess.check_output(['rocm-smi','--showpids'], text=True).lower()
    for group in groups:
        card, bus, uid = DEVICES[group]
        row = cards[f'card{card}']
        if row['PCI Bus'].lower() != bus or row['Unique ID'].lower() != uid:
            raise RuntimeError(f'{group} physical card identity differs: {row}')
        if (bus in pids or re.search(rf"hcu index:.*'{card}'", pids)
                or float(row['HCU use (%)']) > 0 or int(row['vram Total Used Memory (MiB)']) > 16):
            raise RuntimeError(f'{group} physical GPU {card} is occupied: {row}')


def map_device(group):
    import torch
    card, bus, uid = DEVICES[group]
    if (os.environ.get('ROCR_VISIBLE_DEVICES') != str(card)
            or any(k in os.environ for k in ('HIP_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES'))
            or not torch.version.hip or not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise RuntimeError('Unexpected HIP device visibility')
    paths = [line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
             if '/libamdhip64.so' in line]
    hip = ctypes.CDLL(paths[0] if paths else 'libamdhip64.so')
    actual = ctypes.create_string_buffer(32)
    if hip.hipDeviceGetPCIBusId(actual, len(actual), 0) or actual.value.decode().lower() != bus:
        raise RuntimeError(f'Logical GPU does not map to {bus}')
    assert torch.ones(1, device='cuda:0').item() == 1
    return dict(physical_gpu=card, logical_gpu='cuda:0', pci_bus=bus, unique_id=uid,
                hip=torch.version.hip, torch=str(torch.__version__), gpu=torch.cuda.get_device_name(0))


def audit():
    idle(DEVICES)
    prior = pot.audit()
    cohort = select_targets()
    historical = read_json(HISTORY / 'COHORT.json')
    if {w:{str(i):v for i,v in frames.items()} for w,frames in cohort.items()} != historical:
        raise ValueError('Fixed P0 cohort differs from its historical file')
    if [len(cohort[w]) for w in ('fog','snow')] != [91,165]:
        raise ValueError('Historical target frame counts differ')
    results = read_json(HISTORY / 'RESULTS.json')
    if results['weight_sha256'] != sha256(pot.WEIGHT) or results['radius_m'] != 12.:
        raise ValueError('Historical context definition differs')
    samples = {}
    for weather, frames in cohort.items():
        for index, ids in frames.items():
            file = HISTORY / weather / f'{index:06d}.json'
            record = read_json(file)
            if sorted(record['targets']) != sorted(ids) or record['weight_sha256'] != results['weight_sha256']:
                raise ValueError(f'Historical sample mismatch: {file}')
            samples[str(file.relative_to(HISTORY))] = sha256(file)
    source = {p.relative_to(HERE).as_posix():sha256(p) for p in HERE.rglob('*')
              if p.is_file() and p.suffix in ('.py','.sh')}
    if any(b'\r' in (HERE/p).read_bytes() for p in source):
        raise ValueError('Expected LF source files')
    reference = {str(p):sha256(p) for p in (HISTORY/'COHORT.json', HISTORY/'RESULTS.json',
        pot.LR0/'final_evaluation/prediction_manifest.json', POT/'parallel/p0_failure.py')}
    contract = dict(sources=source, historical_samples=samples, historical=reference,
                    LR0=prior, devices=DEVICES, frame_count=829, targets_per_weather=6235)
    OUT.mkdir(parents=True, exist_ok=True)
    pot.bind(OUT/'CONTRACT.json', json.loads(json.dumps(contract)))
    atomic_json(OUT/'COHORT.json',historical)
    print('AUDIT PASSED',identity(contract),flush=True)
