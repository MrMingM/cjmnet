"""Resolve HIP's logical device using its PCI identity before running any model."""
import ctypes
import json
import os
from pathlib import Path
import re
import socket
import subprocess


PHYSICAL_PCI = '0000:1d:00.0'
PHYSICAL_UNIQUE_ID = '7137d9b2a09a2061'


def pre_device():
    if (socket.gethostname().split('.')[0] != 'admin1' or os.environ.get('ROCR_VISIBLE_DEVICES') != '4'
            or any(var in os.environ for var in ('HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES'))):
        raise RuntimeError('Require admin1 with ROCR_VISIBLE_DEVICES=4 and no competing mask')
    rows = json.loads(subprocess.check_output(
        ['rocm-smi', '--showbus', '--showuniqueid', '--showuse', '--showmeminfo', 'vram', '--json'], text=True))
    card = rows['card4']
    if card['PCI Bus'].lower() != PHYSICAL_PCI or card['Unique ID'].lower() != PHYSICAL_UNIQUE_ID:
        raise RuntimeError(f'Physical card 4 changed identity: {card}')
    pids = subprocess.check_output(['rocm-smi', '--showpids'], text=True)
    if (PHYSICAL_PCI in pids.lower() or re.search(r"HCU Index:.*['\"]4['\"]", pids)
            or float(card['HCU use (%)']) > 0 or int(card['vram Total Used Memory (MiB)']) > 16):
        raise RuntimeError(f'Physical GPU 4 is occupied: {card}')
    return card


def map_device():
    import torch

    if not torch.version.hip or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Require one visible ROCm GPU')
    libraries = [line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                 if '/libamdhip64.so' in line]
    hip = ctypes.CDLL(libraries[0] if libraries else 'libamdhip64.so')
    pci = ctypes.create_string_buffer(32)
    status = hip.hipDeviceGetPCIBusId(pci, len(pci), 0)
    logical = pci.value.decode('ascii').lower()
    if status or logical != PHYSICAL_PCI:
        raise RuntimeError(f'Logical HIP 0 PCI check failed: {status}, {logical}')
    x = torch.ones(16, device='cuda:0')
    assert x.device.type == 'cuda' and float(x.sum()) == 16
    return dict(physical_index=4, logical_device='cuda:0', pci_bus=logical,
                unique_id=PHYSICAL_UNIQUE_ID, gpu=torch.cuda.get_device_name(0),
                hip=torch.version.hip, total_bytes=torch.cuda.get_device_properties(0).total_memory)
