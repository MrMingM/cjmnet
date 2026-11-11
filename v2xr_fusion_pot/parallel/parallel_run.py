"""Continue the frozen POT-CF experiment with one group per physical GPU."""
import os
import json
from pathlib import Path
import re
import socket
import subprocess
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run
from common import atomic_json, identity, lock, read_json
import device

CARDS = {
    'P1': (4, '0000:1d:00.0', '7137d9b2a09a2061'),
    'P2': (5, '0000:20:00.0', '7137d9aff40e38c1'),
    'P3': (6, '0000:25:00.0', '71010022e8ce1081'),
}


def main(group):
    card, device.PHYSICAL_PCI, device.PHYSICAL_UNIQUE_ID = CARDS['P1' if group == 'report' else group]
    if os.environ.get('ROCR_VISIBLE_DEVICES') != str(card):
        raise RuntimeError('GPU mask differs from the assigned card')
    if socket.gethostname().split('.')[0] != 'admin1' or any(
            k in os.environ for k in ('HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES')):
        raise RuntimeError('Unexpected ROCm host or device mask')
    rows = json.loads(subprocess.check_output(
        ['rocm-smi', '--showbus', '--showuniqueid', '--showuse', '--showmeminfo', 'vram', '--json'], text=True))
    row = rows[f'card{card}']
    pids = subprocess.check_output(['rocm-smi', '--showpids'], text=True)
    if (row['PCI Bus'].lower() != device.PHYSICAL_PCI
            or row['Unique ID'].lower() != device.PHYSICAL_UNIQUE_ID
            or device.PHYSICAL_PCI in pids.lower() or re.search(rf"HCU Index:.*'{card}'", pids)
            or float(row['HCU use (%)']) > 0 or int(row['vram Total Used Memory (MiB)']) > 16):
        raise RuntimeError(f'Physical GPU {card} is busy or its identity changed: {row}')
    contract = read_json(run.OUT / 'CONTRACT.json')
    if run.audit() != {k: v for k, v in contract.items() if k != 'device'}:
        raise RuntimeError('Frozen formal contract changed')
    mapped = dict(device.map_device(), physical_index=card)
    experiment_id = identity(contract)
    print('DEVICE VERIFIED', group, mapped, flush=True)
    import engine
    import diagnostics
    from geometry_warnings import checked_evaluation
    from diagnostics import evaluate_formal, report
    engine.evaluate = checked_evaluation(engine.evaluate)
    diagnostics.evaluate = checked_evaluation(diagnostics.evaluate)

    if group == 'report':
        results = {'P0': read_json(run.LR0 / 'FINAL_RESULTS.json')}
        for name in CARDS:
            result = read_json(run.OUT / name / 'FINAL_RESULTS.json')
            if result['status'] != 'complete' or result['contract'] != experiment_id:
                raise ValueError(f'{name} has no compatible completed result')
            results[name] = result
        report(run.OUT, run.CONFIG, experiment_id, results, run.make_model, run.SETTINGS)
        return
    destination = run.OUT / group
    run.bind(destination / 'PARALLEL_DEVICE.json', mapped)
    atomic_json(destination / 'PARALLEL_STATUS.json', dict(status='training', group=group, device=mapped))
    with run.use_group(group):
        rows = engine.train(destination, run.CONFIG, experiment_id)
    atomic_json(destination / 'PARALLEL_STATUS.json', dict(status='evaluating', group=group, device=mapped))
    evaluate_formal(destination, run.CONFIG, experiment_id, group, rows, run.make_model)
    atomic_json(destination / 'PARALLEL_STATUS.json', dict(status='complete', group=group, device=mapped))


if __name__ == '__main__':
    group = sys.argv[1]
    destination = run.OUT / ('parallel-report' if group == 'report' else group)
    with lock(destination / '.parallel.lock'):
        try:
            main(group)
        except Exception:
            atomic_json(destination / 'PARALLEL_STATUS.json',
                        dict(status='failed', group=group, traceback=traceback.format_exc()))
            raise
