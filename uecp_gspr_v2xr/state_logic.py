"""Data/GPU-independent resume identities and checked stage commits."""
from pathlib import Path

from .common import complete, identity, sha


def verify_resume(state, contract, variant, epoch, micro, order):
    if (state['contract_identity'] != contract or state['variant'] != variant or state['epoch'] != epoch
            or state['microbatch'] != micro or state['order_identity'] != identity(order)
            or state['accumulation_position'] != 0 or state['position'] % 8
            or not 0 <= state['position'] <= len(order)//8*8):
        raise ValueError('Recovery config/source/data/sampler/accumulation contract differs')


def valid_stage(row):
    if row.get('status') != 'COMPLETE' or not row.get('artifacts'):
        return False
    for path, digest in row['artifacts'].items():
        if sha(path) != digest:
            raise ValueError('Completed stage artifact changed: ' + path)
        if Path(path).name.endswith('complete.json'):
            complete(path)
    return True


def choose(rows):
    if not rows or any(r['split'] != 'train_dev' or r['condition'] != 'clean' or r['protocol'] != 'O' for r in rows):
        raise ValueError('Selection only accepts train_dev Clean O')
    return min(rows, key=lambda r: (-r['AP']['ap70'], r['epoch']))
