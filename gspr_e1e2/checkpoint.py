"""Use the original complete-state checkpoint engine."""
from bridge import load
_engine = load('checkpoint', 'gspr_repe/checkpoint.py')
for _name in ('seed', 'rng', 'restore_rng', 'save', 'state', 'restore', 'committed', 'ACCUMULATORS'):
    globals()[_name] = getattr(_engine, _name)
