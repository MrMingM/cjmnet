"""Exact REPE loop; initialize all A1 tensors from pristine E0 epoch zero."""
from bridge import adapt_training,load
from baseline import initialize_model,initialization_reference,restore_initial_rng
_engine=load('training','gspr_repe/training.py')
_engine.initialize_model=initialize_model
_engine.initialization_reference=initialization_reference
_engine.restore_initial_rng=restore_initial_rng
adapt_training(_engine)
config=_engine.config

def prepare(args):
    result=_engine.prepare(args)
    if args.command!='preflight':
        from baseline import guard_reference
        guard_reference(args)
    return result

def run(args):
    if args.variant not in ('E1','E2'):raise ValueError('E0 must never train')
    return _engine.run(args)
