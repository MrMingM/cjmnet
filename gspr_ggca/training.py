"""Reuse the fixed REPE training loop with two audited initialization insertions."""
from bridge import adapt_training, load
from baseline import initialize_model, restore_initial_rng, initialization_reference

_engine = load('training', 'gspr_repe/training.py')
_engine.initialize_model = initialize_model
_engine.restore_initial_rng = restore_initial_rng
_engine.initialization_reference = initialization_reference
adapt_training(_engine)
config = _engine.config


def prepare(args):
    prepared = _engine.prepare(args)
    if args.command != 'preflight':
        from baseline import guard_reference
        guard_reference(args)
    return prepared


def run(args):
    from auditlib import METHOD
    if args.variant not in METHOD['trainable_variants']:
        raise ValueError('A0 training is forbidden')
    return _engine.run(args)
