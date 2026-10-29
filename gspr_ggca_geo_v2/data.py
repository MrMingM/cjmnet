"""Keep B0's exact mixed-weather reads and RNG-isolated reference reads.

The reference is not used by GGCA or its loss. Keeping the wrapper also keeps
loader construction/randomness directly comparable with B0.
"""
from bridge import load
_engine = load('data', 'gspr_repe/data.py')
build_train = _engine.build_train
paired_inventory = _engine.paired_inventory
tensor_digest = _engine.tensor_digest
