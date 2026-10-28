"""Retain the accepted 8083-frame extraction contract."""
from bridge import load
_engine = load('train_acceptance', 'gspr_repe/train_acceptance.py')
required_train_count = _engine.required_train_count
verify_manifest_acceptance = _engine.verify_manifest_acceptance
