"""Retain the original accepted 8083-frame policy and evidence path."""
from auditlib import BASE, load_frozen
inherited = load_frozen('train_acceptance', 'train_acceptance.py')
inherited.POLICY_PATH = BASE / 'TRAIN_DATA_ACCEPTANCE.json'
required_train_count = inherited.required_train_count
verify_manifest_acceptance = inherited.verify_manifest_acceptance
