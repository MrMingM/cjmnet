"""Reuse original 8083-frame acceptance, retaining its historical policy path."""
from auditlib import BASE, load_frozen

inherited = load_frozen('train_acceptance', 'train_acceptance.py')
# The historical module imports auditlib.HERE. Point its policy back to the
# original immutable acceptance file, never create a second authorization.
inherited.POLICY_PATH = BASE / 'TRAIN_DATA_ACCEPTANCE.json'
required_train_count = inherited.required_train_count
verify_manifest_acceptance = inherited.verify_manifest_acceptance
