"""Execute the ORIGINAL run_oracle code object with a per-call key adapter.

The original module globals are never mutated. Original shortlist, state,
postprocess, target order, composition, strict > tie break and Shared clones
remain in the original function. Only ineligible candidates get KEEP's key,
so they cannot beat the legal KEEP candidate. All candidates are evaluated.
"""
from collections import Counter
from types import FunctionType
from .common import KEEP

VARIANTS = ('FULL', 'A_ONLY', 'B_ONLY')


def partition(key, keep_key):
    if len(key) != 9 or len(keep_key) != 9:
        raise ValueError('Expected original nine-field Oracle keys')
    key, keep_key = tuple(key), tuple(keep_key)
    return 'A' if key[:6] != keep_key[:6] else 'B' if key[:8] != keep_key[:8] else 'E'


def eligible(variant, group):
    if variant not in VARIANTS or group not in ('A', 'B', 'E', 'KEEP'):
        raise ValueError('Unknown trajectory/group')
    return group == 'KEEP' or variant == 'FULL' or (variant == 'A_ONLY' and group == 'A') or (
        variant == 'B_ONLY' and group == 'B')


class KeyPolicy:
    def __init__(self, original, variant):
        if variant not in VARIANTS:
            raise ValueError('Unknown trajectory')
        self.original = original
        self.variant = variant
        self.targets = []

    def __call__(self, before, after, cls_source, reg_source):
        key = self.original(before, after, cls_source, reg_source)
        if cls_source == KEEP and reg_source == KEEP:
            if before is not after:
                raise ValueError('Original current-state KEEP contract changed')
            self.keep_key = key
            self.best_key = key
            self.targets.append({'keep_key': list(key), 'selected_group': 'KEEP',
                                 'selected_key': list(key), 'shortlisted': Counter(),
                                 'actions': []})
            return key
        if not self.targets:
            raise ValueError('Current-state KEEP must be evaluated first')
        row = self.targets[-1]
        group = partition(key, self.keep_key)
        allowed = eligible(self.variant, group)
        row['shortlisted'][group] += 1
        row['actions'].append(dict(cls_source=cls_source, reg_source=reg_source,
                                   group=group, key=list(key), allowed=allowed))
        effective = key if allowed else self.keep_key
        if effective > self.best_key:
            self.best_key = effective
            row.update(selected_group=group, selected_key=list(key))
        return effective


def run_restricted(original_module, variant, *args, **kwargs):
    policy = KeyPolicy(original_module._action_key, variant)
    original = original_module.run_oracle
    namespace = {**original.__globals__, '_action_key': policy}
    adapter = FunctionType(original.__code__, namespace, original.__name__,
                           original.__defaults__, original.__closure__)
    adapter.__kwdefaults__ = original.__kwdefaults__
    prediction, post, records = adapter(*args, **kwargs)
    if len(records) != len(policy.targets):
        raise ValueError('Oracle target audit lost alignment')
    for record, audit in zip(records, policy.targets):
        if record['changed'] != (audit['selected_group'] != 'KEEP'):
            raise ValueError('Adapter selection differs from original trajectory')
        audit['target_index'] = record['target_index']
        audit['shortlisted'] = dict(audit['shortlisted'])
    return prediction, post, records, policy.targets


def action_statistics(records, audits):
    counts = Counter({key: 0 for key in (
        'targets', 'changed', 'shortlisted_A', 'shortlisted_B', 'shortlisted_E',
        'selected_A', 'selected_B', 'selected_E', 'selected_KEEP', 'task_separated',
        'A_available_targets', 'B_available_targets', 'E_available_targets',
        'step_recovered', 'step_lost', 'step_new_fp')})
    for row, audit in zip(records, audits):
        counts['targets'] += 1
        counts['changed'] += int(row['changed'])
        counts['task_separated'] += int(row['task_separated'])
        counts['selected_' + audit['selected_group']] += 1
        for group in ('A', 'B', 'E'):
            count = audit['shortlisted'].get(group, 0)
            counts['shortlisted_' + group] += count
            counts[group + '_available_targets'] += int(count > 0)
        chosen = audit['selected_key']
        counts['step_recovered'] += chosen[4]
        counts['step_lost'] += -chosen[1]
        counts['step_new_fp'] += -chosen[2]
    return counts
