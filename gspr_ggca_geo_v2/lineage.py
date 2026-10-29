"""Explicit multi-version references. Never rewrite a historical contract."""
from pathlib import Path
from auditlib import HERE, atomic_json, complete, identity, read_json, sha256


def plan(root):
    row = complete(Path(root) / 'migration_plan_v2.json')
    if identity({k:v for k,v in row.items() if k != 'identity'}) != row['identity']:
        raise ValueError('Migration plan identity mismatch')
    if sha256(HERE/'source_snapshot.json') != row['new_source_snapshot_sha256']:
        raise ValueError('New geometry source differs from its explicit migration revision')
    if sha256(Path(root)/'migration_plan.json') != row['previous_plan_sha256']:
        raise ValueError('Previous deployment plan changed')
    for name, digest in row['old_source_files'].items():
        if sha256(Path(row['old_source']) / name) != digest:
            raise ValueError('Preserved A1 source changed: ' + name)
    for name, digest in row['inherited_artifacts'].items():
        if sha256(name) != digest:
            raise ValueError('Inherited completed artifact changed: ' + name)
    if sha256(Path(row['old_root'])/'A1/contract.json') != row['A1_contract_sha256']:
        raise ValueError('Preserved A1 contract changed')
    return row


def training_root(root, variant):
    return Path(plan(root)['old_root'])/'A1' if variant == 'A1' else Path(root)/variant


def selected(root, variant):
    if variant != 'A1':
        from evaluation import selection
        return selection(Path(root)/variant)
    folder = training_root(root, variant)
    from checkpoint import committed
    contract = read_json(folder/'contract.json')
    rows = committed(folder, 20, contract['identity'])
    if len(rows) != 20 or complete(folder/'training_summary.json')['epochs'] != 20:
        raise ValueError('Preserved A1 has not completed all 20 epochs')
    choice = complete(folder/'selected.json')
    best = min(rows, key=lambda r:(-r['AP70'],r['epoch']))
    if (choice['checkpoint_sha256'] != best['checkpoint_sha256'] or choice['epoch'] != best['epoch']
            or sha256(choice['checkpoint']) != choice['checkpoint_sha256']
            or choice['training_contract_sha256'] != sha256(folder/'contract.json')):
        raise ValueError('Preserved A1 selection/weight mismatch')
    return choice


def assert_fair(root):
    row = plan(root)
    baseline = complete(Path(root)/'baseline_reference.json')['original_training_contract']
    fields = ('epochs','policy','train_manifest_sha256','train_manifest_identity','train_data_acceptance',
              'pair_identity','config_sha256','input_identity','frozen_gt','loader','scheduler',
              'training_GT','gspr_frozen_settings')
    contracts = {}
    evaluations = {}
    reference_evaluation=complete(Path(root)/'A0/evaluation_reference.json')
    reference_protocol=read_json(Path(reference_evaluation['root'])/'protocol.json')
    for variant in ('A1','A2','A3'):
        folder = training_root(root, variant)
        c = read_json(folder/'contract.json')
        for name in fields:
            if c[name] != baseline[name]:
                raise ValueError('Training fairness mismatch: '+variant+':'+name)
        # CONFIG variants/source fields differ by declared version; training
        # settings, model family, optimizer YAML and initial shared state do not.
        for name in ('epochs','seed'):
            if c['settings'][name] != baseline['settings'][name]:
                raise ValueError('Seed/epoch settings differ')
        if c['ggca_initialization']['initial_state_sha256'] != complete(Path(root)/'baseline_reference.json')['initial_state_sha256']:
            raise ValueError('Original initialization differs')
        a = complete(folder/'initialization_audit.json')
        if not a['shared_tensors_exact'] or a['trained_A0_weight_used']:
            raise ValueError('Invalid shared initialization')
        choice=selected(root, variant)
        summary=complete(folder/'final_evaluation/summary.json')
        protocol=read_json(folder/'final_evaluation/protocol.json')
        if (summary['checkpoint_sha256']!=choice['checkpoint_sha256'] or summary['frame_count']!=829
                or not summary['full_validation'] or summary['conditions']!=['clean','fog','snow']
                or protocol['protocols']!=reference_protocol['protocols']
                or protocol['input_manifest_identity']!=reference_protocol['input_manifest_identity']):
            raise ValueError('Evaluation selection/protocol/data differs: '+variant)
        evaluations[variant]=dict(summary=str(folder/'final_evaluation/summary.json'),
            sha256=sha256(folder/'final_evaluation/summary.json'),protocol_sha256=sha256(folder/'final_evaluation/protocol.json'))
        contracts[variant] = dict(path=str(folder/'contract.json'),sha256=sha256(folder/'contract.json'),
            source_revision='original-A1' if variant=='A1' else 'pair-exchange-v2')
    record = dict(status='complete', unchanged_fields=list(fields), contracts=contracts,evaluations=evaluations,
                  migration_identity=row['identity'],A0_retrained=False,A1_restarted_from_epoch_zero=False)
    atomic_json(Path(root)/'reports/VERSION_FAIRNESS_AUDIT.json',record)
    return record
