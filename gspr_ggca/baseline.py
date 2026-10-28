"""Strict, read-only B0 discovery. Never select a replacement or train A0."""
from pathlib import Path
from auditlib import (CONFIG, DEFAULTS, METHOD, ROOT, SELECTION, SETTINGS, atomic_json,
    bound_json, complete, frozen_c, identity, no_test, read_json, sha256)


def check_identity(record):
    if record.get('identity') != identity({k: v for k, v in record.items() if k != 'identity'}):
        raise ValueError('Artifact identity differs')


def validate_recipe(contract):
    check_identity(contract)
    if contract['variant'] != 'B0' or contract['smoke_only'] or contract['epochs'] != 20:
        raise ValueError('A0 must be the formal 20-epoch B0')
    for key in ('seed', 'epochs', 'batch_size', 'workers', 'sampling', 'train_frames'):
        if contract['settings'][key] != CONFIG[key]:
            raise ValueError('A0 training recipe differs: ' + key)
    if contract['policy'] != SELECTION or contract['gspr_frozen_settings'] != SETTINGS:
        raise ValueError('A0 selection/GSPR configuration differs')
    if contract['loader'] != dict(batch_size=8, num_workers=16, shuffle=True, drop_last=True,
                                  pin_memory=True, prefetch_factor=4):
        raise ValueError('A0 loader recipe differs')
    if contract['environment']['backend'] != 'rocm':
        raise ValueError('A0 backend must be ROCm for this matched run')
    if contract['scheduler'] != 'official end-of-epoch scheduler.step(zero_based_epoch)':
        raise ValueError('A0 scheduler semantics differ')


def audit(args, manifest, evidence):
    import yaml
    root = no_test(args.a0_root).resolve(strict=True)
    contract = read_json(root / 'contract.json')
    validate_recipe(contract)
    expected_models = sha256(ROOT / 'gspr_repe/models.py')
    for filename in ('models.py', 'training.py', 'data.py', 'checkpoint.py', 'settings.json'):
        if contract['sources']['files'][filename] != sha256(ROOT / 'gspr_repe' / filename):
            raise ValueError('A0 model/training implementation differs from reviewed B0: ' + filename)
    for key in ('attfuse_gspr/reliability.py', 'v2xr_gspr_clean/integration.py'):
        if contract['sources']['dependencies'].get(key) != sha256(ROOT / key):
            raise ValueError('A0 original GSPR implementation differs: ' + key)
    from training import config
    expected = config(args, Path(args.output_root))
    old_config = yaml.safe_load((root / 'config.yaml').read_text(encoding='utf-8'))
    # config() returns the parsed/expanded hypes; compare original YAML bytes instead.
    new_config = yaml.safe_load((Path(args.output_root) / 'config.yaml').read_text(encoding='utf-8'))
    if old_config != new_config or sha256(root / 'config.yaml') != contract['config_sha256']:
        raise ValueError('A0 official config/training split differs')
    if contract['train_manifest_sha256'] != sha256(Path(args.train_root) / 'manifest.json'):
        raise ValueError('A0 train extraction identity differs')
    if contract['train_manifest_identity'] != manifest['identity'] or contract['train_data_acceptance'] != manifest['train_data_acceptance']:
        raise ValueError('A0 accepted train partition differs')
    from data import paired_inventory
    if contract['pair_identity'] != paired_inventory(manifest, args.train_root)['identity']:
        raise ValueError('A0 Clean/Fog inventory differs')
    if contract['input_identity'] != evidence['input_manifest']['identity'] or contract['frozen_gt'] != frozen_c(args)['gt']['identity']:
        raise ValueError('A0 validation split/GT protocol differs')
    if contract['environment']['versions'] != evidence['a']['versions']:
        raise ValueError('A0 package versions differ')
    import torch
    if contract['environment']['hip'] != torch.version.hip:
        raise ValueError('A0 HIP runtime differs')
    from checkpoint import committed
    rows = committed(root, 20, contract['identity'])
    if len(rows) != 20 or any(r.get('smoke_only') or r['train_updates'] != 1010 for r in rows):
        raise ValueError('A0 has not completed 20 full training epochs; A0 retraining is forbidden')
    done = complete(root / 'training_summary.json')
    chosen = complete(root / 'selected.json')
    check_identity(chosen)
    if done['epochs'] != 20 or done['selected'] != chosen or chosen['weather_used'] or chosen['candidate_count'] != 20:
        raise ValueError('A0 training completion/selection differs')
    if chosen['training_contract_sha256'] != sha256(root / 'contract.json') or chosen['policy'] != SELECTION:
        raise ValueError('A0 selected weight is not bound to the training contract')
    for row in rows:
        clean = complete(Path(row['clean_summary']))
        if clean['frame_count'] != 829 or clean['conditions'] != ['clean'] or not clean['full_validation']:
            raise ValueError('A0 selection did not use full Clean validation')
        if clean['checkpoint_sha256'] != row['checkpoint_sha256'] or row['AP70'] != clean['results']['clean']['AP']['O']['ap70']:
            raise ValueError('A0 epoch AP/checkpoint mismatch')
    best = min(rows, key=lambda r: (-r['AP70'], r['epoch']))
    if (chosen['epoch'], chosen['checkpoint_sha256'], chosen['clean_O_AP70']) != (best['epoch'], best['checkpoint_sha256'], best['AP70']):
        raise ValueError('A0 Clean-only model selection cannot be reproduced')
    checkpoint = no_test(chosen['checkpoint']).resolve(strict=True)
    if not checkpoint.is_relative_to(root) or sha256(checkpoint) != chosen['checkpoint_sha256']:
        raise ValueError('A0 selected checkpoint missing/changed/outside the designated B0')
    initial = root / 'resume_states/00.pth'
    marker = read_json(root / 'resume_states/00.json')
    if sha256(initial) != marker['sha256']:
        raise ValueError('A0 initial training state missing or changed')
    state = torch.load(initial, map_location='cpu', weights_only=False)
    if state['epoch'] != 0 or state['contract'] != contract['identity'] or state['optimizer']['state']:
        raise ValueError('A0 initialization is not a pristine complete training state')
    if not all(k in state for k in ('model', 'optimizer', 'scheduler', 'criterion', 'accumulators', 'rng')):
        raise ValueError('A0 full initial state unavailable; no trained-weight fallback')
    paths = [root / n for n in ('selected.json', 'training_summary.json', 'contract.json',
             'config.yaml', 'history.json', 'selection_policy.json', 'resume_states/00.json')]
    if read_json(root / 'selection_policy.json') != SELECTION:
        raise ValueError('A0 selection policy file differs')
    if read_json(root / 'history.json')['epochs'] != rows:
        raise ValueError('A0 epoch commit/history mismatch')
    reused = existing_evaluation(root, chosen, contract, evidence, args)
    reference = dict(status='complete', a0_root=str(root), checkpoint=str(checkpoint),
        checkpoint_sha256=chosen['checkpoint_sha256'], selected=chosen, config=old_config,
        original_training_contract=contract, initial_state=str(initial), initial_state_sha256=marker['sha256'],
        initial_model_keys=sorted(state['model']), training_epochs=20, training_forbidden=True,
        artifacts={str(p): sha256(p) for p in paths}, evaluation_reused=reused is not None,
        evaluation_root=str(reused or Path(args.experiment_root) / 'A0/supplemental_evaluation'),
        original_model_sha256=expected_models, method=METHOD)
    reference['identity'] = identity(reference)
    bound_json(Path(args.experiment_root) / 'baseline_reference.json', reference)
    atomic_json(Path(args.output_root) / 'A0_REUSE_AUDIT.json', reference)
    (Path(args.output_root) / 'A0_REUSE_AUDIT.md').write_text(
        '# A0复用审计\n\n训练20轮、Clean选模、配置/数据/源码/初始状态哈希检查通过。\n\n'
        + '指定B0：`' + str(root) + '`\n\n权重：`' + str(checkpoint) + '`\n\nSHA256：`'
        + chosen['checkpoint_sha256'] + '`\n\n正式评测复用：' + str(reused is not None)
        + '\n\nA0没有重新训练；旧文件只读。缺少正式评测时仅在新目录补评测。\n', encoding='utf-8')
    return reference


def existing_evaluation(root, chosen, contract, evidence, args):
    folder = root / 'final_evaluation'
    if not (folder / 'summary.json').is_file():
        return None
    raw = read_json(folder / 'summary.json')
    if raw.get('status') != 'complete' or not raw.get('full_validation') or raw.get('conditions') != ['clean', 'fog', 'snow']:
        return None
    summary = complete(folder / 'summary.json')
    protocol = read_json(folder / 'protocol.json')
    from support import DEFINITIONS
    if (summary['frame_count'] != 829 or summary['checkpoint_sha256'] != chosen['checkpoint_sha256']
            or summary['frozen_d1c_gt_identity'] != contract['frozen_gt'] or protocol['protocols'] != DEFINITIONS
            or protocol['input_manifest_identity'] != evidence['input_manifest']['identity']
            or summary['associated_d1d_training']['training_contract_sha256'] != sha256(root / 'contract.json')):
        raise ValueError('A0 formal evaluation protocol/data/checkpoint cannot be matched')
    from evaluation import inherited
    inherited.assert_frozen_gt(read_json(folder / 'gt_manifest.json'), frozen_c(args), ['clean', 'fog', 'snow'], 829)
    for row in read_json(folder / 'prediction_manifest.json')['files']:
        if sha256(no_test(row['path'])) != row['sha256']:
            raise ValueError('A0 formal prediction cache changed')
    for weather in ('clean', 'fog', 'snow'):
        if set(summary['results'][weather]['AP']) != {'O', 'O*', 'F'}:
            raise ValueError('A0 evaluation lacks O/O*/F protocols')
    return folder


def initial_state(args):
    import torch
    reference = complete(Path(args.experiment_root) / 'baseline_reference.json')
    if sha256(reference['initial_state']) != reference['initial_state_sha256']:
        raise ValueError('A0 initialization changed since preflight')
    return torch.load(reference['initial_state'], map_location='cpu', weights_only=False)


def initialization_reference(args):
    reference = complete(Path(args.experiment_root) / 'baseline_reference.json')
    return dict(a0_root=reference['a0_root'], initial_state_sha256=reference['initial_state_sha256'],
                baseline_reference_identity=reference['identity'], source_epoch=0, trained_A0_weight_used=False)


def guard_reference(args):
    reference = complete(Path(args.experiment_root) / 'baseline_reference.json')
    check_identity(reference)
    for name, digest in dict(reference['artifacts'], **{
            reference['checkpoint']: reference['checkpoint_sha256'],
            reference['initial_state']: reference['initial_state_sha256']}).items():
        if sha256(no_test(name)) != digest:
            raise ValueError('Audited A0 artifact changed: ' + name)
    path = Path(args.experiment_root) / 'A0/evaluation_reference.json'
    if path.exists():
        evaluation = complete(path)
        if sha256(Path(evaluation['root']) / 'summary.json') != evaluation['summary_sha256']:
            raise ValueError('Audited A0 evaluation changed')


def initialize_model(model, args):
    import torch
    if args.variant not in METHOD['trainable_variants']:
        raise ValueError('A0 training forbidden')
    source = initial_state(args)['model']
    target = model.state_dict()
    if set(k for k in target if not k.startswith('ggca.')) != set(source):
        raise ValueError('A0/shared model keys differ; no partial loading allowed')
    for key, value in source.items():
        if value.shape != target[key].shape or value.dtype != target[key].dtype:
            raise ValueError('A0/shared tensor shape/dtype differs: ' + key)
        target[key] = value.to(target[key].device)
    model.load_state_dict(target, strict=True)
    if not all(torch.equal(model.state_dict()[k].cpu(), v) for k, v in source.items()):
        raise ValueError('Shared initial tensors differ after loading')
    atomic_json(Path(args.output_root) / 'initialization_audit.json', dict(status='complete',
        source=str(Path(args.a0_root) / 'resume_states/00.pth'), source_epoch=0,
        shared_tensors_exact=True, shared_tensor_count=len(source), trained_A0_weight_used=False))
    return model


def restore_initial_rng(args):
    from checkpoint import restore_rng
    restore_rng(initial_state(args)['rng'])
