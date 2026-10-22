"""Server bridge to the frozen weather trainer and frozen D-1D evaluator."""
import argparse
import copy
from pathlib import Path
import sys
from adapters import adapted_run
from e01_common import HERE, PROJECT, SETTINGS, atomic, read, run, sha, sources


def defaults(cli):
    sys.path.insert(0, str(PROJECT / 'v2xr_weather_l012'))
    import auditlib
    args = argparse.Namespace(**auditlib.DEFAULTS)
    args.output_root = str(cli.model_root)
    args.train_root = str(cli.train_root)
    args.variant, args.recipe = cli.variant, cli.recipe
    args.smoke, args.resume = cli.smoke, True
    args.parent_root = None
    return args


def install(cli):
    """Install only in-memory adapters in a fresh worker, with original byte checks intact."""
    args = defaults(cli)
    import auditlib
    original_sources = auditlib.sources
    original_sources()
    new_sources = sources()
    auditlib.CONFIG = dict(copy.deepcopy(auditlib.CONFIG), e01=dict(recipe=cli.recipe,
        fractions=SETTINGS['recipes'][cli.recipe], draw_order=SETTINGS['draw_order'][cli.recipe],
        settings=SETTINGS, training_changes='per-CAV weather suffix only'))
    def own_sources():
        return dict(frozen=original_sources(), extension=new_sources,
            recipe=cli.recipe, scope='E1 C/D; original loss/architecture/training loop')
    auditlib.sources = own_sources
    import training
    import data
    import checkpoint
    import evaluation
    # Imports above see the extended immutable contract; frozen source checks still check original bytes.
    from e1_data import build_adapter, inventory
    source = Path(args.runtime_root) / 'opencood/data_utils/datasets/opv2v/basedataset.py'
    data.build_train = lambda h, m: build_adapter(data, h, m, cli.recipe, source)
    data.paired_inventory = inventory
    evaluation.clean_ap = lambda a, c, d: clean_ap(cli, a, c, d)
    training.sources = own_sources
    return args, auditlib, training, checkpoint, evaluation


def evaluate(cli):
    args, audit, _, _, evaluation = install(cli)
    audit.gpu()
    # Frozen generator verifies exact keys, shapes, dtypes and every loaded tensor.
    from e1_targets import observed_evaluation
    result = observed_evaluation(cli, args, evaluation)
    if result['frame_count'] != (12 if cli.smoke else 829):
        raise ValueError('Evaluation returned the wrong frame count')
    return result


def clean_ap(cli, args, checkpoint, destination):
    command = [sys.executable, '-B', HERE / 'worker.py', 'evaluate',
        '--model-root', args.output_root, '--train-root', args.train_root,
        '--recipe', cli.recipe, '--variant', cli.variant, '--checkpoint', checkpoint,
        '--destination', destination, '--clean-only']
    if cli.smoke:
        command.append('--smoke')
    run(command, destination.with_suffix('.log'))
    result = read(destination / 'summary.json')
    if (result['status'] != 'complete' or result['checkpoint_sha256'] != sha(checkpoint)
            or result['conditions'] != ['clean'] or result['frame_count'] != (12 if cli.smoke else 829)):
        raise ValueError('Clean selection evidence differs')
    return result


def exact(a, b, path='state'):
    import numpy as np
    import torch
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor) or a.shape != b.shape or a.dtype != b.dtype or not torch.equal(a.cpu(), b.cpu()):
            raise ValueError('State roundtrip differs: '+path)
    elif isinstance(a, np.ndarray):
        if not isinstance(b, np.ndarray) or a.dtype != b.dtype or not np.array_equal(a, b):
            raise ValueError('State roundtrip differs: '+path)
    elif isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b):
            raise ValueError('State keys differ: '+path)
        for k in a:
            exact(a[k], b[k], path+'/'+str(k))
    elif isinstance(a, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            raise ValueError('State sequence differs: '+path)
        for i, (x, y) in enumerate(zip(a, b)):
            exact(x, y, path+'/'+str(i))
    elif type(a) is not type(b) or a != b:
        raise ValueError('State scalar differs: '+path)


def install_restore_check(checkpoint, root, smoke):
    """Exercise the actual original load function, including Adam/RNG, without a training update."""
    import torch
    original_restore, original_save, original_state = checkpoint.restore, checkpoint.save, checkpoint.state
    context = {}
    def checked_restore(path, model, optimizer, scheduler, criterion, contract, epoch):
        expected = torch.load(path, map_location='cpu', weights_only=False)
        original_restore(path, model, optimizer, scheduler, criterion, contract, epoch)
        actual = original_state(model, optimizer, scheduler, criterion, epoch, contract)
        exact(expected, actual)
        atomic(root / 'restore_audit' / f'{epoch:02d}.json', dict(status='complete',
            source_sha256=sha(path), exact_model_Adam_scheduler_criterion_RNG=True,
            epoch=epoch, contract=contract, scope='actual restore; no optimizer update'))
    def state(*a, **kw):
        result = original_state(*a, **kw)
        context['objects'] = a
        return result
    def save(path, value):
        original_save(path, value)
        if isinstance(value, dict) and 'rng' in value and 'model' in value and (smoke or value['epoch'] == 0):
            model, optimizer, scheduler, criterion, epoch, contract = context['objects']
            checked_restore(path, model, optimizer, scheduler, criterion, contract, epoch)
    checkpoint.restore, checkpoint.state, checkpoint.save = checked_restore, state, save


def train(cli):
    args, audit, training, checkpoint, evaluation = install(cli)
    audit.gpu()
    root = Path(args.output_root)
    args.prepared = training.prepare(args)
    import torch
    import models
    original_create = models.create
    def create(hypes, variant):
        model = original_create(hypes, variant)
        if variant == 'D':
            croot = root.parent / 'C'
            initial = croot / 'resume_states/00.pth'
            if sha(initial) != read(croot / 'resume_states/00.json')['sha256']:
                raise ValueError('Paired C initial state changed')
            base = torch.load(initial, map_location='cpu', weights_only=False)['model']
            actual = model.state_dict()
            shared = {k: v for k, v in actual.items() if not k.startswith('gspr.')}
            exact(base, shared, 'shared_initial_backbone')
            atomic(root / 'initialization_audit.json', dict(status='complete', exact=True,
                C_initial_sha256=sha(initial), shared_tensor_count=len(base),
                seed=SETTINGS['seed'], note='D-only GSPR initialization preserves shared RNG'))
        return model
    models.create = create
    install_restore_check(checkpoint, root, cli.smoke)
    original_atomic = training.atomic_json
    def commit(path, value):
        path = Path(path)
        if path.name == 'complete.json' and path.parent.parent.name == 'epochs':
            if cli.variant == 'D':
                from e1_data import compare_epoch
                peer = root.parent / 'C/epochs' / path.parent.name / 'complete.json'
                crow = audit.complete(peer)
                if sha(crow['trace']) != crow['trace_sha256']:
                    raise ValueError('Paired C trace changed')
                pairing = compare_epoch(crow['trace'], value['trace'], 32 if cli.smoke else 8080)
                atomic(root / 'pairing' / (path.parent.name+'.json'), pairing, immutable=True)
                if crow['sampling_counts'] != value['sampling_counts'] or crow['gt_total'] != value['gt_total']:
                    raise ValueError('C/D epoch sampling/GT totals differ')
            value['weather_fractions'] = {w: value['sampling_counts'][w]/sum(value['sampling_counts'].values())
                for w in SETTINGS['weather_order']}
        original_atomic(path, value)
    training.atomic_json = commit
    adapted, adapter = adapted_run(training)
    root.mkdir(parents=True, exist_ok=True)
    atomic(root / 'adapter.json', dict(status='complete', training=adapter,
        original_training_sha256=sha(PROJECT / 'v2xr_weather_l012/training.py'),
        original_retrieve_sha256=sha(Path(args.runtime_root) / 'opencood/data_utils/datasets/opv2v/basedataset.py'),
        per_CAV_draw=SETTINGS['draw_order'][cli.recipe]), immutable=True)
    # Compile after replacing the commit callback: the function globals hold this callback.
    adapted(args)
    if cli.smoke:
        rows = checkpoint.committed(root, 2, read(root / 'contract.json')['identity'])
        if len(rows) != 2 or any(r['train_updates'] != 4 for r in rows):
            raise ValueError('Smoke must complete two epochs / eight updates')
        best = min(rows, key=lambda r: (-r['AP70'], r['epoch']))
        atomic(root / 'smoke_selected.json', dict(status='complete', eligible_for_formal_selection=False,
            checkpoint=best['checkpoint'], checkpoint_sha256=best['checkpoint_sha256'],
            epoch=best['epoch'], AP70=best['AP70'], validation_frames=12))
    else:
        evaluation.selection(root)
