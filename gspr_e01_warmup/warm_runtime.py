"""Keep original E1 computation; pause after committed formal epochs, without early selection."""
from pathlib import Path
import sys

from warm_common import HERE, POLICY, bridge, read, sha


def activate(cli):
    base = bridge()
    import e1_runtime
    from warm_adapter import adapted_run
    limit = POLICY['gate_epochs'] if cli.stage == 'warmup' else None
    e1_runtime.adapted_run = lambda module: adapted_run(module, limit)

    def clean_ap(original_cli, args, checkpoint, destination):
        command = [sys.executable, '-B', HERE / 'worker.py', 'evaluate', '--model-root', args.output_root,
            '--train-root', args.train_root, '--original-root', cli.original_root,
            '--recipe', cli.recipe, '--variant', cli.variant, '--checkpoint', checkpoint,
            '--destination', destination, '--clean-only']
        base.run(command, destination.with_suffix('.log'))
        result = read(destination / 'summary.json')
        if (result['status'] != 'complete' or result['checkpoint_sha256'] != sha(checkpoint)
                or result['conditions'] != ['clean'] or result['frame_count'] != 829 or not result['full_validation']):
            raise ValueError('Original full Clean selection evaluation is incomplete')
        return result
    e1_runtime.clean_ap = clean_ap
    if limit is not None:
        original_install = e1_runtime.install
        def install(original_cli):
            args, audit, training, checkpoint, evaluation = original_install(original_cli)
            original_selection = evaluation.selection
            def deferred(root):
                rows = checkpoint.committed(Path(root), 20, read(Path(root) / 'contract.json')['identity'])
                if 2 <= len(rows) < 20:
                    return None  # No selected file, scores or receipt is synthesized.
                return original_selection(root)
            evaluation.selection = deferred
            return args, audit, training, checkpoint, evaluation
        e1_runtime.install = install
        original_restore_audit = e1_runtime.install_restore_check
        # This flag only audits actual full-state save/load; cli.smoke remains False everywhere.
        e1_runtime.install_restore_check = lambda checkpoint, root, smoke: original_restore_audit(checkpoint, root, True)
    return e1_runtime


def audit_warmup(root, recipe, variant):
    """Only immutable first-two-epoch artifacts; later epochs cannot invalidate the gate."""
    base = bridge()
    root = Path(root)
    contract = read(root / 'contract.json')
    if (base.frozen_identity({k: v for k, v in contract.items() if k != 'identity'}) != contract['identity']
            or contract['smoke_only'] or contract['epochs'] != 20 or contract['variant'] != variant
            or contract['settings']['e01']['recipe'] != recipe
            or contract['settings']['e01']['settings']['formal_startup'] != POLICY
            or sha(root / 'config.yaml') != contract['config_sha256']):
        raise ValueError('Warmup must be the original formal 20-epoch model contract')
    files = {}
    def remember(path, digest=None):
        path = base.safe(path).resolve(strict=True)
        if root.resolve() not in path.parents:
            raise ValueError('Formal artifact outside its own model: ' + str(path))
        actual = sha(path)
        if digest is not None and actual != digest:
            raise ValueError('Immutable formal artifact changed: ' + str(path))
        files[str(path)] = actual
        return path
    for name in ('contract.json', 'config.yaml', 'selection_policy.json', 'adapter.json'):
        remember(root / name)
    for epoch in range(3):
        audit_path = remember(root / 'restore_audit' / f'{epoch:02d}.json')
        audit = read(audit_path)
        state = root / 'resume_states' / f'{epoch:02d}.pth'
        remember(state, audit['source_sha256'])
        if (audit['status'] != 'complete' or audit['epoch'] != epoch or audit['contract'] != contract['identity']
                or not audit['exact_model_Adam_scheduler_criterion_RNG']):
            raise ValueError('Exact original full-state restore proof is missing')
    initial = read(remember(root / 'resume_states/00.json'))
    remember(root / 'resume_states/00.pth', initial['sha256'])
    for epoch in (1, 2):
        marker = remember(root / 'epochs' / f'{epoch:02d}' / 'complete.json')
        row = read(marker)
        if (row['status'] != 'complete' or row['epoch'] != epoch or row['contract'] != contract['identity']
                or row['train_updates'] != 1010 or row['consumed_input_frames'] != 8080
                or row['accepted_train_frames'] != 8083 or row['dropped_frames'] != 3 or row['smoke_only']):
            raise ValueError('Formal full-data epoch is incomplete')
        for field in ('checkpoint', 'resume_state', 'trace', 'clean_summary'):
            remember(row[field], row[field + '_sha256'])
        summary = read(row['clean_summary'])
        if (summary['status'] != 'complete' or summary['frame_count'] != 829 or not summary['full_validation']
                or summary['conditions'] != ['clean'] or summary['checkpoint_sha256'] != row['checkpoint_sha256']
                or summary['frozen_d1c_gt_identity'] != contract['frozen_gt']
                or summary['results']['clean']['counts']['fixed_gt'] != 6224
                or summary['results']['clean']['counts']['official_gt'] != 6224
                or summary['results']['clean']['AP'] != row['clean_AP']
                or summary['results']['clean']['AP']['O']['ap70'] != row['AP70']):
            raise ValueError('Full original Clean evaluation/GT identity differs')
        destination = Path(row['clean_summary']).parent
        expected_outputs = {'contract.json', 'protocol.json', 'prediction_manifest.json', 'gt_manifest.json',
            'per_frame_metrics.jsonl', 'removed_gt_metrics.jsonl', 'RESULTS.md', 'predictions/clean/manifest.json',
            'stat_clean_O.json', 'stat_clean_Ostar.json', 'stat_clean_F.json'}
        if set(summary['output_hashes']) != expected_outputs:
            raise ValueError('Clean evaluation output manifest differs from the frozen evaluator')
        for relative, digest in summary['output_hashes'].items():
            remember(destination / relative, digest)
        predictions = read(destination / 'prediction_manifest.json')['files']
        if len(predictions) != 829 or {(r['condition'], r['index']) for r in predictions} != {('clean', i) for i in range(829)}:
            raise ValueError('Clean prediction cache is incomplete')
        for record in predictions:
            remember(record['path'], record['sha256'])
        gt = read(destination / 'gt_manifest.json')['frames']
        if len(gt) != 829 or {(r['condition'], r['index']) for r in gt} != {('clean', i) for i in range(829)}:
            raise ValueError('Clean GT manifest is incomplete')
        if variant == 'D':
            pairing = read(remember(root / 'pairing' / f'{epoch:02d}.json'))
            peer = read(root.parent / 'C/epochs' / f'{epoch:02d}' / 'complete.json')
            if (not pairing['exact'] or pairing['consumed_frames'] != 8080
                    or pairing['D_trace_sha256'] != row['trace_sha256']
                    or pairing['C_trace_sha256'] != peer['trace_sha256']):
                raise ValueError('C/D full-data trace pairing differs')
    if variant == 'D':
        initial_audit = read(remember(root / 'initialization_audit.json'))
        if not initial_audit['exact'] or initial_audit['C_initial_sha256'] != sha(root.parent / 'C/resume_states/00.pth'):
            raise ValueError('C/D shared initial model was not verified')
    return dict(status='formal_warmup_complete', recipe=recipe, variant=variant,
        contract_identity=contract['identity'], completed_gate_epochs=[1, 2], total_formal_epochs=20,
        validation_frames=829, updates_per_epoch=1010, no_BN_changes=True,
        first_two_epochs_count_towards_twenty=True, files=files)


def publish_warmup(root, recipe, variant):
    base = bridge()
    result = audit_warmup(root, recipe, variant)
    path = Path(root) / 'formal_warmup/receipt.json'
    base.atomic(path, result, immutable=True)
    return path
