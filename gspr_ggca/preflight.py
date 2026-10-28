"""Validate real data with the existing engine, then audit the specified A0."""
from pathlib import Path
from auditlib import atomic_json, complete, frozen_c, identity, read_json, sha256, sources


def run(args):
    from baseline import audit
    manifest, evidence = args.prepared
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    reference = audit(args, manifest, evidence)
    refs = dict(reference['artifacts'])
    refs[reference['checkpoint']] = reference['checkpoint_sha256']
    refs[reference['initial_state']] = reference['initial_state_sha256']
    train = Path(args.train_root)
    for row in manifest['files']:
        refs[str(train / 'train' / row['relative'])] = row['sha256']
    for file in ('manifest.json', 'data_acceptance.json', 'plan.json'):
        refs[str(train / file)] = sha256(train / file)
    for row in evidence['input_manifest']['files']:
        refs[row['source']] = row['sha256']
        refs[str(Path(evidence['input_manifest']['overlay']) / row['relative'])] = row['sha256']
    atomic_json(root / 'external_inputs.json', refs)
    trees = {}
    for folder in (train / 'train', Path(evidence['input_manifest']['overlay']), Path(args.view_root) / 'validate'):
        trees[str(folder)] = dict(files=sorted(p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()),
                                 directories=sorted(p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_dir()))
    atomic_json(root / 'input_trees.json', trees)
    atomic_json(root / 'summary.json', dict(status='complete', sources=sources(),
        data_identity=manifest['identity'], input_identity=evidence['input_manifest']['identity'],
        frozen_gt=frozen_c(args)['gt']['identity'], baseline_reference_identity=reference['identity']))


def reference_evaluation(args):
    from evaluation import evaluate
    from auditlib import bound_json
    reference = complete(Path(args.experiment_root) / 'baseline_reference.json')
    folder = Path(reference['evaluation_root'])
    if reference['evaluation_reused']:
        summary = complete(folder / 'summary.json')
    else:
        args.variant = 'A0'
        summary = evaluate(args, reference['checkpoint'], folder)
    if summary['checkpoint_sha256'] != reference['checkpoint_sha256'] or summary['frame_count'] != 829:
        raise ValueError('A0 evaluation is not the audited selected weight/full validation')
    record = dict(status='complete', root=str(folder), summary_sha256=sha256(folder / 'summary.json'),
                  checkpoint_sha256=reference['checkpoint_sha256'], reused=reference['evaluation_reused'],
                  a0_training_executed=False)
    bound_json(Path(args.output_root) / 'evaluation_reference.json', record)
    return [folder / 'summary.json', Path(args.experiment_root) / 'baseline_reference.json']
