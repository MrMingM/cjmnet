"""Official O model selection; frozen inference/GT/AP reused for the new GSPR model."""
from pathlib import Path

from auditlib import (SELECTION, SETTINGS, atomic_json, bound_json, complete, identity,
                      load_frozen, read_json, sha256, sources)

inherited = load_frozen('evaluation', 'evaluation.py')


def selection(args):
    root = Path(args.output_root)
    if read_json(root / 'selection_policy.json') != SELECTION:
        raise ValueError('Registered GSPR O selection policy changed')
    contract = read_json(root / 'contract.json')
    if contract['smoke_only'] or contract['sources'] != sources() or contract['gspr_adapter'] != SETTINGS:
        raise ValueError('Selection requires completed full GSPR training with unchanged sources')
    candidates = []
    for epoch in range(1, 21):
        row = complete(root / 'epochs' / f'{epoch:02d}' / 'complete.json')
        if (row['smoke_only'] or row['contract_identity'] != contract['identity'] or row['epoch'] != epoch
                or row['checkpoint_sha256'] != sha256(root / f'net_epoch{epoch}.pth')):
            raise ValueError('Epoch/checkpoint contract differs')
        ap = complete(Path(row['clean_ap_summary']))
        if (sha256(row['clean_ap_summary']) != row['clean_ap_summary_sha256']
                or ap['conditions'] != ['clean'] or ap['frame_count'] != 829
                or ap['checkpoint_sha256'] != row['checkpoint_sha256']
                or ap['gspr_adapter'] != SETTINGS):
            raise ValueError('Selection requires complete matching Clean-only GSPR AP')
        candidates.append(dict(epoch=epoch, AP70=ap['results']['clean']['AP']['O']['ap70'],
            checkpoint_sha256=row['checkpoint_sha256'], clean_ap_summary_sha256=row['clean_ap_summary_sha256']))
    best = min(candidates, key=lambda r: (-r['AP70'], r['epoch']))
    chosen = dict(status='complete', policy=SELECTION, epoch=best['epoch'],
        checkpoint=str(root / f'net_epoch{best["epoch"]}.pth'), checkpoint_sha256=best['checkpoint_sha256'],
        clean_O_AP70=best['AP70'], candidate_count=20, candidate_metrics=candidates,
        training_contract_sha256=sha256(root / 'contract.json'),
        policy_sha256=sha256(root / 'selection_policy.json'), weather_used=False)
    chosen['identity'] = identity(chosen)
    bound_json(root / 'selected.json', chosen)
    return chosen


def evaluate(args, own, checkpoint, destination, conditions=('clean', 'fog', 'snow'), limit=0):
    # The inherited evaluator installs auditlib.runtime's explicit model adapter in this fresh process.
    summary = inherited.evaluate(args, own, checkpoint, destination, conditions=conditions, limit=limit)
    if summary.get('main_protocol') != 'O':
        summary.update(stage='V2XR_GSPR_CLEAN_O', main_protocol='O', gspr_adapter=SETTINGS,
                       actual_model_class='integration.PointPillarGsprOfficial')
        atomic_json(Path(destination) / 'summary.json', summary)
    if summary['gspr_adapter'] != SETTINGS:
        raise ValueError('Completed GSPR evaluation settings changed')
    return summary


def final(args, own):
    chosen = selection(args)
    root = Path(args.output_root)
    name = f'final_evaluation_smoke{args.limit_frames}' if args.limit_frames else 'final_evaluation'
    summary = evaluate(args, own, chosen['checkpoint'], root / name, limit=args.limit_frames)
    if not args.limit_frames:
        from results import comparison
        comparison(args, summary)
    return summary
