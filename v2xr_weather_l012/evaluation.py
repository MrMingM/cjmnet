"""Call the frozen official inference generator and O/O*/F replay, without new AP code."""
from pathlib import Path
import subprocess
import sys
from auditlib import (DEFAULTS, HERE, SELECTION, atomic_json, bound_json, complete,
                      identity, load_frozen, read_json, sha256, sources)

inherited = load_frozen('evaluation', 'evaluation.py')


def evaluate(args, checkpoint, destination, clean_only=False, smoke=False):
    own = dict(sources=sources(), variant=args.variant)
    summary = inherited.evaluate(args, own, checkpoint, destination,
        conditions=('clean',) if clean_only else ('clean', 'fog', 'snow'), limit=12 if smoke else 0)
    summary.update(stage='V2XR_WEATHER_L012', main_protocol='O', model_variant=args.variant)
    if args.variant.endswith('FiLM'):
        from models import OBSERVED_FILM
        path = Path(destination) / 'film_coefficients.json'
        if OBSERVED_FILM:
            atomic_json(path, dict(status='complete', observations=OBSERVED_FILM,
                scope='actual uncached inference forwards in this invocation; completed cached records retain their original report',
                input='current sample spatial feature mean/std only'))
        if path.exists():
            summary['output_hashes']['film_coefficients.json'] = sha256(path)
    atomic_json(Path(destination) / 'summary.json', summary)
    return summary


def clean_ap(args, checkpoint, destination):
    command = [sys.executable, '-B', str(HERE / 'main.py'), 'evaluate', '--variant', args.variant,
        '--checkpoint', str(checkpoint), '--evaluation-root', str(destination), '--clean-only', '--resume']
    if args.smoke:
        command.append('--smoke')
    for key in DEFAULTS:
        command += ['--' + key.replace('_', '-'), str(getattr(args, key))]
    with destination.with_suffix('.log').open('a', encoding='utf-8') as log:
        subprocess.run(command, check=True, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    summary = complete(destination / 'summary.json')
    if (summary['checkpoint_sha256'] != sha256(checkpoint) or summary['conditions'] != ['clean'] or
            summary['frame_count'] != (12 if args.smoke else 829)):
        raise ValueError('Clean selection evidence mismatch')
    return summary


def selection(root):
    contract = read_json(root / 'contract.json')
    if identity({k: v for k, v in contract.items() if k != 'identity'}) != contract['identity']:
        raise ValueError('Training contract content/identity differs')
    if contract['smoke_only'] or contract['sources'] != sources():
        raise ValueError('Smoke cannot select a formal checkpoint')
    if contract['config_sha256'] != sha256(root / 'config.yaml'):
        raise ValueError('Selected model configuration changed')
    from checkpoint import committed
    rows = committed(root, contract['epochs'], contract['identity'])
    if len(rows) != contract['epochs']:
        raise ValueError('All registered epochs must be committed before selection')
    for row in rows:
        ap = complete(Path(row['clean_summary']))
        if (not ap['full_validation'] or ap['conditions'] != ['clean'] or ap['frame_count'] != 829 or
                ap['checkpoint_sha256'] != row['checkpoint_sha256'] or
                row['AP70'] != ap['results']['clean']['AP']['O']['ap70']):
            raise ValueError('Invalid Clean-only checkpoint selection')
    best = min(rows, key=lambda row: (-row['AP70'], row['epoch']))
    chosen = dict(status='complete', epoch=best['epoch'], checkpoint=best['checkpoint'],
        checkpoint_sha256=best['checkpoint_sha256'], clean_O_AP70=best['AP70'],
        policy=SELECTION, weather_used=False, candidate_count=len(rows),
        candidates=[dict(epoch=r['epoch'], AP70=r['AP70'], sha256=r['checkpoint_sha256']) for r in rows],
        training_contract_sha256=sha256(root / 'contract.json'))
    chosen['identity'] = identity(chosen)
    bound_json(root / 'selected.json', chosen)
    return chosen
