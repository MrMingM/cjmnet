"""One official inference cache per condition, replayed through frozen O/O*/F functions."""
import copy
import gc
import os
from pathlib import Path

from .common import bound, identity, read, sha, write
from .runtime import arguments, evidence, frozen_gt


class SmokeInterruption(RuntimeError):
    """Expected one-time recovery fault, confined to independent smoke output."""


def run(output, group, epoch, smoke=False, interrupt_once=False):
    import torch
    from predict import generate
    from evaluate import replay
    from support import DEFINITIONS
    root = Path(output)
    ep = root / ('smoke' if smoke else 'groups') / group / 'epochs' / f'{epoch:02d}'
    train = read(ep / 'train_complete.json')
    for name, digest in train['output_hashes'].items():
        if sha(ep / name) != digest:
            raise ValueError('Training commit changed before evaluation: ' + name)
    ev, args = evidence(root), arguments(root)
    destination = ep / 'evaluation'
    destination.mkdir(exist_ok=True)
    count = read(root / 'config.json')['smoke_frames'] if smoke else 829
    if os.environ.get('PYTHONHASHSEED') != '20261016':
        raise ValueError('Frozen evaluation requires its separate pre-launch PYTHONHASHSEED=20261016')
    contract = dict(parent_contract=read(root / 'contract.json')['identity'], group=group, epoch=epoch,
        checkpoint_sha256=sha(ep / 'detector.pth'), frame_keys=ev['b']['selected_frame_keys'][:count],
        smoke_only=smoke, protocols=DEFINITIONS, seed=20261016,
        frozen_F_identity=read(root / 'contract.json')['inputs']['frozen_F'])
    contract['identity'] = identity(contract)
    bound(destination / 'contract.json', contract)
    if (destination / 'summary.json').exists():
        result = read(destination / 'summary.json')
        if result['contract_identity'] != contract['identity']:
            raise ValueError('Evaluation completion contract differs')
        for name, digest in result['output_hashes'].items():
            if sha(destination / name) != digest:
                raise ValueError('Completed evaluation artifact changed: ' + name)
        prediction = read(destination / 'prediction_manifest.json')
        for row in prediction['files']:
            if sha(row['path']) != row['sha256']:
                raise ValueError('Completed immutable prediction changed')
        frozen_gt(root, read(destination / 'gt_manifest.json'), count)
        return result
    alias = destination / 'detector_only_model'
    alias.mkdir(exist_ok=True)
    config = Path(args.model_dir) / 'config.yaml'
    from .common import atomic_bytes
    if not (alias / 'config.yaml').exists():
        atomic_bytes(alias / 'config.yaml', config.read_bytes())
    if sha(alias / 'config.yaml') != sha(config):
        raise ValueError('Official evaluation config changed')
    from common import link
    for name in ('0.pth', 'net_epoch0.pth'):
        link(ep / 'detector.pth', alias / name)
    args.model_dir = str(alias)
    manifests = {}
    for condition in ('clean', 'fog', 'snow'):
        manifests[condition] = generate(condition, destination, args, ev, contract)
        if condition == 'clean' and interrupt_once:
            if not smoke:
                raise ValueError('Fault injection is forbidden in formal evaluation')
            marker = destination / 'expected_interruption.json'
            if not marker.exists():
                write(marker, dict(status='expected_smoke_interruption',
                    train_commit_sha256=sha(ep / 'train_complete.json'), checkpoint_sha256=sha(ep / 'detector.pth'),
                    clean_prediction_manifest_sha256=sha(destination / 'predictions/clean/manifest.json')))
                raise SmokeInterruption('Expected smoke interruption after committed Clean cache; resume must not retrain')
        gc.collect(); torch.cuda.empty_cache()
    results, prediction, gt = replay(destination, ev, contract, manifests)
    frozen_gt(root, gt, count)
    if (destination / 'expected_interruption.json').exists():
        old = read(destination / 'expected_interruption.json')
        if (old['train_commit_sha256'] != sha(ep / 'train_complete.json')
                or old['checkpoint_sha256'] != sha(ep / 'detector.pth')
                or old['clean_prediction_manifest_sha256'] != sha(destination / 'predictions/clean/manifest.json')):
            raise ValueError('Evaluation recovery changed committed training/weights/Clean predictions')
        write(destination / 'recovery_check.json', dict(status='complete', resumed_predictions=True,
            training_and_weights_unchanged=True, cached_clean_prediction_manifest_unchanged=True))
    summary = dict(status='complete', contract_identity=contract['identity'], group=group, epoch=epoch,
        full_validation=count == 829, smoke_only=smoke, frame_count=count,
        checkpoint_sha256=contract['checkpoint_sha256'], results=results,
        metric='BEV AP, VOC2010, global_sort=False, fixed frame order',
        reused_functions=['v2xr_protocol_audit.predict.generate', 'v2xr_protocol_audit.evaluate.replay'])
    summary['output_hashes'] = {n: sha(destination / n) for n in
        ('contract.json', 'gt_manifest.json', 'prediction_manifest.json', 'per_frame_metrics.jsonl', 'removed_gt_metrics.jsonl')}
    write(destination / 'protocol.json', dict(**contract, metric=summary['metric'], reused=summary['reused_functions']))
    write(destination / 'summary.json', summary)
    from .report import evaluation_markdown
    from .common import atomic_bytes
    atomic_bytes(destination / 'RESULTS.md', evaluation_markdown(summary).encode())
    return summary
