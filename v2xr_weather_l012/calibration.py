"""Fog-only, no-augmentation, fixed-order BN buffer recalibration."""
import json
import os
from pathlib import Path
import time
from auditlib import CONFIG, atomic_json, bound_json, complete, identity, jsonable, sha256, sources


def run(args):
    import torch
    from torch.utils.data import DataLoader, Subset
    from opencood.tools import train_utils
    from checkpoint import save, seed
    from data import build_train
    from evaluation import selection
    from models import bn_difference, create, start_bn
    from training import config
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    manifest, evidence = args.prepared
    parent = selection(Path(args.parent_root)) if not args.smoke else complete(Path(args.parent_root) / 'smoke_checkpoint.json')
    if sha256(parent['checkpoint']) != parent['checkpoint_sha256']:
        raise ValueError('BN parent checkpoint changed')
    hypes = config(args, root)
    contract = dict(sources=sources(), variant=args.variant, settings=CONFIG['bn'],
        parent=parent, manifest_identity=manifest['identity'],
        train_data_acceptance=manifest['train_data_acceptance'],
        train_manifest_sha256=sha256(Path(args.train_root) / 'manifest.json'),
        validation_inputs_not_used_for_calibration=True, smoke_only=args.smoke,
        config_sha256=sha256(root / 'config.yaml'))
    contract['identity'] = identity(contract)
    bound_json(root / 'contract.json', contract)
    seed()
    dataset = build_train(hypes, manifest, fog=True)
    loader = DataLoader(Subset(dataset, range(2)) if args.smoke else dataset,
        batch_size=1, num_workers=16, shuffle=False, pin_memory=True, drop_last=False,
        prefetch_factor=4, collate_fn=dataset.collate_batch_train)
    model = create(hypes, args.variant).to('cuda:0')
    before = torch.load(parent['checkpoint'], map_location='cpu', weights_only=False)
    model.load_state_dict(before, strict=True)
    model.eval()
    probe = None
    initial_output = None
    trace = root / 'calibration_reads.jsonl'
    started = time.monotonic()
    count = 0
    with torch.no_grad(), trace.with_suffix('.jsonl.tmp').open('w', encoding='utf-8', newline='\n') as stream:
        layers = None
        for batch in loader:
            audits = batch.pop('_audit')
            if len(audits) != 1 or audits[0]['index'] != count:
                raise ValueError('BN must traverse every training frame exactly once in fixed order')
            stream.write(json.dumps(jsonable(audits), allow_nan=False) + '\n')
            batch = train_utils.to_device(batch, torch.device('cuda:0'))
            if probe is None:
                probe = batch['ego']
                initial_output = {k: v.detach().cpu() for k, v in model(probe).items() if k in ('psm', 'rm')}
                layers = start_bn(model)
            result = model(batch['ego'])
            if any(not torch.isfinite(result[k]).all() for k in ('psm', 'rm')):
                raise ValueError('Nonfinite BN calibration output')
            count += 1
            if count % 100 == 0:
                print(f'{args.variant} BN calibration {count}/{len(loader)}', flush=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(trace.with_suffix('.jsonl.tmp'), trace)
    if count != (2 if args.smoke else CONFIG['bn']['frames']):
        raise ValueError('BN calibration did not visit all registered train frames')
    audit = bn_difference(before, model.state_dict(), layers)
    model.eval()
    with torch.no_grad():
        output = model(probe)
    changes = {k: float((output[k].cpu() - initial_output[k]).abs().max()) for k in initial_output}
    if any(not torch.isfinite(output[k]).all() for k in initial_output) or max(changes.values()) == 0:
        raise ValueError('Recalibrated BN did not change finite model outputs')
    checkpoint = root / 'calibrated.pth'
    save(checkpoint, model.state_dict())
    loaded = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if any(not torch.equal(v.detach().cpu(), loaded[k]) for k, v in model.state_dict().items()):
        raise ValueError('BN saved state mismatch')
    audit.update(status='complete', contract=contract['identity'], calibration_frames=count,
        output_change_max=changes, calibration_seconds=time.monotonic()-started,
        checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
        train_only=True, no_optimizer=True, no_grad=True, no_augmentation=True,
        same_statistics_for_all_evaluation_conditions=True, sampling_trace_sha256=sha256(trace),
        before_state_sha256=parent['checkpoint_sha256'], source_parameter_count=sum(p.numel() for p in model.parameters()))
    atomic_json(root / 'adaptation.json', audit)
    atomic_json(root / 'artifact.json', dict(status='complete', checkpoint=str(checkpoint),
        checkpoint_sha256=sha256(checkpoint), parent=parent, contract=contract['identity'],
        output_hashes={'adaptation.json': sha256(root / 'adaptation.json'), trace.name: sha256(trace)}))
