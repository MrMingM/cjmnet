"""Official single-GPU operations with an independent weather/full-state commit adapter."""
import json
import os
from pathlib import Path
import re
import time

from auditlib import (CONFIG, SELECTION, SETTINGS, atomic_json, bound_json, complete, frozen_c,
                      identity, jsonable, read_json, runtime, sha256, sources, verify_train)


def config(args, output):
    from opencood.hypes_yaml import yaml_utils
    import yaml
    source = Path(args.runtime_root) / 'opencood/hypes_yaml/V2X-R/LiDAR/V2XR_attfuse.yaml'
    text = source.read_text(encoding='utf-8')
    expected = yaml.safe_load(text)
    changes = dict(root_dir=str(Path(args.train_root) / 'train'),
        validate_dir=str(Path(args.view_root) / 'validate'), train_sim=True,
        eval_sim=False, sim_weather='_fog_0.060')
    expected.update(changes)
    for key, value in changes.items():
        text, count = re.subn('^' + key + r':[^\n]*$', lambda m: key + ': ' + json.dumps(value), text, flags=re.MULTILINE)
        if count != 1:
            raise ValueError('Unexpected official config field: ' + key)
    if yaml.safe_load(text) != expected:
        raise ValueError('Unregistered official configuration mutation')
    destination = output / 'config.yaml'
    if destination.exists() and destination.read_text(encoding='utf-8') != text:
        raise ValueError('Training YAML identity changed')
    destination.write_text(text, encoding='utf-8', newline='\n')
    h = yaml_utils.load_yaml(str(destination))
    if (h['train_params'] != dict(batch_size=8, epoches=20, eval_freq=1, save_freq=1) or
            h['optimizer'] != dict(core_method='Adam', lr=.002, args=dict(eps=1e-10, weight_decay=1e-4)) or
            h['lr_scheduler'] != dict(core_method='multistep', gamma=.1, step_size=[10, 15]) or
            h['preprocess']['args']['voxel_size'] != [.4, .4, 4] or
            h['model']['args']['pillar_vfe']['num_filters'] != [64] or
            h['model']['args']['point_pillar_scatter']['num_features'] != 64):
        raise ValueError('Official hyperparameter/shape contract differs')
    return h


def prepare(args):
    if os.environ.get('BACKEND') == 'cuda':
        from cuda_runtime import prepare as cuda_prepare
        return cuda_prepare(args)
    from provenance import verify_inputs
    evidence = verify_inputs(args)
    frozen_c(args)
    manifest = verify_train(args)
    # Retain the original link/hash validation for the validation-loss view too.
    from auditlib import load_frozen
    load_frozen('training', 'training.py').verify_clean_loss_view(args, evidence)
    train_keys = {(s['scene'], t) for s in manifest['scenes'] for t in s['timestamps']}
    if train_keys & {tuple(k) for k in evidence['b']['selected_frame_keys']}:
        raise ValueError('Training and validation frame keys overlap')
    runtime(args, evidence['a'])
    import torch
    if not torch.cuda.is_available() or not torch.version.hip or torch.cuda.device_count() != 1:
        raise ValueError('Require exactly one visible HIP GPU; device cuda:0')
    return manifest, evidence


def run(args):
    import torch
    from torch.utils.data import DataLoader, Subset
    from tensorboardX import SummaryWriter
    from opencood.data_utils.datasets import build_dataset
    from opencood.tools import train_utils
    from checkpoint import committed, restore, save, seed, state
    from data import build_train, paired_inventory
    from evaluation import clean_ap, selection
    from models import create, criterion, frozen_backbone, strict_backbone

    manifest, evidence = args.prepared
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    hypes = config(args, root)
    film = args.variant.endswith('FiLM')
    epochs = CONFIG['film']['epochs'] if film else CONFIG['epochs']
    base = None
    parent = None
    if film:
        parent = selection(Path(args.parent_root)) if not args.smoke else complete(Path(args.parent_root) / 'smoke_checkpoint.json')
        if sha256(parent['checkpoint']) != parent['checkpoint_sha256']:
            raise ValueError('FiLM parent checkpoint changed')
        base = torch.load(parent['checkpoint'], map_location='cpu', weights_only=False)
    contract = dict(sources=sources(), settings=CONFIG, variant=args.variant, epochs=epochs,
        policy=SELECTION, train_manifest_sha256=sha256(Path(args.train_root) / 'manifest.json'),
        train_manifest_identity=manifest['identity'], train_data_acceptance=manifest['train_data_acceptance'],
        pair_identity=paired_inventory(manifest, args.train_root)['identity'],
        config_sha256=sha256(root / 'config.yaml'), smoke_only=args.smoke,
        parent=parent, input_identity=evidence['input_manifest']['identity'],
        upstream_commit=evidence['a']['upstream_commit'],
        frozen_gt=frozen_c(args)['gt']['identity'],
        environment=dict(versions=evidence['a']['versions'], hip=torch.version.hip,
            gpu=torch.cuda.get_device_name(0), backend=os.environ.get('BACKEND','rocm'),
            cuda=torch.version.cuda, physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES',os.environ.get('ROCR_VISIBLE_DEVICES','platform default')),
            PYTHONHASHSEED=os.environ['PYTHONHASHSEED'],
            threads={k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
            cudnn_benchmark=torch.backends.cudnn.benchmark, cudnn_deterministic=torch.backends.cudnn.deterministic,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32, matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32),
        loader=dict(batch_size=8, num_workers=16, shuffle=True, drop_last=True, pin_memory=True, prefetch_factor=4),
        scheduler='official end-of-epoch scheduler.step(zero_based_epoch)' if not film else 'constant lr',
        training_GT='unmodified official current-point-cloud FOV/native point filter, including empty-frame resampling')
    contract['augmentation_audit'] = 'official IntermediateFusionDataset constructs DataAugmentor but never invokes its world flip/rotation/scaling path; preserve and count actual calls'
    contract['gspr_frozen_settings'] = SETTINGS
    contract['identity'] = identity(contract)
    bound_json(root / 'contract.json', contract)
    bound_json(root / 'selection_policy.json', SELECTION)
    rows = committed(root, 1 if args.smoke else epochs, contract['identity'])
    completed = len(rows)
    if completed == (1 if args.smoke else epochs):
        if not args.smoke:
            if film:
                film_audit(root, rows, parent)
            finish(root, rows, selection(root))
        elif not (root / 'smoke_checkpoint.json').exists():
            atomic_json(root / 'smoke_checkpoint.json', dict(status='complete', checkpoint=rows[-1]['checkpoint'],
                checkpoint_sha256=rows[-1]['checkpoint_sha256'], eligible_for_selection=False))
        return
    seed(CONFIG['seed'])
    dataset = build_train(hypes, manifest)
    validation = build_dataset(hypes, visualize=False, train=False)
    if len(validation) != 829 or validation.eval_sim or validation.data_augmentor.train:
        raise ValueError('Loss validation must remain Clean with augmentation disabled')
    options = dict(batch_size=8, num_workers=16, shuffle=True, pin_memory=True, drop_last=True, prefetch_factor=4)
    # Match the actual official construction/order, including train collate on validation loss.
    train_loader = DataLoader(Subset(dataset, range(16)) if args.smoke else dataset,
        collate_fn=dataset.collate_batch_train, **options)
    val_loader = DataLoader(Subset(validation, range(16)) if args.smoke else validation,
        collate_fn=dataset.dataset.collate_batch_train, **options)
    model = create(hypes, args.variant).to('cuda:0')
    loss_fn = criterion(hypes, args.variant, film).to('cuda:0')
    if film:
        combined = dict(model.state_dict())
        combined.update(base)
        model.load_state_dict(combined, strict=True)
        strict_backbone(model, base)
        frozen_backbone(model)
        cfg = CONFIG['film']
        optimizer = torch.optim.Adam(model.film.parameters(), lr=cfg['lr'], eps=cfg['eps'], weight_decay=cfg['weight_decay'])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.)
    else:
        optimizer = train_utils.setup_optimizer(hypes, model)
        scheduler = train_utils.setup_lr_schedular(hypes, optimizer, n_iter_per_epoch=len(train_loader))
    writer = SummaryWriter(str(root / 'tensorboard'))
    initial = root / 'resume_states/00.pth'
    if completed:
        restore(rows[-1]['resume_state'], model, optimizer, scheduler, loss_fn, contract['identity'], completed)
    elif initial.exists() and (root / 'resume_states/00.json').exists():
        marker = read_json(root / 'resume_states/00.json')
        if marker['sha256'] != sha256(initial):
            raise ValueError('Initial model/optimizer/RNG changed')
        restore(initial, model, optimizer, scheduler, loss_fn, contract['identity'], 0)
    else:
        save(initial, state(model, optimizer, scheduler, loss_fn, 0, contract['identity']))
        atomic_json(root / 'resume_states/00.json', dict(sha256=sha256(initial)))
    try:
        for epoch in range(completed, 1 if args.smoke else epochs):
            started = time.monotonic()
            directory = root / 'epochs' / f'{epoch+1:02d}'
            directory.mkdir(parents=True, exist_ok=True)
            trace = directory / 'sampling.jsonl'
            temporary = trace.with_suffix('.jsonl.tmp')
            counts = dict(clean=0, fog=0)
            losses = []
            module_rows = []
            gt_total = 0
            resamples = 0
            augment_calls = 0
            film_rows = []
            with temporary.open('w', encoding='utf-8', newline='\n') as stream:
                for step, batch in enumerate(train_loader):
                    audits = batch.pop('_audit')
                    stream.write(json.dumps(jsonable(audits), allow_nan=False) + '\n')
                    for audit in audits:
                        for read in audit['reads']:
                            counts[read['condition']] += 1
                        gt_total += audit['final_gt_count']
                        resamples += sum('requested_index' in c for c in audit['gt_calls']) - 1
                        augment_calls += audit['configured_world_augmentation_calls']
                    frozen_backbone(model) if film else model.train()
                    model.zero_grad()
                    optimizer.zero_grad()
                    batch = train_utils.to_device(batch, torch.device('cuda:0'))
                    batch['ego']['epoch'] = epoch
                    prediction = model(batch['ego'])
                    loss = loss_fn(prediction, batch['ego']['label_dict'])
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite training loss')
                    loss_fn.logging(epoch, step, len(train_loader), writer)
                    loss.backward()
                    model.repe_check_gradients()
                    gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
                    if not gradients or any(not torch.isfinite(g).all() for g in gradients):
                        raise ValueError('Missing/nonfinite training gradients')
                    optimizer.step()
                    module_rows.append(model.repe_statistics(loss_fn))
                    losses.append(float(loss.detach()))
                    if film:
                        film_rows.append(model.film.last)
                        model.film.observations.clear()
                        from models import OBSERVED_FILM
                        OBSERVED_FILM.clear()
                    if step % 25 == 0:
                        print(f'{args.variant} epoch={epoch+1}/{epochs} batch={step}/{len(train_loader)} loss={losses[-1]:.6f}', flush=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, trace)
            if min(counts.values()) == 0:
                raise ValueError('Mixed epoch did not read both Clean and Fog')
            if film:
                strict_backbone(model, base)
            # Official save occurs before validation and the explicit-epoch scheduler call.
            raw = root / f'net_epoch{epoch+1}.pth'
            save(raw, model.state_dict())
            weight_hash = sha256(raw)
            weight = directory / ('checkpoint_' + weight_hash + '.pth')
            if not weight.exists():
                os.link(raw, weight)
            if sha256(weight) != weight_hash:
                raise ValueError('Immutable selection checkpoint changed')
            val_losses = []
            module_rows = []
            with torch.no_grad():
                for batch in val_loader:
                    model.zero_grad()
                    optimizer.zero_grad()
                    model.eval()
                    batch = train_utils.to_device(batch, torch.device('cuda:0'))
                    batch['ego']['epoch'] = epoch
                    value = loss_fn(model(batch['ego']), batch['ego']['label_dict'])
                    if not torch.isfinite(value):
                        raise ValueError('Nonfinite validation loss')
                    val_losses.append(float(value))
                    if film:
                        model.film.observations.clear()
                        from models import OBSERVED_FILM
                        OBSERVED_FILM.clear()
                    torch.cuda.empty_cache()
            validation_loss = sum(val_losses) / len(val_losses)
            writer.add_scalar('Validate_Loss', validation_loss, epoch)
            destination = directory / ('clean_ap_' + weight_hash)
            torch.cuda.empty_cache()
            ap = clean_ap(args, weight, destination)
            # Preserve the official order and zero-based argument exactly for L0/L1.
            scheduler.step(epoch)
            if film:
                strict_backbone(model, base)
            resume_state = root / 'resume_states' / f'{epoch+1:02d}.pth'
            save(resume_state, state(model, optimizer, scheduler, loss_fn, epoch+1, contract['identity']))
            row = dict(status='complete', epoch=epoch+1, contract=contract['identity'],
                checkpoint=str(weight), checkpoint_sha256=weight_hash, resume_state=str(resume_state),
                resume_state_sha256=sha256(resume_state), clean_summary=str(destination / 'summary.json'),
                clean_summary_sha256=sha256(destination / 'summary.json'), trace=str(trace), trace_sha256=sha256(trace),
                train_loss=sum(losses)/len(losses), train_updates=len(losses), validation_loss=validation_loss,
                AP70=ap['results']['clean']['AP']['O']['ap70'], clean_AP=ap['results']['clean']['AP'],
                sampling_counts=counts, fog_fraction=counts['fog']/sum(counts.values()),
                consumed_input_frames=len(losses)*8, accepted_train_frames=8083, dropped_frames=3 if not args.smoke else 0,
                gt_total=gt_total, official_resampling_attempts=resamples,
                configured_world_augmentation_calls=augment_calls,
                next_lr=[group['lr'] for group in optimizer.param_groups], duration_seconds=time.monotonic()-started,
                parameter_counts=dict(total=sum(p.numel() for p in model.parameters()),
                    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)),
                repe=module_rows, peak_memory_bytes=torch.cuda.max_memory_allocated(),
                smoke_only=args.smoke, film=film_rows if film else None)
            atomic_json(directory / 'complete.json', row)
            rows.append(row)
            atomic_json(root / 'history.json', dict(status='running', epochs=rows))
            atomic_json(root / 'progress.json', dict(status='running', completed_epoch=epoch+1, total_epochs=epochs))
            print(f'{args.variant} EPOCH COMMITTED {epoch+1}/{epochs}: Clean O AP70={row["AP70"]}, sampling={counts}', flush=True)
        if args.smoke:
            atomic_json(root / 'smoke_checkpoint.json', dict(status='complete', checkpoint=rows[-1]['checkpoint'],
                checkpoint_sha256=rows[-1]['checkpoint_sha256'], eligible_for_selection=False))
        else:
            if film:
                film_audit(root, rows, parent)
            finish(root, rows, selection(root))
    finally:
        writer.close()


def finish(root, rows, selected):
    atomic_json(root / 'history.json', dict(status='complete', epochs=rows))
    atomic_json(root / 'training_summary.json', dict(status='complete', epochs=len(rows), selected=selected,
        output_hashes={name: sha256(root / name) for name in
            ('contract.json', 'config.yaml', 'selection_policy.json', 'selected.json', 'history.json')}))


def film_audit(root, rows, parent):
    summaries = [entry for row in rows for entry in row['film']]
    atomic_json(root / 'adaptation.json', dict(status='complete', frozen_backbone_exact=True,
        frozen_bn_exact=True, parent_checkpoint_sha256=parent['checkpoint_sha256'],
        film_parameters=rows[-1]['parameter_counts']['trainable'],
        total_parameters=rows[-1]['parameter_counts']['total'], coefficient_observations=summaries,
        ever_nonidentity=any(r['delta_rms'] > 0 or r['beta_rms'] > 0 for r in summaries),
        observed_input_variation_between_cavs=any(r['between_cav_gamma_rms'] > 1e-12 or
            r['between_cav_beta_rms'] > 1e-12 for r in summaries),
        interpretation='Zero/constant modulation is reported as measured null behavior, not an improvement claim.'))
