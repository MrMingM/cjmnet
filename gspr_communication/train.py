"""python -m gspr_communication.train --help"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Train isolated communication selectors only')
    parser.add_argument('--config', default='gspr_communication/experiment.yaml')
    parser.add_argument('--frontend-config', required=True)
    parser.add_argument('--frontend-checkpoint', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--variant', choices=['a1b0', 'a0b1', 'a1b1', 'residual'])
    args = parser.parse_args()
    import torch
    import yaml
    from opencood.loss.point_pillar_loss import PointPillarLoss
    from opencood.tools.train_utils import to_device
    from . import runtime as rt
    rt.verify_frozen()
    options, hypes = rt.load_config(args.config, args.frontend_config)
    if args.variant:
        options['communication']['variant'] = args.variant
    if options['communication']['variant'] not in ('a1b0', 'a0b1', 'a1b1', 'residual'):
        raise ValueError('Rule baselines need evaluation only, no optimizer')
    rt.seed_all(int(options['seed']))
    target = rt.device()
    model, frontend_hash = rt.load_model(hypes, options, args.frontend_checkpoint, target)
    train_ds, train_loader, train_ids = rt.make_loader(hypes, options, train=True)
    _, val_loader, val_ids = rt.make_loader(hypes, options)
    # Prevent accidentally selecting an overlapping scenario under different roots.
    def selected_names(root, setting):
        names = sorted(p.name for p in Path(root).iterdir() if p.is_dir())
        return set(names if setting is None else [names[i] for i in setting])
    if selected_names(hypes['root_dir'], options.get('train_scene_indices')) & selected_names(hypes['validate_dir'], options.get('validation_scene_indices')):
        raise ValueError('Train/validation scene names overlap; supply disjoint scenario splits')
    out = rt.new_output(args.run_dir)
    (out/'experiment.yaml').write_text(yaml.safe_dump(options, sort_keys=False), encoding='utf-8')
    # Preserve raw frontend YAML (anchors / parser semantics) without overwriting it.
    (out/'frontend_config.yaml').write_bytes(Path(args.frontend_config).read_bytes())
    rt.write_json(out/'manifest.json', {'frontend_checkpoint': str(Path(args.frontend_checkpoint).resolve()),
        'frontend_sha256': frontend_hash, 'train_indices': train_ids, 'validation_indices': val_ids,
        'frozen_sources': rt.verify_frozen(), 'seed': options['seed'],
        'communication_source_sha256': {str(p.relative_to(rt.ROOT)): rt.sha256(p)
            for p in (rt.ROOT/'gspr_communication').glob('*.py')}})
    criterion = PointPillarLoss({'cls_weight': 1., 'reg': 2.})
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=float(options['learning_rate']))
    best = float('inf')
    total_steps = 0
    for epoch in range(int(options['epochs'])):
        if hasattr(train_ds, 'weather_augmentation_epoch'):
            train_ds.weather_augmentation_epoch = epoch
        model.train()
        total, samples, updates = 0., 0, 0
        for batch in train_loader:
            batch = to_device(batch, target)
            output = model(rt.model_input(batch['ego'], options))
            loss = criterion(output, batch['ego']['label_dict'])
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            optimizer.zero_grad(set_to_none=True)
            if loss.requires_grad:
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
                    raise RuntimeError('Non-finite selector gradients')
                torch.nn.utils.clip_grad_norm_(parameters, 5.)
                optimizer.step()
                updates += 1
            n = len(batch['ego']['record_len'])
            total += float(loss.detach())*n
            samples += n
            total_steps += 1
            if total_steps % 10 == 0:
                progress = {'step': total_steps, 'loss': float(loss.detach()),
                            'bytes': [d['total_bytes'] for d in model.last_diagnostics]}
                if model.correction is not None:
                    progress['replacement_fraction'] = [d['replacement_fraction'] for d in model.last_diagnostics]
                    progress['mean_abs_score_adjustment'] = [d['mean_abs_score_adjustment'] for d in model.last_diagnostics]
                print(json.dumps(progress), flush=True)
            if options.get('steps_per_epoch') and updates >= int(options['steps_per_epoch']):
                break
        if not updates:
            raise RuntimeError('No trainable communication batches: check peers and budget (must select some but not all blocks)')
        model.eval()
        val_total, val_n = 0., 0
        with torch.no_grad():
            for batch in val_loader:
                batch = to_device(batch, target)
                loss = criterion(model(rt.model_input(batch['ego'], options)), batch['ego']['label_dict'])
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite validation loss')
                n = len(batch['ego']['record_len'])
                val_total += float(loss)*n
                val_n += n
        val = val_total / val_n
        record = {'epoch': epoch+1, 'train_loss': total/samples, 'validation_loss': val, 'updates': updates}
        print(json.dumps(record), flush=True)
        with (out/'history.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record)+'\n')
        checkpoint = {'request': model.request.state_dict(), 'response': model.response.state_dict(),
                      'frontend_sha256': frontend_hash, 'communication': options['communication'],
                      'frontend_config_sha256': rt.sha256(args.frontend_config),
                      'epoch': epoch+1, 'validation_loss': val}
        if model.correction is not None:
            checkpoint['correction'] = model.correction.state_dict()
        torch.save(checkpoint, out/f'selectors_epoch{epoch+1}.pth')
        if val < best:
            best = val
            torch.save(checkpoint, out/'selectors_best.pth')
    rt.verify_frozen()
    print('Finished. Selector checkpoints only; frozen frontend was not saved or modified.', flush=True)


if __name__ == '__main__':
    main()
