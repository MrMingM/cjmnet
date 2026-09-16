"""Train only the cooperative point-weight reviewer through exact hard fusion."""
import argparse
import json
import random
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='gspr_review/experiment.yaml')
    p.add_argument('--frontend-config', required=True)
    p.add_argument('--frontend-checkpoint', required=True)
    p.add_argument('--run-dir', required=True)
    args = p.parse_args()
    import torch
    import yaml
    from opencood.loss.point_pillar_loss import PointPillarLoss
    from opencood.tools.train_utils import to_device
    from . import runtime as rt
    rt.verify_frozen()
    options, hypes = rt.load_config(args.config, args.frontend_config)
    if options['communication']['variant'] != 'a0b0':
        raise ValueError('Review requires fixed A0B0 feature selection')
    clean_probability = float(options.get('clean_train_probability', .25))
    if not 0 <= clean_probability <= 1:
        raise ValueError('Invalid clean training probability')
    rt.seed_all(int(options['seed']))
    target = rt.device()
    model, digest = rt.load_model(hypes, options, args.frontend_checkpoint, target)
    # Different heads consume different initialization RNG draws. Reset so paired
    # architectures see the same sampler order and clean/weather branch draws.
    rt.seed_all(int(options['seed']))
    ds, train_loader, train_ids = rt.make_loader(hypes, options, train=True)
    _, val_loader, val_ids = rt.make_loader(hypes, options)
    def names(root, indices):
        all_names = sorted(x.name for x in Path(root).iterdir() if x.is_dir())
        return set(all_names if indices is None else [all_names[i] for i in indices])
    if names(hypes['root_dir'], options.get('train_scene_indices')) & names(hypes['validate_dir'], options.get('validation_scene_indices')):
        raise ValueError('Overlapping train/validation scenes')
    out = rt.new_output(args.run_dir)
    (out/'experiment.yaml').write_text(yaml.safe_dump(options, sort_keys=False), encoding='utf-8')
    (out/'frontend_config.yaml').write_bytes(Path(args.frontend_config).read_bytes())
    rt.write_json(out/'manifest.json', {'frontend_checkpoint': str(Path(args.frontend_checkpoint).resolve()),
        'frontend_sha256': digest, 'train_indices': train_ids, 'validation_indices': val_ids,
        'frozen_sources': rt.verify_frozen(), 'seed': options['seed'],
        'source_sha256': {str(f.relative_to(rt.ROOT)): rt.sha256(f) for folder in ('gspr_review', 'gspr_communication') for f in (rt.ROOT/folder).glob('*.py')},
        'point_labels': 'Not available for this synchronized multi-CAV loader. Detection supervision only.',
        'selection': 'Fixed rules; no STE or full-peer-feature gradient surrogate'})
    if options['review'].get('architecture', 'legacy') != 'legacy':
        (out/'source_adaptation_hashes.json').write_bytes((rt.ROOT/'gspr_review/source_adaptation_hashes.json').read_bytes())
        (out/'SOURCE_ADAPTATION.md').write_bytes((rt.ROOT/'gspr_review/SOURCE_ADAPTATION.md').read_bytes())
    if options['review'].get('architecture') == 'bev_contrast':
        (out/'BEV_LOCATION.md').write_bytes((rt.ROOT/'gspr_review/BEV_LOCATION.md').read_bytes())
    criterion = PointPillarLoss({'cls_weight': 1., 'reg': 2.})
    parameters = list(model.reviewer.parameters())
    optimizer = torch.optim.Adam(parameters, lr=float(options['learning_rate']))
    frozen_buffers = {k: v.clone() for k, v in model.base.named_buffers()}
    best = float('inf')
    for epoch in range(int(options['epochs'])):
        ds.weather_augmentation_epoch = epoch
        model.train()
        total = updates = seen = 0
        for batch in train_loader:
            batch = to_device(batch, target)
            branch = 'processed_lidar' if random.random() < clean_probability else options['lidar_key']
            prediction = model(rt.input_branch(batch['ego'], options, branch))
            supported = sum(d['eligible_points'] for d in model.last_diagnostics)
            seen += 1
            if not supported:
                if seen % 20 == 0:
                    print(f'Skipped {seen} seen batches: no received evidence at ego point cells', flush=True)
                continue
            detection = criterion(prediction, batch['ego']['label_dict'])
            loss = detection + float(options['correction_regularization'])*model.correction_penalty
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite review loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not any(x.grad is not None and x.grad.abs().sum() > 0 for x in parameters):
                raise RuntimeError('No reviewer gradient despite eligible evidence; inspect re-encoding path')
            if any(x.grad is not None and not torch.isfinite(x.grad).all() for x in parameters):
                raise RuntimeError('Nonfinite reviewer gradient')
            torch.nn.utils.clip_grad_norm_(parameters, 5.)
            optimizer.step()
            total += float(loss.detach())
            updates += 1
            if updates % 10 == 0:
                print(json.dumps({'epoch': epoch+1, 'update': updates, 'loss': float(loss.detach()),
                    'branch': branch, 'diagnostics': model.last_diagnostics}), flush=True)
            if options.get('steps_per_epoch') and updates >= int(options['steps_per_epoch']):
                break
        if not updates:
            raise RuntimeError('No matched evidence to train on; inspect query budget and cross-view support')
        model.eval()
        totals, n = {'clean': 0., 'weather': 0.}, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = to_device(batch, target)
                for name, branch in (('clean', 'processed_lidar'), ('weather', options['lidar_key'])):
                    prediction = model(rt.input_branch(batch['ego'], options, branch))
                    value = criterion(prediction, batch['ego']['label_dict'])
                    if not torch.isfinite(value):
                        raise RuntimeError('Nonfinite validation loss')
                    totals[name] += float(value)
                n += 1
        val = (totals['clean']+totals['weather'])/(2*n)
        row = {'epoch': epoch+1, 'updates': updates, 'seen': seen, 'train_loss': total/updates,
               'validation_loss': val, **{name+'_loss': value/n for name, value in totals.items()}}
        print(json.dumps(row), flush=True)
        with (out/'history.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(row)+'\n')
        for key, value in model.base.named_buffers():
            torch.testing.assert_close(value, frozen_buffers[key], atol=0, rtol=0)
        assert all(not x.requires_grad and x.grad is None for x in model.base.parameters())
        state = {'reviewer': model.reviewer.state_dict(), 'frontend_sha256': digest,
                 'frontend_config_sha256': rt.sha256(args.frontend_config),
                 'communication': options['communication'], 'review': options['review'], **row}
        torch.save(state, out/f'reviewer_epoch{epoch+1}.pth')
        if val < best:
            best = val
            torch.save(state, out/'reviewer_best.pth')
    rt.verify_frozen()
    print('Finished. Only reviewer checkpoints saved; original frontend and experiments unchanged.', flush=True)


if __name__ == '__main__':
    main()
