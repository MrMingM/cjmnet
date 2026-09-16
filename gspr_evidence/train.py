"""Train dense risk first, then directly supervised signed candidate gains."""
import argparse
import json
import random
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train-cache', required=True)
    p.add_argument('--validation-cache', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--variant', choices=('matching', 'concat', 'no_u'), default='matching')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    import torch
    import yaml
    import torch.nn.functional as F
    from torch import nn
    from . import runtime as rt, protocol
    from .heads import RiskHead, GainHead, risk_loss, pair_ranking_loss
    from .model import flat, pose_features
    rt.verify_frozen()
    caches = {name: Path(path) for name, path in [('train', args.train_cache), ('validation', args.validation_cache)]}
    manifests = {name: json.loads((path/'manifest.json').read_text()) for name, path in caches.items()}
    for name, manifest in manifests.items():
        if not manifest['complete'] or manifest['split'] != name:
            raise ValueError('Requires COMPLETE full train/validation caches')
    if manifests['train']['contract'] != manifests['validation']['contract']:
        raise ValueError('Train/validation cache contracts differ')
    contract = manifests['train']['contract']
    rt.validate_contract(contract)
    options = contract['options']
    rt.seed_all(options['seed'])
    target = rt.device()
    heads = nn.ModuleDict({'risk': RiskHead(), 'gain': GainHead(matching=args.variant != 'concat')}).to(target)
    out = Path(args.output_dir) if args.resume else rt.new_output(args.output_dir)
    resume = torch.load(out/'last.pth', map_location=target, weights_only=True) if args.resume else None
    if resume and (resume['contract'] != contract or resume['variant'] != args.variant):
        raise ValueError('Resume training contract differs')
    if resume:
        heads.load_state_dict(resume['heads'])
    (out/'experiment.yaml').write_text(yaml.safe_dump(options, sort_keys=False), encoding='utf-8')
    rt.write_json(out/'training_protocol.json', dict(contract=contract, variant=args.variant,
                                                   manifests=manifests, full_frames=True))
    for phase in ('risk', 'gain'):
        if resume and phase == 'risk' and resume['phase'] == 'gain':
            continue
        module = heads[phase]
        optimizer = torch.optim.AdamW(module.parameters(), lr=options['learning_rate'], weight_decay=1e-4)
        best = float('inf')
        begin = 0
        if resume and resume['phase'] == phase:
            optimizer.load_state_dict(resume['optimizer'])
            begin, best = resume['epoch']+1, resume['best']
        epochs = options[phase+'_epochs']
        for epoch in range(begin, epochs):
            rt.seed_all(options['seed']+epoch + (10000 if phase == 'gain' else 0))
            metrics = {}
            for split in ('train', 'validation'):
                training = split == 'train'
                heads.eval()
                module.train(training)
                ids = manifests[split]['indices'].copy()
                if training:
                    random.shuffle(ids)
                total, frames, groups_seen, correct_pairs, all_pairs, abs_error = 0., 0, 0, 0, 0, 0.
                regret, oracle_over_rule, predicted_over_rule, positive = 0., 0., 0., 0
                for frame_index in ids:
                    record = torch.load(caches[split]/f'{frame_index:08d}.pt', map_location='cpu', weights_only=True)
                    losses = []
                    with torch.set_grad_enabled(training):
                        for branch in ('clean', 'weather'):
                            row = record[branch]
                            obs = row['obs'].to(target).float()
                            if args.variant == 'no_u':
                                obs[:, 1] = 0
                            semantics = row['semantics'].to(target).float()
                            if phase == 'risk':
                                predicted = heads['risk'](semantics[:1], obs[:1])
                                truth = row['target'].to(target).float()
                                losses.append(risk_loss(predicted, truth, row['foreground'].to(target)))
                            else:
                                with torch.no_grad():
                                    risk = heads['risk'](semantics[:1], obs[:1])
                                    profile = torch.cat([obs[:1], risk], 1)[0].cpu().numpy()
                                poses = row['poses'].to(target)
                                for group in row['groups']:
                                    peer = group['peer']
                                    packet = protocol.pack_request(profile, group['ids'].numpy(), row['poses'][0].numpy())
                                    req, received_ids, pose, _ = protocol.unpack_request(packet)
                                    block_ids = torch.as_tensor(received_ids, device=target)
                                    predicted = heads['gain'](torch.as_tensor(req, device=target), flat(obs[peer])[block_ids],
                                        flat(semantics[peer])[block_ids], pose_features(poses[peer], torch.as_tensor(pose, device=target)))
                                    truth = group['gains'].to(target)*options['gain_scale']
                                    loss = F.smooth_l1_loss(predicted, truth) + .1*pair_ranking_loss(predicted, truth)
                                    losses.append(loss)
                                    i, j = torch.triu_indices(len(truth), len(truth), 1, device=target)
                                    keep = (truth[i]-truth[j]).abs() > 1e-4
                                    all_pairs += int(keep.sum())
                                    correct_pairs += int((((truth[i]-truth[j])*(predicted[i]-predicted[j]) > 0) & keep).sum())
                                    abs_error += float((predicted.detach()-truth).abs().mean())
                                    chosen = int(predicted.detach().argmax())
                                    regret += float(truth.max()-truth[chosen])/options['gain_scale']
                                    oracle_over_rule += float(truth.max()-truth[0])/options['gain_scale']
                                    predicted_over_rule += float(truth[chosen]-truth[0])/options['gain_scale']
                                    positive += int(truth[chosen] > 0)
                                    groups_seen += 1
                        if losses:
                            loss = torch.stack(losses).mean()
                            if not torch.isfinite(loss):
                                raise FloatingPointError(f'Nonfinite {phase} loss, frame {frame_index}')
                            if training:
                                optimizer.zero_grad(set_to_none=True)
                                loss.backward()
                                nn.utils.clip_grad_norm_(module.parameters(), 5.)
                                optimizer.step()
                            total += float(loss.detach())
                        frames += 1
                        if frames % 500 == 0:
                            print(f'{args.variant} {phase} epoch {epoch+1} {split}: {frames}/{len(ids)}', flush=True)
                if phase == 'gain' and groups_seen == 0:
                    raise RuntimeError('No valid multi-agent gain supervision in full split')
                metrics[split] = dict(loss=total/max(frames, 1), frames=frames, candidate_groups=groups_seen,
                                      pair_accuracy=correct_pairs/max(all_pairs, 1),
                                      signed_gain_mae=abs_error/max(groups_seen, 1)/options['gain_scale'],
                                      candidate_regret=regret/max(groups_seen, 1),
                                      oracle_over_rule=oracle_over_rule/max(groups_seen, 1),
                                      predicted_over_rule=predicted_over_rule/max(groups_seen, 1),
                                      positive_chosen_fraction=positive/max(groups_seen, 1))
            score = metrics['validation']['loss']
            improved = score < best
            best = min(best, score)
            state = dict(heads=heads.state_dict(), optimizer=optimizer.state_dict(), phase=phase, epoch=epoch,
                         best=best, contract=contract, variant=args.variant)
            if improved:
                torch.save(state, out/f'{phase}_best.partial')
                (out/f'{phase}_best.partial').replace(out/f'{phase}_best.pth')
            torch.save(state, out/'last.partial')
            (out/'last.partial').replace(out/'last.pth')
            with (out/'history.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(dict(variant=args.variant, phase=phase, epoch=epoch+1, **metrics))+'\n')
            print(json.dumps(dict(phase=phase, epoch=epoch+1, **metrics)), flush=True)
        # Freeze the validation-selected risk predictor for responder training.
        best_state = torch.load(out/f'{phase}_best.pth', map_location=target, weights_only=True)
        heads.load_state_dict(best_state['heads'])
        if phase == 'risk':
            heads['risk'].requires_grad_(False)
        resume = None
    print(f'Finished: {out}/gain_best.pth; no frontend weights saved or changed', flush=True)
    rt.verify_frozen()


if __name__ == '__main__':
    main()
