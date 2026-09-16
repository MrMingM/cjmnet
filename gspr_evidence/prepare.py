"""One expensive, resumable full-split teacher pass; compact CPU tensors only."""
import argparse
import bisect
import json
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='gspr_evidence/experiment.yaml')
    p.add_argument('--frontend-config', required=True)
    p.add_argument('--frontend-checkpoint', required=True)
    p.add_argument('--split', choices=('train', 'validation'), required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    import torch
    import yaml
    from opencood.tools.train_utils import to_device
    from opencood.loss.point_pillar_loss import PointPillarLoss
    from . import runtime as rt
    from .supervision import dense_targets, candidate_targets
    rt.verify_frozen()
    options, hypes = rt.load_config(args.config, args.frontend_config)
    rt.seed_all(options['seed'])
    target = rt.device()
    model, digest = rt.load_model(hypes, options, args.frontend_checkpoint, target)
    criterion = PointPillarLoss(hypes['loss']['args'])
    ds, loader, indices = rt.make_loader(hypes, options, train=args.split == 'train')
    specification = dict(contract=rt.contract(options, args.frontend_config, digest), split=args.split,
                         indices=indices, scene_ends=ds.len_record, complete=False)
    out = Path(args.output_dir)
    if args.resume:
        old = json.loads((out/'manifest.json').read_text())
        if {k: v for k, v in old.items() if k != 'complete'} != {k: v for k, v in specification.items() if k != 'complete'}:
            raise ValueError('Resume cache contract differs')
    else:
        out = rt.new_output(out)
        rt.write_json(out/'manifest.json', specification)
        (out/'experiment.yaml').write_text(yaml.safe_dump(options, sort_keys=False), encoding='utf-8')
    start = time.perf_counter()
    done = 0
    print(f'{args.split}: {len(indices)} frames, {len(ds.len_record)} scenes; full split, paired clean/weather', flush=True)
    with torch.no_grad():
        for batch in loader:
            index = int(batch['ego']['communication_sample_index'][0])
            path = out/f'{index:08d}.pt'
            if path.exists() and args.resume:
                continue
            batch = to_device(batch, target)
            ego = batch['ego']
            records = {}
            clean_risk = None
            for branch in ('clean', 'weather'):
                encoded = model.encode(rt.input_branch(ego, branch))
                ego_output, _ = model.detect(encoded, model.empty_masks(encoded), serialize=False)
                risk, fg = dense_targets(ego_output, ego['label_dict'], model.grid)
                if branch == 'clean':
                    clean_risk = risk.clone()
                restoration = (risk-clean_risk).clamp_min(0).amax(1, keepdim=True)
                groups = candidate_targets(model, encoded, ego['label_dict'], criterion, options['seed']+index)
                records[branch] = dict(semantics=encoded['semantics'].cpu().half(), obs=encoded['obs'].cpu().half(),
                                       poses=encoded['poses'].cpu(), target=torch.cat([risk, restoration], 1).cpu().half(),
                                       foreground=fg.cpu().to(torch.uint8), groups=groups)
                del encoded
            records.update(sample_index=index, scene=bisect.bisect_right(ds.len_record, index))
            temporary = path.with_suffix('.partial')
            torch.save(records, temporary)
            temporary.replace(path)
            done += 1
            if done == 1 or done % 20 == 0:
                seconds = time.perf_counter()-start
                remaining = len(indices) - sum(1 for _ in out.glob('*.pt'))
                print(f'{args.split} {len(indices)-remaining}/{len(indices)}; {seconds/done:.1f}s/new frame; '
                      f'ETA {remaining*seconds/done/3600:.2f}h; estimated cache {path.stat().st_size*len(indices)/2**30:.1f}GiB', flush=True)
    if any(not (out/f'{i:08d}.pt').is_file() for i in indices):
        raise RuntimeError('Incomplete cache')
    specification['complete'] = True
    rt.write_json(out/'manifest.json', specification)
    rt.verify_frozen()
    print(f'Complete: {out}/manifest.json', flush=True)


if __name__ == '__main__':
    main()
