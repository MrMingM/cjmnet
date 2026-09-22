"""Generate resumable single-action labels from the official train/validation split."""
import argparse
import bisect
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='local_fusion_utility/experiment.yaml')
    parser.add_argument('--frontend-config', required=True)
    parser.add_argument('--frontend-checkpoint', required=True)
    parser.add_argument('--split', choices=('train', 'validation'), required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', type=int, default=0, help='Development only; first N frames')
    args = parser.parse_args()

    import torch
    import yaml
    from opencood.tools.train_utils import to_device
    from opencood.loss.point_pillar_loss import PointPillarLoss
    from gspr_communication.runtime import new_output, seed_all, device, verify_frozen, write_json
    from gspr_evidence import runtime as evidence_runtime
    from . import runtime as rt
    from .supervision import action_targets

    verify_frozen()
    options, hypes = rt.load_config(args.config, args.frontend_config)
    utility = rt.settings(options)
    seed_all(options['seed'])
    target = device()
    model, digest = evidence_runtime.load_model(hypes, options, args.frontend_checkpoint, target)
    criterion = PointPillarLoss(hypes['loss']['args']).to(target)
    dataset, loader, indices = evidence_runtime.make_loader(
        hypes, options, train=args.split == 'train', smoke=args.smoke)
    specification = {
        'contract': rt.contract(options, args.frontend_config, digest),
        'split': args.split,
        'indices': indices,
        'scene_ends': dataset.len_record,
        'branches': ['clean', 'weather'],
        'sampling_uses_gt': False,
        'target_gt_use': 'post-NMS recovered GT, lost baseline GT, and newly introduced FP only',
        'smoke': int(args.smoke),
        'complete': False,
    }
    output = Path(args.output_dir)
    if args.resume:
        old = json.loads((output/'manifest.json').read_text(encoding='utf-8'))
        left = {key: value for key, value in old.items() if key != 'complete'}
        right = {key: value for key, value in specification.items() if key != 'complete'}
        if left != right:
            raise ValueError('Resume cache contract differs')
    else:
        output = new_output(output)
        write_json(output/'manifest.json', specification)
        (output/'experiment.yaml').write_text(yaml.safe_dump(options, sort_keys=False), encoding='utf-8')

    started, written = time.perf_counter(), 0
    print(f'{args.split}: {len(indices)} frames, paired clean/weather, '
          f'{utility["actions_per_view"]} actions per branch', flush=True)
    with torch.no_grad():
        for batch in loader:
            index = int(batch['ego']['communication_sample_index'][0])
            path = output/f'{index:08d}.pt'
            if args.resume and path.is_file():
                continue
            batch = to_device(batch, target)
            ego = batch['ego']
            record = {}
            for branch_index, branch in enumerate(('clean', 'weather')):
                context = rt.prepare_context(model, ego, branch, utility, verify=True)
                record[branch] = action_targets(
                    model.engine.base, context, batch, dataset, ego['label_dict'], criterion,
                    utility, options['seed']+index*2+branch_index)
                record[branch]['raw_feature_bytes'] = rt.raw_feature_bytes(context)
                del context
            record.update(sample_index=index, scene=bisect.bisect_right(dataset.len_record, index))
            temporary = path.with_suffix('.partial')
            torch.save(record, temporary)
            temporary.replace(path)
            written += 1
            if written == 1 or written % 20 == 0:
                complete = sum(1 for _ in output.glob('*.pt'))
                elapsed = time.perf_counter()-started
                remaining = len(indices)-complete
                print(f'{args.split} {complete}/{len(indices)}; {elapsed/written:.1f}s/new frame; '
                      f'ETA {remaining*elapsed/written/3600:.2f}h', flush=True)
    if any(not (output/f'{index:08d}.pt').is_file() for index in indices):
        raise RuntimeError('Incomplete cache')
    specification['complete'] = True
    write_json(output/'manifest.json', specification)
    verify_frozen()
    print('Complete:', output/'manifest.json', flush=True)


if __name__ == '__main__':
    main()

