"""Smoke checks only: results are NOT representative AP experiments."""
import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='gspr_evidence/experiment.yaml')
    p.add_argument('--frontend-config', required=True)
    p.add_argument('--frontend-checkpoint', required=True)
    args = p.parse_args()
    import torch
    from opencood.tools.train_utils import to_device
    from . import runtime as rt
    rt.verify_frozen()
    options, hypes = rt.load_config(args.config, args.frontend_config)
    rt.seed_all(options['seed'])
    device = rt.device()
    model, _ = rt.load_model(hypes, options, args.frontend_checkpoint, device)
    buffers = {name: value.clone() for name, value in model.engine.base.named_buffers()}
    _, loader, _ = rt.make_loader(hypes, options, smoke=8)
    count = 0
    with torch.no_grad():
        for batch in loader:
            ego = to_device(batch, device)['ego']
            if int(ego['record_len'][0]) < 2:
                continue
            inp = rt.input_branch(ego, 'weather')
            encoded = model.encode(inp)
            for mode in ('none', 'full', 'a0b0'):
                output, diag = model.run(encoded, mode)
                model.engine.variant = mode
                reference = model.engine(dict(processed_lidar=inp['processed_lidar'], record_len=inp['record_len']))
                for key in output:
                    torch.testing.assert_close(output[key], reference[key], atol=2e-5, rtol=2e-4)
                if mode != 'full':
                    assert diag['total_bytes'] <= model.engine.budget
            model.engine.variant = 'a0b0'
            exchange = model.request(encoded)
            learned, ld = model.run(encoded, 'learned', exchange)
            rule, rd = model.run(encoded, 'protocol', exchange)
            assert ld['total_bytes'] == rd['total_bytes'] <= model.engine.budget
            masks = model.empty_masks(encoded)
            masks[1, 0, 0, 0] = 1
            wire, _ = model.detect(encoded, masks, True)
            replay, _ = model.detect(encoded, masks, False)
            for key in wire:
                torch.testing.assert_close(wire[key], replay[key], atol=2e-5, rtol=2e-4)
            assert all(not x.requires_grad for x in model.engine.parameters())
            assert not model.engine.base.training
            for name, value in model.engine.base.named_buffers():
                torch.testing.assert_close(value, buffers[name], atol=0, rtol=0)
            count += 1
            print(f'PASS frame {count}: old controls, decoded transport, equal bytes, frozen frontend', flush=True)
    if not count:
        raise RuntimeError('No multi-agent frame in smoke check')
    rt.verify_frozen()
    print('Connectivity only. Next: full-split prepare/train/evaluate.', flush=True)


if __name__ == '__main__':
    main()
