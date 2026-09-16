"""Server preflight on real multi-CAV data. Fails before training on broken wiring."""
import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='gspr_review/experiment.yaml')
    p.add_argument('--frontend-config', required=True)
    p.add_argument('--frontend-checkpoint', required=True)
    args = p.parse_args()
    import torch
    from opencood.tools.train_utils import to_device
    from . import runtime as rt
    rt.verify_frozen()
    options, hypes = rt.load_config(args.config, args.frontend_config)
    rt.seed_all(int(options['seed']))
    target = rt.device()
    model, _ = rt.load_model(hypes, options, args.frontend_checkpoint, target)
    _, loader, _ = rt.make_loader(hypes, options, test=True)
    model.eval()
    inp = None
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = to_device(batch, target)
            candidate = rt.input_branch(batch['ego'], options)
            output = model(candidate)
            if sum(d['eligible_points'] for d in model.last_diagnostics):
                inp = candidate
                break
            if i >= 19:
                break
    if inp is None:
        raise RuntimeError('First 20 validation frames contain no matched review evidence; inspect budget, queries and voxel support before training')
    buffers = {k: v.clone() for k, v in model.base.named_buffers()}
    def close(a, b):
        for key in ('psm', 'rm'):
            torch.testing.assert_close(a[key], b[key], atol=2e-4, rtol=2e-4)
    with torch.no_grad():
        model.mode = 'full'
        close(model(inp), model.base(inp))
        print('PASS: full communication matches original frozen frontend', flush=True)
        model.mode = 'none'
        none = model(inp)
        assert model.last_diagnostics[0]['total_bytes'] == 0
        levels, _, _, _ = model.encode(inp)
        features = torch.cat([d(x[:1]) for d, x in zip(model.base.backbone.deblocks, levels)], 1)
        close(none, {'psm': model.base.cls_head(features), 'rm': model.base.reg_head(features)})
        print('PASS: none uses ego only and zero bytes', flush=True)
        model.mode = 'protocol'
        reference = model(inp)
        reference_bytes = model.last_diagnostics[0]['total_bytes']
        model.mode = 'review'
        initial = model(inp)
        close(initial, reference)
        assert model.last_diagnostics[0]['changed_points'] == 0
        assert model.last_diagnostics[0]['total_bytes'] == reference_bytes <= model.budget
        print('PASS: zero-initialized reviewer matches same-packet protocol control', flush=True)
        # Nonzero correction must actually change point weights and detector outputs.
        probe_state = {k: v.clone() for k, v in model.reviewer.state_dict().items()}
        if options['review'].get('architecture', 'legacy') == 'legacy':
            model.reviewer.net[-1].bias.fill_(.5)
        else:
            # Deterministic path responding to br-bn; a bias would cancel in the
            # contrast head and therefore cannot be used as a connection probe.
            score = model.reviewer.net.score
            for parameter in score.parameters():
                parameter.zero_()
            score[0].weight[0, 7] = 1.
            score[0].weight[0, 8] = -1.
            score[2].weight[0, 0] = 1.
            score[4].weight[0, 0] = 1.
        changed = model(inp)
        change_key = 'bev_changed_cells' if model.bev_location else 'changed_points'
        assert model.last_diagnostics[0][change_key] > 0
        if model.bev_location:
            assert model.last_diagnostics[0]['changed_points'] == 0
        assert any((changed[k]-initial[k]).abs().max() > 0 for k in ('psm', 'rm'))
        model.reviewer.load_state_dict(probe_state)
        print('PASS: received evidence changes the selected correction location AND detector outputs', flush=True)
    model.train()
    output = model(inp)
    objective = output['psm'].square().mean()+output['rm'].square().mean()
    objective.backward()
    gradients = [x.grad for x in model.reviewer.parameters() if x.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients) or sum(float(g.abs().sum()) for g in gradients) <= 0:
        raise RuntimeError('No finite reviewer gradient through frozen re-encoding')
    assert all(not x.requires_grad and x.grad is None for x in model.base.parameters())
    assert all(not x.requires_grad and x.grad is None for module in (model.request, model.response) for x in module.parameters())
    for name, value in model.base.named_buffers():
        torch.testing.assert_close(value, buffers[name], atol=0, rtol=0)
    print('PASS: exact hard-fusion gradient reaches reviewer; original parameters/BN remain frozen', flush=True)
    with torch.no_grad():
        model.eval()
        close(output, model(inp))
        saved = model.budget
        model.budget = 0
        model.mode = 'review'
        zero = model(inp)
        assert model.last_diagnostics[0]['total_bytes'] == model.last_diagnostics[0]['changed_points'] == 0
        model.mode = 'none'
        close(zero, model(inp))
        model.budget = saved
    print('PASS: train/eval packet path agrees; zero budget cannot modify ego points', flush=True)
    rt.verify_frozen()


if __name__ == '__main__':
    main()
