"""Real-data server gate: exact endpoints, frozen state and both selector gradients."""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='gspr_communication/experiment.yaml')
    parser.add_argument('--frontend-config', required=True)
    parser.add_argument('--frontend-checkpoint', required=True)
    parser.add_argument('--variant', choices=['a1b1', 'residual'], default='a1b1')
    args = parser.parse_args()
    import torch
    from opencood.tools.train_utils import to_device
    from . import runtime as rt
    rt.verify_frozen()
    options, hypes = rt.load_config(args.config, args.frontend_config)
    options['communication']['variant'] = args.variant
    options['communication']['value_bytes'] = 4
    rt.seed_all(int(options['seed']))
    target = rt.device()
    model, _ = rt.load_model(hypes, options, args.frontend_checkpoint, target)
    _, loader, _ = rt.make_loader(hypes, options, test=True)
    inp = None
    for i, batch in enumerate(loader):
        if int(batch['ego']['record_len'][0]) > 1:
            inp = rt.model_input(to_device(batch, target)['ego'], options)
            break
        if i >= 19:
            break
    if inp is None:
        raise RuntimeError('No multi-CAV sample in first 20 validation frames')
    buffers = {k: v.clone() for k, v in model.base.named_buffers()}
    model.eval()
    with torch.no_grad():
        original = model.base(inp)
        model.variant = 'full'
        full = model(inp)
        for key in ('psm', 'rm'):
            torch.testing.assert_close(full[key], original[key], atol=2e-4, rtol=2e-4)
        print('PASS: full communication matches frozen original', flush=True)
        levels, _, _, _ = model.encode(inp)
        ego_features = torch.cat([d(x[:1]) for d, x in zip(model.base.backbone.deblocks, levels)], 1)
        model.variant = 'none'
        none = model(inp)
        torch.testing.assert_close(none['psm'], model.base.cls_head(ego_features), atol=2e-4, rtol=2e-4)
        torch.testing.assert_close(none['rm'], model.base.reg_head(ego_features), atol=2e-4, rtol=2e-4)
        assert model.last_diagnostics[0]['total_bytes'] == 0
        print('PASS: zero communication matches ego-only path', flush=True)
    if args.variant == 'residual':
        with torch.no_grad():
            from . import codec
            from .selection import hard_topk
            # Reuse ONE encoder result so floating-point variation between
            # separate full forwards cannot hide a selector implementation bug.
            _, sem, stats, conf = model.encode(inp)
            peers = int(inp['record_len'][0]) - 1
            active, quotas = codec.allocate(model.budget, peers, *model.grid, model.cost)
            if not active or not any(0 < k < model.grid[0]*model.grid[1] for k in quotas):
                raise RuntimeError('Identity check requires an intermediate communication budget')
            model.variant = 'a0b0'
            request = model._request(sem[:1], stats[:1], conf[:1])
            model.variant = 'residual'
            residual_request = model._request(sem[:1], stats[:1], conf[:1])
            torch.testing.assert_close(residual_request, request, atol=0, rtol=0)
            packet = codec.pack_request(request[0, 0].cpu().numpy())
            request = torch.as_tensor(codec.unpack_request(packet), device=sem.device,
                                      dtype=sem.dtype)[None, None]
            for peer, k in enumerate(quotas, 1):
                model.variant = 'a0b0'
                rule_score = model._score(sem[peer:peer+1], stats[peer:peer+1], conf[peer:peer+1], request)
                model.variant = 'residual'
                residual_score = model._score(sem[peer:peer+1], stats[peer:peer+1], conf[peer:peer+1], request)
                torch.testing.assert_close(residual_score, rule_score, atol=0, rtol=0)
                torch.testing.assert_close(hard_topk(residual_score, k), hard_topk(rule_score, k), atol=0, rtol=0)
            print('PASS: shared-input requests, scores and selected blocks exactly match A0B0', flush=True)
            model.variant = 'a0b0'
            reference = model(inp)
            reference_bytes = model.last_diagnostics[0]['total_bytes']
            model.variant = 'residual'
            initial = model(inp)
            assert model.last_diagnostics[0]['total_bytes'] == reference_bytes
            assert model.last_diagnostics[0]['replaced_blocks_vs_a0b0'] == 0
            for key in ('psm', 'rm'):
                error = float((initial[key]-reference[key]).abs().max())
                print(f'CHECK: residual vs A0B0 {key} max_abs_error={error:.8g}', flush=True)
                # Same FP32 tolerance as the original/full/ego checks above.
                torch.testing.assert_close(initial[key], reference[key], atol=2e-4, rtol=2e-4)
        print('PASS: detection outputs match A0B0 within FP32 tolerance; bytes match exactly', flush=True)
    model.variant = args.variant
    model.train()
    output = model(inp)
    objective = output['psm'].square().mean() + output['rm'].square().mean()
    if not objective.requires_grad:
        raise RuntimeError('Budget yields no differentiable selection; choose an intermediate budget')
    objective.backward()
    modules = [('correction', model.correction)] if args.variant == 'residual' else [('A', model.request), ('B', model.response)]
    for name, module in modules:
        magnitude = sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)
        if not magnitude > 0 or not all(torch.isfinite(p.grad).all() for p in module.parameters() if p.grad is not None):
            raise RuntimeError(name + ' has missing/nonfinite gradients')
        print('PASS:', name, 'gradient sum', magnitude, flush=True)
    if args.variant == 'residual':
        assert all(not p.requires_grad and p.grad is None for m in (model.request, model.response) for p in m.parameters())
    assert all(p.grad is None and not p.requires_grad for p in model.base.parameters())
    for key, value in model.base.named_buffers():
        torch.testing.assert_close(value, buffers[key], atol=0, rtol=0)
    print('PASS: frozen parameters and BN buffers unchanged', flush=True)
    # Exact hard forward used both in training and serialized evaluation.
    with torch.no_grad():
        model.eval()
        evaluated = model(inp)
        for key in ('psm', 'rm'):
            torch.testing.assert_close(output[key], evaluated[key], atol=2e-4, rtol=2e-4)
    print('PASS: training hard forward matches serialized inference', flush=True)
    rt.verify_frozen()


if __name__ == '__main__':
    main()
