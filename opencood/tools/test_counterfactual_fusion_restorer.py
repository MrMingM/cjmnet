"""Minimal checks for post-fusion CURE restoration."""

import argparse

import torch

from opencood.models.sub_modules.counterfactual_fusion_restorer import (
    CounterfactualFusionRestorer,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(20260725)
    module = CounterfactualFusionRestorer(
        channels=256, hidden_channels=32).to(device)
    fused = torch.randn(
        2, 256, 12, 20, device=device, requires_grad=True)
    ego = torch.randn(2, 256, 12, 20, device=device)
    restored, aux = module(fused, ego)
    assert torch.equal(restored, fused), (
        'zero-initialized post-fusion CURE must be an exact identity')
    for name in [
            'correction', 'recoverability_logit', 'harm_logit',
            'utility', 'restoration_gate']:
        assert torch.isfinite(aux[name]).all(), (
            '%s contains non-finite values' % name)
    assert aux['correction'].shape == fused.shape
    assert aux['recoverability_logit'].shape == (2, 1, 12, 20)
    restored.square().mean().backward()
    assert module.correction_head.weight.grad is not None
    assert torch.isfinite(module.correction_head.weight.grad).all()
    print(
        'Post-fusion CURE unit checks passed: identity=True '
        'gate=%.4f harm=%.4f' %
        (aux['restoration_gate'].mean().item(),
         torch.sigmoid(aux['harm_logit']).mean().item()))


if __name__ == '__main__':
    main()
