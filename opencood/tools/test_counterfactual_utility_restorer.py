"""Minimal CPU/GPU checks for the CURE feature-restoration module."""

import argparse

import torch

from opencood.models.sub_modules.counterfactual_utility_restorer import (
    CounterfactualUtilityRestorer,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(20260724)
    module = CounterfactualUtilityRestorer(
        channels=64, hidden_channels=32,
        fusion_gate_floor=0.5).to(device)
    feature = torch.randn(
        5, 64, 20, 28, device=device, requires_grad=True)
    record_len = torch.tensor([3, 2], device=device)
    restored, aux = module(feature, record_len)

    assert restored.shape == feature.shape
    assert torch.equal(restored, feature), (
        'zero-initialized CURE module must be an exact identity')
    for name in [
            'recoverability_logit', 'harm_logit', 'utility',
            'restoration_gate', 'fusion_weight']:
        assert aux[name].shape == (5, 1, 20, 28), (
            '%s has unexpected shape %s' % (
                name, tuple(aux[name].shape)))
        assert torch.isfinite(aux[name]).all(), (
            '%s contains a non-finite value' % name)
    assert aux['ego_mask'].tolist() == [True, False, False, True, False]
    assert torch.equal(
        aux['fusion_weight'][aux['ego_mask']],
        torch.ones_like(aux['fusion_weight'][aux['ego_mask']])), (
            'Ego fusion weights must remain one')

    loss = restored.square().mean()
    loss.backward()
    assert module.correction[-1].weight.grad is not None
    assert torch.isfinite(module.correction[-1].weight.grad).all()
    with torch.no_grad():
        module.correction[-1].bias.fill_(0.25)
        changed, changed_aux = module(feature.detach(), record_len)
    assert torch.equal(
        changed[changed_aux['ego_mask']],
        feature.detach()[changed_aux['ego_mask']]), (
            'CURE must never modify Ego features')
    assert not torch.equal(
        changed[~changed_aux['ego_mask']],
        feature.detach()[~changed_aux['ego_mask']]), (
            'a nonzero correction must affect cooperative-agent features')
    print(
        'CURE unit checks passed: identity=True ego_fixed=True agents=5 '
        'recoverability=%.4f harm=%.6f fusion=%.4f' %
        (aux['restoration_gate'][~aux['ego_mask']].mean().item(),
         torch.sigmoid(
             aux['harm_logit'][~aux['ego_mask']]).mean().item(),
         aux['fusion_weight'][~aux['ego_mask']].mean().item()))


if __name__ == '__main__':
    main()
