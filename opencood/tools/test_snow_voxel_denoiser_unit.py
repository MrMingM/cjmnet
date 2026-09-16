"""CPU unit checks for the pillar-level snow voxel denoiser."""

import torch
import torch.nn.functional as F

from opencood.models.sub_modules.snow_voxel_denoiser import (
    SnowVoxelDenoiser, snow_voxel_novelty_labels)


def main():
    torch.manual_seed(20260727)
    clean_coords = torch.tensor([
        [0, 0, 10, 20],
        [0, 0, 11, 20],
        [1, 0, 10, 20],
    ], dtype=torch.int32)
    weather_coords = torch.tensor([
        [0, 0, 11, 20],
        [0, 0, 12, 20],
        [1, 0, 10, 20],
    ], dtype=torch.int32)
    novelty = snow_voxel_novelty_labels(
        clean_coords, weather_coords)
    expected = torch.tensor([0.0, 1.0, 0.0])
    assert torch.equal(novelty.cpu(), expected), (
        novelty, expected)

    model = SnowVoxelDenoiser(
        channels=64, hidden_channels=16, max_points=32,
        grid_size=[176, 48, 1], gate_floor=0.05)
    features = torch.randn(3, 64)
    counts = torch.tensor([8, 1, 5], dtype=torch.int32)
    gated, aux = model(features, counts, weather_coords)

    # Enabling the module must be exactly identity before training.
    assert torch.equal(gated, features)
    assert torch.equal(aux['keep_weight'], torch.ones(3))
    assert torch.equal(
        aux['noise_logit'], torch.zeros_like(aux['noise_logit']))

    loss = F.binary_cross_entropy_with_logits(
        aux['noise_logit'], novelty)
    loss.backward()
    final_layer = model.classifier[-1]
    assert final_layer.weight.grad is not None
    assert float(final_layer.weight.grad.abs().sum()) > 0.0
    assert final_layer.bias.grad is not None

    print(
        'Snow voxel denoiser unit checks passed: '
        'novel=%d/%d identity_keep=%.3f gradient=%.6f' %
        (int(novelty.sum().item()), novelty.numel(),
         float(aux['keep_weight'].mean().item()),
         float(final_layer.weight.grad.abs().sum().item())))


if __name__ == '__main__':
    main()
