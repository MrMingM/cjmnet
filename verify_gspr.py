"""Small server-side forward/backward check for the GSPR front end."""

import torch

from attfuse_gspr.reliability import GeometryFirstPillarReliability


def main():
    cfg = {
        "hidden_dim": 16, "context_dim": 8,
        "max_points_per_voxel": 8, "spectral_downsample": 4,
        "grid_size": [32, 16, 1], "initial_reliability": 0.95,
    }
    model = GeometryFirstPillarReliability(
        cfg, [0.4, 0.4, 4.0], [-6.4, -3.2, -3.0, 6.4, 3.2, 1.0])
    points = torch.randn(24, 8, 4, requires_grad=True)
    counts = torch.randint(1, 9, (24,))
    coords = torch.stack((
        torch.arange(24) % 3, torch.zeros(24, dtype=torch.long),
        torch.arange(24) % 16, torch.arange(24) % 32), dim=1)
    output = model(points, counts, coords)
    assert output["point_reliability"].shape == (24, 8)
    assert output["point_uncertainty"].shape == (24, 8)
    valid = output["point_valid_mask"]
    assert torch.all(output["point_reliability"][~valid] == 0)
    loss = output["point_reliability"][valid].mean() + \
        output["point_uncertainty"][valid].mean()
    loss.backward()
    assert model.fusion_bias.grad is not None
    print("GSPR forward/backward check passed")


if __name__ == "__main__":
    main()
