"""Pure/synthetic checks for the B0 task-gap benefit audit."""
import numpy as np
import torch

from .audit_task_gap import (
    _auc,
    _category,
    _weight_maps,
    target_metrics,
)


def main():
    assert _category({0}, set(), 0) == 'split_only'
    assert _category(set(), {0}, 0) == 'reference_only'
    assert _category({0}, {0}, 0) == 'both'
    assert _category(set(), set(), 0) == 'neither'

    assert _auc([3.0, 4.0], [1.0, 2.0]) == 1.0
    assert _auc([1.0], [1.0]) == 0.5
    assert _auc([], [1.0]) is None

    # Two sources, two spatial cells. Cell (0,0) is identical; cell (0,1)
    # swaps the top source completely.
    a = torch.tensor([[[[.8, .9]]], [[[.2, .1]]]], dtype=torch.float32)
    b = torch.tensor([[[[.8, .1]]], [[[.2, .9]]]], dtype=torch.float32)
    maps = _weight_maps([a, a], [b, b])
    np.testing.assert_allclose(
        maps[0]['total_variation'].numpy(), [[0.0, 0.8]], atol=1e-6)
    np.testing.assert_allclose(
        maps[0]['top_disagreement'].numpy(), [[0.0, 1.0]], atol=0)

    # A box covering the full synthetic 1x2 grid must recover the average.
    corners = np.array([
        [-1.0, -1.0, 0.0], [1.0, -1.0, 0.0],
        [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0],
        [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0],
    ])
    metrics = target_metrics(
        maps, corners, [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])
    assert abs(
        metrics['box']['combined']['total_variation_mean'] - 0.4
    ) < 1e-6
    assert abs(
        metrics['box']['combined']['top_disagreement_mean'] - 0.5
    ) < 1e-6

    print('task-gap audit tests passed', flush=True)


if __name__ == '__main__':
    main()
