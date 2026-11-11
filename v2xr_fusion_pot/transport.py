"""Entropy-regularized transport on a reciprocal, bounded BEV neighborhood."""
from typing import NamedTuple

import torch


class Plan(NamedTuple):
    weights: torch.Tensor
    row: torch.Tensor
    col: torch.Tensor
    row_remaining: torch.Tensor
    col_remaining: torch.Tensor
    residual: torch.Tensor


class Grid:
    def __init__(self, height, width, window, device):
        if window not in (3, 5):
            raise ValueError('Expected a 3x3 or 5x5 correspondence window')
        radius = window // 2
        cells = torch.arange(height * width, device=device)
        offsets = torch.arange(-radius, radius + 1, device=device)
        dy, dx = (t.flatten() for t in torch.meshgrid(offsets, offsets, indexing='ij'))
        y = cells[:, None] // width + dy
        x = cells[:, None] % width + dx
        self.valid = (y >= 0) & (y < height) & (x >= 0) & (x < width)
        self.neighbor = (y.clamp(0, height - 1) * width + x.clamp(0, width - 1)).long()
        self.reverse = torch.arange(window * window - 1, -1, -1, device=device)
        self.dy, self.dx = dy, dx

    def transpose(self, edges):
        return edges[:, self.neighbor, self.reverse].masked_fill(~self.valid, 0)


def log_reduce(logits, allowed):
    # logsumexp(-inf, ..., -inf) has an undefined backward derivative.
    exists = allowed.any(-1, keepdim=True)
    safe = logits.masked_fill(~allowed, -torch.inf)
    safe = torch.where(exists, safe, 0.)
    return safe.logsumexp(-1).masked_fill(~exists.squeeze(-1), 0.)


def finish(weights, a, b, grid, residual):
    row = weights.sum(-1)
    col = grid.transpose(weights).sum(-1)
    return Plan(weights, row, col, (a - row).clamp_min(0), (b - col).clamp_min(0), residual)


def unbalanced(cost, a, b, grid, epsilon, rho_ego, rho_neighbor, steps):
    """Minimize <P,C> + eps*entropy(P) + rho_ego*KL(P1,a) + rho_neighbor*KL(P^T1,b)."""
    cost, a, b = cost.float(), a.float(), b.float()
    allowed = grid.valid & (a[..., None] > 0) & (b[:, grid.neighbor] > 0)
    incoming = grid.transpose(allowed)
    kernel = -cost / epsilon
    transposed = grid.transpose(kernel)
    alpha, beta = rho_ego / (rho_ego + epsilon), rho_neighbor / (rho_neighbor + epsilon)
    log_a, log_b = a.clamp_min(1e-30).log(), b.clamp_min(1e-30).log()
    active_a, active_b = allowed.any(-1), incoming.any(-1)
    # Reuse the static edge masks in the loop. Inactive rows have constant finite logits.
    kernel = torch.where(active_a[..., None], kernel.masked_fill(~allowed, -torch.inf), 0.)
    transposed = torch.where(active_b[..., None], transposed.masked_fill(~incoming, -torch.inf), 0.)
    v = torch.zeros_like(b)
    for _ in range(steps):
        u = (alpha * (log_a - (kernel + v[:, grid.neighbor]).logsumexp(-1))).masked_fill(~active_a, 0.)
        v = (beta * (log_b - (transposed + u[:, grid.neighbor]).logsumexp(-1))).masked_fill(~active_b, 0.)
    log_plan = kernel + u[..., None] + v[:, grid.neighbor]
    weights = log_plan.masked_fill(~allowed, -torch.inf).exp()
    with torch.no_grad():
        next_u = alpha * (log_a - (kernel + v[:, grid.neighbor]).logsumexp(-1))
        residual = (u - next_u).abs().masked_fill(~active_a, 0.).amax()
    return finish(weights, a, b, grid, residual)


def attention(cost, a, b, grid, temperature):
    """P1 uses the same edges and scores, with ordinary row-wise Softmax."""
    allowed = grid.valid & (a[..., None] > 0) & (b[:, grid.neighbor] > 0)
    logits = -cost.float() / temperature
    weights = (logits - log_reduce(logits, allowed)[..., None]).masked_fill(~allowed, -torch.inf).exp()
    return finish(weights * a[..., None], a, b, grid, cost.new_zeros(()))


def dustbin(cost, a, b, grid, epsilon, bin_score, steps):
    """Balanced comparison with an extra row/column, each with the opposite total mass."""
    cost, a, b = cost.float(), a.float(), b.float()
    kernel = -cost / epsilon
    transposed = grid.transpose(kernel)
    observed_a, observed_b = a > 0, b > 0
    allowed = grid.valid & observed_a[..., None] & observed_b[:, grid.neighbor]
    incoming = grid.transpose(allowed)
    # The virtual-to-virtual edge is needed for the augmented problem's equal totals.
    mass_a, mass_b = a.sum(-1, keepdim=True), b.sum(-1, keepdim=True)
    valid_pair = (mass_a > 0) & (mass_b > 0)
    observed_a = observed_a & valid_pair
    observed_b = observed_b & valid_pair
    log_a, log_b = a.clamp_min(1e-30).log(), b.clamp_min(1e-30).log()
    v, v_bin = torch.zeros_like(b), torch.zeros_like(mass_a)

    def real_update(logits, allowed_edges, bin_logits, log_mass, observed):
        logits = torch.cat((logits.masked_fill(~allowed_edges, -torch.inf),
                            bin_logits.expand_as(log_mass)[..., None]), -1)
        return (log_mass - logits.logsumexp(-1)).masked_fill(~observed, 0.)

    for _ in range(steps):
        u = real_update(kernel + v[:, grid.neighbor], allowed, bin_score + v_bin, log_a, observed_a)
        u_bin = mass_b.clamp_min(1e-30).log() - bin_score - torch.cat(
            (v.masked_fill(~observed_b, -torch.inf), v_bin), -1).logsumexp(-1, keepdim=True)
        v = real_update(transposed + u[:, grid.neighbor], incoming, bin_score + u_bin, log_b, observed_b)
        v_bin = mass_a.clamp_min(1e-30).log() - bin_score - torch.cat(
            (u.masked_fill(~observed_a, -torch.inf), u_bin), -1).logsumexp(-1, keepdim=True)
    weights = (kernel + u[..., None] + v[:, grid.neighbor]).masked_fill(
        ~(allowed & valid_pair[..., None]), -torch.inf).exp()
    plan = finish(weights, a, b, grid, cost.new_zeros(()))
    with torch.no_grad():
        error = (plan.row + (bin_score + u + v_bin).exp() - a).abs()
        residual = (error / a.clamp_min(1e-30)).masked_fill(~observed_a, 0.).amax()
    return plan._replace(residual=residual)
