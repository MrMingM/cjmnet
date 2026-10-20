"""Reuse UECP learned blocks; separate within-CAV coverage pooling and cross-CAV precision."""
import ast
import inspect
import textwrap

import torch
from torch.nn import functional as F

from .upstream import module
from .quality_adapter import describe

native = module('models.utils.pyramid_fusion_modulev1')


def enable_precision_gradient(fusion):
    """Exactly one AST edit in the pinned upstream forward, only for the named E2 groups."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(native.UGPrecisionFusion.forward)))
    class RemoveDetach(ast.NodeTransformer):
        edits = 0
        def visit_Call(self, node):
            self.generic_visit(node)
            if (isinstance(node.func, ast.Attribute) and node.func.attr == 'detach'
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == 'u_map'
                    and not node.args and not node.keywords):
                self.edits += 1
                return node.func.value
            return node
    edit = RemoveDetach(); tree = edit.visit(tree)
    if edit.edits != 1:
        raise ValueError('Upstream uncertainty detach contract changed')
    env = dict(vars(native))
    exec(compile(ast.fix_missing_locations(tree), '<UECP precision detach removed once>', 'exec'), env)
    from types import MethodType
    for block in fusion.blocks:
        block.forward = MethodType(env['forward'], block)


def _masked_pool(feature, coverage, quality, valid, scale):
    mask = valid.float()
    if scale == 1:
        return feature*mask, coverage*mask, quality*mask, valid
    pool = lambda x: F.avg_pool2d(x, scale, scale)
    weight = coverage.clamp_min(1e-3)*mask
    denominator = pool(weight)
    f = pool(feature*weight)/denominator.clamp_min(1e-12)
    support = pool(mask)
    c = pool(coverage*mask)  # Spatial occupancy fraction remains in C.
    q = pool(quality*mask)/support.clamp_min(1e-12)
    v = support > 0
    return f*v, c*v, q*v, v


class DualFusion(native.MSUGPrecisionFusionV1):
    @classmethod
    def from_original(cls, original):
        original.__class__ = cls  # Same parameters, buffers, names and initialization.
        original.last_stats = []
        return original

    def forward(self, feature, coverage, quality, valid, record_len, affine_matrix):
        prior = None
        self.last_stats = []
        height, width = feature.shape[-2:]
        lmax = int(record_len.max())
        for scale, block, refine in zip(self.scales, self.blocks, self.refine):
            fs, cs, qs, vs = _masked_pool(feature, coverage, quality, valid, scale)
            fb, cavmask = native._regroup(fs, record_len, lmax)
            qb, _ = native._regroup(qs, record_len, lmax)
            vb, _ = native._regroup(vs.float(), record_len, lmax)
            cb, _ = native._regroup(cs, record_len, lmax)
            results, stats, unions = [], [], []
            for b in range(len(record_len)):
                n = int(cavmask[b].sum()); hw = fs.shape[-2:]
                warp = lambda x: native._warp_affine(x, affine_matrix[b, 0, :n], hw)
                observed_mass = warp(vb[b, :n])
                support = observed_mass[:, 0] > 1e-6
                # Divide by interpolated observed mass; empty neighbors never dilute quality.
                qw = warp(qb[b, :n]*vb[b, :n])/observed_mass.clamp_min(1e-12)
                fw = warp(fb[b, :n]*vb[b, :n])/observed_mass.clamp_min(1e-12)
                precision = (block.eps + F.softplus(block.log_gamma)*qw[:, 0].clamp(0, 1))*support
                weights = precision/precision.sum(0, keepdim=True).clamp_min(1e-12)
                weights = block._blur3(weights[None])[0]*support
                weights = weights/weights.sum(0, keepdim=True).clamp_min(1e-12)
                any_valid = support.any(0)[None]
                fused = block._reduce(weights[:, None]*fw)
                ego = fw[0]*support[0]
                residual = block.post((fused-ego)[None])[0]*any_valid
                result = (ego + torch.sigmoid(block.logit_beta)*residual)*any_valid
                results.append(result)
                unions.append(any_valid)
                stats.append(dict(weights=describe(weights, support),
                    empty_weight_mass=float((weights*~support).sum()),
                    all_missing_cells=int((~any_valid).sum()),
                    maximum_source_weight=float(weights.max()),
                    normalized_error=float((weights.sum(0)[any_valid[0]]-1).abs().max()) if any_valid.any() else 0.))
            fused_s = torch.stack(results)
            ego_valid = vb[:, 0] > 0
            if prior is not None:
                fused_s = fused_s + cb[:, 0]*ego_valid*self._up(prior, fs.shape[-2:])
            # 1x1 refine has no bias, then explicit all-source mask prevents invented support.
            observed = torch.stack(unions)
            fused_s = refine(fused_s)*observed
            prior = fused_s
            self.last_stats.append(dict(scale=scale, batches=stats))
        return self._up(prior, (height, width)) if prior.shape[-2:] != (height, width) else prior
