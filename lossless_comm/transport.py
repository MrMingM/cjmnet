"""Full-feature lossless transport and level-0 transport + receiver recompute."""
from __future__ import annotations

import time
from typing import Dict, List

import numpy as np
import torch

from gspr_communication.masked_attfuse import fuse
from .codec import pack_tensors, unpack_tensors


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    aa = a.detach().cpu().contiguous().numpy()
    bb = b.detach().cpu().contiguous().numpy()
    return (
        aa.dtype == np.float32
        and bb.dtype == np.float32
        and aa.shape == bb.shape
        and np.array_equal(
            aa.view(np.uint32).reshape(-1),
            bb.view(np.uint32).reshape(-1),
        )
    )


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).detach().abs().max().cpu()) if a.numel() else 0.0


@torch.no_grad()
def _detect_from_levels(model, received: List[List[torch.Tensor]]):
    n = sum(1 for _ in received[0])
    masks = received[0][0].new_ones((n, 1, *model.grid))
    fused = [fuse(torch.cat(parts, 0), masks) for parts in received]
    base = model.engine.base
    joined = torch.cat(
        [deblock(x) for deblock, x in zip(base.backbone.deblocks, fused)], 1
    )
    return {"psm": base.cls_head(joined), "rm": base.reg_head(joined)}


@torch.no_grad()
def lossless_full(model, encoded: Dict, backend="auto", level=3):
    """Transmit all three complete feature scales, but entropy-code them losslessly."""
    levels = encoded["levels"]
    device = levels[0].device
    received = [[x[:1]] for x in levels]

    total_packet = total_raw = 0
    encode_ms = decode_ms = 0.0
    peer_stats = []

    for peer in range(1, levels[0].shape[0]):
        host = [x[peer].detach().cpu().contiguous().numpy() for x in levels]

        t0 = time.perf_counter()
        packet, stat = pack_tensors(host, backend=backend, level=level)
        encode_ms += (time.perf_counter() - t0) * 1000.0

        shapes = [tuple(x.shape) for x in host]
        t0 = time.perf_counter()
        decoded = unpack_tensors(packet, shapes)
        decode_ms += (time.perf_counter() - t0) * 1000.0

        total_packet += len(packet)
        total_raw += stat["raw_bytes"]
        peer_stats.append(stat)

        for i, array in enumerate(decoded):
            tensor = torch.from_numpy(array).to(device=device)
            if not _bits_equal(tensor, levels[i][peer]):
                raise AssertionError(
                    f"lossless decode changed bits: peer={peer}, level={i}"
                )
            received[i].append(tensor[None])

    output = _detect_from_levels(model, received)
    return output, {
        "mode": "lossless_full",
        "total_bytes": int(total_packet),
        "raw_feature_bytes": int(total_raw),
        "compression_ratio": total_raw / max(total_packet, 1),
        "encode_ms": encode_ms,
        "decode_ms": decode_ms,
        "recompute_ms": 0.0,
        "peer_codec_stats": peer_stats,
        "max_level0_error": 0.0,
        "max_level1_error": 0.0,
        "max_level2_error": 0.0,
    }


@torch.no_grad()
def level0_recompute(model, encoded: Dict, backend="auto", level=3,
                     check_atol=2e-5, check_rtol=2e-5):
    """Send complete level-0 only; reconstruct level-1/2 using frozen blocks 1/2."""
    levels = encoded["levels"]
    base = model.engine.base
    device = levels[0].device
    received = [[levels[i][:1]] for i in range(3)]

    total_packet = total_raw = 0
    encode_ms = decode_ms = recompute_ms = 0.0
    max_err = [0.0, 0.0, 0.0]
    peer_stats = []

    for peer in range(1, levels[0].shape[0]):
        host0 = levels[0][peer].detach().cpu().contiguous().numpy()

        t0 = time.perf_counter()
        packet, stat = pack_tensors([host0], backend=backend, level=level)
        encode_ms += (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        [decoded0] = unpack_tensors(packet, [host0.shape])
        decode_ms += (time.perf_counter() - t0) * 1000.0

        total_packet += len(packet)
        total_raw += stat["raw_bytes"]
        peer_stats.append(stat)

        l0 = torch.from_numpy(decoded0).to(device=device)[None]
        if not _bits_equal(l0[0], levels[0][peer]):
            raise AssertionError(f"level0 decode changed bits for peer={peer}")

        _sync()
        t0 = time.perf_counter()
        l1 = base.backbone.blocks[1](l0)
        l2 = base.backbone.blocks[2](l1)
        _sync()
        recompute_ms += (time.perf_counter() - t0) * 1000.0

        originals = [
            levels[0][peer:peer+1],
            levels[1][peer:peer+1],
            levels[2][peer:peer+1],
        ]
        rebuilt = [l0, l1, l2]
        for i, (a, b) in enumerate(zip(rebuilt, originals)):
            max_err[i] = max(max_err[i], _max_abs(a, b))
            torch.testing.assert_close(
                a, b, atol=float(check_atol), rtol=float(check_rtol)
            )
            received[i].append(a)

    output = _detect_from_levels(model, received)
    return output, {
        "mode": "level0_recompute",
        "total_bytes": int(total_packet),
        "raw_feature_bytes": int(total_raw),
        "compression_ratio": total_raw / max(total_packet, 1),
        "encode_ms": encode_ms,
        "decode_ms": decode_ms,
        "recompute_ms": recompute_ms,
        "peer_codec_stats": peer_stats,
        "max_level0_error": max_err[0],
        "max_level1_error": max_err[1],
        "max_level2_error": max_err[2],
    }
