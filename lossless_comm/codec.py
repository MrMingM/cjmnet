"""Exact lossless codec for float32 BEV feature tensors.

This module does not modify GSPR or AttFuse. It preserves every IEEE-754 bit.
Encoding:
1) optional +0.0 bitmap compaction,
2) byte shuffle of remaining float32 words,
3) Zstandard when available, otherwise zlib.

The packet stores the backend, so decoding is deterministic.
"""
from __future__ import annotations

import struct
import zlib
from typing import Iterable, Sequence

import numpy as np

MAGIC = b"LCF1"
VERSION = 1
HEAD = struct.Struct("<4sBBH")      # magic, version, backend, tensor count
TENSOR = struct.Struct("<BIII")    # mode, words, nnz, compressed bytes

BACKENDS = {"zstd": 1, "zlib": 2}
BACKEND_NAMES = {v: k for k, v in BACKENDS.items()}
MODE_DENSE = 0
MODE_SPARSE_ZERO = 1


def resolve_backend(backend: str) -> str:
    if backend == "auto":
        try:
            import zstandard  # noqa: F401
            return "zstd"
        except ImportError:
            return "zlib"
    if backend not in BACKENDS:
        raise ValueError(f"backend must be auto/zstd/zlib, got {backend}")
    return backend


def _compress(data: bytes, backend: str, level: int) -> bytes:
    if backend == "zlib":
        return zlib.compress(data, level=max(0, min(int(level), 9)))
    if backend == "zstd":
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise RuntimeError(
                "Install zstandard in the opencood environment: pip install zstandard"
            ) from exc
        return zstd.ZstdCompressor(level=int(level)).compress(data)
    raise ValueError(backend)


def _decompress(data: bytes, backend: str, expected_size: int) -> bytes:
    if backend == "zlib":
        raw = zlib.decompress(data)
    elif backend == "zstd":
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise RuntimeError("Install zstandard to decode this packet") from exc
        raw = zstd.ZstdDecompressor().decompress(
            data, max_output_size=int(expected_size)
        )
    else:
        raise ValueError(backend)
    if len(raw) != int(expected_size):
        raise ValueError(f"decompressed size mismatch: {len(raw)} != {expected_size}")
    return raw


def _as_words(array) -> np.ndarray:
    a = np.asarray(array)
    if a.dtype != np.float32:
        raise ValueError(f"v1 exact codec requires float32, got {a.dtype}")
    if not np.isfinite(a).all():
        raise ValueError("feature tensor contains non-finite values")
    # Current training/eval server is little-endian x86/HIP. Enforce C layout.
    return np.ascontiguousarray(a).view(np.uint32).reshape(-1)


def _shuffle(words: np.ndarray) -> bytes:
    """Reversible byte shuffle of 32-bit words."""
    if words.size == 0:
        return b""
    b = np.ascontiguousarray(words).view(np.uint8).reshape(-1, 4)
    return np.ascontiguousarray(b.T).tobytes()


def _unshuffle(blob: bytes, count: int) -> np.ndarray:
    count = int(count)
    if len(blob) != count * 4:
        raise ValueError("invalid shuffled byte count")
    if count == 0:
        return np.empty(0, dtype=np.uint32)
    b = np.frombuffer(blob, dtype=np.uint8).reshape(4, count).T.copy()
    return b.reshape(-1).view(np.uint32)


def _pack_one(array, backend: str, level: int):
    words = _as_words(array)
    n = int(words.size)

    dense_raw = _shuffle(words)
    dense_cmp = _compress(dense_raw, backend, level)

    # Preserve -0.0 exactly. Only the +0.0 bit pattern is omitted.
    nz = words != np.uint32(0)
    nnz = int(nz.sum())
    bitmap = np.packbits(nz.astype(np.uint8), bitorder="little").tobytes()
    sparse_raw = bitmap + _shuffle(words[nz])
    sparse_cmp = _compress(sparse_raw, backend, level)

    if len(sparse_cmp) < len(dense_cmp):
        mode, payload = MODE_SPARSE_ZERO, sparse_cmp
    else:
        mode, payload = MODE_DENSE, dense_cmp

    meta = TENSOR.pack(mode, n, nnz, len(payload))
    stat = {
        "mode": "sparse_zero" if mode == MODE_SPARSE_ZERO else "dense_shuffle",
        "raw_bytes": n * 4,
        "packet_component_bytes": len(meta) + len(payload),
        "zero_fraction": 1.0 - nnz / max(n, 1),
    }
    return meta + payload, stat


def pack_tensors(
    tensors: Iterable[np.ndarray], backend: str = "auto", level: int = 3
):
    tensors = list(tensors)
    backend = resolve_backend(backend)
    parts = [HEAD.pack(MAGIC, VERSION, BACKENDS[backend], len(tensors))]
    stats = []
    for tensor in tensors:
        chunk, stat = _pack_one(tensor, backend, int(level))
        parts.append(chunk)
        stats.append(stat)
    packet = b"".join(parts)
    return packet, {
        "backend": backend,
        "level": int(level),
        "packet_bytes": len(packet),
        "raw_bytes": sum(x["raw_bytes"] for x in stats),
        "compression_ratio": (
            sum(x["raw_bytes"] for x in stats) / max(len(packet), 1)
        ),
        "tensors": stats,
    }


def unpack_tensors(packet: bytes, shapes: Sequence[Sequence[int]]):
    if len(packet) < HEAD.size:
        raise ValueError("truncated packet")
    magic, version, backend_code, count = HEAD.unpack_from(packet)
    if magic != MAGIC or version != VERSION or backend_code not in BACKEND_NAMES:
        raise ValueError("invalid packet header")
    if count != len(shapes):
        raise ValueError("tensor count does not match pre-shared shapes")

    backend = BACKEND_NAMES[backend_code]
    offset = HEAD.size
    outputs = []

    for shape in shapes:
        if offset + TENSOR.size > len(packet):
            raise ValueError("truncated tensor metadata")
        mode, words_count, nnz, payload_len = TENSOR.unpack_from(packet, offset)
        offset += TENSOR.size
        if offset + payload_len > len(packet):
            raise ValueError("truncated compressed payload")
        payload = packet[offset:offset + payload_len]
        offset += payload_len

        expected_words = int(np.prod(shape))
        if words_count != expected_words or nnz > words_count:
            raise ValueError("tensor metadata does not match expected shape")

        if mode == MODE_DENSE:
            raw = _decompress(payload, backend, words_count * 4)
            words = _unshuffle(raw, words_count)
        elif mode == MODE_SPARSE_ZERO:
            bitmap_bytes = (words_count + 7) // 8
            raw = _decompress(payload, backend, bitmap_bytes + nnz * 4)
            mask = np.unpackbits(
                np.frombuffer(raw[:bitmap_bytes], dtype=np.uint8),
                bitorder="little",
            )[:words_count].astype(bool)
            if int(mask.sum()) != nnz:
                raise ValueError("sparse bitmap nnz mismatch")
            words = np.zeros(words_count, dtype=np.uint32)
            words[mask] = _unshuffle(raw[bitmap_bytes:], nnz)
        else:
            raise ValueError(f"unknown tensor mode {mode}")

        array = words.view(np.float32).reshape(tuple(int(x) for x in shape)).copy()
        outputs.append(array)

    if offset != len(packet):
        raise ValueError("trailing bytes in packet")
    return outputs
