"""Sparse review queries and quantized geometric evidence. Unicast byte accounting."""
import struct
import numpy as np

HEADER = struct.Struct('<4sIII')
SIDE = 8
CHANNELS = 4  # reliable belief, noise belief, unknown, retained support


def query_count(budget, peers, fraction, maximum, eligible, z_bins):
    if budget < 0 or peers < 0 or not 0 <= fraction < 1 or maximum < 0 or eligible < 0 or z_bins < 1:
        raise ValueError('Invalid review budget')
    if not peers:
        return 0
    per_query = 8 + SIDE * SIDE * z_bins * CHANNELS
    return max(0, min(maximum, eligible, (int(budget * fraction) - peers * 2 * HEADER.size) // (peers * per_query)))


def _ids(indices, grid):
    h, w = map(int, grid)
    ids = np.asarray(indices, dtype=np.int64)
    if h <= 0 or w <= 0 or ids.ndim != 1 or len(np.unique(ids)) != len(ids) or (ids < 0).any() or (ids >= h*w).any():
        raise ValueError('Invalid review block indices/grid')
    return ids


def pack_query(indices, grid):
    ids = _ids(indices, grid)
    return HEADER.pack(b'GRQ1', len(ids), *grid) + ids.astype('<u4').tobytes()


def unpack_query(packet, grid):
    if len(packet) < HEADER.size:
        raise ValueError('Truncated query')
    magic, n, h, w = HEADER.unpack_from(packet)
    if magic != b'GRQ1' or (h, w) != tuple(grid) or len(packet) != HEADER.size + n*4:
        raise ValueError('Malformed query')
    return _ids(np.frombuffer(packet, '<u4', n, HEADER.size), grid).copy()


def pack_evidence(indices, values, grid, z_bins):
    ids = _ids(indices, grid)
    a = np.asarray(values)
    if z_bins < 1 or a.shape != (len(ids), SIDE, SIDE, z_bins, CHANNELS) or not np.isfinite(a).all():
        raise ValueError('Invalid evidence shape/values')
    q = np.rint(a.clip(0, 1) * 255).astype('u1')
    return HEADER.pack(b'GRE1', len(ids), z_bins, CHANNELS) + ids.astype('<u4').tobytes() + q.tobytes()


def unpack_evidence(packet, grid, z_bins, requested):
    if len(packet) < HEADER.size:
        raise ValueError('Truncated evidence')
    magic, n, z, c = HEADER.unpack_from(packet)
    size = SIDE * SIDE * z_bins * CHANNELS
    if magic != b'GRE1' or z != z_bins or c != CHANNELS or len(packet) != HEADER.size + n*(4+size):
        raise ValueError('Malformed evidence')
    ids = _ids(np.frombuffer(packet, '<u4', n, HEADER.size), grid)
    if not np.array_equal(ids, _ids(requested, grid)):
        raise ValueError('Unrequested/reordered evidence')
    values = np.frombuffer(packet, 'u1', n*size, HEADER.size+n*4).reshape(n, SIDE, SIDE, z, c)
    return values.astype(np.float32) / 255
