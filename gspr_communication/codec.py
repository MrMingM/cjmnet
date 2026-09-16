"""Versioned application payloads; excludes radio/IP/MAC overhead.

Requests are unicast, uint8. Responses: 16-byte header, uint32 block indices,
and fixed-layout float32/float16 multi-scale blocks in little-endian order.
No uncharged scores or quality maps are exchanged. Grid/layout is pre-shared.
"""
import struct

HEADER = struct.Struct("<4sIII")


def request_bytes(h, w):
    return HEADER.size + int(h) * int(w)


def block_bytes(channels, patches, value_bytes=4):
    if value_bytes not in (2, 4):
        raise ValueError("payload values must be float16 or float32")
    return 4 + value_bytes * sum(c * p * p for c, p in zip(channels, patches))


def allocate(budget, peers, h, w, cost):
    """Reserve a request and response header per peer; equal fixed quotas."""
    if budget < 0 or peers < 0 or cost <= 0:
        raise ValueError("Invalid budget layout")
    if peers == 0:
        return False, []
    overhead = peers * (request_bytes(h, w) + HEADER.size)
    # Do not send an unaffordable request with no feature response.
    if budget < overhead + cost:
        return False, [0] * peers
    total = min((budget - overhead) // cost, peers * h * w)
    return True, [total // peers + (i < total % peers) for i in range(peers)]


def pack_request(values):
    import numpy as np
    a = np.asarray(values)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError("Expected finite HxW request")
    q = np.rint(np.clip(a, 0, 1) * 255).astype(np.uint8)
    return HEADER.pack(b"GSR1", *q.shape, 0) + q.tobytes()


def unpack_request(packet):
    import numpy as np
    magic, h, w, reserved = HEADER.unpack_from(packet)
    if magic != b"GSR1" or reserved or len(packet) != request_bytes(h, w):
        raise ValueError("Malformed request")
    return np.frombuffer(packet, dtype=np.uint8, offset=HEADER.size).reshape(h, w).copy() / 255.0


def pack_response(levels, indices, grid, value_bytes=4):
    import numpy as np
    h, w = grid
    ids = np.asarray(indices, dtype=np.int64).reshape(-1)
    if len(set(ids.tolist())) != len(ids) or (ids < 0).any() or (ids >= h*w).any():
        raise ValueError("Invalid block indices")
    dtype = "<f4" if value_bytes == 4 else "<f2"
    if value_bytes not in (2, 4):
        raise ValueError("Invalid payload dtype")
    parts = [HEADER.pack(b"GSF1", len(ids), len(levels), value_bytes), ids.astype("<u4").tobytes()]
    for level in levels:
        a = np.asarray(level)
        c, lh, lw = a.shape
        if lh % h or lw % w or lh // h != lw // w:
            raise ValueError("Incompatible multi-scale block grid")
        p = lh // h
        blocks = a.reshape(c, h, p, w, p).transpose(1, 3, 0, 2, 4).reshape(h*w, c, p, p)
        payload = blocks[ids].astype(dtype)
        if not np.isfinite(payload).all():
            raise ValueError("Non-finite/overflowed payload")
        parts.append(payload.tobytes())
    return b"".join(parts)


def unpack_response(packet, shapes, grid):
    import numpy as np
    magic, n, count, width = HEADER.unpack_from(packet)
    if magic != b"GSF1" or count != len(shapes) or width not in (2, 4):
        raise ValueError("Malformed response header")
    h, w = grid
    patches = [s[1] // h for s in shapes]
    expected = HEADER.size + n * block_bytes([s[0] for s in shapes], patches, width)
    if len(packet) != expected:
        raise ValueError("Malformed response length")
    ids = np.frombuffer(packet, dtype="<u4", count=n, offset=HEADER.size)
    if len(set(ids.tolist())) != n or (ids >= h*w).any():
        raise ValueError("Malformed response indices")
    offset = HEADER.size + n*4
    result = []
    for (c, lh, lw), p in zip(shapes, patches):
        if lh != h*p or lw != w*p:
            raise ValueError("Incompatible grid")
        size = n*c*p*p
        values = np.frombuffer(packet, dtype="<f4" if width == 4 else "<f2", count=size, offset=offset)
        if not np.isfinite(values).all():
            raise ValueError("Non-finite response")
        offset += size*width
        blocks = np.zeros((h*w, c, p, p), dtype=np.float32)
        blocks[ids] = values.reshape(n, c, p, p)
        result.append(blocks.reshape(h, w, c, p, p).transpose(2, 0, 3, 1, 4).reshape(c, lh, lw))
    return result, ids.copy()
