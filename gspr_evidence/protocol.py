"""Sparse, quantized requests. Only decoded fields enter the responder."""
import struct
import numpy as np
from gspr_communication.codec import HEADER, block_bytes, pack_response, unpack_response

FIELDS = 17  # p,u,support,known,height support x4, ray samples x4,conf,risk x3,restore
HEAD = struct.Struct('<4sIIII')
POSE_BYTES = 64


def request_bytes(count):
    return HEAD.size + POSE_BYTES + int(count) * (4 + FIELDS)


def pack_request(profile, ids, pose):
    profile = np.asarray(profile, dtype=np.float32)
    ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    pose = np.asarray(pose, dtype=np.float32)
    if profile.ndim != 3 or profile.shape[0] != FIELDS or pose.shape != (4, 4):
        raise ValueError('Invalid request shape')
    h, w = profile.shape[1:]
    if (not np.isfinite(profile).all() or not np.isfinite(pose).all()
            or len(set(ids.tolist())) != len(ids) or (ids < 0).any() or (ids >= h*w).any()):
        raise ValueError('Nonfinite/duplicate/out of range request')
    values = np.rint(profile.reshape(FIELDS, -1)[:, ids].T.clip(0, 1)*255).astype('u1')
    return (HEAD.pack(b'GED1', h, w, len(ids), FIELDS) + pose.astype('<f4').tobytes()
            + ids.astype('<u4').tobytes() + values.tobytes())


def unpack_request(packet):
    if len(packet) < HEAD.size + POSE_BYTES:
        raise ValueError('Truncated request')
    magic, h, w, n, fields = HEAD.unpack_from(packet)
    if magic != b'GED1' or fields != FIELDS or not h or not w or n > h*w or len(packet) != request_bytes(n):
        raise ValueError('Malformed request')
    pose = np.frombuffer(packet, '<f4', 16, HEAD.size).copy().reshape(4, 4)
    ids = np.frombuffer(packet, '<u4', n, HEAD.size + POSE_BYTES).copy().astype(np.int64)
    if not np.isfinite(pose).all() or len(set(ids.tolist())) != n or (ids >= h*w).any():
        raise ValueError('Invalid request contents')
    values = np.frombuffer(packet, 'u1', n*FIELDS, HEAD.size + POSE_BYTES + 4*n).reshape(n, FIELDS).copy()/255.
    return values.astype(np.float32), ids, pose, (h, w)


def allocate(budget, peers, queries, cost):
    if peers < 1:
        return False, []
    overhead = peers * (request_bytes(queries) + HEADER.size)
    k = min(peers*queries, max(0, (int(budget) - overhead)//cost))
    if k < peers:  # symmetric protocol requires >=1 response block per peer
        return False, [0]*peers
    return True, [k//peers + (i < k % peers) for i in range(peers)]
