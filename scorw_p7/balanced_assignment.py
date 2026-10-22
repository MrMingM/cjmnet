"""Finite-pool cyclic design indexed by stable vehicle identity, never input order."""
from __future__ import annotations

from scorw_p7.core import identity


def assignment(cavs, ego, size=20, shift=1):
    if size < 2 or len(cavs) != len(set(cavs)) or ego not in cavs or not cavs:
        raise ValueError("Invalid pool/CAV/ego identity")
    order = [ego] + sorted(c for c in cavs if c != ego)
    if len(order) > size:
        raise ValueError("More CAVs than fields; distinct offsets impossible")
    offsets = {c: i for i, c in enumerate(order)}
    schedules = {"E2": {}, "E3": {}, "E2_shift": {}}
    for c in order:
        schedules["E2"][c] = list(range(size))
        schedules["E3"][c] = [(j + offsets[c]) % size for j in range(size)]
        schedules["E2_shift"][c] = [(j + shift) % size for j in range(size)]
    for schedule in schedules.values():
        if any(sorted(v) != list(range(size)) for v in schedule.values()):
            raise ValueError("A vehicle did not traverse the entire pool exactly once")
    if any(len({schedules["E3"][c][j] for c in order}) != len(order) for j in range(size)):
        raise ValueError("E3 field collision in a repetition")
    data = dict(cavs=order, ego=ego, offsets=offsets, size=size, schedules=schedules,
                rule="ego offset0; lexically sorted non-ego vehicle identities offsets1..N-1")
    return dict(data, sha256=identity(data))
