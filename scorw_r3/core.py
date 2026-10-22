"""Reuse immutable transactions; R3 never requires Git or changes old contracts."""
from pathlib import Path
import hashlib
import numpy as np

from scorw_repair.core import (REPO,Blocked,read_json,digest,identity,seed_for,atomic_json,atomic_text,
    atomic_bytes,lock,begin,transaction,save_arrays,server_only,server_train)
HERE=Path(__file__).resolve().parent


def sources():
    snap=read_json(HERE/"source_snapshot.json")
    pins=snap["sources"]
    wrong=[n for n,h in pins.items() if not (REPO/n).is_file() or digest(REPO/n)!=h]
    actual={p.relative_to(REPO).as_posix() for p in HERE.rglob("*") if p.is_file()
        and p.name!="source_snapshot.json" and not {"local_validation","__pycache__"}.intersection(p.relative_to(HERE).parts)}
    expected={n for n in pins if n.startswith("scorw_r3/")}
    if wrong or actual!=expected:raise Blocked("R3_SOURCE_MISMATCH: "+str(dict(changed=wrong,inventory=sorted(actual^expected))))
    return dict(identity=identity(pins),snapshot_sha256=digest(HERE/"source_snapshot.json"),files=pins,git_required=False)


def array_hash(a):
    a=np.ascontiguousarray(a)
    return identity(dict(shape=list(a.shape),dtype=str(a.dtype),bytes=hashlib.sha256(a.tobytes()).hexdigest()))


def load_arrays(path):
    with np.load(path,allow_pickle=False) as data:return {k:data[k] for k in data.files}


def check_inputs(rows):
    for r in rows:
        for name in ("pcd","yaml"):
            if not Path(r[name]).is_file() or digest(r[name])!=r[name+"_sha256"]:
                raise Blocked("R3_INPUT_CHANGED: "+r["scene"]+"/"+r["timestamp"]+"/"+r["cav"])


def quantiles(values):
    a=np.asarray(values,dtype=float)
    return np.quantile(a,[0,.01,.1,.5,.9,.99,1]).tolist() if a.size else None
