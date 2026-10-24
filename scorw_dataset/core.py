"""Content identities and transactions, independent of historical split guards."""
from pathlib import Path
import os
import platform
from shared_weather_benchmark.provenance import atomic_json,atomic_text,atomic_bytes,digest,identity,seed_for
from scorw_p7.core import read_json,lock,begin,transaction,save_arrays

HERE=Path(__file__).resolve().parent
REPO=HERE.parent
SPLITS=("train","validate","test")


def server_only():
    if platform.system()!="Linux" or REPO!=Path("/home/cjm/OpenCOOD-main/cjmnet").resolve():
        raise RuntimeError("SCOR-W dataset execution is restricted to the designated Linux server")


def source_identity():
    pins=read_json(HERE/"source_snapshot.json")["sources"]
    wrong=[p for p,h in pins.items() if not (REPO/p).is_file() or digest(REPO/p)!=h]
    actual={p.relative_to(REPO).as_posix() for p in HERE.rglob("*") if p.is_file() and
        p.name!="source_snapshot.json" and not {"local_validation","__pycache__"}.intersection(p.relative_to(HERE).parts)}
    if wrong or actual!={p for p in pins if p.startswith("scorw_dataset/")}:
        raise ValueError("SOURCE_MISMATCH: synchronize the complete delivery; do not edit a server snapshot: "+str(wrong))
    # Native dataset loaders are needed only in D5's isolated process. Their actual
    # server bytes are frozen here, including the historically different basedataset.
    native={p.relative_to(REPO).as_posix():digest(p) for folder in ("opencood","attfuse_code")
        for p in sorted((REPO/folder).rglob("*.py")) if "__pycache__" not in p.parts}
    return dict(pinned=pins,native_runtime=native,identity=identity(dict(pins=pins,native=native)),
        native_policy="actual server source bytes frozen before IO; historical dataset source is not rewritten",git_required=False)


def load_config():
    c=read_json(HERE/"config.json");r3=read_json(REPO/"scorw_r3/config.json")
    if c["field"]!=r3["field"] or c["receiver"]!=r3["receiver"] or c["model"]!=r3["model"]:
        raise ValueError("Dataset physics/model differs from frozen R3")
    if c["receiver"]["algorithm"]!="single-scattering-finite-pulse-strongest-total-peak-v1":raise ValueError("Unexpected receiver")
    return c


def source_path(root,relative):
    root=Path(root).resolve(strict=True);p=(root/relative).resolve(strict=True)
    if p!=root and root not in p.parents:raise ValueError("Source link escapes original OPV2V root")
    return p


def verify_input(row):
    for key in ("pcd","yaml"):
        if digest(row[key])!=row[key+"_sha256"]:raise ValueError("INPUT_CHANGED: "+row[key])


def immutable_json(path,value):
    path=Path(path)
    if path.exists():
        if read_json(path)!=value:raise ValueError("Completed identity record changed: "+str(path))
    else:atomic_json(path,value)


def ensure_link(source,target):
    source=Path(source).resolve(strict=True);target=Path(target)
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.is_symlink():
        if target.resolve(strict=True)!=source:raise ValueError("LINK_IDENTITY_CHANGED: "+str(target))
    elif target.exists():raise ValueError("Expected a metadata/Clean-view symbolic link: "+str(target))
    else:
        temporary=target.with_name(target.name+".partial")
        if temporary.is_symlink():temporary.unlink()
        temporary.symlink_to(source,target_is_directory=source.is_dir());os.replace(temporary,target)


def field_from(manifest):
    from shared_weather_benchmark.world_field import WorldField
    f=WorldField(**{k:manifest[k] for k in ("mean_alpha","length_m","contrast","seed","kind","bounds")},modes=len(manifest["coefficients"]))
    if f.manifest()!=manifest:raise ValueError("FIELD_IDENTITY_CHANGED")
    return f
