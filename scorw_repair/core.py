"""Reuse existing atomic transactions and identity machinery, without import guards."""
from pathlib import Path
import platform
import sys

from scorw_p7.core import (REPO, atomic_json, atomic_text, atomic_bytes, read_json, digest,
    identity, seed_for, lock, begin, transaction, verify_inventory, save_arrays, server_train)

HERE = Path(__file__).resolve().parent


class Blocked(RuntimeError):
    """Public diagnostic messages must contain filenames/statuses, never URLs."""


def server_only():
    if platform.system() != "Linux" or REPO != Path("/home/cjm/OpenCOOD-main/cjmnet").resolve():
        raise Blocked("SERVER_ONLY: execute this entry on the designated Linux server")


class _DatasetImportGuard:
    def find_spec(self,fullname,path=None,target=None):
        if fullname=="opencood.data_utils.datasets" or fullname.startswith("opencood.data_utils.datasets."):
            raise Blocked("R2_FORBIDDEN_HISTORICAL_DATASET_IMPORT")
        return None


def install_dataset_guard():
    if any(n=="opencood.data_utils.datasets" or n.startswith("opencood.data_utils.datasets.") for n in sys.modules):
        raise Blocked("Historical dataset modules already loaded; use a fresh repair process")
    if not any(isinstance(f,_DatasetImportGuard) for f in sys.meta_path):
        sys.meta_path.insert(0,_DatasetImportGuard())


def sources():
    pins = read_json(HERE/"source_snapshot.json")["sources"]
    changed = [n for n,h in pins.items() if not (REPO/n).is_file() or digest(REPO/n) != h]
    actual = {p.relative_to(REPO).as_posix() for p in HERE.rglob("*") if p.is_file()
        and (p.suffix in {".py", ".sh", ".json", ".md", ".txt"} or p.name=="LICENSE")
        and not {"local_validation", "__pycache__", "_private", "official_stf"}.intersection(p.relative_to(HERE).parts)
        and p.name != "source_snapshot.json"}
    expected = {n for n in pins if n.startswith("scorw_repair/")}
    if changed or actual != expected:
        raise Blocked("SOURCE_MISMATCH: sync full delivery; never rewrite a server snapshot: "
                      + str(dict(changed=changed, inventory=sorted(actual ^ expected))))
    return dict(identity=identity(pins), snapshot_sha256=digest(HERE/"source_snapshot.json"),
                files=pins, git_required=False)


def load_config():
    config=read_json(HERE/"config.json")
    original=read_json(REPO/"scorw_p7/config/protocol.json")
    if config["field"]!=original["field"] or config["pool_size"]!=20:
        raise Blocked("Frozen world-field parameters or full twenty-field pool changed")
    if config.get("real_reference",{}).get("enabled") is not False:
        raise Blocked("Real reference IO/download is disabled in this revision")
    if config.get("physical_calibration_status")!="NOT_VALIDATED_NO_REAL_REFERENCE":
        raise Blocked("Real calibration cannot be promoted without a new reviewed protocol")
    return config


def checked_path(root, relative):
    p = Path(relative)
    if p.is_absolute() or ".." in p.parts or "\\" in str(relative):
        raise Blocked("Unsafe archive member path")
    target = Path(root)/p
    if Path(root).resolve() not in target.resolve().parents:
        raise Blocked("Archive member escapes extraction root")
    return target
