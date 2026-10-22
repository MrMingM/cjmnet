"""Export only the five registered small P6-A reports; no Git or data copying."""
from __future__ import annotations

import argparse
from pathlib import Path

from shared_weather_benchmark_p6.core import (REPO,read_json,digest,atomic_bytes,atomic_json,ensure_json)

NAMES = ["P6A_AUDIT.json","P6A_AUDIT.md","P6A_SCENE_TABLE.csv",
         "P6A_MATCH_FAILURES.json","P6A_ARTIFACT_MANIFEST.json"]


def export(source,destination,max_total_bytes=25*1024*1024):
    source,destination = Path(source).resolve(strict=True),Path(destination).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Export must be separate from audit source")
    commit = read_json(source/"COMMIT.json")
    if commit["status"] != "passed":
        raise ValueError("P6-A is not committed")
    registered = {r["path"]:r for r in commit["products"]}
    checked = []
    for name in NAMES:
        path = source/name
        entry = registered.get(str(path.resolve()))
        if entry is None or digest(path) != entry["sha256"]:
            raise ValueError("Export artifact not registered or changed: " + name)
        checked.append(dict(name=name,source=str(path),sha256=entry["sha256"],bytes=path.stat().st_size))
    if sum(r["bytes"] for r in checked) > max_total_bytes:
        raise ValueError("Audit export exceeds small-file budget; do not copy large data")
    for row in checked:
        target = destination/row["name"]
        if target.exists() and digest(target) != row["sha256"]:
            raise ValueError("Export destination already has a different result")
    destination.mkdir(parents=True,exist_ok=True)
    for row in checked:
        target = destination/row["name"]
        if not target.exists():
            atomic_bytes(target,Path(row["source"]).read_bytes())
    manifest = dict(status="verified_small_export",source_contract=commit["contract"],
                    files=checked,large_data_copied=False,git_commands_executed=False)
    ensure_json(destination/"EXPORT_MANIFEST.json",manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",default="/data/cjm/datasets/logs/scorw-p6-v1/audit")
    parser.add_argument("--destination",default=str(REPO/"logs/SCOR-W/P6A"))
    args = parser.parse_args()
    print(export(args.source,args.destination))


if __name__ == "__main__":
    main()
