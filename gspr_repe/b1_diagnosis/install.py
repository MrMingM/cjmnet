"""Install an external read-only tool link without changing REPE snapshots."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def hashes(folder):
    return {p.relative_to(folder).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts
            and p.name!='source_snapshot.json'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',default='/home/cjm/OpenCOOD-main/cjmnet')
    args=parser.parse_args()
    if sys.platform!='linux':raise SystemExit('Install on server only')
    source=Path(__file__).resolve().parent
    manifest=json.loads((source/'TOOL_MANIFEST.json').read_text())
    for name,expected_hash in manifest['files'].items():
        if Path(name).name!=name or hashlib.sha256((source/name).read_bytes()).hexdigest()!=expected_hash:
            raise SystemExit('Diagnostic tool hash mismatch: '+name)
    package=Path(args.project).resolve(strict=True)/'gspr_repe'
    if package in source.parents:raise SystemExit('Extract tools outside gspr_repe first; do not overwrite snapshots')
    expected=json.loads((package/'source_snapshot.json').read_text())['files']
    before=hashes(package)
    if before!=expected:raise SystemExit('Existing REPE source snapshot mismatch; installation refused')
    link=package/'b1_diagnosis'
    if link.is_symlink():
        if link.resolve()!=source:raise SystemExit('Existing link points elsewhere; preserved')
    elif link.exists():raise SystemExit('Existing diagnostic directory preserved; use external deployment')
    else:link.symlink_to(source,target_is_directory=True)
    if hashes(package)!=before:
        link.unlink()  # Only the new link, never files or a directory.
        raise SystemExit('This Python follows directory symlinks; reverted link')
    print('INSTALLED',str(link),'->',str(source),'REPE_SOURCE_IDENTITY_UNCHANGED')


if __name__=='__main__':main()
