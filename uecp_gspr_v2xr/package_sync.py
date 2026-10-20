"""Create a code-only ZIP from sealed relative source identities; no datasets/checkpoints/Git."""
import argparse
import json
from pathlib import Path
import sys
import zipfile

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from uecp_gspr_v2xr.common import HERE, ROOT, sources, sha


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', default=str(ROOT / 'uecp_gspr_v2xr_sync.zip'))
    parser.add_argument('--update-from', help='Old released sync ZIP: build a package-only pre-request repair ZIP')
    args = parser.parse_args(); state = sources()
    path = Path(args.output).resolve()
    if path.exists():
        raise ValueError('Existing sync archive refused; choose another filename')
    if HERE == path.parent or HERE in path.parents:
        raise ValueError('Do not add an archive to the sealed code directory')
    names = set() if args.update_from else set(state['dependencies'])
    names |= {p.relative_to(ROOT).as_posix() for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(names):
            archive.write(ROOT / name, name)
        if args.update_from:
            with zipfile.ZipFile(args.update_from) as old:
                old_snapshot = old.read('uecp_gspr_v2xr/source_snapshot.json')
                base = json.loads(old_snapshot)['files']
                base = {'uecp_gspr_v2xr/'+n: h for n, h in base.items()}
                import hashlib
                base['uecp_gspr_v2xr/source_snapshot.json'] = hashlib.sha256(old_snapshot).hexdigest()
            meta = dict(kind='uecp_gspr_pre_request_source_scope_fix', base_files=base,
                target_snapshot_sha256=sha(HERE / 'source_snapshot.json'))
            archive.writestr('UPDATE.json', json.dumps(meta, indent=2)+'\n')
            archive.writestr('__main__.py', 'import sys, zipfile\nwith zipfile.ZipFile(sys.argv[0]) as z:\n'
                '    source = z.read("uecp_gspr_v2xr/install_update.py")\n'
                'exec(compile(source, "<UECP source-scope installer>", "exec"), {"__name__": "__main__"})\n')
    print(str(path)); print('files=', len(names), 'bytes=', path.stat().st_size, 'sha256=', sha(path))


if __name__ == '__main__':
    main()
