"""Install a verified package-only fix before request.json exists; preserve historical files."""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import zipfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def apply_update(archive, project, output):
    project = Path(project).resolve(strict=True)
    output = Path(output).resolve(strict=True)
    # The reported failure occurs before request creation. Never rewrite an active run's contract.
    if (output / 'request.json').exists():
        raise ValueError('This output already has request.json; preserve its source contract and use a new output for changed code')
    allowed = {'.controller.lock', 'launcher.log', 'launcher.pid'}
    if any(p.name not in allowed for p in output.iterdir()):
        raise ValueError('Output contains more than startup logs; refusing a preflight-only repair')
    with zipfile.ZipFile(archive) as bundle:
        meta = json.loads(bundle.read('UPDATE.json'))
        if meta['kind'] != 'uecp_gspr_pre_request_source_scope_fix':
            raise ValueError('Unsupported update archive')
        target_snapshot = bundle.read('uecp_gspr_v2xr/source_snapshot.json')
        if digest(target_snapshot) != meta['target_snapshot_sha256']:
            raise ValueError('Update snapshot checksum differs')
        target = json.loads(target_snapshot)['files']
        payload = {'uecp_gspr_v2xr/'+name: bundle.read('uecp_gspr_v2xr/'+name) for name in target}
        for name, expected in target.items():
            if digest(payload['uecp_gspr_v2xr/'+name]) != expected:
                raise ValueError('Corrupt update payload: '+name)
        payload['uecp_gspr_v2xr/source_snapshot.json'] = target_snapshot
        base = meta['base_files']
        expected_names = set(payload) | {'UPDATE.json', '__main__.py'}
        if len(bundle.namelist()) != len(set(bundle.namelist())) or set(bundle.namelist()) != expected_names:
            raise ValueError('Unexpected update archive member')
        dependencies = json.loads(payload['uecp_gspr_v2xr/PINS.json'])['files']
        mismatches = []
        for name, expected in dependencies.items():
            try:
                actual = digest((project / name).read_bytes())
            except OSError as exc:
                mismatches.append(name+': unavailable ('+str(exc)+')')
                continue
            if actual != expected:
                mismatches.append(name+': expected='+expected+' actual='+actual)
        if mismatches:
            raise ValueError('Runtime code differs; no files were updated:\n'+'\n'.join(mismatches))
        changes = []
        for name, data in payload.items():
            path = project / name
            if not path.resolve().is_relative_to(project / 'uecp_gspr_v2xr'):
                raise ValueError('Update path escapes the new package')
            if path.exists():
                current = digest(path.read_bytes())
                if current == digest(data):
                    continue
                if current != base.get(name):
                    raise ValueError('Local changes differ from both released versions; preserve and inspect: '+str(path))
            changes.append((path, data))
        if not changes:
            return dict(status='already_applied', files=0, backup=None)
        # Keep a recoverable copy outside the sealed package and outside the experiment output.
        backup = project / ('uecp_gspr_before_source_fix_'+str(time.time_ns())+'.zip')
        with zipfile.ZipFile(backup, 'x', compression=zipfile.ZIP_DEFLATED) as previous:
            for path, _ in changes:
                if path.exists():
                    previous.write(path, path.relative_to(project).as_posix())
        changes.sort(key=lambda row: row[0].name == 'source_snapshot.json')
        for path, data in changes:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=project, prefix='.uecp_source_fix_', delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(data)
                stream.flush()
            try:
                temporary.replace(path)
            finally:
                if temporary.exists():
                    temporary.unlink()
        return dict(status='updated', files=len(changes), backup=str(backup))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if sys.platform != 'linux':
        raise RuntimeError('Install this fix on the Linux server; local tests exercise only temporary fixtures')
    import fcntl
    with contextlib.ExitStack() as stack:
        lock = stack.enter_context((Path(args.output) / '.controller.lock').open('a+'))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = apply_update(Path(sys.argv[0]), args.project, args.output)
    print(json.dumps(result, ensure_ascii=False))
    print('Only uecp_gspr_v2xr code was updated. Historical GSPR files and startup logs were retained.')


if __name__ == '__main__':
    main()
