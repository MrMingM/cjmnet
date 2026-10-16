"""AST/archive/hash checks only. Safe locally: no Torch, OpenCOOD or real data."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    package = json.loads((HERE / 'source_snapshot.json').read_text(encoding='utf-8'))['files']
    actual = {p.relative_to(HERE).as_posix(): digest(p.read_bytes()) for p in HERE.rglob('*')
              if p.is_file() and p.name != 'source_snapshot.json' and '__pycache__' not in p.parts}
    assert package == actual, 'Package hash mismatch'
    upstream = json.loads((HERE / 'UPSTREAM.json').read_text(encoding='utf-8'))
    with zipfile.ZipFile(HERE / 'upstream_archive.zip') as archive:
        prefix = 'V2X-R-' + upstream['commit'] + '/'
        files = {n[len(prefix):]: digest(archive.read(n)) for n in archive.namelist()
                 if n.startswith(prefix) and not n.endswith('/')}
        assert files == upstream['files'], 'Archive/upstream index mismatch'
        for name, sha in files.items():
            assert digest((HERE / 'upstream' / name).read_bytes()) == sha, name
    for path in HERE.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        assert b'\r' not in path.read_bytes(), f'CRLF: {path}'
    from audit_native_dependencies import audit
    dependencies = audit(HERE / 'upstream')
    assert not dependencies['pointnet2_reached']
    assert 'opencood.pcdet_utils.roiaware_pool3d.roiaware_pool3d_cuda' in dependencies['native_imports']
    shell = (HERE / 'run_server.sh').read_bytes()
    assert b'\r' not in shell
    for forbidden in (b'set -o pipefail', b'[[', b'CUDA_VISIBLE_DEVICES=', b'HIP_VISIBLE_DEVICES='):
        assert forbidden not in shell, forbidden
    frozen = json.loads((HERE / 'frozen_baseline.json').read_text(encoding='utf-8'))
    for name, sha in frozen.items():
        assert digest((HERE.parent / name).read_bytes()) == sha, f'Frozen file changed: {name}'
    print(f'Static checks OK: {len(files)} upstream files byte-identical; '
          f'{len(frozen)} frozen files unchanged; own AST/LF/package hashes verified; '
          f'AttFuse import closure: {len(dependencies["python_modules"])} modules, no PointNet2.')


if __name__ == '__main__':
    main()
