"""Local-safe AST/LF/source/evidence hashes only; no Torch or real data."""
import ast
from pathlib import Path

from support import HERE, REPRO, identity, read_json, sha256, verify_sources


def main():
    files = verify_sources()
    pins = read_json(HERE / 'EVIDENCE.json')
    historical = read_json(REPRO / 'source_snapshot.json')['files']
    if sha256(REPRO / 'source_snapshot.json') != pins['historical_snapshot_sha256']:
        raise ValueError('Historical source snapshot changed')
    for name, digest in historical.items():
        if sha256(REPRO / name) != digest:
            raise ValueError(f'Historical reproduction source changed: {name}')
    for name, digest in read_json(REPRO / 'frozen_baseline.json').items():
        if sha256(REPRO.parent / name) != digest:
            raise ValueError(f'Frozen historical file changed: {name}')
    for path in HERE.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    for name in files:
        path = HERE / name
        if path.suffix in ('.py', '.sh', '.md', '.json') and b'\r' in path.read_bytes():
            raise ValueError(f'Expected LF: {path}')
    shell = (HERE / 'run_server.sh').read_text(encoding='utf-8')
    if 'pipefail' in shell or '[[' in shell:
        raise ValueError('Shell must be POSIX sh')
    local = REPRO.parent / 'logs/D/D1'
    if local.is_dir():
        for name, digest in pins['d1b_artifacts'].items():
            if sha256(local / name) != digest:
                raise ValueError(f'Local D-1B evidence changed: {name}')
        contract = read_json(local / 'contract.json')
        if identity({k: v for k, v in contract.items() if k != 'identity'}) != contract['identity']:
            raise ValueError('Local D-1B contract identity invalid')
        if contract['source_hashes'] != historical:
            raise ValueError('Local D-1B source map differs from historical snapshot')
    print(f'D-1C static checks passed: {len(files)} new package files, '
          f'{len(historical)} historical package files unchanged; frozen files and D-1B evidence hashes checked. '
          'No Torch/OpenCOOD/data/inference/AP execution.')


if __name__ == '__main__':
    main()
