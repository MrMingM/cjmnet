"""Read-only, standard-library package diagnosis. Never regenerate the manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parent.parent


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def differences(expected, actual):
    return dict(missing=sorted(expected.keys() - actual.keys()),
        unexpected=sorted(actual.keys() - expected.keys()),
        changed=[dict(path=name, expected_sha256=expected[name], actual_sha256=actual[name])
                 for name in sorted(expected.keys() & actual.keys()) if expected[name] != actual[name]])


def text_conversion_hint(data, expected):
    """Evidence of transfer/editor conversion; still a strict integrity failure."""
    normalized = data.replace(b'\r\n', b'\n')
    if normalized != data and hashlib.sha256(normalized).hexdigest() == expected:
        return 'bytes match expected only after CRLF -> LF'
    without_bom = data[3:] if data.startswith(b'\xef\xbb\xbf') else data
    if without_bom != data and hashlib.sha256(without_bom).hexdigest() == expected:
        return 'bytes match expected only after removing UTF-8 BOM'
    if without_bom.replace(b'\r\n', b'\n') != data and hashlib.sha256(without_bom.replace(b'\r\n', b'\n')).hexdigest() == expected:
        return 'bytes match expected only after removing UTF-8 BOM and CRLF -> LF'
    return None


def inspect_package(package=PACKAGE):
    package = Path(package).resolve(strict=True)
    manifest = package / 'source_snapshot.json'
    expected = json.loads(manifest.read_text(encoding='utf-8'))['files']
    actual, unreadable = {}, []
    for path in sorted(package.rglob('*')):
        # Exactly the D-1A common.verify_package inventory rules.
        if not path.is_file() or '__pycache__' in path.parts or path.name == 'source_snapshot.json':
            continue
        name = path.relative_to(package).as_posix()
        try:
            actual[name] = digest(path)
        except OSError as error:
            unreadable.append(dict(path=name, error=str(error)))
    result = differences(expected, actual)
    for row in result['changed']:
        path = package / row['path']
        if path.suffix in ('.py', '.sh', '.md', '.json', '.yaml', '.txt'):
            try:
                hint = text_conversion_hint(path.read_bytes(), row['expected_sha256'])
                if hint:
                    row['transfer_conversion_evidence'] = hint
            except OSError:
                pass
    result.update(status='mismatch' if unreadable or any(result.values()) else 'complete',
        package=str(package), manifest_sha256=digest(manifest),
        expected_file_count=len(expected), actual_file_count=len(actual), unreadable=unreadable)
    return expected, result


def verify_package():
    expected, result = inspect_package()
    if result['status'] != 'complete':
        print(json.dumps(result, ensure_ascii=False, indent=2), file=sys.stderr, flush=True)
        raise ValueError('Package hash mismatch; see missing/unexpected/changed above. '
                         'Synchronize the full package including source_snapshot.json; '
                         'do not regenerate the server manifest to accept changed files.')
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=PACKAGE)
    args = parser.parse_args()
    _, result = inspect_package(args.package)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0 if result['status'] == 'complete' else 1


if __name__ == '__main__':
    sys.exit(main())
