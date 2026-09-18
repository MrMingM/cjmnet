"""CPU-only, fail-closed comparison against completed Stage-3A protocols.

Never updates a manifest or historical log. Known new diagnostic modules are
reported separately from changes to files that existed during Stage-3A.
"""
import argparse
import hashlib
import json
from pathlib import Path

NEW_MODULES = {
    'gspr_evidence/stage3b_diagnostic.py',
    'gspr_evidence/stage3_diagnostic_analysis.py',
    'gspr_evidence/stage3_diagnostic_runtime.py',
    'gspr_evidence/stage3_diagnostic_report.py',
}


def digest(path, source=False):
    if not path.is_file():
        return None
    data = path.read_bytes()
    if source:
        data = data.replace(b'\r\n', b'\n')
    return hashlib.sha256(data).hexdigest()


def audit(root, aroot):
    root, aroot = Path(root), Path(aroot)
    result = json.loads((aroot/'stage3a_results.json').read_text(encoding='utf-8'))
    manifest = json.loads((root/'gspr_evidence/stage3_sources.json').read_text(encoding='utf-8'))
    current = {p.relative_to(root).as_posix(): digest(p, source=True)
               for p in (root/'gspr_evidence').glob('stage3*.py')}
    report = dict(stage3a_root=str(aroot.resolve()), source_root=str(root.resolve()),
                  weathers={}, errors=[], audit_script_sha256=digest(Path(__file__), source=True))
    if result['smoke']:
        report['errors'].append('Expected completed FULL Stage-3A, got smoke')
    for weather in ('fog', 'rain', 'snow'):
        protocol = json.loads((aroot/weather/'protocol.json').read_text(encoding='utf-8'))
        summary = json.loads((aroot/weather/'summary.json').read_text(encoding='utf-8'))
        if (protocol['stage'] != '3A' or not protocol['development_only'] or protocol['test_data_used']
                or not summary['complete']):
            report['errors'].append(weather+': incomplete/non-development/wrong-stage A')
        if protocol != result['protocols'][weather]:
            report['errors'].append(weather+': protocol differs from completed report snapshot')
        target_sha = digest(aroot/weather/'targets.jsonl')
        if target_sha != result['weathers'][weather]['targets_sha256']:
            report['errors'].append(weather+': Stage-3A target log changed')
        if manifest != protocol['audited_sources']:
            report['errors'].append(weather+': current source manifest differs from historical A')
        dependency_rows = []
        for name, historical in protocol['audited_sources'].items():
            actual = digest(root/name, source=True)
            if actual != historical:
                dependency_rows.append(dict(file=name, stage3a_sha256=historical, current_sha256=actual))
                report['errors'].append(weather+': audited dependency mismatch: '+name)
        historical = protocol['implementation_sha256']
        rows = []
        for name in sorted(set(historical) | set(current)):
            old, now = historical.get(name), current.get(name)
            if name not in historical and name in NEW_MODULES and now is not None:
                status = 'NEW_DIAGNOSTIC_MODULE'
            elif old is not None and old == now:
                status = 'MATCH'
            else:
                status = 'MISMATCH'
                report['errors'].append(weather+': implementation mismatch: '+name)
            rows.append(dict(file=name, stage3a_sha256=old, current_sha256=now, status=status))
        report['weathers'][weather] = dict(implementation=rows, dependency_mismatches=dependency_rows,
                                          targets_sha256=target_sha)
    report['passed'] = not report['errors']
    return report


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--stage3a-root', required=True)
    parser.add_argument('--output-json')
    args = parser.parse_args()
    report = audit(Path(__file__).resolve().parents[1], args.stage3a_root)
    for weather, data in report['weathers'].items():
        print('\n'+weather+' implementation_sha256:')
        for r in data['implementation']:
            print(r['status'], r['file'])
            print('  Stage-3A:', r['stage3a_sha256'])
            print('  current :', r['current_sha256'])
        for r in data['dependency_mismatches']:
            print('DEPENDENCY_MISMATCH', r)
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    for error in report['errors']:
        print('ERROR:', error)
    print('LINEAGE AUDIT: '+('PASS' if report['passed'] else 'FAIL; nothing was overwritten'))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
