"""Prepare a separate server diagnostic from the already uploaded older tool.

No sealed package, checkpoint, original diagnostic, or pin file is overwritten.
This avoids requiring a second manual file upload during remote troubleshooting.
"""
from pathlib import Path
import sys


SOURCE_CHECK = '''
def reviewed_diagnostic_sources(project, recorded):
    here = project / 'uecp_gspr_v2xr'
    own = read(here / 'source_snapshot.json')['files']
    actual = {p.relative_to(here).as_posix(): sha(p) for p in here.rglob('*')
              if p.is_file() and p.name != 'source_snapshot.json' and '__pycache__' not in p.parts}
    pins = read(here / 'PINS.json')['files']
    if actual != own or own != recorded['own'] or pins != recorded['dependencies']:
        raise ValueError('Sealed package or original dependency pins changed')
    known = ('v2xr_raygen_e0e3/training.py',
             'b6577e83734086bc760cde237eb99c9086ac2a6d514e3fa0bf1ae7a44ba6c129',
             'cc5f523e64cb34814208a5901ef30be45712701ef0f00a0e4892accf9a683793')
    differences = []
    for name, expected in pins.items():
        current = sha(project / name)
        if current == expected:
            continue
        if (name, expected, current) != known:
            raise ValueError('Unreviewed source difference: '+name)
        differences.append(dict(path=name, original_sha256=expected, current_sha256=current,
            scope='DIAGNOSIS_ONLY; original pipeline gate unchanged',
            reused_helpers_reviewed=['tensor_hash', 'atomic_torch', 'rng_state', 'restore_rng'],
            review='Exact whole-file version pair reviewed against original delivery; helper bodies '
                   'and module-level statements are unchanged'))
    from uecp_gspr_v2xr.dependency_scope import validate_scope
    validate_scope(project, pins)
    return dict(dependencies_checked=len(pins), sealed_package_exact=True,
                original_pins_unchanged=True, reviewed_differences=differences)


'''


def prepare_source(source):
    old = ("    from uecp_gspr_v2xr.common import sources\n"
           "    if sources() != read(root / 'request.json')['sources']:\n"
           "        raise ValueError('Current source hashes differ from the original experiment request')\n")
    anchor = 'def restore_load(root, project, paths, commits, torch, numpy_tree):\n'
    parser = '    parser = argparse.ArgumentParser(description=__doc__)\n'
    final = "    print('DIAGNOSIS_FINISHED: existing FAILED recovery gate was retained.', flush=True)\n"
    returned = 'loader_source_sha256=sha(loader_path), extracted_original_loader=True, checkpoints=reports,'
    for text in (old, anchor, parser, final, returned):
        if source.count(text) != 1:
            raise ValueError('Uploaded diagnostic differs from the inspected old version; no file changed')
    source = source.replace(old, "    reviewed = reviewed_diagnostic_sources(project, read(root / 'request.json')['sources'])\n")
    source = source.replace(anchor, SOURCE_CHECK+anchor)
    source = source.replace(parser, parser+"    parser.add_argument('--version', action='version', version='uecp-resume-diagnose 2026-10-20.3')\n")
    source = source.replace(returned, returned+'\n        source_review=reviewed,')
    source = source.replace(final, "    if audit_restore:\n"
        "        print('RESTORE_RESULT_BEGIN', flush=True)\n"
        "        print(json.dumps(report['restore_load_audit'], indent=2), flush=True)\n"
        "        print('RESTORE_RESULT_END', flush=True)\n"+final)
    compile(source, '<prepared independent diagnostic>', 'exec')
    return source


def main():
    if sys.platform != 'linux':
        raise RuntimeError('Diagnostic preparation is for the Linux server only')
    original = Path('/home/cjm/uecp_gspr_resume_diagnose.py')
    target = original.with_name('uecp_gspr_resume_diagnose_ready.py')
    source = prepare_source(original.read_text(encoding='utf-8'))
    if target.exists():
        if target.read_text(encoding='utf-8') != source:
            raise ValueError('Prepared diagnostic already exists with different content; preserved')
    else:
        with target.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(source)
    print(target)


if __name__ == '__main__':
    main()
