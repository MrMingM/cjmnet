"""Frozen file listing plus content-verified directory-only numeric-CAV placeholders."""
from pathlib import Path
import re
import subprocess

from auditlib import atomic_json, no_test, sha256
from prepare import listing as frozen_listing, safe_member


class ArchiveEntries(dict):
    def __init__(self, evidence):
        super().__init__()
        self.placeholder_evidence = evidence


def directory_placeholder(text, archive):
    """Require conclusive directory-only evidence, independent of the numeric CAV ID sign."""
    archive = Path(archive)
    if not re.fullmatch(r'-?\d+', archive.stem):
        return None
    text = text.replace('\r\n', '\n')
    if '\n----------\n' not in text:
        return None
    header, payload = text.split('\n----------\n', 1)
    attributes = {}
    for line in header.splitlines():
        if ' = ' in line:
            key, value = line.split(' = ', 1)
            attributes[key] = value
    if attributes.get('Type') != '7z' or attributes.get('Blocks') != '0':
        return None
    records, row = [], {}
    for line in payload.splitlines() + ['']:
        if not line.strip():
            if row:
                records.append(row)
            row = {}
        elif ' =' in line:
            # Empty fields may be printed as either "Method = " or "Method =".
            key, value = line.split(' =', 1)
            if value.startswith(' '):
                value = value[1:]
            if key in row:
                return None
            row[key] = value
        else:
            return None
    if len(records) != 1:
        return None
    row = records[0]
    if (row.get('Path') != archive.stem or row.get('Size') != '0'
            or row.get('Packed Size') != '0' or row.get('Folder') == '-'
            or not (row.get('Folder') == '+' or row.get('Attributes', '').split(' ', 1)[0] == 'D')
            or any(k in row for k in ('Symbolic Link', 'Hard Link'))):
        return None
    safe_member(row['Path'])
    return dict(kind='empty_directory_placeholder', cav=archive.stem,
        directory=row['Path'], regular_file_count=0, data_blocks=0,
        excluded_from_loader=True,
        rule='numeric CAV of either sign; exactly its root directory; size/packed-size zero; 7z Blocks=0; no file or link')


def listing(seven, archive, diagnostics=None):
    archive = no_test(archive)
    print('Reading archive listing:', archive, flush=True)
    try:
        return frozen_listing(seven, archive)
    except (ValueError, UnicodeError, subprocess.CalledProcessError) as error:
        command = [seven, 'l', '-slt', '-sccUTF-8', str(archive)]
        result = subprocess.run(command, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        # Capture the original bytes, including an unexpected encoding or newline format.
        text = result.stdout.decode('utf-8-sig', errors='replace')
        proof = directory_placeholder(text, archive) if result.returncode == 0 else None
        if proof is not None:
            proof.update(archive_sha256=sha256(archive), listing_stdout_sha256=__import__('hashlib').sha256(result.stdout).hexdigest())
        if diagnostics is None:
            if proof is not None:
                return ArchiveEntries(proof)
            raise ValueError(f'Archive listing failed: {archive}: {error}') from error
        folder = Path(diagnostics)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'stdout.txt').write_bytes(result.stdout)
        (folder / 'stderr.txt').write_bytes(result.stderr)
        paths = [line.split(' = ', 1)[1] for line in text.splitlines() if line.startswith('Path = ')]
        fields = [line for line in text.splitlines() if line.startswith(('Folder = ', 'Attributes = ', 'Size = '))]
        diagnostic = dict(status='classified_empty_placeholder' if proof else 'failed', archive=str(archive), archive_size=archive.stat().st_size,
            executable=seven, executable_sha256=sha256(seven), command=command,
            original_error=repr(error), diagnostic_listing_returncode=result.returncode,
            diagnostic_listing_is_second_read=True,
            stdout_sha256=sha256(folder / 'stdout.txt'), stderr_sha256=sha256(folder / 'stderr.txt'),
            path_record_count=len(paths), first_paths=paths[:12], first_record_fields=fields[:36],
            stdout=str(folder / 'stdout.txt'), stderr=str(folder / 'stderr.txt'),
            placeholder_evidence=proof,
            interpretation='Content-verified directory-only CAV is recorded and excluded from the loader view; all other listing failures stop.')
        atomic_json(folder / 'diagnostic.json', diagnostic)
        if proof is not None:
            print('DIRECTORY-ONLY CAV PLACEHOLDER:', archive, '(recorded; no training CAV directory will be created)', flush=True)
            return ArchiveEntries(proof)
        raise ValueError(f'Archive listing failed: {archive}; original={error}; '
            f'raw diagnostics={folder / "diagnostic.json"}; no extraction performed for this archive') from error


def inspect_only(args, output):
    """Inspect only the explicit train split; keep existing extraction request untouched."""
    raw = no_test(args.data_root)
    if raw.name != 'train' or not raw.is_dir():
        raise ValueError('DATA_ROOT must name the original train split')
    scenes = sorted(no_test(p) for p in raw.iterdir() if p.is_dir())
    if args.smoke:
        scenes = scenes[:1]
    count, empty, failed, ordinary = 0, [], [], 0
    for scene in scenes:
        packs = sorted(scene.glob('*.7z'))
        if not packs:
            failed.append(dict(scene=str(scene), archive=None, error='No CAV archives'))
            continue
        for pack in packs:
            count += 1
            diagnostics = output / scene.name / pack.stem
            try:
                entries = listing(args.seven_zip, pack, diagnostics)
            except (ValueError, OSError, UnicodeError, subprocess.CalledProcessError) as error:
                failed.append(dict(scene=str(scene), archive=str(pack), error=repr(error),
                    diagnostic=str(diagnostics / 'diagnostic.json'), stdout=str(diagnostics / 'stdout.txt')))
                print('LISTING FAILED (inspection continues):', pack, repr(error), flush=True)
                continue
            if getattr(entries, 'placeholder_evidence', None):
                empty.append(dict(archive=str(pack), **entries.placeholder_evidence))
            else:
                ordinary += 1
            print(f'LISTING OK: {pack}: {len(entries)} regular files', flush=True)
    if count == 0 and not failed:
        failed.append(dict(archive=None, error='No train archives found'))
    summary = dict(status='failed' if failed else 'complete', scan_completed=True, archive_count=count,
        ordinary_archive_count=ordinary, failed_archive_count=sum(r['archive'] is not None for r in failed),
        issue_count=len(failed), failures=failed,
        empty_cav_archive_count=len(empty), empty_cav_archives=empty,
        scope='first scene' if args.smoke else 'all train scenes', extraction_performed=False,
        note='All archive listing failures are collected. scan_completed does not imply data integrity or extraction completion.')
    atomic_json(output / 'inspection_summary.json', summary)
    print('ARCHIVE INSPECTION SUMMARY:', output / 'inspection_summary.json',
          f'(ordinary={ordinary}, verified_empty={len(empty)}, issues={len(failed)})', flush=True)
    if failed:
        raise ValueError(f'Archive inspection finished with {len(failed)} issues; see {output / "inspection_summary.json"}; no extraction performed')
    atomic_json(output / 'inspection_complete.json', summary)
