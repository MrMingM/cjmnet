"""Prepare the exact files read by the official Clean validation loader."""
from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import zlib

from common import atomic_json, identity, link, lock, no_test, read_json, sha256


def safe_member(name):
    if '\\' in name or any(x in name for x in ('\n', '\r', '\x00', ':')):
        raise ValueError(f'Unsafe archive member: {name!r}')
    path = PurePosixPath(name)
    if path.is_absolute() or any(p in ('.', '..') for p in path.parts) or not path.parts:
        raise ValueError(f'Unsafe archive member: {name!r}')
    return path


def listing(seven, archive):
    result = subprocess.run([seven, 'l', '-slt', '-sccUTF-8', str(archive)],
        check=True, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, encoding='utf-8')
    text = result.stdout.replace('\r\n', '\n')
    if '\n----------\n' not in text:
        raise ValueError('Unrecognized 7z technical listing')
    entries, entry = {}, {}
    for line in text.split('\n----------\n', 1)[1].splitlines() + ['']:
        if not line.strip():
            if 'Path' in entry:
                path = safe_member(entry['Path'])
                directory = entry.get('Folder') == '+' or entry.get('Attributes', '').startswith('D')
                if not directory:
                    if any(k in entry for k in ('Symbolic Link', 'Hard Link')):
                        raise ValueError('Archive links are refused')
                    if path.name in entries:
                        raise ValueError(f'Flattening collision: {path.name}')
                    entries[path.name] = dict(member=entry['Path'], size=int(entry['Size']),
                                              crc=entry.get('CRC', ''))
            entry = {}
        elif ' = ' in line:
            key, val = line.split(' = ', 1)
            if key in entry:
                raise ValueError('Duplicate archive listing field')
            entry[key] = val
    if not entries or any(e['size'] < 0 for e in entries.values()):
        raise ValueError('Empty/invalid archive listing')
    return entries


def scene_inventory(root):
    """Use official lexical agent ordering, moving the negative RSU to the end."""
    result = []
    for scene in sorted(p for p in root.iterdir() if p.is_dir()):
        if not re.fullmatch(r'[A-Za-z0-9_-]+', scene.name):
            raise ValueError(f'Unexpected scene directory: {scene}')
        agents = sorted(p.name for p in scene.iterdir() if p.is_dir())
        if not agents or any(not re.fullmatch(r'-?\d+', a) for a in agents):
            raise ValueError(f'Expected numeric CAV directories: {scene}')
        if int(agents[0]) < 0:
            agents = agents[1:] + agents[:1]
        if not 2 <= len(agents) <= 5:
            raise ValueError(f'Unexpected agent count: {scene}/{agents}')
        stamps = None
        for agent in agents:
            folder = scene / agent
            current = sorted(p.stem for p in folder.glob('*.yaml') if 'additional' not in p.name)
            if not current or any(not t.isdigit() for t in current):
                raise ValueError(f'Unexpected timestamps: {folder}')
            if stamps is not None and current != stamps:
                raise ValueError(f'Agent timestamps differ: {folder}')
            stamps = current
        result.append(dict(scene=scene.name, agents=agents, ego=agents[0], timestamps=stamps))
    if len(result) != 6 or sum(len(s['timestamps']) for s in result) != 829:
        raise ValueError('Expected exactly 6 scenes / 829 cooperative validation frames')
    if sum(len(s['agents']) for s in result) != 20:
        raise ValueError('Expected the recorded 20 scene-agent archives')
    return result


def required_files(stamp):
    # Official load_camera_data opens all four images, then selects camera0.
    return [stamp + '.yaml', stamp + '.pcd', stamp + '_radar.pcd'] + [
        f'{stamp}_camera{i}.png' for i in range(4)]


def crc32(path):
    crc = 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            crc = zlib.crc32(block, crc)
    return f'{crc & 0xffffffff:08X}'


def prepare_data(args):
    raw, paired, output = (no_test(p).resolve() for p in
                          (args.data_root, args.paired_root, args.view_root))
    split, clean = raw / 'validate', paired / 'clean' / 'validate'
    if not split.is_dir():
        raise FileNotFoundError(split)
    output.mkdir(parents=True, exist_ok=True)
    with lock(output.with_suffix('.lock')):
        report = dict(status='preparing', raw_validate=str(split), paired_validate=str(clean),
            layout='validate/<scene>/<agent>/<timestamp>{.yaml,.pcd,_radar.pcd,_camera0..3.png}',
            raw_scene_entries={p.name: sorted(q.name for q in p.iterdir())
                               for p in sorted(split.iterdir()) if p.is_dir()})
        atomic_json(output / 'inspection.json', report)
        # Prefer already decompressed official files; otherwise link Clean/YAML from paired.
        try:
            scenes = scene_inventory(split)
            primary = split
            mode = 'already_extracted_official_validate'
        except ValueError:
            if not clean.is_dir():
                raise ValueError('Raw validate is still compressed and paired Clean view is missing')
            scenes = scene_inventory(clean)
            primary = clean
            mode = 'paired_clean_plus_original_auxiliary_files'
        keys = [[s['scene'], t] for s in scenes for t in s['timestamps']]
        if set(report['raw_scene_entries']) != {s['scene'] for s in scenes}:
            raise ValueError('Original and prepared validation scene sets differ')
        plan = read_json(paired / 'plan.json') if (paired / 'plan.json').exists() else None
        if plan is not None:
            p = plan['protocol']
            manifest = read_json(paired / 'manifest.json')
            from hashlib import sha256 as hash_bytes
            import json
            old_hash = hash_bytes(json.dumps(p, sort_keys=True, separators=(',', ':'),
                allow_nan=False).encode()).hexdigest()
            if (old_hash != plan['identity'] or manifest['identity'] != plan['identity']
                    or manifest['status'] != 'complete' or p['split'] != 'validate'
                    or p['frame_keys'] != keys or p['selected_cooperative_frames'] != 829):
                raise ValueError('Paired plan/manifest/frame identities mismatch')
        elif primary == clean:
            raise ValueError('Paired data requires the original plan/manifest/extraction records')
        seven = args.seven_zip or shutil.which('7zz') or shutil.which('7z') or shutil.which('7za')
        file_rows, archive_rows = [], []
        for s in scenes:
            for agent in s['agents']:
                folder = primary / s['scene'] / agent
                names = [n for t in s['timestamps'] for n in required_files(t)]
                pending = [n for n in names if not (folder / n).is_file()]
                archive = split / s['scene'] / (agent + '.7z')
                payload = output / 'auxiliary' / s['scene'] / agent
                extracted = {}
                if pending:
                    if not seven or not archive.is_file():
                        raise RuntimeError(f'Need 7zz and original archive {archive} to supply {pending[:3]}')
                    no_test(archive)
                    archive_sha = sha256(archive)
                    if plan:
                        original = next(r for r in plan['protocol']['archives']
                                        if r['scene'] == s['scene'] and r['cav'] == agent)
                        if archive_sha != original['sha256']:
                            raise ValueError(f'Original archive changed: {archive}')
                    entries = listing(seven, archive)
                    if any(n not in entries for n in pending):
                        raise ValueError(f'Official loader required files absent from {archive}')
                    record_path = output / 'records' / s['scene'] / (agent + '.json')
                    if record_path.exists():
                        record = read_json(record_path)
                        if record['archive_sha256'] != archive_sha or set(record['files']) != set(pending):
                            raise ValueError('Auxiliary record selection changed')
                        for name, digest in record['files'].items():
                            if sha256(no_test(payload / name)) != digest:
                                raise ValueError(f'Existing auxiliary file changed: {payload / name}')
                        extracted = record['files']
                    else:
                        stage = output / '.partial' / s['scene'] / agent
                        stage.mkdir(parents=True, exist_ok=True)
                        needed = sum(entries[n]['size'] for n in pending)
                        if shutil.disk_usage(output).free < needed + 512 * 1024 ** 2:
                            raise OSError('Insufficient data disk space for this auxiliary archive')
                        include = stage / 'members.txt'
                        include.write_text(''.join(entries[n]['member'] + '\n' for n in pending),
                                           encoding='utf-8', newline='\n')
                        print(f'Extract official dependencies: {archive} ({len(pending)} files)', flush=True)
                        with (stage / '7z.log').open('w') as log:
                            subprocess.run([seven, 'e', str(archive), '@' + str(include),
                                '-o' + str(stage), '-y', '-aoa', '-spd', '-scsUTF-8',
                                '-sccUTF-8', '-mmt=2'], check=True, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT)
                        payload.mkdir(parents=True, exist_ok=True)
                        for name in pending:
                            temp, final = stage / name, payload / name
                            entry = entries[name]
                            if (temp.is_symlink() or temp.stat().st_size != entry['size']
                                    or (entry['crc'] and crc32(temp) != entry['crc'].upper())):
                                raise ValueError(f'Extracted CRC/size mismatch: {temp}')
                            digest = sha256(temp)
                            if final.exists():
                                if sha256(final) != digest:
                                    raise FileExistsError(final)
                                temp.unlink()
                            else:
                                os.replace(temp, final)
                            extracted[name] = digest
                        atomic_json(record_path, dict(archive=str(archive), archive_sha256=archive_sha,
                                                      files=extracted))
                    archive_rows.append(dict(path=str(archive), sha256=archive_sha))
                old_record = None
                if primary == clean:
                    old_record = read_json(paired / 'records' / 'validate' / s['scene'] / (agent + '.json'))
                    if old_record['identity'] != plan['identity']:
                        raise ValueError('Paired extraction record identity mismatch')
                for name in names:
                    source = folder / name if (folder / name).is_file() else payload / name
                    no_test(source)
                    digest = sha256(source)
                    if old_record and name in old_record['files'] and digest != old_record['files'][name]:
                        raise ValueError(f'Paired Clean/YAML hash changed: {source}')
                    destination = output / 'validate' / s['scene'] / agent / name
                    link(source, destination)
                    file_rows.append(dict(view=str(destination), source=str(source.resolve()),
                                          sha256=digest, size=source.stat().st_size))
                print(f'Official view checked: {s["scene"]}/{agent}', flush=True)
        actual = scene_inventory(output / 'validate')
        if actual != scenes:
            raise ValueError('Constructed view changed scene/CAV/ego/timestamp order')
        expected_files = {r['view'] for r in file_rows}
        actual_files = {str(p) for p in (output / 'validate').rglob('*') if p.is_file()}
        if actual_files != expected_files:
            raise ValueError('Unexpected extra/missing files in official validation view')
        report.update(status='complete', mode=mode, scenes=scenes, frame_keys=keys,
            validation_frame_count=829, scene_count=len(scenes), scene_agent_count=20,
            single_agent_frame_count=sum(len(s['agents']) * len(s['timestamps']) for s in scenes),
            archives=archive_rows, files=file_rows)
        report['identity'] = identity(report)
        atomic_json(output / 'manifest.json', report)
        return report
