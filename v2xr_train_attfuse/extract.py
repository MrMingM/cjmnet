"""Inspect actual train archives, atomically extract and verify all their members."""
import os
from pathlib import Path
import re
import shutil
import subprocess

from auditlib import (atomic_json, bound_json, check_scene_partition, identity, no_test, read_json,
                      request, sha256, verify_train)
from prepare import required_files, crc32
from archive_listing import listing
from train_acceptance import bind_train_acceptance, required_train_count


def extraction_request(args, output, own):
    """Migrate only explicitly pinned attempts which failed before any data plan existed."""
    desired = dict(source_hashes=own, data_root=args.data_root,
        seven_zip_sha256=sha256(args.seven_zip), smoke=args.smoke)
    marker = output / 'request.json'
    if args.resume and marker.exists() and read_json(marker) != desired:
        from auditlib import HERE
        old = read_json(marker)
        known = read_json(HERE / 'pre_empty_placeholder_requests.json')['accepted_source_hashes']
        failure = read_json(output / 'last_failure.json') if (output / 'last_failure.json').exists() else {}
        allowed = {'request.json', 'last_failure.json', 'listing_diagnostics', 'request_history',
                   'inventory_scan.json', 'inventory.json', 'inventory_summary.json', 'frame_keys.jsonl', 'scene_counts.tsv', 'archive_counts.tsv'}
        if (old.get('source_hashes') not in known
                or {k: v for k, v in old.items() if k != 'source_hashes'} !=
                   {k: v for k, v in desired.items() if k != 'source_hashes'}
                or failure.get('command') != 'extract'
                or not any(s in failure.get('error', '') for s in
                    ('Empty/invalid archive listing', 'Archive listing failed',
                     'Scene has no data CAVs; no scene may be silently discarded:',
                     'Train cooperative frames 8083 != official 8084; do not train'))
                or {p.name for p in output.iterdir()} - allowed):
            raise ValueError('Changed request cannot migrate: only known listing/zero-frame-scene/8083-count failures before plan/data creation are eligible; keep existing artifacts and use an independent root')
        old_id, new_id = identity(old), identity(desired)
        bound_json(output / 'request_history' / (old_id + '.json'), old)
        bound_json(output / 'request_history' / (old_id + '_upgrade.json'), dict(
            reason='Explicit pre-extraction diagnostic upgrade; source-count/placeholder guards remain active; no plan, records, data or completion existed',
            old_request_identity=old_id, new_request_identity=new_id,
            original_failure=failure, resumed_payload=False))
        atomic_json(marker, desired)
        print('UPGRADED FAILED PRE-EXTRACTION REQUEST:', old_id, '->', new_id,
              '(original request preserved; no existing payload/cache reused)', flush=True)
    request(output, desired, args.resume)


def inspect(args, output, inventory_only=False):
    raw = no_test(args.data_root)
    if raw.name != 'train' or not raw.is_dir():
        raise ValueError('DATA_ROOT must be the explicit original train split directory')
    from inventory import official_reference_hashes, write_inventory
    references = official_reference_hashes()  # Fail before expensive archive scanning on reference errors.
    if not inventory_only:
        from train_acceptance import approved_reference
        approved_reference()  # Preserve/check the originally approved report before another long scan.
    directories = sorted(p for p in raw.iterdir() if p.is_dir())
    if not directories:
        raise ValueError('No train scenes found')
    archives, scenes, scene_metadata, empty_scenes = [], [], [], []
    inspected_scene_count = 0
    for directory in directories:
        inspected_scene_count += 1
        no_test(directory)
        if not re.fullmatch(r'[A-Za-z0-9_-]+', directory.name):
            raise ValueError('Unsafe scene name')
        packs = sorted(directory.glob('*.7z'))
        if not packs or any(not re.fullmatch(r'-?\d+', p.stem) for p in packs):
            raise ValueError('Expected train/<scene>/<numeric-CAV>.7z: ' + str(directory))
        if any(p.is_dir() for p in directory.iterdir()):
            raise ValueError('Mixed extracted/compressed train layout requires explicit inspection')
        for file in directory.iterdir():
            if file in packs:
                continue
            if file.name != 'data_protocol.yaml' or not file.is_file():
                raise ValueError('Unrecognized train scene entry (do not silently omit a CAV/archive): ' + str(file))
            scene_metadata.append(dict(path=str(no_test(file)), sha256=sha256(file)))
        scene_archive_start = len(archives)
        agents, empty_agents = [], []
        stamps = None
        for pack in packs:
            no_test(pack)
            entries = listing(args.seven_zip, pack, output / 'listing_diagnostics' / directory.name / pack.stem)
            placeholder = getattr(entries, 'placeholder_evidence', None)
            if placeholder is not None:
                empty_agents.append(pack.stem)
                archives.append(dict(scene=directory.name, cav=pack.stem, path=str(pack),
                    sha256=sha256(pack), size=pack.stat().st_size, entries={}, original_member_count=0,
                    timestamps=[], weather_inventory=dict(fog=0, snow=0),
                    kind='empty_directory_placeholder', excluded_from_loader=True, placeholder_evidence=placeholder))
                continue
            agents.append(pack.stem)
            if {'members.txt', '7z_test.log', '7z_extract.log'} & entries.keys():
                raise ValueError('Archive member collides with extraction control files: ' + str(pack))
            for row in entries.values():
                if any(p.lower() in ('test', 'testing') for p in Path(row['member']).parts):
                    raise ValueError('Test archive member refused')
            current = sorted(Path(n).stem for n in entries if n.endswith('.yaml') and 'additional' not in n)
            if not current or any(not t.isdigit() for t in current):
                raise ValueError('Unexpected YAML timestamps: ' + str(pack))
            normal = sorted(Path(n).stem for n in entries if re.fullmatch(r'\d+\.pcd', n))
            if normal != current:
                raise ValueError('Clean PCD/YAML timestamp sets differ: ' + str(pack))
            if stamps is not None and current != stamps:
                raise ValueError('CAV timestamps are not aligned: ' + str(pack))
            stamps = current
            selected = current[:2] if args.smoke else current
            required = {name for t in selected for name in required_files(t)}
            missing = sorted(required - entries.keys())
            if missing:
                atomic_json(output / 'inspection_failure.json', dict(missing=missing, archive=str(pack)))
                raise ValueError('Missing official loader dependencies: ' + str(pack))
            chosen = {n: e for n, e in entries.items() if not args.smoke or any(n.startswith(t + '.') or n.startswith(t + '_') for t in selected)}
            archives.append(dict(scene=directory.name, cav=pack.stem, path=str(pack),
                sha256=sha256(pack), size=pack.stat().st_size, entries=chosen,
                kind='data_cav', excluded_from_loader=False, placeholder_evidence=None,
                original_member_count=len(entries), timestamps=selected,
                weather_inventory=dict(fog=sum(n.endswith('_fog_0.060.pcd') for n in entries),
                                       snow=sum(n.endswith('_snow_2.5_2.0.pcd') for n in entries))))
            print('Inspected train archive', pack, len(entries), 'members', flush=True)
        if not agents or stamps is None:
            observed = archives[scene_archive_start:]
            if (len(observed) != len(packs) or len(empty_agents) != len(packs)
                    or any(a['kind'] != 'empty_directory_placeholder' or not a['placeholder_evidence']
                           for a in observed)):
                raise ValueError('Scene has no data CAVs but lacks complete directory-only archive evidence: ' + str(directory))
            empty_scenes.append(dict(scene=directory.name, empty_agents=empty_agents, timestamps=[],
                cooperative_frames=0, excluded_from_loader=True, kind='zero_frame_scene_placeholder',
                archives=[dict(cav=a['cav'], path=a['path'], sha256=a['sha256'],
                               placeholder_evidence=a['placeholder_evidence']) for a in observed],
                reason='Every original CAV archive is content-verified directory-only; no regular files or timestamps exist',
                integrity='All archives remain in extraction plan; 7zz t is mandatory before manifest completion'))
            print('ZERO-FRAME SCENE PLACEHOLDER:', directory, f'({len(observed)} recorded empty archives; no loader scene directory)', flush=True)
            continue
        if int(agents[0]) < 0:
            agents = agents[1:] + agents[:1]
        if int(agents[0]) < 0:
            raise ValueError('Scene lacks a non-RSU ego: ' + str(directory))
        scenes.append(dict(scene=directory.name, agents=agents, empty_agents=empty_agents, ego=agents[0],
            timestamps=stamps[:2] if args.smoke else stamps, used_max_cav=7))
        if args.smoke:
            break  # First data-bearing scene, after recording any preceding zero-frame scenes.
    if not scenes:
        raise ValueError('No data-bearing train scene found; zero-frame placeholders cannot constitute a training/smoke dataset')
    count = sum(len(s['timestamps']) for s in scenes)
    atomic_json(output / 'inventory_scan.json', dict(status='scan_recorded',
        data_root=str(raw), smoke=args.smoke, actual_cooperative_frames=count,
        source_scene_count=inspected_scene_count, raw_source_scene_count=len(directories),
        scenes=scenes, empty_scenes=empty_scenes, archives=archives, scene_metadata=scene_metadata,
        official_references=references, extraction_performed=False, training_allowed=False))
    inventory = write_inventory(output, args, scenes, empty_scenes, archives, scene_metadata, len(directories))
    if inventory_only:
        return inventory
    if not args.smoke and count != required_train_count():
        raise ValueError(f'Train cooperative frames {count} != approved {required_train_count()}; do not train; '
                         f'full inventory saved at {output / "inventory_summary.json"}')
    acceptance = bind_train_acceptance(output, args, scenes, empty_scenes, archives, scene_metadata)
    plan = dict(stage='D-1D', full_train=not args.smoke, scenes=scenes, empty_scenes=empty_scenes,
        source_scene_count=inspected_scene_count, raw_source_scene_count=len(directories),
        archives=archives, scene_metadata=scene_metadata, train_data_acceptance=acceptance,
        cooperative_frames=count, definition='one scene/timestamp, all CAVs aligned; agent files counted separately',
        extraction='all safe regular members for full train; weather files stored but never selected for Clean training')
    plan['identity'] = identity(plan)
    bound_json(output / 'plan.json', plan)
    return plan


def extract_archive(args, output, archive):
    record_path = output / 'records' / archive['scene'] / (archive['cav'] + '.json')
    record = read_json(record_path) if record_path.exists() else dict(
        archive_sha256=archive['sha256'], status='pending', crc_test_passed=False, files={})
    if record['archive_sha256'] != archive['sha256']:
        raise ValueError('Archive changed since extraction began')
    destination = output / 'train' / archive['scene'] / archive['cav']
    empty = archive.get('kind') == 'empty_directory_placeholder'
    if empty:
        if (archive['entries'] or not archive.get('placeholder_evidence')
                or archive['placeholder_evidence']['archive_sha256'] != archive['sha256']):
            raise ValueError('Invalid empty-CAV classification evidence')
        if destination.exists() or destination.is_symlink():
            raise ValueError('Empty CAV must not have a directory in the loader view: ' + str(destination))
    else:
        if not archive['entries']:
            raise ValueError('Data archive has no regular files')
        destination.mkdir(parents=True, exist_ok=True)
    stage = output / '.partial' / archive['scene'] / archive['cav']
    stage.mkdir(parents=True, exist_ok=True)
    if output.resolve() not in destination.resolve().parents or output.resolve() not in stage.resolve().parents:
        raise ValueError('Extraction destination/stage escapes the independent data root')
    try:
        record.update(kind='empty_directory_placeholder' if empty else 'data_cav', excluded_from_loader=empty)
        if not record['crc_test_passed']:
            with (stage / '7z_test.log').open('w') as log:
                subprocess.run([args.seven_zip, 't', archive['path'], '-mmt=2'], check=True,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            record['crc_test_passed'] = True
            atomic_json(record_path, record)
        for name, row in record['files'].items():
            if name not in archive['entries'] or sha256(no_test(destination / name)) != row['sha256']:
                raise ValueError('Previously committed extracted file changed: ' + name)
        pending = []
        for name, entry in archive['entries'].items():
            file = no_test(destination / name)
            if name in record['files']:
                continue
            if file.exists():
                # Recover atomic publish interrupted before the per-file journal commit.
                validate_crc(file, entry)
                record['files'][name] = dict(sha256=sha256(file), size=file.stat().st_size)
                atomic_json(record_path, record)
            else:
                pending.append(name)
        record['status'] = 'extracting'
        atomic_json(record_path, record)
        # Small batches limit repeated decompression after interruption; committed files are never requested again.
        for offset in range(0, len(pending), 128):
            batch = pending[offset:offset + 128]
            needed = sum(archive['entries'][n]['size'] for n in batch)
            if shutil.disk_usage(output).free < needed + 1024 ** 3:
                raise OSError('Insufficient space on data disk')
            include = stage / 'members.txt'
            include.write_text(''.join(archive['entries'][n]['member'] + '\n' for n in batch), encoding='utf-8', newline='\n')
            with (stage / '7z_extract.log').open('a') as log:
                subprocess.run([args.seven_zip, 'e', archive['path'], '@' + str(include),
                    '-o' + str(stage), '-y', '-aoa', '-spd', '-scsUTF-8', '-sccUTF-8', '-mmt=2'],
                    check=True, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            for name in batch:
                temp = no_test(stage / name)
                validate_crc(temp, archive['entries'][name])
                row = dict(sha256=sha256(temp), size=temp.stat().st_size)
                os.replace(temp, destination / name)
                record['files'][name] = row
                atomic_json(record_path, record)
            print('Extract', archive['scene'], archive['cav'], len(record['files']), '/', len(archive['entries']), flush=True)
        if set(record['files']) != set(archive['entries']):
            raise ValueError('Archive completion has missing members')
        record['status'] = 'complete'
        record.pop('error', None)
        atomic_json(record_path, record)
        return record
    except Exception as error:
        record.update(status='failed', error=repr(error))
        atomic_json(record_path, record)
        raise


def validate_crc(file, entry):
    if (file.is_symlink() or file.stat().st_size != entry['size']
            or (entry['crc'] and crc32(file) != entry['crc'].upper())):
        raise ValueError('Extracted size/CRC mismatch: ' + str(file))


def content(file):
    """Decode Clean data using the official readers; archive CRC covers weather payloads."""
    import numpy as np
    from opencood.hypes_yaml import yaml_utils
    from opencood.utils import pcd_utils
    from PIL import Image
    if file.suffix == '.yaml' and file.stem.isdigit():
        import yaml
        def unique(node):
            if isinstance(node, yaml.MappingNode):
                keys = [k.value for k, _ in node.value if isinstance(k, yaml.ScalarNode)]
                if len(keys) != len(set(keys)):
                    raise ValueError('Duplicate YAML mapping/object ID: ' + str(file))
                for k, value in node.value:
                    unique(value)
            elif isinstance(node, yaml.SequenceNode):
                for value in node.value:
                    unique(value)
        unique(yaml.compose(file.read_text(encoding='utf-8'), Loader=yaml.Loader))
        data = yaml_utils.load_yaml(str(file))
        if not {'lidar_pose', 'ego_speed', 'vehicles'} <= data.keys() or len(data['lidar_pose']) != 6:
            raise ValueError('Incomplete Clean YAML: ' + str(file))
        if not np.isfinite(data['lidar_pose']).all() or not isinstance(data['vehicles'], dict):
            raise ValueError('Invalid Clean YAML: ' + str(file))
        for obj in data['vehicles'].values():
            if any(len(obj[k]) != 3 or not np.isfinite(obj[k]).all() for k in ('location', 'angle', 'center', 'extent')):
                raise ValueError('Invalid object geometry: ' + str(file))
    elif file.suffix == '.pcd' and file.stem.isdigit():
        declared = pcd_header(file)
        points = pcd_utils.pcd_to_np(str(file))
        if points.ndim != 2 or points.shape != (declared, 4) or not np.isfinite(points).all():
            raise ValueError('Broken/nonfinite Clean PCD: ' + str(file))
    elif re.fullmatch(r'\d+_camera[0-3]\.png', file.name):
        with Image.open(file) as image:
            image.verify()
    elif file.suffix == '.pcd':
        pcd_header(file)


def pcd_header(file):
    header = {}
    with file.open('rb') as stream:
        for _ in range(64):
            line = stream.readline(4096).strip()
            if not line or line.startswith(b'#'):
                continue
            key, _, value = line.partition(b' ')
            header[key] = value.split()
            if key == b'DATA':
                break
    if (not {b'FIELDS', b'POINTS', b'WIDTH', b'HEIGHT', b'DATA'} <= header.keys()
            or not {b'x', b'y', b'z'} <= set(header[b'FIELDS'])
            or header[b'DATA'] not in ([b'ascii'], [b'binary'], [b'binary_compressed'])):
        raise ValueError('Broken PCD header: ' + str(file))
    count = int(header[b'POINTS'][0])
    if count < 0 or count != int(header[b'WIDTH'][0]) * int(header[b'HEIGHT'][0]):
        raise ValueError('PCD POINTS/WIDTH/HEIGHT differ: ' + str(file))
    return count


def run(args, own):
    output = Path(args.train_root)
    extraction_request(args, output, own)
    plan = inspect(args, output)
    files, records = [], []
    try:
        if (output / 'manifest.json').exists():
            verify_train(args, require_full=not args.smoke)
            print('Completed train extraction hashes verified; no re-extraction', flush=True)
            return
        from observe import bootstrap, module_inventory
        from auditlib import REPRO
        for name, digest in read_json(REPRO / 'UPSTREAM.json')['files'].items():
            if sha256(no_test(Path(args.runtime_root) / name)) != digest:
                raise ValueError('Official extraction reader source changed: ' + name)
        runtime_root = bootstrap(args.runtime_root)
        for archive in plan['archives']:
            record = extract_archive(args, output, archive)
            records.append(dict(path=str((output / 'records' / archive['scene'] / (archive['cav'] + '.json')).relative_to(output)),
                                sha256=sha256(output / 'records' / archive['scene'] / (archive['cav'] + '.json'))))
            for name, row in record['files'].items():
                relative = '/'.join((archive['scene'], archive['cav'], name))
                file = no_test(output / 'train' / relative)
                content(file)
                files.append(dict(relative=relative, **row))
        n = lambda suffix: sum(r['relative'].endswith(suffix) for r in files)
        counts = dict(scene_count=len(plan['scenes']), cooperative_frames=plan['cooperative_frames'],
            source_scene_count=plan['source_scene_count'], raw_source_scene_count=plan['raw_source_scene_count'],
            empty_scene_count=len(plan['empty_scenes']),
            source_archive_count=len(plan['archives']),
            empty_cav_archive_count=sum(a['kind'] == 'empty_directory_placeholder' for a in plan['archives']),
            scene_agent_count=sum(len(s['agents']) for s in plan['scenes']),
            agent_frames=sum(len(s['agents']) * len(s['timestamps']) for s in plan['scenes']),
            clean_pcd=sum(bool(re.fullmatch(r'\d+\.pcd', Path(r['relative']).name)) for r in files),
            pcd=n('.pcd'), yaml=n('.yaml'), fog=n('_fog_0.060.pcd'), snow=n('_snow_2.5_2.0.pcd'),
            radar=n('_radar.pcd'), png=n('.png'))
        if counts['clean_pcd'] != counts['agent_frames']:
            raise ValueError('Agent-frame and Clean PCD counts differ')
        imported_modules = module_inventory(runtime_root)
        expected = {r['relative'] for r in files}
        if {p.relative_to(output / 'train').as_posix() for p in (output / 'train').rglob('*') if p.is_file()} != expected:
            raise ValueError('Unexpected files in extracted train view')
        manifest = dict(status='complete', full_train=not args.smoke, counts=counts, scenes=plan['scenes'],
            train_data_acceptance=plan['train_data_acceptance'], integrity_complete=True,
            empty_scenes=plan['empty_scenes'],
            plan_identity=plan['identity'], files=files, archives=[{k: a[k] for k in
                ('scene', 'cav', 'path', 'sha256', 'kind', 'excluded_from_loader', 'placeholder_evidence', 'weather_inventory')}
                for a in plan['archives']],
            empty_cav_policy='Content-verified directory-only numeric CAV archives of either sign are CRC-tested and hashed, recorded, never materialized in the official loader view',
            empty_scene_policy='Only scenes whose every archive is a verified directory placeholder are recorded as zero frames and excluded from loader view; nonempty/broken/incomplete scenes cannot be omitted',
            missing=[], duplicate=[], broken_archive=[],
            imported_modules=imported_modules,
            decoded='official Clean YAML/PCD readers; all four PNGs; other PCD headers; all members archive CRC and SHA256',
            output_hashes={'plan.json': sha256(output / 'plan.json'),
                'data_acceptance.json': sha256(output / 'data_acceptance.json'),
                **{r['path']: r['sha256'] for r in records}})
        check_scene_partition(manifest)
        manifest['identity'] = identity(manifest)
        atomic_json(output / 'manifest.json', manifest)  # Only successful final verification publishes completion.
        print('Train extraction complete:', counts, 'SMOKE ONLY' if args.smoke else 'FULL TRAIN', flush=True)
    except Exception as error:
        atomic_json(output / 'failure.json', dict(status='failed', error=repr(error),
            missing='see inspection_failure.json when present', duplicate='listing collisions fail before extraction',
            broken_archive=[a['path'] for a in plan['archives'] if
                (output / 'records' / a['scene'] / (a['cav'] + '.json')).exists() and
                read_json(output / 'records' / a['scene'] / (a['cav'] + '.json'))['status'] == 'failed']))
        raise
