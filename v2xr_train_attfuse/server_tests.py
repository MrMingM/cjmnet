"""Server-only synthetic checks of official AP and epoch-boundary recovery."""
import copy
from pathlib import Path
import random
import subprocess
import tempfile

from auditlib import runtime


def run(args):
    test_empty_placeholders(args)
    from provenance import verify_inputs
    evidence = verify_inputs(args)
    runtime(args, evidence['a'])
    from test_contract import run_tests
    run_tests()  # Frozen D-1C official matching/None/subset tests.
    import numpy as np
    import torch
    from opencood.tools import train_utils
    from opencood.hypes_yaml import yaml_utils
    from training import atomic_torch
    config = yaml_utils.load_yaml(str(Path(args.runtime_root) / 'opencood/hypes_yaml/V2X-R/LiDAR/V2XR_attfuse.yaml'))
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    model = torch.nn.Linear(2, 1)
    optimizer = train_utils.setup_optimizer(config, model)
    scheduler = train_utils.setup_lr_schedular(config, optimizer, n_iter_per_epoch=2)
    def epoch(number):
        for _ in range(2):
            optimizer.zero_grad()
            x = torch.randn(8, 2) + random.random() + np.random.random()
            model(x).square().mean().backward()
            optimizer.step()
        scheduler.step(number)
    for n in range(11):
        epoch(n)
    checkpoint = copy.deepcopy(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(), python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state()))
    epoch(11)
    expected = copy.deepcopy(model.state_dict())
    expected_lr = optimizer.param_groups[0]['lr']
    # Include our instrumentation's instance step method, then serialize only the scheduler data.
    scheduler.step = lambda epoch=None: None
    filtered = {k: v for k, v in scheduler.state_dict().items() if k != 'step'}
    with tempfile.TemporaryDirectory(prefix='d1d_synthetic_', dir=__import__('os').environ['TMPDIR']) as directory:
        path = Path(directory) / 'state.pth'
        atomic_torch(path, dict(checkpoint=checkpoint, scheduler=filtered), torch.save)
        loaded = torch.load(path, map_location='cpu', weights_only=False)['checkpoint']
    model = torch.nn.Linear(2, 1)
    model.load_state_dict(loaded['model'])
    optimizer = train_utils.setup_optimizer(config, model)
    scheduler = train_utils.setup_lr_schedular(config, optimizer, n_iter_per_epoch=2)
    optimizer.load_state_dict(loaded['optimizer'])
    scheduler.load_state_dict(loaded['scheduler'])
    random.setstate(loaded['python'])
    np.random.set_state(loaded['numpy'])
    torch.set_rng_state(loaded['cpu'])
    epoch(11)
    if not all(torch.equal(expected[k], model.state_dict()[k]) for k in expected) or optimizer.param_groups[0]['lr'] != expected_lr:
        raise ValueError('Full optimizer/scheduler/RNG recovery differs from uninterrupted training')
    from extract import validate_crc, pcd_header, extract_archive
    with tempfile.TemporaryDirectory(prefix='d1d_crc_', dir=__import__('os').environ['TMPDIR']) as directory:
        file = Path(directory) / '1.pcd'
        file.write_bytes(b'FIELDS x y z rgb\nWIDTH 1\nHEIGHT 1\nPOINTS 1\nDATA ascii\n0 0 0 0\n')
        if pcd_header(file) != 1:
            raise ValueError('PCD header validation failed')
        try:
            validate_crc(file, dict(size=file.stat().st_size, crc='00000000'))
        except ValueError:
            pass
        else:
            raise ValueError('Corrupt extracted member was accepted')
    from prepare import listing
    from auditlib import read_json, sha256
    with tempfile.TemporaryDirectory(prefix='d1d_archive_', dir=__import__('os').environ['TMPDIR']) as directory:
        root = Path(directory)
        source = root / 'members'
        source.mkdir()
        for name in ('000001.pcd', '000001.yaml', '000002.pcd'):
            (source / name).write_bytes(('synthetic ' + name + '\n').encode())
        pack = root / '1.7z'
        subprocess.run([args.seven_zip, 'a', str(pack), '000001.pcd', '000001.yaml', '000002.pcd'], cwd=source, check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        archive = dict(scene='scene_1', cav='1', path=str(pack), sha256=sha256(pack), entries=listing(args.seven_zip, pack))
        output = root / 'extracted'
        first = extract_archive(args, output, archive)
        if first['status'] != 'complete' or len(first['files']) != 3:
            raise ValueError('Synthetic archive did not complete')
        files = sorted((output / 'train/scene_1/1').iterdir())
        before = {p.name: (p.stat().st_mtime_ns, sha256(p)) for p in files}
        extract_archive(args, output, archive)
        if before != {p.name: (p.stat().st_mtime_ns, sha256(p)) for p in files}:
            raise ValueError('Resume re-extracted/modified a committed member')
        files[0].write_bytes(b'corrupt')
        try:
            extract_archive(args, output, archive)
        except ValueError:
            pass
        else:
            raise ValueError('Committed file corruption was silently accepted')
        if read_json(output / 'records/scene_1/1.json')['status'] != 'failed':
            raise ValueError('Failed extraction retained complete status')
        from unittest.mock import patch
        import archive_listing
        raw = b'7-Zip\n----------\nPath = empty_dir\nFolder = +\nSize = 0\n\n'
        captured = subprocess.CompletedProcess([], 0, stdout=raw, stderr=b'')
        diagnostics = root / 'listing_diagnostics'
        with patch.object(archive_listing, 'frozen_listing', side_effect=ValueError('Empty/invalid archive listing')), \
                patch.object(archive_listing.subprocess, 'run', return_value=captured):
            try:
                archive_listing.listing(args.seven_zip, pack, diagnostics)
            except ValueError as error:
                if str(pack) not in str(error) or 'diagnostic.json' not in str(error):
                    raise ValueError('Listing error lost archive/diagnostic context')
            else:
                raise ValueError('Empty/invalid archive was silently skipped')
        if (diagnostics / 'stdout.txt').read_bytes() != raw:
            raise ValueError('Listing raw output was not preserved exactly')
        diagnostic = read_json(diagnostics / 'diagnostic.json')
        if diagnostic['status'] != 'failed' or diagnostic['diagnostic_listing_returncode'] != 0:
            raise ValueError('Listing parse failure was confused with 7-Zip exit status')
    print('SERVER SYNTHETIC CHECKS PASSED: official O/O*/F AP; full epoch recovery; scheduler serialization; archive resume without re-extraction; corrupted-member rejection.', flush=True)


def test_empty_placeholders(args):
    """Server-only regression checks: classify only proven directory placeholders, preserve source identity."""
    import argparse
    from unittest.mock import patch
    from auditlib import HERE, atomic_json, read_json, sha256
    from archive_listing import ArchiveEntries, directory_placeholder, listing
    from extract import extract_archive, extraction_request, inspect
    from prepare import required_files
    import inventory as inventory_module
    references = inventory_module.official_reference_hashes()
    if 'readme.md' not in references or 'README.md' in references:
        raise ValueError('Frozen official readme name case differs')
    with patch.object(inventory_module, 'OFFICIAL_REFERENCE_FILES', ('README.md',)), \
            patch.object(inventory_module, 'sha256', side_effect=AssertionError('Wrong-case path reached file hashing')):
        try:
            inventory_module.official_reference_hashes()
        except ValueError as error:
            if 'spelling/case' not in str(error):
                raise
        else:
            raise ValueError('Wrong-case official reference was accepted')
    observed = ('7-Zip\n--\nType = 7z\nPhysical Size = 82\nHeaders Size = 82\nBlocks = 0\n'
        '\n----------\nPath = -1\nSize = 0\nPacked Size = 0\nAttributes = D drwxrwxr-x\nMethod =\nBlock = \n\n')
    if directory_placeholder(observed, Path('-1.7z')) is None:
        raise ValueError('Observed directory-only negative CAV was not recognized')
    positive = observed.replace('Path = -1', 'Path = 1084')
    if directory_placeholder(positive, Path('1084.7z')) is None:
        raise ValueError('Positive CAV with the same conclusive directory-only evidence was rejected')
    for text, name in ((observed, '1.7z'), (observed.replace('Blocks = 0', 'Blocks = 1'), '-1.7z'),
                       (observed.replace('Attributes = D drwxrwxr-x', 'Folder = -'), '-1.7z'),
                       (observed + 'Path = -1/empty.yaml\nSize = 0\nPacked Size = 0\n\n', '-1.7z'),
                       (observed.replace('Attributes = D drwxrwxr-x', 'Attributes = D\nSymbolic Link = elsewhere'), '-1.7z')):
        if directory_placeholder(text, Path(name)) is not None:
            raise ValueError('Non-placeholder archive was wrongly accepted')
    with tempfile.TemporaryDirectory(prefix='d1d_empty_rsu_', dir=__import__('os').environ['TMPDIR']) as directory:
        root = Path(directory)
        source = root / 'empty_members'
        (source / '-1').mkdir(parents=True)
        pack = root / '-1.7z'
        subprocess.run([args.seven_zip, 'a', str(pack), '--', '-1'], cwd=source, check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        entries = listing(args.seven_zip, pack, root / 'diagnostics')
        if entries or not getattr(entries, 'placeholder_evidence', None):
            raise ValueError('Actual 7-Zip empty directory fixture was not classified')
        archive = dict(scene='scene_1', cav='-1', path=str(pack), sha256=sha256(pack), entries={},
            kind='empty_directory_placeholder', placeholder_evidence=entries.placeholder_evidence)
        output = root / 'extracted'
        for _ in range(2):
            record = extract_archive(args, output, archive)
            if not record['crc_test_passed'] or record['status'] != 'complete' or not record['excluded_from_loader']:
                raise ValueError('Empty archive lacks CRC/completion/exclusion evidence')
            if (output / 'train/scene_1/-1').exists():
                raise ValueError('Empty RSU was materialized into the loader view')
        (source / '1084').mkdir()
        positive_pack = root / '1084.7z'
        subprocess.run([args.seven_zip, 'a', str(positive_pack), '--', '1084'], cwd=source, check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        positive_entries = listing(args.seven_zip, positive_pack, root / 'positive_diagnostics')
        if positive_entries or not getattr(positive_entries, 'placeholder_evidence', None):
            raise ValueError('Actual positive-ID empty archive was not content-classified')
        positive_archive = dict(scene='scene_1', cav='1084', path=str(positive_pack),
            sha256=sha256(positive_pack), entries={}, kind='empty_directory_placeholder',
            placeholder_evidence=positive_entries.placeholder_evidence)
        positive_record = extract_archive(args, output, positive_archive)
        if (not positive_record['crc_test_passed'] or positive_record['status'] != 'complete'
                or (output / 'train/scene_1/1084').exists()):
            raise ValueError('Positive directory placeholder lacks CRC evidence or entered the loader view')
        if (output / 'train/scene_1').exists():
            raise ValueError('An all-placeholder scene was created in the loader view')
        raw = root / 'train'
        scene = raw / 'scene_1'
        scene.mkdir(parents=True)
        for name in ('-1.7z', '20.7z', '21.7z', '22.7z'):
            (scene / name).write_bytes(b'synthetic archive identity')
        files = {n: dict(member=n, size=1, crc='') for t in ('000001', '000002') for n in required_files(t)}
        synthetic = argparse.Namespace(data_root=str(raw), smoke=True, seven_zip=args.seven_zip, resume=True)
        def fixture(seven, path, diagnostics):
            if path.stem in ('-1', '20') or path.parent.name == '000_empty_scene':
                return ArchiveEntries(dict(entries.placeholder_evidence, cav=path.stem,
                    directory=path.stem, archive_sha256=sha256(path)))
            return files
        planned = root / 'planned'
        planned.mkdir()
        with patch('extract.listing', side_effect=fixture), \
                patch('train_acceptance.approved_reference'), \
                patch('extract.bind_train_acceptance', return_value={'synthetic_fixture_only': True}):
            plan = inspect(synthetic, planned)
        if plan['scenes'][0]['agents'] != ['21', '22'] or plan['scenes'][0]['empty_agents'] != ['-1', '20'] or plan['cooperative_frames'] != 2:
            raise ValueError('Empty archive altered active CAV order/cooperative frame count')
        # Exercise real ID/hash/context matching separately from the production-only 8083-frame policy.
        from train_acceptance import compare_observation
        reference = read_json(planned / 'inventory.json')
        observation = dict(scenes=plan['scenes'], empty_scenes=plan['empty_scenes'],
            archives=plan['archives'], metadata=plan['scene_metadata'])
        compare_observation(reference, **observation)
        for mutation in ('archive_hash', 'timestamp', 'ego', 'missing_archive'):
            changed = copy.deepcopy(observation)
            if mutation == 'archive_hash':
                changed['archives'][0]['sha256'] = '0' * 64
            elif mutation == 'timestamp':
                changed['scenes'][0]['timestamps'][0] = '999999'
            elif mutation == 'ego':
                changed['scenes'][0]['ego'] = '22'
            else:
                changed['archives'].pop()
            try:
                compare_observation(reference, **changed)
            except ValueError:
                pass
            else:
                raise ValueError('Train data acceptance silently allowed ' + mutation)
        from archive_listing import inspect_only
        def failed_fixture(seven, path, diagnostics):
            if path.stem in ('-1', '20'):
                return fixture(seven, path, diagnostics)
            raise ValueError('synthetic unclassified listing failure')
        collected = root / 'collected_inspection'
        collected.mkdir()
        with patch('archive_listing.listing', side_effect=failed_fixture):
            try:
                inspect_only(synthetic, collected)
            except ValueError:
                pass
            else:
                raise ValueError('Inspection with failures reported success')
        summary = read_json(collected / 'inspection_summary.json')
        if (not summary['scan_completed'] or summary['status'] != 'failed'
                or summary['archive_count'] != 4 or summary['failed_archive_count'] != 2
                or summary['empty_cav_archive_count'] != 2
                or (collected / 'inspection_complete.json').exists()):
            raise ValueError('Inspection did not collect every failure or falsely published completion')
        # A zero-frame scene sorts first: smoke must record it, then select real data.
        empty_scene = raw / '000_empty_scene'
        empty_scene.mkdir()
        for name in ('-1.7z', '1084.7z'):
            (empty_scene / name).write_bytes(b'synthetic directory-only archive identity')
        prefixed = root / 'prefixed_plan'
        prefixed.mkdir()
        with patch('extract.listing', side_effect=fixture), \
                patch('train_acceptance.approved_reference'), \
                patch('extract.bind_train_acceptance', return_value={'synthetic_fixture_only': True}):
            prefixed_plan = inspect(synthetic, prefixed)
        if (prefixed_plan['source_scene_count'] != 2 or len(prefixed_plan['empty_scenes']) != 1
                or prefixed_plan['scenes'][0]['scene'] != 'scene_1'
                or prefixed_plan['cooperative_frames'] != 2
                or prefixed_plan['empty_scenes'][0]['cooperative_frames'] != 0
                or len(prefixed_plan['archives']) != 6):
            raise ValueError('Zero-frame scene was not preserved as evidence or changed smoke frame selection')
        from inventory import timestamp_overview
        gap = timestamp_overview(['000060', '000062', '000066', '000068'])
        if gap['count'] != 4 or gap['modal_step'] != 2 or gap['candidate_gaps'][0]['candidate_timestamps'] != ['000064']:
            raise ValueError('Timestamp diagnostics changed the actual count or failed to preserve a candidate gap')
        full_args = argparse.Namespace(**vars(synthetic))
        full_args.smoke = False
        count_failure = root / 'count_mismatch'
        count_failure.mkdir()
        with patch('extract.listing', side_effect=fixture), patch('train_acceptance.approved_reference'):
            try:
                inspect(full_args, count_failure)
            except ValueError as error:
                if 'Train cooperative frames 2 != approved 8083' not in str(error):
                    raise
            else:
                raise ValueError('Frame count mismatch was silently accepted')
        count_report = read_json(count_failure / 'inventory_summary.json')
        if (count_report['actual_cooperative_frames'] != 2 or count_report['official_filename_rule_ego_count'] != 2
                or count_report['status'] != 'count_mismatch' or count_report['training_allowed']
                or (count_failure / 'plan.json').exists() or (count_failure / 'manifest.json').exists()):
            raise ValueError('Count failure lost evidence or published experiment completion')
        inventory_only_output = root / 'inventory_only'
        inventory_only_output.mkdir()
        with patch('extract.listing', side_effect=fixture):
            inventory_report = inspect(full_args, inventory_only_output, inventory_only=True)
        if inventory_report['status'] != 'count_mismatch' or (inventory_only_output / 'plan.json').exists():
            raise ValueError('Read-only inventory treated a count mismatch as an accepted data plan')
        from auditlib import check_scene_partition
        partition = dict(scenes=prefixed_plan['scenes'], empty_scenes=prefixed_plan['empty_scenes'],
            archives=prefixed_plan['archives'], counts=dict(source_scene_count=2, empty_scene_count=1,
                scene_count=1, cooperative_frames=2))
        check_scene_partition(partition)
        bad_partition = copy.deepcopy(partition)
        bad_partition['empty_scenes'][0]['cooperative_frames'] = 1
        try:
            check_scene_partition(bad_partition)
        except ValueError:
            pass
        else:
            raise ValueError('Nonzero frame count was allowed in an excluded scene')
        all_empty_raw = root / 'all_empty_source/train'
        all_empty_scene = all_empty_raw / '000_empty_scene'
        all_empty_scene.mkdir(parents=True)
        (all_empty_scene / '-1.7z').write_bytes(b'synthetic directory-only archive identity')
        all_empty_args = argparse.Namespace(**vars(synthetic))
        all_empty_args.data_root = str(all_empty_raw)
        all_empty_output = root / 'all_empty_plan'
        all_empty_output.mkdir()
        with patch('extract.listing', side_effect=fixture), patch('train_acceptance.approved_reference'):
            try:
                inspect(all_empty_args, all_empty_output)
            except ValueError as error:
                if 'No data-bearing train scene found' not in str(error):
                    raise
            else:
                raise ValueError('An all-empty source was accepted as a training/smoke dataset')
        if (all_empty_output / 'plan.json').exists():
            raise ValueError('All-empty source published an experiment plan')
        migration = root / 'migration'
        migration.mkdir()
        old = dict(source_hashes=read_json(HERE / 'pre_empty_placeholder_requests.json')['accepted_source_hashes'][0],
            data_root=synthetic.data_root, seven_zip_sha256=sha256(args.seven_zip), smoke=True)
        failure = dict(command='extract', error="ValueError('Empty/invalid archive listing')")
        atomic_json(migration / 'request.json', old)
        atomic_json(migration / 'last_failure.json', failure)
        extraction_request(synthetic, migration, {'synthetic.py': '0' * 64})
        if read_json(migration / 'request.json')['source_hashes'] != {'synthetic.py': '0' * 64} or not list((migration / 'request_history').glob('*.json')):
            raise ValueError('Failed pre-extraction request upgrade was not journaled')
        blocked = root / 'blocked_migration'
        blocked.mkdir()
        atomic_json(blocked / 'request.json', old)
        atomic_json(blocked / 'last_failure.json', failure)
        atomic_json(blocked / 'plan.json', {})
        try:
            extraction_request(synthetic, blocked, {'synthetic.py': '0' * 64})
        except ValueError:
            pass
        else:
            raise ValueError('Source identity was upgraded despite an existing experiment plan')
        scene_failure = root / 'zero_scene_migration'
        scene_failure.mkdir()
        atomic_json(scene_failure / 'request.json', old)
        atomic_json(scene_failure / 'last_failure.json', dict(command='extract',
            error="ValueError('Scene has no data CAVs; no scene may be silently discarded: scene')"))
        extraction_request(synthetic, scene_failure, {'synthetic.py': '0' * 64})
        if read_json(scene_failure / 'request.json')['source_hashes'] != {'synthetic.py': '0' * 64}:
            raise ValueError('Known zero-frame scene failure did not preserve upgrade/recovery support')
        count_migration = root / 'count_migration'
        count_migration.mkdir()
        atomic_json(count_migration / 'request.json', old)
        atomic_json(count_migration / 'last_failure.json', dict(command='extract',
            error="ValueError('Train cooperative frames 8083 != official 8084; do not train')"))
        extraction_request(synthetic, count_migration, {'synthetic.py': '0' * 64})
        if read_json(count_migration / 'request.json')['source_hashes'] != {'synthetic.py': '0' * 64}:
            raise ValueError('Count diagnostic upgrade was not preserved as a data-free request transition')
    print('EMPTY-CAV REGRESSION CHECKS PASSED: observed format, zero-byte file/link rejection, actual 7z CRC, no loader directory, active CAV/frame count, narrowly journaled request upgrade.', flush=True)
