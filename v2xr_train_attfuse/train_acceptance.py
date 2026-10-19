"""Bind the explicitly accepted 8083 frames to the user's exact source inventory."""
from pathlib import Path

from auditlib import HERE, bound_json, identity, no_test, read_json, sha256

POLICY_PATH = HERE / 'TRAIN_DATA_ACCEPTANCE.json'


def policy():
    value = read_json(POLICY_PATH)
    if (value['status'] != 'user_accepted' or value['accepted_cooperative_frames'] != 8083
            or value['official_reference_cooperative_frames'] != 8084 or value['difference_from_reference'] != -1):
        raise ValueError('The explicit user-approved train data contract changed')
    return value


def required_train_count():
    return policy()['accepted_cooperative_frames']


def approved_reference():
    approved = policy()
    root = no_test(approved['inventory_evidence_root']).resolve(strict=True)
    expected = approved['approved_artifact_sha256']
    candidates = []
    for path in sorted(root.rglob('inventory.json')):
        no_test(path)
        if sha256(path) == expected['inventory.json']:
            candidates.append(path.parent)
    if not candidates:
        raise ValueError('The user-approved inventory.json hash was not found. Preserve the original successful inventory under ' + str(root))
    source = candidates[0]
    for name, digest in expected.items():
        if sha256(no_test(source / name)) != digest:
            raise ValueError('User-approved inventory evidence changed: ' + str(source / name))
    reference = read_json(source / 'inventory.json')
    if (reference['identity'] != approved['approved_inventory_identity']
            or identity({k: v for k, v in reference.items() if k != 'identity'}) != reference['identity']):
        raise ValueError('Approved inventory identity differs')
    if any(reference['summary'][k] != v for k, v in approved['approved_counts'].items()):
        raise ValueError('Approved inventory summary counts differ from the user-reviewed facts')
    return approved, reference, source


def archive_projection(rows):
    result = {}
    for row in rows:
        key = (row['scene'], row['cav'])
        if key in result:
            raise ValueError('Duplicate scene/CAV archive in train acceptance inventory')
        result[key] = dict(scene=row['scene'], cav=row['cav'],
            path=row.get('path', row.get('archive')),
            sha256=row.get('sha256', row.get('archive_sha256')),
            kind=row['kind'], excluded_from_loader=row['excluded_from_loader'])
    return result


def scene_projection(rows):
    return [dict(scene=s['scene'], ego=s['ego'], agents=s['agents'],
        registered_cavs=s['agents'][:7], empty_agents=s['empty_agents']) for s in rows]


def actual_frame_keys(scenes):
    return [dict(scene=s['scene'], timestamp=t, ego=s['ego'], available_cavs=s['agents'],
        official_registered_cavs=s['agents'][:7]) for s in scenes for t in s['timestamps']]


def compare_observation(reference, scenes, empty_scenes, archives, metadata, smoke=False):
    """Compare physical IDs/context and all source archive bytes, independent of diagnostic stdout."""
    expected, observed = archive_projection(reference['archives']), archive_projection(archives)
    if (not smoke and expected.keys() != observed.keys()) or not observed.keys() <= expected.keys():
        raise ValueError('Train archive set differs from the explicitly accepted inventory')
    for key, row in observed.items():
        if row != expected[key]:
            raise ValueError('Train archive path/hash/classification changed: ' + str(key))
    frame_keys = actual_frame_keys(scenes)
    expected_frames = reference['frame_keys'][:len(frame_keys)] if smoke else reference['frame_keys']
    if frame_keys != expected_frames:
        raise ValueError('Train physical frame IDs/order/ego/CAV context differs from the accepted inventory')
    expected_scenes = {s['scene']: s for s in scene_projection(reference['scenes'])}
    observed_scenes = scene_projection(scenes)
    if not smoke and len(observed_scenes) != len(expected_scenes):
        raise ValueError('Data-bearing train scene set changed')
    if any(expected_scenes.get(s['scene']) != s for s in observed_scenes):
        raise ValueError('Train scene ego/CAV selection changed')
    expected_empty = {s['scene']: s['empty_agents'] for s in reference['empty_scenes']}
    observed_empty = {s['scene']: s['empty_agents'] for s in empty_scenes}
    if (not smoke and observed_empty != expected_empty) or any(expected_empty.get(k) != v for k, v in observed_empty.items()):
        raise ValueError('Zero-frame scene inventory changed')
    expected_metadata = {r['path']: r['sha256'] for r in reference['scene_metadata']}
    observed_metadata = {r['path']: r['sha256'] for r in metadata}
    if (not smoke and observed_metadata != expected_metadata) or any(expected_metadata.get(k) != v for k, v in observed_metadata.items()):
        raise ValueError('Original scene metadata bytes changed')
    projection = dict(archives=[observed[k] for k in sorted(observed)],
        frame_keys=frame_keys, scenes=observed_scenes, empty_scenes=observed_empty, metadata=observed_metadata)
    return identity(projection)


def bind_train_acceptance(output, args, scenes, empty_scenes, archives, metadata):
    approved, reference, source = approved_reference()
    if no_test(args.data_root).resolve() != no_test(approved['source_train_root']).resolve():
        raise ValueError('Raw train root differs from the accepted source inventory')
    projection_id = compare_observation(reference, scenes, empty_scenes, archives, metadata, args.smoke)
    selected_count = sum(len(s['timestamps']) for s in scenes)
    if not args.smoke and selected_count != approved['accepted_cooperative_frames']:
        raise ValueError('Approved full source must have exactly 8083 actual frames')
    contract = dict(status='user_accepted_inventory_verified',
        official_reference_cooperative_frames=approved['official_reference_cooperative_frames'],
        accepted_cooperative_frames=approved['accepted_cooperative_frames'],
        actual_selected_cooperative_frames=selected_count, difference_from_reference=-1,
        full_source_scope=not args.smoke, acceptance_policy_sha256=sha256(POLICY_PATH),
        approved_inventory_identity=approved['approved_inventory_identity'],
        approved_artifact_sha256=approved['approved_artifact_sha256'],
        approved_evidence_directory=str(source), source_projection_identity=projection_id,
        authorization=approved['authorization'],
        integrity_requirement='This count acceptance alone does not certify CRC/content integrity; the extraction manifest must certify it separately.',
        interpretation='Count deviation is explicitly accepted; archive CRC and extracted content checks are still mandatory.')
    contract['identity'] = identity(contract)
    bound_json(output / 'data_acceptance.json', contract)
    print('APPROVED TRAIN INVENTORY MATCHED:', selected_count, 'actual / 8084 published; source archive/frame hashes locked', flush=True)
    return contract


def verify_manifest_acceptance(manifest, root):
    approved, reference, source = approved_reference()
    contract = manifest['train_data_acceptance']
    if (contract != read_json(root / 'data_acceptance.json')
            or identity({k: v for k, v in contract.items() if k != 'identity'}) != contract['identity']
            or contract['status'] != 'user_accepted_inventory_verified'
            or not manifest.get('integrity_complete')
            or contract['official_reference_cooperative_frames'] != 8084
            or contract['accepted_cooperative_frames'] != required_train_count()
            or contract['acceptance_policy_sha256'] != sha256(POLICY_PATH)
            or contract['approved_inventory_identity'] != approved['approved_inventory_identity']
            or not contract['full_source_scope'] or contract['actual_selected_cooperative_frames'] != required_train_count()
            or contract['approved_artifact_sha256'] != approved['approved_artifact_sha256']):
        raise ValueError('Completed train manifest lacks the exact accepted data contract')
    metadata = read_json(root / 'plan.json')['scene_metadata']
    projection_id = compare_observation(reference, manifest['scenes'], manifest['empty_scenes'], manifest['archives'], metadata)
    if projection_id != contract['source_projection_identity']:
        raise ValueError('Completed train source/frame inventory differs from the accepted contract')
