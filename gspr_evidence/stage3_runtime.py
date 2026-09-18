"""Read-only lineage checks and deterministic replay shared by Stage-3A/B."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
from .stage3_analysis import read_json, read_rows, unique_rows, row_key, WEATHERS

ROOT = Path(__file__).resolve().parents[1]
VALIDATION = '/data/scd/datasets/opv2v_official_data_dumping/validate'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def safe_output(path, create=False):
    out = Path(path).resolve()
    allowed = Path('/data/cjm/datasets/logs').resolve()
    if allowed not in out.parents:
        raise ValueError('Stage-3 output must be below /data/cjm/datasets/logs')
    if create:
        out.mkdir(parents=True, exist_ok=False)
    return out


def validate_sources():
    manifest = read_json(ROOT / 'gspr_evidence/stage3_sources.json')
    for name, digest in manifest.items():
        if source_sha(ROOT / name) != digest:
            raise ValueError('Stage-3 audited source changed: '+name)
    return manifest


def source_sha(path):
    """Portable source fingerprint: normalize CRLF only, not spaces/content."""
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def load_lineage(stage2_root, weather):
    """Use actual Stage-2 source-valid cases, never strong/weak proxies."""
    root = Path(stage2_root).resolve()
    result = read_json(root / 'evidence_validity_results.json')
    stage1 = Path(result['stage1_root']).resolve()
    first = read_json(stage1 / weather / 'protocol.json')
    second = read_json(root / 'peer' / weather / 'protocol.json')
    for p in (first, second):
        if not p.get('development_only') or p.get('test_data_used', True) or p['weather'] != weather:
            raise ValueError('Only development weather lineage is allowed')
    if first.get('smoke') or first.get('region_scale', 1.0) != 1.0:
        raise ValueError('Require the full Stage-1 run with canonical region scale')
    if Path(first['data_root']).resolve() != Path(VALIDATION).resolve():
        raise ValueError('Stage-1 root is not official validation')
    if Path(second['stage1_root']).resolve() != stage1:
        raise ValueError('Stage-2/Stage-1 root mismatch')
    for field in ('sample_indices', 'frontend_sha256', 'frontend_config_sha256'):
        if first[field] != second[field]:
            raise ValueError('Stage-1/2 mismatch: '+field)
    if first['collector_sha256'] != sha(ROOT/'qa_observation_diagnostic/collect.py'):
        raise ValueError('Stage-1 collector version changed')
    if second['source_sha256'] != sha(ROOT/'qa_evidence_validity/peer_audit.py'):
        raise ValueError('Stage-2 collector version changed')
    rows = unique_rows(read_rows(root/'peer'/weather/'peer_targets.jsonl'))
    stats = unique_rows(read_rows(stage1/weather/'targets.jsonl'))
    clean = unique_rows(read_rows(stage1/'clean'/'targets.jsonl'))
    eligible = {k for k, r in stats.items() if k in clean and clean[k]['ego_detected'] and not r['ego_detected']}
    if set(rows) != eligible or len(rows) != second['candidate_targets']:
        raise ValueError('Stage-2 target coverage is incomplete or inconsistent')
    for k, r in rows.items():
        peers = r['peers']
        valid = any(p['matched'] for p in peers)
        indices = [int(p['peer_index']) for p in peers if p['matched']]
        if [int(p['peer_index']) for p in peers] != list(range(1, len(peers)+1)):
            raise ValueError('Peer order is not contiguous')
        if (r['any_peer_alone_detected'] != valid or r['detected_peer_indices'] != indices
                or r['source_valid_full_miss'] != (valid and not r['full']['matched'])
                or r['full']['matched'] != stats[k]['full_detected'] or r['ego']['matched']):
            raise ValueError('Stage-2 flags do not match branch observations')
    actual_n = sum(r['source_valid_full_miss'] for r in rows.values())
    expected = result['peer_validity'][weather]['task_observed']
    if (expected['source_valid_full_miss_n'] != actual_n or
            expected['any_peer_alone_detected_n'] != sum(r['any_peer_alone_detected'] for r in rows.values())):
        raise ValueError('Stage-2 summary/log count mismatch')
    paths = [root/'evidence_validity_results.json', root/'peer'/weather/'peer_targets.jsonl',
             root/'peer'/weather/'protocol.json', stage1/weather/'protocol.json',
             stage1/weather/'targets.jsonl', stage1/'clean'/'targets.jsonl']
    return dict(root=root, stage1=stage1, first=first, second=second, rows=rows, stats=stats,
                input_files={str(p): sha(p) for p in paths})


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--stage2-root', required=True)
    p.add_argument('--weather', choices=WEATHERS, required=True)
    p.add_argument('--config', default='qa_observation_diagnostic/experiment.yaml')
    p.add_argument('--frontend-config', required=True)
    p.add_argument('--frontend-checkpoint', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--smoke', type=int, default=0, help='First N candidate frames per weather; 0 = all')
    return p


def prepare(args):
    import platform
    import torch
    import shapely
    from gspr_evidence import runtime as rt
    if args.smoke < 0:
        raise ValueError('Negative smoke count')
    rt.verify_frozen()
    sources = validate_sources()
    lineage = load_lineage(args.stage2_root, args.weather)
    # Old runs lack an experiment-config digest. Only allow the audited historical
    # config snapshot; do not silently accept alternative workers/seeds/weather.
    if source_sha(args.config) != sources['qa_observation_diagnostic/experiment.yaml']:
        raise ValueError('Replay config differs from audited Stage-1/2 config')
    if sha(args.frontend_config) != lineage['first']['frontend_config_sha256']:
        raise ValueError('Frontend config SHA mismatch')
    options, hypes = rt.load_config(args.config, args.frontend_config)
    if Path(hypes['validate_dir']).resolve() != Path(VALIDATION).resolve():
        raise ValueError('Refusing non-validation data')
    rt.seed_all(options['seed'])
    device = rt.device()
    model, digest = rt.load_model(hypes, options, args.frontend_checkpoint, device)
    model.eval()
    if digest != lineage['first']['frontend_sha256']:
        raise ValueError('Frontend checkpoint SHA mismatch')
    for path in (args.config, args.frontend_config, args.frontend_checkpoint):
        lineage['input_files'][str(Path(path).resolve())] = sha(path)
    if model.engine.value_bytes != 4:
        raise ValueError('Stage-2 peer/full comparison requires float32 unquantized features')
    rt.seed_all(options['seed'])
    ds, loader, indices = rt.make_loader(hypes, options, weather=args.weather)
    if indices != lineage['first']['sample_indices']:
        raise ValueError('Full frame queue mismatch; never subset loader for smoke')
    if list(ds.len_record) != lineage['first']['scene_ends']:
        raise ValueError('Scene boundaries/order changed')
    candidates = {}
    for k, r in lineage['rows'].items():
        if r['source_valid_full_miss']:
            candidates.setdefault(k[0], {})[k[1]] = r
    chosen_frames = sorted(candidates)
    if args.smoke:
        chosen_frames = chosen_frames[:args.smoke]
    candidates = {i: candidates[i] for i in chosen_frames}
    out = safe_output(args.output_dir, create=True)
    implementation = {str(p.relative_to(ROOT)).replace('\\', '/'): source_sha(p)
                      for p in (ROOT/'gspr_evidence').glob('stage3*.py')}
    lineage['implementation'] = implementation
    protocol = dict(schema=1, development_only=True, test_data_used=False,
        weather=args.weather, smoke=args.smoke, sample_indices=indices,
        selected_candidate_frames=chosen_frames, candidate_targets=sum(map(len, candidates.values())),
        stage2_root=str(lineage['root']), stage1_root=str(lineage['stage1']),
        frontend_sha256=digest, frontend_config_sha256=sha(args.frontend_config),
        experiment_config_sha256=sha(args.config), options=options,
        audited_sources=sources, input_files=lineage['input_files'],
        implementation_sha256=implementation,
        software=dict(python=platform.python_version(), torch=torch.__version__,
                      numpy=np.__version__, shapely=shapely.__version__),
        postprocessing=dict(order=ds.post_processor.params['order'],
            score_threshold=ds.post_processor.params['target_args']['score_threshold'],
            nms_threshold=ds.post_processor.params['nms_thresh']),
        audited_source_hash_algorithm='sha256 after CRLF to LF normalization; historical log/checkpoint SHA remains raw',
        replay_limit='Historical Stage-1/2 did not save per-frame input hashes or experiment config SHA. '
                     'Historical bitwise weather identity cannot be proved. Audited config, original '
                     'queue, source versions, per-target sensing statistics and all source detection '
                     'details are checked; Stage-3 now records input hashes.',
        input_hash_algorithm='sha256 over sorted tensor keys, dtype, shape, contiguous bytes',
        nms_topk=1000, match_iou=.7, consistency='Every traced branch checked against ds.post_process')
    return rt, lineage, model, device, ds, loader, candidates, out, protocol


def tensor_digest(data):
    h = hashlib.sha256()
    def visit(value, name):
        if isinstance(value, dict):
            for k in sorted(value):
                visit(value[k], name+'/'+str(k))
        elif isinstance(value, (list, tuple)):
            for i, x in enumerate(value):
                visit(x, name+'/'+str(i))
        else:
            arr = value.detach().cpu().contiguous().numpy()
            h.update(name.encode()); h.update(str(arr.dtype).encode())
            h.update(str(arr.shape).encode()); h.update(arr.tobytes())
    visit(data, '')
    return h.hexdigest()


def compare_details(actual, expected, label):
    for field in ('matched', 'matched_score', 'matched_iou', 'best_post_iou', 'score_at_best_post_iou'):
        a, b = actual[field], expected[field]
        if a is None or b is None:
            if a is not b:
                raise AssertionError(label+': '+field)
        elif field == 'matched':
            if a != b:
                raise AssertionError(label+': matched state changed')
        elif not np.isclose(a, b, atol=2e-5, rtol=2e-5):
            raise AssertionError(label+': '+field+' changed')


def check_sensing(model, inp, encoded, gt, frame_rows, lineage):
    """Check all retained-point/observation statistics for replayed candidate targets."""
    from qa_observation_diagnostic.collect import (
        _box_geometry, _grid_target_cells, _extract_retained, _target_agent_stats)
    processed = inp['processed_lidar']
    base = model.engine.base
    rel = base.gspr(processed['voxel_features'], processed['voxel_num_points'], processed['voxel_coords'])
    retained = _extract_retained(processed, rel, len(encoded['levels'][0]))
    for j, r in frame_rows.items():
        old = lineage['stats'][row_key(r)]
        geom = _box_geometry(gt[j])
        if np.linalg.norm(geom['center'] - [old['center_x'], old['center_y']]) > 1e-3:
            raise AssertionError('GT center/order changed')
        if len(retained) != old['agents']:
            raise AssertionError('Agent count changed')
        ids = _grid_target_cells(geom, model.grid, model.lidar_range)
        for a, expected in enumerate([old['ego']] + old['peers']):
            observed = _target_agent_stats(inp['clouds'][a], retained[a], geom, ids,
                                          encoded['obs'][a], encoded['semantics'][a])
            for field, value in expected.items():
                equal = (observed[field] == value if isinstance(value, int) else
                         np.isclose(observed[field], value, atol=2e-5, rtol=2e-5))
                if not equal:
                    raise AssertionError(f'Stage-1 sensing replay changed target {j}, source {a}, {field}')


def final_guards(lineage):
    from gspr_evidence.runtime import verify_frozen
    verify_frozen()
    validate_sources()
    for name, digest in lineage.get('implementation', {}).items():
        if source_sha(ROOT/name) != digest:
            raise RuntimeError('Stage-3 implementation changed while running: '+name)
    for name, digest in lineage['input_files'].items():
        if sha(name) != digest:
            raise RuntimeError('Input log changed while running: '+name)


def replay_frame(model, ds, batch, encoded, inp, index, lineage):
    """Reproduce ALL Stage-2 observations in this selected frame, not just hits."""
    import torch
    from qa_evidence_validity.peer_audit import _source_only_prediction, _greedy_target_details
    frame_rows = {k[1]: r for k, r in lineage['rows'].items() if k[0] == index}
    full, _ = model.detect(encoded, torch.ones_like(model.empty_masks(encoded)), serialize=False)
    original = model.engine.base(inp)
    for name in ('psm', 'rm'):
        torch.testing.assert_close(full[name], original[name], atol=2e-4, rtol=2e-4)
    predictions = {'full': full, 'ego': _source_only_prediction(model, encoded, 0)}
    for a in range(1, len(encoded['levels'][0])):
        predictions['peer_'+str(a)] = _source_only_prediction(model, encoded, a)
    none, _ = model.detect(encoded, model.empty_masks(encoded), serialize=False)
    for name in ('psm', 'rm'):
        torch.testing.assert_close(predictions['ego'][name], none[name], atol=2e-4, rtol=2e-4)
    reference_gt = None
    observations = {}
    for branch, prediction in predictions.items():
        boxes, scores, gt = ds.post_process(batch, {'ego': prediction})
        if reference_gt is None:
            reference_gt = gt
        else:
            torch.testing.assert_close(gt, reference_gt)
        details, fp = _greedy_target_details(boxes, scores, gt)
        observations[branch] = (details, fp)
        for j, r in frame_rows.items():
            if j >= len(gt):
                raise AssertionError('GT index missing')
            expected = r[branch] if branch in ('ego', 'full') else r['peers'][int(branch[5:])-1]
            compare_details(details[j], expected, f'{index}/{j}/{branch}')
            if fp != expected['frame_fp_count']:
                print(
                    f'WARNING: Stage-2 frame FP count changed: '
                    f'frame={index}, branch={branch}, target={j}, '
                    f'old={expected["frame_fp_count"]}, new={fp}',
                    flush=True,
                )
    check_sensing(model, inp, encoded, reference_gt.detach().cpu().numpy(), frame_rows, lineage)
    return predictions, observations, reference_gt
