"""One frozen source-pool forward per frame for both label policies."""
from collections import Counter
from pathlib import Path
import numpy as np
from .common import (GROUPS, FEATURES, WEATHERS, Manifest, atomic_json, atomic_torch,
                     cli, load_cache, protocol, read_json, report)
from .inputs import control_source
from .probes import load_probe
from local_fusion_action_utility_audit.s3_replay import predict_actions, conflict_resolver, execute
from local_fusion_action_utility_audit.s0_counterfactual import proposals, prediction_hash, frame_ap_stats

POLICIES = ('Assisted-Task-Greedy', 'Proposal-Task-Greedy', 'Proposal-Task-CommonConservative')
METHODS = ('Shared',) + tuple(g + '-' + policy for g in GROUPS for policy in POLICIES)


def check_ap(actual, expected, label):
    differences = {}
    for key in ('ap30', 'ap50', 'ap70'):
        if actual[key] is None or expected[key] is None or not np.isfinite([actual[key], expected[key]]).all():
            raise ValueError('Invalid AP: ' + label + '.' + key)
        differences[key] = abs(actual[key] - expected[key])
        if differences[key] > 1e-6:
            raise RuntimeError(f'AP reproduction failed {label}.{key}: {differences[key]}')
    return differences


def action_record(action, mask=False):
    import torch
    result = {k: action[k] for k in ('position', 'anchor_id', 'shared_score', 'cls_source',
                                     'reg_source', 'cls_margin', 'reg_margin')}
    if mask:
        result.update({'margin': action['margin'], 'mask_shape': list(action['mask'].shape),
                       'accepted_cell_ids': torch.tensor(np.flatnonzero(action['mask']), dtype=torch.int32)})
    return result


def replay(run):
    import torch
    from ceif_audit.scoring import ap_values, empty_stats, merge
    from opencood.tools.train_utils import to_device
    from opencood.utils import eval_utils
    from local_fusion_task_split_pilot.proposal_constrained_oracle import assign_proposals
    from local_fusion_utility_v2.outcomes import compare_predictions
    from .runtime import ReplayRuntime
    p = protocol(run)
    source = control_source(run, p)
    if source.readiness(baseline=True):
        raise RuntimeError('Source replay inputs incomplete: ' + ', '.join(source.readiness(True)))
    source.artifact('baseline_reproduction.json', 'baseline')
    runtime = ReplayRuntime(p['source_run'])
    probes = {g: {t: load_probe(run, g, t) for t in ('cls', 'reg')} for g in GROUPS}
    manifest = Manifest(run)
    conditions, shared_checks = {}, {}
    with torch.no_grad():
        for weather in WEATHERS:
            folder = Path(run) / 'cache' / 'replay' / weather
            summary_path = folder / 'summary.pt'
            if manifest.complete('replay-weather', 'validation', weather):
                summary = load_cache(summary_path)
                conditions[weather], shared_checks[weather] = summary['methods'], summary['shared_check']
                continue
            dataset, loader = runtime.loader('validation', weather)
            stats = {method: empty_stats() for method in METHODS}
            totals = {method: Counter() for method in METHODS}
            seen = []
            for ordinal, batch in enumerate(loader):
                index = int(batch['ego']['communication_sample_index'][0])
                seen.append(index)
                if index != source.protocol['validation_indices'][ordinal]:
                    raise RuntimeError('Original validation order differs')
                path = folder / f'{index:08d}.pt'
                if manifest.complete('replay-frame', 'validation', weather, index):
                    row = load_cache(path)
                else:
                    with manifest.work('replay-frame', 'validation', weather, index):
                        batch = to_device(batch, runtime.target)
                        context, shared, pool = runtime.predict(batch, weather, verify=ordinal == 0)
                        cached = source.frame('validation', weather, index)
                        base_cache = load_cache(source.artifact(f'cache/baseline/{weather}/{index:08d}.pt',
                                                               'baseline-frame', 'validation', weather, index))
                        shared_hash = prediction_hash(shared)
                        if shared_hash != cached['prediction_hash'] or shared_hash != base_cache['prediction_hash']:
                            raise RuntimeError('Shared tensor differs from source S0/baseline')
                        ids, scores, corners, masks, names, features = proposals(dataset, batch, shared, pool, context, runtime)
                        features = {t: torch.from_numpy(x) for t, x in features.items()}
                        if names != cached['source_names'] or ids.tolist() != cached['proposals']['anchor_ids']:
                            raise RuntimeError('Original source/proposal order differs')
                        for task in ('cls', 'reg'):
                            if not torch.equal(features[task], cached['features'][task]):
                                raise RuntimeError('Inference feature drift: ' + task)
                        # All source choices and proposal conflicts precede GT evaluation.
                        strategies, accepted = {}, {}
                        for g in GROUPS:
                            strategies[g] = {}
                            for policy, conservative in (('Greedy', False), ('CommonConservative', True)):
                                actions = predict_actions(ids, scores, masks, features, names,
                                    probes[g]['cls'], probes[g]['reg'], FEATURES, conservative)
                                strategies[g][policy] = actions
                                accepted[g + '-Proposal-Task-' + policy] = conflict_resolver(actions)
                        shared_post = dataset.post_process(batch, {'ego': shared})
                        association, _ = assign_proposals(shared_post[2].detach().cpu().numpy(), ids, scores,
                                                          corners, source.spec['minimum_gt_proposal_iou'])
                        anchors = {a['anchor_id'] for a in association if a is not None}
                        allowed = {position for position, anchor in enumerate(ids) if int(anchor) in anchors}
                        for g in GROUPS:
                            accepted[g + '-Assisted-Task-Greedy'] = conflict_resolver(strategies[g]['Greedy'], allowed_positions=allowed)
                        row = {'frame': index, 'weather': weather, 'shared_prediction_hash': shared_hash,
                               'stats': {'Shared': frame_ap_stats(shared_post)},
                               'counts': {'Shared': {'proposals': len(ids), 'changed_proposals': 0,
                                                    'recovered': 0, 'lost': 0, 'new_fp': 0}},
                               'proposals': cached['proposals'], 'source_names': names,
                               'strategies': {g: {policy: [action_record(a) for a in actions]
                                                 for policy, actions in policies.items()} for g, policies in strategies.items()},
                               'accepted_actions': {}}
                        for method, (actions, counts) in accepted.items():
                            post = dataset.post_process(batch, {'ego': execute(shared, pool, actions)})
                            row['stats'][method] = frame_ap_stats(post)
                            difference = compare_predictions(shared_post, post, source.spec['target_iou'], source.spec['fp_identity_iou'])
                            row['counts'][method] = {**counts, **difference,
                                'partial_overlap_rejections': counts['rejected_overlap_actions'] - counts['fully_rejected_overlap_actions']}
                            row['accepted_actions'][method] = [action_record(a, True) for a in actions]
                        atomic_torch(path, row)
                    manifest.mark('replay-frame', 'complete', 'validation', weather, index, [path])
                for method in METHODS:
                    merge(stats[method], row['stats'][method])
                    totals[method].update(row['counts'][method])
                if ordinal == 0 or (ordinal + 1) % 10 == 0:
                    print(f'REPLAY {weather} {ordinal+1}/{len(loader)} complete', flush=True)
            if seen != source.protocol['validation_indices']:
                raise RuntimeError('Incomplete replay frame sequence')
            aps = {m: ap_values(stats[m], eval_utils) for m in METHODS}
            baseline = read_json(source.root / 'baseline_reproduction.json')[weather]['Shared']
            shared_checks[weather] = check_ap(aps['Shared'], baseline, weather + '.Shared')
            results = {}
            for method in METHODS:
                c = totals[method]
                n = max(1, c['proposals'])
                results[method] = {**aps[method], 'counts': dict(c),
                    'delta_ap_pp_vs_Shared': {k: 100 * (aps[method][k] - aps['Shared'][k]) for k in aps[method]},
                    'KEEP_ratio': 1 - c['changed_proposals'] / n, 'cls_changed_ratio': c['cls_changed'] / n,
                    'reg_changed_ratio': c['reg_changed'] / n, 'task_separated_ratio': c['task_separated'] / n,
                    'task_separated_among_changed': c['task_separated'] / max(1, c['changed_proposals']), 'frames': len(seen)}
            conditions[weather] = results
            atomic_torch(summary_path, {'methods': results, 'shared_check': shared_checks[weather]})
            manifest.mark('replay-weather', 'complete', 'validation', weather, artifacts=[summary_path])
            del dataset, loader
    paired = {w: {policy: {key: 100 * (conditions[w][GROUPS[1] + '-' + policy][key]
                                                - conditions[w][GROUPS[0] + '-' + policy][key])
                           for key in ('ap30', 'ap50', 'ap70')} for policy in POLICIES} for w in WEATHERS}
    md = '# 两标签完整回放\n\n每帧只构建一次冻结 Shared/source pool，两组共享同一输入。AP 为原帧顺序口径，变化以百分点（pp）计。\n\n'
    md += '| Weather | Method | AP30 | AP50 | AP70 | ΔAP70 pp | recovered/lost/newFP | proposed/accepted | partial/full reject | changed | KEEP | cls/reg changed | cells | cls/reg不同 |\n|---|---|---:|---:|---:|---:|---|---|---|---:|---:|---|---:|---:|\n'
    for weather, methods in conditions.items():
        for method, r in methods.items():
            c = r['counts']
            md += (f"| {weather} | {method} | {r['ap30']:.6f} | {r['ap50']:.6f} | {r['ap70']:.6f} | {r['delta_ap_pp_vs_Shared']['ap70']:+.4f} | "
                   f"{c.get('recovered', 0)}/{c.get('lost', 0)}/{c.get('new_fp', 0)} | {c.get('proposed_actions', 0)}/{c.get('accepted_actions', 0)} | "
                   f"{c.get('partial_overlap_rejections', 0)}/{c.get('fully_rejected_overlap_actions', 0)} | {c.get('changed_proposals', 0)} | "
                   f"{r['KEEP_ratio']:.2%} | {r['cls_changed_ratio']:.2%}/{r['reg_changed_ratio']:.2%} | {c.get('modified_cells', 0)} | {r['task_separated_ratio']:.2%} |\n")
    md += '\n新标签减原标签的配对 AP 变化：\n\n| Weather | Policy | ΔAP30 pp | ΔAP50 pp | ΔAP70 pp |\n|---|---|---:|---:|---:|\n'
    for w, policies in paired.items():
        for policy, r in policies.items():
            md += f"| {w} | {policy} | {r['ap30']:+.4f} | {r['ap50']:+.4f} | {r['ap70']:+.4f} |\n"
    md += '\nGT 只用于 Assisted 位置集合和最终评价，Proposal 的来源选择及冲突处理先于 GT。保持原 ROI、margin 排序及部分重叠处理；逐帧策略、接受 cell 和后果在 cache/replay。无 bootstrap 或显著性推断。\n'
    report(run, 'REPLAY_RESULTS', {'conditions': conditions, 'paired_delta_ap_pp_new_minus_original': paired,
        'shared_reproduction': shared_checks, 'frame_order_ap': True, 'frozen_pool_for_both_groups': True}, md)


def compare_original_s3(run, required=True):
    p = protocol(run)
    source = control_source(run, p)
    source.refresh()
    if not source.complete('S3'):
        atomic_json(Path(run) / 'SOURCE_S3_CHECK.json', {'status': 'pending',
            'reason': 'Original S3 must complete before FINAL; rerun the same RUN with mode all'})
        if required:
            raise RuntimeError('Original S3 reference pending; final completion blocked. Resume this RUN after original S3 completes.')
        return False
    path = source.artifact('S3_RESULTS.json', 'S3')
    original = read_json(path)['conditions']
    actual = read_json(Path(run) / 'REPLAY_RESULTS.json')['conditions']
    checks = {}
    for w in WEATHERS:
        checks[w] = {}
        for policy, old_method in (('Assisted-Task-Greedy', 'Assisted-Task'),
                                   ('Proposal-Task-Greedy', 'Proposal-Task-Greedy')):
            checks[w][policy] = check_ap(actual[w]['OriginalLabel-' + policy], original[w][old_method], w + '.' + old_method)
    from .common import digest
    atomic_json(Path(run) / 'SOURCE_S3_CHECK.json', {'status': 'matched', 'absolute_ap_differences': checks,
                'reference_file': str(path), 'reference_sha256': digest(path), 'tolerance': 1e-6})
    return True


if __name__ == '__main__':
    parser = cli(__doc__)
    parser.add_argument('--mode', choices=('replay', 'reference'), default='replay')
    args = parser.parse_args()
    (replay if args.mode == 'replay' else compare_original_s3)(args.run)
