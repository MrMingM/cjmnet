"""S3 assisted and fully inference-driven joint action replay; no training."""
from collections import Counter
from pathlib import Path
import numpy as np
from .common import (KEEP, WEATHERS, Runtime, atomic_json, atomic_torch,
                     load_cache, parser, read_json, write_report)
from .features import columns
from .linear_ranker import load_probe, selection
from .s0_counterfactual import frame_ap_stats, prediction_hash, proposals

ASSISTED = ('Shared', 'Assisted-Cls', 'Assisted-Reg', 'Assisted-Task')
PROPOSAL = ('Shared', 'Proposal-Cls-Greedy', 'Proposal-Reg-Greedy',
            'Proposal-Task-Greedy', 'Proposal-Task-Conservative')


def predict_actions(ids, shared_scores, masks, features, names, cls_probe, reg_probe, variant, conservative=False):
    """Deployable decision boundary. GT/associations/outcomes are not arguments."""
    result = []
    for p, anchor in enumerate(ids):
        actions, margins = {}, {}
        for task, probe in (('cls', cls_probe), ('reg', reg_probe)):
            threshold = probe.state['calibration']['threshold'] if conservative else 0.
            scores = probe.scores(features[task][p][:, columns(task, variant)])
            index, margin, _ = selection(scores, names, threshold)
            actions[task], margins[task] = names[index], margin if index else 0.
        result.append({'position': p, 'anchor_id': int(anchor), 'shared_score': float(shared_scores[p]),
                       'mask': np.asarray(masks[p], dtype=bool),
                       'cls_source': actions['cls'], 'reg_source': actions['reg'],
                       'cls_margin': margins['cls'], 'reg_margin': margins['reg']})
    return result


def conflict_resolver(actions, task='task', allowed_positions=None):
    """First action owns each output cell; later actions use only unoccupied cells.

    One combined cls/reg action occupies cells for both heads. Overlap rejection
    counts include partial rejections; fully rejected counts are also reported.
    """
    rows = []
    total = Counter(proposals=len(actions))
    for action in actions:
        if allowed_positions is not None and action['position'] not in allowed_positions:
            continue
        row = dict(action)
        if task == 'cls':
            row['reg_source'], row['reg_margin'] = KEEP, 0.
        elif task == 'reg':
            row['cls_source'], row['cls_margin'] = KEEP, 0.
        elif task != 'task':
            raise ValueError('Unknown replay task')
        if row['cls_source'] == KEEP and row['reg_source'] == KEEP:
            continue
        row['margin'] = max(row['cls_margin'], row['reg_margin'])
        rows.append(row)
    rows.sort(key=lambda r: (-r['margin'], -r['shared_score'], r['anchor_id']))
    total['proposed_actions'] = len(rows)
    accepted = []
    occupied = np.zeros_like(actions[0]['mask'], bool) if actions else np.zeros((0, 0), bool)
    for row in rows:
        mask = row['mask']
        if mask.shape != occupied.shape:
            raise ValueError('ROI shape differs')
        available = mask & ~occupied
        if (mask & occupied).any():
            total['rejected_overlap_actions'] += 1
            total['rejected_overlap_cells'] += int((mask & occupied).sum())
        if not available.any():
            total['fully_rejected_overlap_actions'] += 1
            continue
        total['accepted_actions'] += 1
        total['changed_proposals'] += 1
        total['cls_changed'] += row['cls_source'] != KEEP
        total['reg_changed'] += row['reg_source'] != KEEP
        total['task_separated'] += row['cls_source'] != row['reg_source']
        total['modified_cells'] += int(available.sum())
        total['cls_modified_cells'] += int(available.sum()) if row['cls_source'] != KEEP else 0
        total['reg_modified_cells'] += int(available.sum()) if row['reg_source'] != KEEP else 0
        occupied |= available
        accepted.append({**row, 'mask': available})
    for key in ('proposed_actions', 'accepted_actions', 'rejected_overlap_actions', 'fully_rejected_overlap_actions',
                'modified_cells', 'changed_proposals', 'cls_changed', 'reg_changed', 'task_separated'):
        total.setdefault(key, 0)
    return accepted, dict(total)


def execute(shared, pool, actions):
    from local_fusion_task_source_oracle.oracle import compose_prediction
    # Accepted masks are disjoint. All source tensors remain the original frame pool.
    current = shared
    for row in actions:
        current = compose_prediction(current, pool, row['mask'], row['cls_source'], row['reg_source'])
    return current


def replay(run, mode):
    import torch
    from ceif_audit.scoring import ap_values, empty_stats, merge
    from opencood.tools.train_utils import to_device
    from opencood.utils import eval_utils
    from local_fusion_task_split_pilot.proposal_constrained_oracle import assign_proposals
    from local_fusion_utility_v2.outcomes import compare_predictions
    runtime = Runtime(run)
    cls_probe, reg_probe = load_probe(run, 'cls'), load_probe(run, 'reg')
    variant = runtime.spec['primary_features']
    methods = ASSISTED if mode == 'assisted' else PROPOSAL
    summaries = {}
    with torch.no_grad():
        for weather in WEATHERS:
            stage = 'S3-' + mode
            folder = runtime.run / 'cache' / stage / weather
            folder.mkdir(parents=True, exist_ok=True)
            if runtime.manifest.complete(stage + '-weather', 'validation', weather):
                summaries[weather] = load_cache(folder / 'summary.pt')
                continue
            dataset, loader = runtime.loader('validation', weather)
            stats = {m: empty_stats() for m in methods}
            totals = {m: Counter() for m in methods}
            with runtime.manifest.work(stage + '-weather', 'validation', weather):
                for ordinal, batch in enumerate(loader):
                    index = int(batch['ego']['communication_sample_index'][0])
                    if index != runtime.protocol['validation_indices'][ordinal]:
                        raise RuntimeError('Replay frame order differs from B0')
                    path = folder / f'{index:08d}.pt'
                    if runtime.manifest.complete(stage + '-frame', 'validation', weather, index):
                        row = load_cache(path)
                    else:
                        with runtime.manifest.work(stage + '-frame', 'validation', weather, index):
                            batch = to_device(batch, runtime.target)
                            context, shared, pool = runtime.predict(batch, weather, verify=ordinal == 0)
                            cached = load_cache(runtime.run / 'cache' / 'validation' / weather / f'{index:08d}.pt')
                            if prediction_hash(shared) != cached['prediction_hash']:
                                raise RuntimeError('Replay Shared state differs from S0/baseline')
                            ids, scores, corners, masks, names, features = proposals(dataset, batch, shared, pool, context, runtime)
                            features = {task: torch.from_numpy(value) for task, value in features.items()}
                            if names != cached['source_names'] or ids.tolist() != cached['proposals']['anchor_ids']:
                                raise RuntimeError('Replay proposal/source pool drift')
                            for task in ('cls', 'reg'):
                                if not torch.equal(features[task], cached['features'][task]):
                                    raise RuntimeError('Replay inference features drift')
                            greedy = predict_actions(ids, scores, masks, features, names, cls_probe, reg_probe, variant)
                            conservative = predict_actions(ids, scores, masks, features, names, cls_probe, reg_probe, variant, True)
                            # Proposal-driven choices/conflicts are fixed before any GT evaluation.
                            accepted_methods = {}
                            if mode == 'proposal':
                                for method, actions, task in (
                                    ('Proposal-Cls-Greedy', greedy, 'cls'), ('Proposal-Reg-Greedy', greedy, 'reg'),
                                    ('Proposal-Task-Greedy', greedy, 'task'),
                                    ('Proposal-Task-Conservative', conservative, 'task')):
                                    accepted_methods[method] = conflict_resolver(actions, task)
                            shared_post = dataset.post_process(batch, {'ego': shared})
                            if mode == 'assisted':
                                gt = shared_post[2].detach().cpu().numpy()
                                association, _ = assign_proposals(gt, ids, scores, corners,
                                                                  runtime.spec['minimum_gt_proposal_iou'])
                                allowed_anchors = {r['anchor_id'] for r in association if r is not None}
                                allowed_positions = {p for p, anchor in enumerate(ids) if int(anchor) in allowed_anchors}
                                # GT restricts positions only; never changes the learned actions/margins/order.
                                for method, task in (('Assisted-Cls', 'cls'), ('Assisted-Reg', 'reg'), ('Assisted-Task', 'task')):
                                    accepted_methods[method] = conflict_resolver(greedy, task, allowed_positions)
                            row = {'stats': {'Shared': frame_ap_stats(shared_post)},
                                   'counts': {'Shared': {'proposals': len(ids), 'changed_proposals': 0,
                                                        'recovered': 0, 'lost': 0, 'new_fp': 0}}, 'actions': {}}
                            for method, (accepted, counts) in accepted_methods.items():
                                post = dataset.post_process(batch, {'ego': execute(shared, pool, accepted)})
                                difference = compare_predictions(shared_post, post, runtime.spec['target_iou'],
                                                                 runtime.spec['fp_identity_iou'])
                                row['stats'][method] = frame_ap_stats(post)
                                row['counts'][method] = {**counts, **difference}
                                row['actions'][method] = [{k: action[k] for k in
                                    ('position', 'anchor_id', 'shared_score', 'cls_source', 'reg_source', 'margin')}
                                    | {'accepted_cells': int(action['mask'].sum())} for action in accepted]
                            atomic_torch(path, row)
                        runtime.manifest.mark(stage + '-frame', 'complete', 'validation', weather, index, [path])
                    for method in methods:
                        merge(stats[method], row['stats'][method])
                        totals[method].update(row['counts'][method])
                    if ordinal == 0 or (ordinal + 1) % 10 == 0:
                        print(f'{stage} {weather} {ordinal + 1}/{len(loader)}', flush=True)
                results = {m: ap_values(stats[m], eval_utils) for m in methods}
                baseline = read_json(runtime.run / 'baseline_reproduction.json')[weather]['Shared']
                if any(abs(results['Shared'][metric] - baseline[metric]) > 1e-6 for metric in ('ap30', 'ap50', 'ap70')):
                    raise RuntimeError('Replay Shared AP differs from complete B0 reproduction')
                summaries[weather] = {}
                for method in methods:
                    counts = totals[method]
                    n = max(1, counts['proposals'])
                    summaries[weather][method] = {
                        **results[method], 'delta_ap_pp_vs_Shared': {metric: 100 * (results[method][metric] - baseline[metric])
                                                                  for metric in ('ap30', 'ap50', 'ap70')},
                        'counts': dict(counts), 'KEEP_ratio': 1 - counts['changed_proposals'] / n,
                        'cls_changed_ratio': counts['cls_changed'] / n, 'reg_changed_ratio': counts['reg_changed'] / n,
                        'task_separated_ratio': counts['task_separated'] / n,
                        'task_separated_among_changed': counts['task_separated'] / max(1, counts['changed_proposals']),
                        'frames': len(loader)}
                atomic_torch(folder / 'summary.pt', summaries[weather])
            runtime.manifest.mark(stage + '-weather', 'complete', 'validation', weather,
                                  artifacts=[folder / 'summary.pt'])
            del dataset, loader
    atomic_json(runtime.run / ('s3_' + mode + '.json'), {'conditions': summaries})


def s3_verdict(conditions, spec):
    method = 'Proposal-Task-Conservative'
    gains = {w: conditions[w][method]['delta_ap_pp_vs_Shared']['ap70'] for w in WEATHERS}
    recovered = sum(conditions[w][method]['counts'].get('recovered', 0) for w in ('fog', 'rain', 'snow'))
    lost = sum(conditions[w][method]['counts'].get('lost', 0) for w in ('fog', 'rain', 'snow'))
    checks = {'adverse_mean_gain': np.mean([gains[w] for w in ('fog', 'rain', 'snow')]) >= spec['s3_weather_mean_ap70_gain_pp'],
              'positive_weathers': sum(gains[w] > 0 for w in ('fog', 'rain', 'snow')) >= spec['s3_min_positive_adverse_weathers'],
              'clean_preserved': gains['clean'] >= -spec['s3_clean_max_drop_pp'], 'recovered_gt_lost': recovered > lost}
    return {'verdict': 'S3_PASS' if all(checks.values()) else 'S3_FAIL', 'checks': {k: bool(v) for k, v in checks.items()},
            'ap70_gains_pp': gains, 'adverse_recovered': recovered, 'adverse_lost': lost,
            'count_gate_scope': 'Fog/Rain/Snow summed; clean constrained separately'}


def summarize(run):
    assisted = read_json(Path(run) / 's3_assisted.json')['conditions']
    proposal = read_json(Path(run) / 's3_proposal.json')['conditions']
    conditions = {w: {**assisted[w], **proposal[w]} for w in WEATHERS}
    for weather in WEATHERS:
        if assisted[weather]['Shared'] != proposal[weather]['Shared']:
            raise RuntimeError('Assisted and proposal Shared summaries differ')
    spec = read_json(Path(run) / 'protocol.json')['config']
    decision = s3_verdict(conditions, spec)
    md = f'# S3：来源选择是否兑现为检测收益\n\n判定：**{decision["verdict"]}**。Conservative 阈值已经在 train 校准固定。\n\n'
    md += '| Weather | Method | AP30 | AP50 | AP70 | ΔAP70 pp | recovered/lost/newFP | changed | KEEP | cls/reg changed | task-separated | accepted/overlap rejected | modified cells |\n|---|---|---:|---:|---:|---:|---|---:|---:|---|---:|---|---:|\n'
    for weather, methods in conditions.items():
        for method, row in methods.items():
            c = row['counts']
            md += (f"| {weather} | {method} | {row['ap30']:.6f} | {row['ap50']:.6f} | {row['ap70']:.6f} | "
                   f"{row['delta_ap_pp_vs_Shared']['ap70']:+.4f} | {c.get('recovered', 0)}/{c.get('lost', 0)}/{c.get('new_fp', 0)} | "
                   f"{c.get('changed_proposals', 0)} | {row['KEEP_ratio']:.2%} | {row['cls_changed_ratio']:.2%}/{row['reg_changed_ratio']:.2%} | "
                   f"{row['task_separated_ratio']:.2%} | {c.get('accepted_actions', 0)}/{c.get('rejected_overlap_actions', 0)} | {c.get('modified_cells', 0)} |\n")
    md += '\nAssisted 使用 GT 关联来限定 proposal；来源、KEEP/MODIFY、排序与冲突规则都由推理证据决定。Proposal 路径在 GT 评价前固定全部动作。重叠后仅接受剩余 cell，部分重叠也计为一次 rejected-overlap，JSON 另给完全拒绝数。\n'
    md += '\nLearned-Task-Greedy / Learned-Task-Conservative 分别对应 Proposal-Task-Greedy / Proposal-Task-Conservative。所有 AP 使用原后处理和帧顺序口径。\n'
    write_report(run, 'S3', {'conditions': conditions, 'decision': decision,
                            'aliases': {'Learned-Task-Greedy': 'Proposal-Task-Greedy',
                                        'Learned-Task-Conservative': 'Proposal-Task-Conservative'},
                            'global_sort': False}, md)


def main():
    cli = parser(__doc__)
    cli.add_argument('--mode', choices=('assisted', 'proposal', 'summarize'), required=True)
    args = cli.parse_args()
    if args.mode == 'summarize':
        summarize(args.run)
    else:
        replay(args.run, args.mode)


if __name__ == '__main__':
    main()
