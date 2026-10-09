"""Frozen original inference pools, old conflict resolution/NMS/AP, new scores only."""
from collections import Counter
from pathlib import Path
import numpy as np
from .common import (KEEP, WEATHERS, Manifest, atomic_torch, base_frame, contract,
                     evidence_path, load_cache, report)
from .features import combine
from .ranker import load_probe, selection

METHODS = ('Shared', 'Evidence-Assisted-Cls', 'Evidence-Assisted-Reg', 'Evidence-Assisted-Task',
           'Evidence-Proposal-Cls-Greedy', 'Evidence-Proposal-Reg-Greedy',
           'Evidence-Proposal-Task-Greedy', 'Evidence-Proposal-Task-Conservative')


def predict_actions(ids, scores, masks, features, names, cls_probe, reg_probe, conservative=False):
    """Inference decision: GT, labels, associations, outcomes are not arguments."""
    rows = []
    for p, anchor in enumerate(ids):
        sources, margins = {}, {}
        for task, probe in (('cls', cls_probe), ('reg', reg_probe)):
            threshold = probe.state['calibration']['threshold'] if conservative else 0.
            selected, margin, _ = selection(probe.scores(features[task][p]), names, threshold)
            sources[task], margins[task] = names[selected], margin if selected else 0.
        rows.append({'position': p, 'anchor_id': int(anchor), 'shared_score': float(scores[p]),
                     'mask': np.asarray(masks[p], bool), 'cls_source': sources['cls'], 'reg_source': sources['reg'],
                     'cls_margin': margins['cls'], 'reg_margin': margins['reg']})
    return rows


def run_replay(run):
    import torch
    from ceif_audit.scoring import ap_values, empty_stats, merge
    from local_fusion_action_utility_audit.common import Runtime
    from local_fusion_action_utility_audit.reproducibility import load_reference, assert_ap_reproduction, assert_frame_reproduction
    from local_fusion_action_utility_audit.s0_counterfactual import frame_ap_stats, prediction_hash, proposals
    from local_fusion_action_utility_audit.s3_replay import conflict_resolver, execute, s3_verdict
    from local_fusion_utility_v2.outcomes import compare_predictions
    from opencood.tools.train_utils import to_device
    from opencood.utils import eval_utils
    protocol = contract(run)
    runtime = Runtime(protocol['base']['base_run'])
    manifest = Manifest(run)
    cls_probe, reg_probe = load_probe(run, 'cls'), load_probe(run, 'reg')
    conditions = {}
    with torch.no_grad():
        for weather in WEATHERS:
            folder = Path(run)/'cache'/'replay'/weather
            summary_path = folder/'summary.pt'
            if manifest.complete('replay-weather', 'validation', weather):
                conditions[weather] = load_cache(summary_path)
                continue
            dataset, loader = runtime.loader('validation', weather)
            stats = {m: empty_stats() for m in METHODS}
            totals = {m: Counter() for m in METHODS}
            seen = []
            for ordinal, batch in enumerate(loader):
                index = int(batch['ego']['communication_sample_index'][0])
                seen.append(index)
                if index != runtime.protocol['validation_indices'][ordinal]:
                    raise RuntimeError('Replay frame order drift')
                path = folder/f'{index:08d}.pt'
                if manifest.complete('replay-frame', 'validation', weather, index):
                    row = load_cache(path)
                else:
                    with manifest.work('replay-frame', 'validation', weather, index):
                        old = base_frame(run, 'validation', weather, index)
                        if not manifest.complete('evidence-frame', 'validation', weather, index):
                            raise ValueError('Missing validation evidence')
                        evidence = load_cache(evidence_path(run, 'validation', weather, index))
                        if evidence['identity'] != protocol['identity']:
                            raise ValueError('Replay evidence identity differs')
                        from local_fusion_action_utility_audit.reproducibility import tensor_tree_hash
                        if evidence['original_labels_hash'] != tensor_tree_hash(old['labels']):
                            raise ValueError('Replay original S0 label hash differs')
                        batch = to_device(batch, runtime.target)
                        pool, reference = load_reference(runtime, batch, weather, index)
                        shared = pool[KEEP]
                        if evidence['input_hash'] != reference['input_hash'] or prediction_hash(shared) != old['prediction_hash']:
                            raise ValueError('Replay inference input/Shared differs from S0')
                        ids, scores, corners, masks, names, output = proposals(dataset, batch, shared, pool, {'levels': []}, runtime)
                        if ids.tolist() != old['proposals']['anchor_ids'] or names != old['source_names'] or not np.array_equal(masks, evidence['masks'].numpy()):
                            raise ValueError('Replay proposal/source/ROI drift')
                        for task in ('cls', 'reg'):
                            if not np.array_equal(output[task], old['features'][task].numpy()):
                                raise ValueError('Replay output-only feature drift')
                        matrices = {task: torch.from_numpy(combine(old['features'][task].numpy(), {k: v.numpy() for k, v in evidence['blocks'].items()}, 'all_evidence')) for task in ('cls', 'reg')}
                        greedy = predict_actions(ids, scores, masks, matrices, names, cls_probe, reg_probe)
                        conservative = predict_actions(ids, scores, masks, matrices, names, cls_probe, reg_probe, True)
                        accepted = {}
                        # Fix deployable decisions and conflicts before consulting labels/GT.
                        for suffix, actions, task in (('Cls-Greedy', greedy, 'cls'), ('Reg-Greedy', greedy, 'reg'),
                                                      ('Task-Greedy', greedy, 'task'), ('Task-Conservative', conservative, 'task')):
                            accepted['Evidence-Proposal-'+suffix] = conflict_resolver(actions, task)
                        # Saved S0 association only restricts Assisted positions, as in S3.
                        # It never changes any predicted source, margin or priority.
                        allowed = {label['proposal_position'] for label in old['labels'] if not label['background']}
                        for suffix, task in (('Cls', 'cls'), ('Reg', 'reg'), ('Task', 'task')):
                            accepted['Evidence-Assisted-'+suffix] = conflict_resolver(greedy, task, allowed)
                        shared_post = dataset.post_process(batch, {'ego': shared})
                        assert_frame_reproduction(reference['evaluation_only']['stats'], frame_ap_stats(shared_post))
                        row = {'stats': {'Shared': frame_ap_stats(shared_post)},
                               'counts': {'Shared': {'proposals': len(ids), 'changed_proposals': 0}}, 'actions': {}}
                        for method, (actions, counts) in accepted.items():
                            post = dataset.post_process(batch, {'ego': execute(shared, pool, actions)})
                            difference = compare_predictions(shared_post, post, runtime.spec['target_iou'], runtime.spec['fp_identity_iou'])
                            row['stats'][method] = frame_ap_stats(post)
                            row['counts'][method] = {**counts, **difference}
                            row['actions'][method] = [{k: action[k] for k in ('position', 'anchor_id', 'shared_score', 'cls_source', 'reg_source', 'margin')}
                                                       | {'accepted_cells': int(action['mask'].sum())} for action in actions]
                        atomic_torch(path, row)
                    manifest.mark('replay-frame', 'complete', 'validation', weather, index, [path])
                for method in METHODS:
                    merge(stats[method], row['stats'][method])
                    totals[method].update(row['counts'][method])
                if ordinal == 0 or (ordinal+1) % 10 == 0:
                    print(f'S4 replay {weather} {ordinal+1}/{len(loader)}', flush=True)
            if seen != runtime.protocol['validation_indices']:
                raise ValueError('Incomplete replay pass')
            ap = {m: ap_values(stats[m], eval_utils) for m in METHODS}
            saved = runtime.protocol['b0_saved_results']['conditions'][weather]['results']['Shared']
            assert_ap_reproduction(saved, ap['Shared'])
            conditions[weather] = {}
            for method in METHODS:
                c = totals[method]
                n = max(1, c['proposals'])
                conditions[weather][method] = {**ap[method], 'delta_ap_pp_vs_Shared': {k: 100*(ap[method][k]-ap['Shared'][k]) for k in ('ap30', 'ap50', 'ap70')},
                                               'counts': dict(c), 'action_rate': c['changed_proposals']/n,
                                               'KEEP_ratio': 1-c['changed_proposals']/n,
                                               'cls_changed_ratio': c['cls_changed']/n, 'reg_changed_ratio': c['reg_changed']/n,
                                               'task_separated_ratio': c['task_separated']/n,
                                               'task_separated_among_changed': c['task_separated']/max(1, c['changed_proposals']),
                                               'frames': len(seen)}
            atomic_torch(summary_path, conditions[weather])
            manifest.mark('replay-weather', 'complete', 'validation', weather, artifacts=[summary_path])
            del dataset, loader
    aliases = {w: {'Proposal-Task-Conservative': conditions[w]['Evidence-Proposal-Task-Conservative']} for w in WEATHERS}
    decision = s3_verdict(aliases, runtime.spec)
    value = {'primary_group': 'all_evidence', 'conditions': conditions, 'decision': decision,
             'inference_gt_fields': 0, 'test_data_used': False, 'global_sort': False,
             'assisted_gt_use': 'original S0 association restricts proposal positions only',
             'pool': 'exact original S0 validation FP32 inference pools; no new detector forward'}
    md = '# S4 整帧回放\n\n主门槛使用 Evidence-Proposal-Task-Conservative：**'+decision['verdict']+'**。\n\n'
    md += '| Weather | Method | AP30 | AP50 | AP70 | ΔAP70 pp | recovered/lost/newFP | changes | KEEP | cls/reg changes | accepted/overlap rejected | modified cells |\n|---|---|---:|---:|---:|---:|---|---:|---:|---|---|---:|\n'
    for weather, methods in conditions.items():
        for method, row in methods.items():
            c = row['counts']
            md += f"| {weather} | {method} | {row['ap30']:.6f} | {row['ap50']:.6f} | {row['ap70']:.6f} | {row['delta_ap_pp_vs_Shared']['ap70']:+.4f} | {c.get('recovered', 0)}/{c.get('lost', 0)}/{c.get('new_fp', 0)} | {c.get('changed_proposals', 0)} | {row['KEEP_ratio']:.2%} | {c.get('cls_changed', 0)}/{c.get('reg_changed', 0)} | {c.get('accepted_actions', 0)}/{c.get('rejected_overlap_actions', 0)} | {c.get('modified_cells', 0)} |\n"
    md += '\n沿用 S3 冲突规则、原始 source pool、原 NMS/AP；Proposal 动作在 GT 评价前固定。Assisted 只使用旧 GT 关联限定候选，来源和动作仍由 evidence ranker 决定。Conservative 阈值在 train calibration 固定；没有可行阈值时明确 KEEP_ALL。\n'
    report(run, 'S4_REPLAY_RESULTS', value, md)


if __name__ == '__main__':
    from .common import parser
    run_replay(parser(__doc__).parse_args().run)
