"""Tie-aware label summaries, mild ROI perturbations and family aggregation."""
import math
import numpy as np
from .common import KEEP, settings
from .bootstrap import metric_bootstrap
from .verdicts import k2_family


def source_id(name):
    if name == KEEP:
        return 'KEEP'
    family, index = name.split(':')
    if family not in ('single', 'query') or not index.isdigit():
        raise ValueError('Unexpected original source action: ' + name)
    return int(index)


def action_id(row, scale=1.):
    expansion = row['expansion']
    return (None if expansion is None else round(float(expansion)/scale, 10), row['cls_source'], row['reg_source'])


def label_summary(evaluation, scale=1.):
    actions = evaluation['actions']
    keep = tuple(evaluation['keep_key'])
    # Baseline P0 A_ONLY, with same current-state eligibility also applied to
    # perturbed labels. B/E are still evaluated and membership is reported.
    legal = [row for row in actions if row['group'] in ('KEEP', 'A')]
    ordered = sorted(legal, key=lambda row: tuple(row['key']), reverse=True)
    best = ordered[0]
    discrete = tuple(best['key'][:6])
    equivalent = [r for r in legal if tuple(r['key'][:6]) == discrete]
    pairs = {(source_id(r['cls_source']), source_id(r['reg_source'])) for r in equivalent}
    same = max((tuple(r['key'][:6]) for r in legal if r['cls_source'] == r['reg_source']), default=keep[:6])
    sources = {source_id(r['cls_source']) for r in equivalent if r['cls_source'] == r['reg_source'] != KEEP}
    return dict(winner=action_id(best, scale), winner_pair=(source_id(best['cls_source']), source_id(best['reg_source'])),
        winner_group=best['group'], modified=best['cls_source'] != KEEP or best['reg_source'] != KEEP,
        discrete_positive=discrete > keep[:6], task_positive=discrete > same,
        sources=sources, pairs=pairs, top2={action_id(r, scale) for r in ordered[:2]},
        membership={action_id(r, scale): r['group'] for r in actions if r['group'] != 'KEEP'},
        winner_key=list(best['key']), same_discrete6=list(same))


def compare_labels(base, changed):
    common = set(base['membership']) | set(changed['membership'])
    membership = sum(base['membership'].get(x) == changed['membership'].get(x) for x in common)/len(common) if common else 1.
    intersection = base['top2'] & changed['top2']
    union = base['top2'] | changed['top2']
    a, b = base['winner'], changed['winner']
    return dict(exact_action_stable=a == b, exact_source_stable=base['winner_pair'] == changed['winner_pair'],
        keep_agreement=base['modified'] == changed['modified'],
        source_retention=bool(base['sources'] & changed['sources']),
        pair_retention=bool(base['pairs'] & changed['pairs']),
        winner_in_baseline_source_set=changed['winner_pair'][0] in base['sources'],
        winner_pair_in_baseline_set=changed['winner_pair'] in base['pairs'],
        positive_retention=changed['task_positive'], task_positive_lost=not changed['task_positive'],
        task_to_same_flip=changed['winner'][1]==changed['winner'][2], task_to_keep_flip=not changed['modified'],
        cls_source_flip=base['winner_pair'][0] != changed['winner_pair'][0],
        reg_source_flip=base['winner_pair'][1] != changed['winner_pair'][1],
        source_family_change=tuple(n.split(':')[0] for n in a[1:]) != tuple(n.split(':')[0] for n in b[1:]),
        source_identity_change=base['winner_pair'] != changed['winner_pair'],
        top2_equal=base['top2'] == changed['top2'], top2_jaccard=len(intersection)/len(union),
        membership_agreement=membership,
        selected_membership_agreement=base['winner_group'] == changed['winner_group'])


def perturb_corners(corners, perturbation):
    family, dx, dy, yaw_degrees, scale = perturbation
    output = np.asarray(corners, dtype=np.float32).copy()
    center = output[:, :2].mean(0)
    angle = math.radians(yaw_degrees)
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]], dtype=np.float32)
    output[:, :2] = (output[:, :2]-center) @ rotation.T + center + np.array([dx, dy], dtype=np.float32)
    return output, scale


def serialize_label(value):
    result = dict(value)
    for key in ('sources', 'pairs', 'top2'):
        result[key] = sorted(value[key], key=str)
    result['membership'] = [dict(action=list(k), group=v) for k, v in value['membership'].items()]
    return result


def aggregate(rows, layer, repetitions=5000):
    variants = settings()['k2']['perturbations']
    result, bootstrap = {}, {}
    primary = ('keep_agreement', 'source_retention', 'membership_agreement') if layer == 'same' else (
        'positive_retention', 'pair_retention')
    all_metrics = ('exact_action_stable', 'exact_source_stable', 'keep_agreement', 'source_retention',
        'pair_retention', 'positive_retention', 'membership_agreement', 'top2_equal', 'top2_jaccard',
        'source_family_change', 'source_identity_change', 'cls_source_flip', 'reg_source_flip',
        'winner_in_baseline_source_set', 'winner_pair_in_baseline_set', 'selected_membership_agreement')
    all_metrics += ('task_positive_lost','task_to_same_flip','task_to_keep_flip')
    for family in ('center', 'yaw', 'roi'):
        names = [k for k, v in variants.items() if v[0] == family]
        per_variant = {}
        for name in names:
            selected = [r for r in rows if r['variant'] == name]
            if not selected:
                raise ValueError('Missing perturbation variant: ' + name)
            metrics = {}
            for key in all_metrics:
                applicable = [r for r in selected if layer != 'task' or key != 'pair_retention' or r['positive_retention']]
                metrics[key] = sum(r[key] for r in applicable)/len(applicable) if applicable else 0.
            stable_field = 'source_retention' if layer == 'same' else 'positive_retention'
            scene = {}
            for row in selected:
                counts = scene.setdefault(row['scene_id'], dict(n=0, stable=0, flip=0))
                counts['n'] += 1
                counts['stable'] += int(row[stable_field])
                counts['flip'] += int(not row[stable_field])
            per_variant[name] = dict(metrics=metrics, n=len(selected),
                stable_count=sum(r[stable_field] for r in selected), flip_count=sum(not r[stable_field] for r in selected),
                per_scene=scene)
        means = {key: float(np.mean([v['metrics'][key] for v in per_variant.values()])) for key in all_metrics}
        worst = {key: min(v['metrics'][key] for v in per_variant.values()) for key in primary}
        ci_keys = ('keep_agreement', 'source_retention') if layer == 'same' else ('positive_retention',)
        ci = {key: metric_bootstrap(rows, key, names, repetitions=repetitions,
                                   seed=settings()['bootstrap']['seed']) for key in ci_keys}
        bootstrap[family] = ci
        result[family] = dict(metrics=means, worst_variant=worst, per_variant=per_variant,
            bootstrap={k: {a: b for a, b in v.items() if a != 'distribution'} for k, v in ci.items()},
            **k2_family(layer, means, ci))
    return result, bootstrap
