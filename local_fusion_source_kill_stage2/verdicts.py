"""Section 42 thresholds only; no outcome-dependent rule changes."""
import math
from collections import Counter
from .common import settings


def k1_family(source_gain, null_gain, ci, valid=True):
    if not valid or not all(math.isfinite(x) for x in (source_gain, null_gain)) or source_gain <= 0:
        return dict(verdict='INCONCLUSIVE', reasons=['Incomplete/unfair input or nonpositive Source gain'])
    if ci is None or not all(math.isfinite(ci[x]) for x in ('lower', 'upper')):
        return dict(verdict='INCONCLUSIVE', reasons=['Paired scene bootstrap unavailable'])
    rule = settings()['k1']
    reasons = []
    if null_gain / source_gain >= rule['null_ratio_fail']:
        reasons.append('Null_gain / Source_gain >= 0.60')
    if source_gain - null_gain < rule['gap_pp_fail_below']:
        reasons.append('Source_gain - Null_gain < 0.50 pp')
    if ci['lower'] <= 0 <= ci['upper']:
        reasons.append('95% paired scene-bootstrap CI includes 0')
    return dict(verdict='FAIL' if reasons else 'PASS', reasons=reasons,
                source_gain_pp=source_gain, null_gain_pp=null_gain,
                difference_pp=source_gain-null_gain, ratio=null_gain/source_gain)


def combine(verdicts):
    if not verdicts or 'INCONCLUSIVE' in verdicts:
        return 'INCONCLUSIVE'
    return 'FAIL' if 'FAIL' in verdicts else 'PASS'


def sample_gate(rows, layer, verify_expected=True):
    rule = settings()['k2'][layer]
    counts = Counter(r['scene_id'] for r in rows)
    n, scenes = len(rows), len(counts)
    maximum = max(counts.values(), default=0)
    if verify_expected and (n, scenes, maximum) != (
            rule['expected_n'], rule['expected_scenes'], rule['expected_max_scene_count']):
        raise ValueError(f'P0 provenance mismatch {layer}: n/scenes/max={n}/{scenes}/{maximum}')
    share = maximum/n if n else 0.
    valid = n >= rule['minimum_n'] and scenes >= rule['minimum_scenes'] and share <= rule['maximum_scene_share']
    return dict(verdict='PASS' if valid else 'INCONCLUSIVE', n=n, scenes=scenes,
                max_scene_share=share, per_scene=dict(counts))


def k2_family(layer, metrics, ci):
    rule = settings()['k2'][layer]
    required = (('keep_agreement', 'keep_ci_lower'), ('source_retention', 'source_ci_lower')) if layer == 'same' else (
        ('positive_retention', 'positive_ci_lower'),)
    reasons = []
    keys = ('keep_agreement', 'source_retention', 'membership_agreement') if layer == 'same' else (
        'positive_retention', 'pair_retention')
    if any(metrics.get(k) is None or not math.isfinite(metrics[k]) for k in keys):
        return dict(verdict='INCONCLUSIVE', reasons=['Undefined metric'])
    for key in keys:
        if metrics[key] < rule[key]:
            reasons.append(f'{key} {metrics[key]:.6f} < {rule[key]}')
    for key, bound in required:
        if key not in ci:
            return dict(verdict='INCONCLUSIVE', reasons=['Scene bootstrap missing'])
        if ci[key]['lower'] < rule[bound]:
            reasons.append(f'{key} scene-bootstrap lower {ci[key]["lower"]:.6f} < {rule[bound]}')
    return dict(verdict='FAIL' if reasons else 'PASS', reasons=reasons)


def final_verdict(k1same, k1task, k2same, k2task):
    states = (k1same, k1task, k2same, k2task)
    if any(v not in ('PASS', 'FAIL', 'INCONCLUSIVE') for v in states):
        raise ValueError('Invalid verdict')
    h1 = k1same == k2same == 'PASS'
    h2 = h1 and k1task == k2task == 'PASS'
    return dict(K1_SAME=k1same, K1_TASK=k1task, K2_SAME=k2same, K2_TASK=k2task,
        H1_SOURCE_SPECIFIC='PASS' if h1 else 'STOP',
        H2_TASK_SPECIFIC='PASS' if h2 else 'STOP' if h1 else 'NOT_APPLICABLE',
        decision='SOURCE UTILITY MAINLINE STOP' if not h1 else 'TASK SEPARATION STOP' if not h2 else
        'Corresponding layers eligible for future K3 data gate review; no K3 execution', k3_frozen=True)
