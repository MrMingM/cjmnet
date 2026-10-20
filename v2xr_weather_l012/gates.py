"""Fixed gates use Decimal comparisons to make exact registered boundaries unambiguous."""
from decimal import Decimal
from auditlib import CONFIG


def delta(a, b):
    return Decimal(100) * (Decimal(str(a)) - Decimal(str(b)))


def l0(ap70):
    gain = delta(ap70, CONFIG['clean_only_O_AP70']['fog'])
    passed = gain >= Decimal(str(CONFIG['L0_fog_gain_pp']))
    return dict(status='L0_GATE_PASSED' if passed else 'L0_GATE_FAILED', passed=passed,
        fog_O_AP70=ap70, gain_pp=float(gain), minimum_gain_pp=CONFIG['L0_fog_gain_pp'],
        failure_class=None if passed else 'performance_gate_not_met; does not establish a pipeline error',
        continue_pipeline=passed)


def l1(c, d):
    gains = {weather: delta(d[weather], c[weather]) for weather in ('clean', 'fog', 'snow')}
    tests = dict(fog=gains['fog'] >= Decimal(str(CONFIG['L1_fog_gain_pp'])),
        nonnegative=sum(v >= 0 for v in gains.values()) >= CONFIG['L1_nonnegative_conditions'],
        clean=gains['clean'] >= Decimal(str(CONFIG['L1_clean_min_pp'])))
    passed = all(tests.values())
    return dict(status='L1_GATE_PASSED' if passed else 'L1_GATE_FAILED', passed=passed,
        gains_pp={k: float(v) for k, v in gains.items()}, tests=tests, continue_pipeline=True,
        technical_validity_required=True)
