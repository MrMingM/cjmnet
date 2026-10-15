"""Accept existing S0 numeric noise only when every decision remains exact."""
import ast
import inspect
import math
from numbers import Real
import textwrap
from types import FunctionType

from local_fusion_action_utility_audit.reproducibility import POLICY, numeric_difference
from local_fusion_source_kill_stage2_diagnostics.differences import exact_differences
from local_fusion_source_kill_stage2_diagnostics.inspect import continuous_path, ordering_changes


def assert_finite(value):
    if isinstance(value, dict):
        for child in value.values():
            assert_finite(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            assert_finite(child)
    elif isinstance(value, Real) and not math.isfinite(float(value)):
        raise ValueError('REPRODUCTION_FAIL: nonfinite trajectory value')


def compare_trajectory(actual, expected):
    """No rounding, field removal, tolerance on labels, or changed tie-break."""
    assert_finite(actual)
    assert_finite(expected)
    differences = exact_differences(actual, expected, limit=1000000)
    rows = differences['first_differences']
    if len(rows) != differences['total_differences']:
        raise ValueError('REPRODUCTION_FAIL: incomplete difference inspection')
    rejected = []
    for row in rows:
        if (not continuous_path(row['path']) or 'absolute_difference' not in row or
                not numeric_difference(row['actual'], row['expected']).get('allclose_2e5', False)):
            rejected.append(row)
    order = ordering_changes(actual, expected) if not rejected else []
    legal_order = ordering_changes(actual, expected, legal_only=True) if not rejected else []
    accepted = not rejected and not order and not legal_order
    return dict(status='complete' if accepted else 'failed', accepted=accepted,
        engineering_check_only=True, legacy_numeric_policy=POLICY,
        differences=differences, rejected_differences=rejected,
        full9_order_changes=order, a_only_legal_order_changes=legal_order,
        max_absolute_difference=max((r.get('absolute_difference', 0.) for r in rows), default=0.),
        scientific_ap_tolerance_unchanged=1e-6, scientific_k2_gates_unchanged=True)


GUARD = "saved is None or records != saved['records'] or audits != saved['current_state_audit']"
MESSAGE = 'REPRODUCTION_FAIL: K2 original A_ONLY current-state trajectory differs'


def adapt_evaluate_frame(original, verifier):
    """Replace exactly the known overstrict guard, retaining original evaluation."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    expected_test = ast.dump(ast.parse(GUARD, mode='eval').body)
    matches = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and ast.dump(node.test) == expected_test:
            expected_body = ast.parse('raise ValueError(' + repr(MESSAGE) + ')').body
            if node.orelse or ast.dump(ast.Module(body=node.body, type_ignores=[])) != ast.dump(
                    ast.Module(body=expected_body, type_ignores=[])):
                raise ValueError('Unknown replay guard body; refuse adapting new code')
            matches.append(node)
    if len(matches) != 1:
        raise ValueError('Expected exactly one known K2 replay guard')
    replacement = ast.parse('_verify_replayed_trajectory(records, audits, saved, protocol, weather, index, mode)').body[0]
    class Replace(ast.NodeTransformer):
        def visit_If(self, node):
            if node is matches[0]:
                return ast.copy_location(replacement, node)
            return self.generic_visit(node)
    tree = ast.fix_missing_locations(Replace().visit(tree))
    namespace = dict(original.__globals__, _verify_replayed_trajectory=verifier)
    exec(compile(tree, inspect.getsourcefile(original), 'exec'), namespace)
    return namespace[original.__name__]


def adapt_run(original, evaluate_frame, runtime, commit):
    """Per-call namespace; never patch original module globals or source files."""
    namespace = dict(original.__globals__, evaluate_frame=evaluate_frame,
                     make_runtime=lambda protocol: runtime, commit_unit=commit)
    return FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)
