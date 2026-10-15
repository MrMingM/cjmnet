"""Implementation audit only. Server scientific gates remain runtime checks."""
import argparse
import ast
import json
from .common import PACKAGE, ROOT, settings
from .preflight import static_check


def audit(shell=True):
    static_check(shell=shell)
    config = settings()
    text = {p.name:p.read_text(encoding='utf-8') for p in PACKAGE.glob('*.py')}
    original = (ROOT/'local_fusion_task_source_oracle/oracle.py').read_text(encoding='utf-8')
    checks = {
        'K1 Source reproduction': "for weather in WEATHERS:" in text['k1_source_specificity.py'] and
            text['k1_source_specificity.py'].index("family == 'Source':\n        reproduction") <
            text['k1_source_specificity.py'].index("for family in FAMILIES:"),
        'N1 whole decoded output vector': 'values[rng.integers' in text['nulls.py'] and
            'delta_vector' in text['nulls.py'] and 'encode(dataset, batch, shared' in text['nulls.py'],
        'N2 valid aligned CAVs': "len(levels[0])" in text['nulls.py'] and 'dirichlet_weights(count, 2*count' in text['nulls.py'],
        'score-only exact geometry': "prediction['rm'] = shared['rm']" in text['nulls.py'],
        'three seeds': config['seeds']==[20261014,20261015,20261016],
        'same original proxy': 'b.build_proxy_pool(dataset, batch, pool)' in text['k1_source_specificity.py'],
        'same original shortlist': 'b._shortlist_actions(*values, **kwargs)' in text['source_adapter.py'],
        'Task union not truncated to six': 'overall + separated + same' in original and 'rows[:6]' not in text['source_adapter.py'],
        'same candidate count': 'assert_fair(source, pool)' in text['k1_source_specificity.py'],
        'scene bootstrap 5000': config['bootstrap']['repetitions']>=5000 and 'scene_draws' in text['bootstrap.py'],
        'separate Same/Task verdict': 'H1_SOURCE_SPECIFIC' in text['verdicts.py'] and 'H2_TASK_SPECIFIC' in text['verdicts.py'],
        'K2 sample gates': config['k2']['same']['minimum_n']==200 and config['k2']['task']['minimum_n']==50 and
            config['k2']['same']['minimum_scenes']==7 and config['k2']['task']['minimum_scenes']==5,
        'eight perturbations': len(config['k2']['perturbations'])==8,
        'equivalent-best set intersection': "base['sources'] & changed['sources']" in text['stability.py'] and
            "base['pairs'] & changed['pairs']" in text['stability.py'],
        'current-state evaluation': 'evaluate_target' in text['k2_label_stability.py'] and
            "run_with_hooks(b, 'A_ONLY', args, focal)" in text['k2_label_stability.py'],
        'K2 frozen thresholds': [config['k2']['same'][k] for k in ('keep_agreement','source_retention','membership_agreement','keep_ci_lower','source_ci_lower')]==[.9,.7,.8,.8,.6] and
            [config['k2']['task'][k] for k in ('positive_retention','pair_retention','positive_ci_lower')]==[.7,.6,.5],
        'formal test realpath rejection': "resolved = raw.resolve()" in text['common.py'] and 'guard_tree(root)' in text['preflight.py'],
        'resource stop not reset': config['resource_stop_at']=='2026-10-27T03:23:56.508866+00:00' and
            "clock != one['project_clock']" in text['preflight.py'],
        'K3 frozen': config['k3_frozen'] is True,
        'resume immutable hashes': 'Source hash drift' in text['common.py'] and 'Resume/hash mismatch' in text['common.py'],
        'separate worker manifests': "with worker(run, 'k1'" in text['k1_source_specificity.py'] and
            "with worker(run, 'k2'" in text['k2_label_stability.py'],
    }
    for name, source in text.items():
        if name.startswith(('test_','fixtures','protocol_audit')):
            continue
        tree=ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr in ('backward','step'):
                checks['K3 frozen']=False
    return dict(status='STAGE2 IMPLEMENTATION READY' if all(checks.values()) else 'NOT READY',
                checks=checks, scope='source/static audit; no claim of server reproduction or scientific PASS')


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--no-shell',action='store_true')
    args=cli.parse_args()
    result=audit(shell=not args.no_shell)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if result['status']=='NOT READY':
        raise SystemExit(1)


if __name__=='__main__':
    main()
