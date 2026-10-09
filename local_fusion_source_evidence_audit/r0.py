"""Replay both old feature variants, all metrics and strata, without new inference."""
from .common import WEATHERS, contract, read_json, report


def compare(expected, actual, path='', tolerance=1e-6):
    if isinstance(expected, dict):
        if set(expected) != set(actual):
            raise ValueError('R0 metric keys differ: '+path)
        return max([compare(v, actual[k], path+'/'+k, tolerance) for k, v in expected.items()] or [0.])
    if expected is None:
        if actual is not None:
            raise ValueError('R0 NA mismatch: '+path)
        return 0.
    if isinstance(expected, (int, float)):
        import math
        delta = abs(expected-actual)
        if not math.isfinite(delta) or delta > tolerance:
            raise ValueError(f'R0 failed {path}: expected={expected}, actual={actual}')
        return delta
    if expected != actual:
        raise ValueError('R0 mismatch: '+path)
    return 0.


def reproduce(run):
    from local_fusion_action_utility_audit.common import VARIANTS
    from local_fusion_action_utility_audit.features import columns
    from local_fusion_action_utility_audit.linear_ranker import groups, load_probe, baseline_scores, Metrics
    protocol = contract(run)
    base = protocol['base']['base_run']
    result = {'pass': True, 'metric_atol': 1e-6, 'inference_rerun': False,
              'comparison': 'same cached S0 outcomes and saved S1/S2 probes; all metrics/counts/strata', 'tasks': {}}
    for task, stage in (('cls', 'S1'), ('reg', 'S2')):
        saved = read_json(__import__('pathlib').Path(base)/(stage+'_RESULTS.json'))
        variants = {}
        for variant in VARIANTS:
            probe = load_probe(base, task, variant)
            conditions, strata_out = {}, {}
            for weather in WEATHERS:
                acc, strata = {}, {'background': {}, 'associated': {}}
                for group in groups(base, 'validation', weather, task, 'with_competition_features'):
                    scores = baseline_scores(group, task)
                    scores['linear'] = probe.scores(group['x'][:, columns(task, variant)])
                    scores['linear_conservative'] = scores['linear']
                    for method, score in scores.items():
                        threshold = probe.state['calibration']['threshold'] if method == 'linear_conservative' else 0.
                        acc.setdefault(method, Metrics()).add(group, score, threshold)
                        strata['background' if group['background'] else 'associated'].setdefault(method, Metrics()).add(group, score, threshold)
                conditions[weather] = {k: v.result() for k, v in acc.items()}
                strata_out[weather] = {k: {m: v.result() for m, v in values.items()} for k, values in strata.items()}
            differences = {'conditions': compare(saved['variants'][variant]['conditions'], conditions),
                           'strata': compare(saved['variants'][variant]['strata'], strata_out)}
            variants[variant] = {'max_absolute_difference': differences, 'conditions': conditions, 'strata': strata_out}
        result['tasks'][task] = variants
    report(run, 'R0_REPRODUCTION', result,
           '# R0 旧结果复现\n\nPASS：相同缓存标签和已保存线性模型，两种旧特征配置的全部指标、计数及分层与 S1/S2 的差值均不超过 1e-6。\n\n原 FP32 推理容差只用于后续重新提取中间证据；R0 没有重新运行 detector，也没有放宽指标容差。\n')


if __name__ == '__main__':
    from .common import parser
    reproduce(parser(__doc__).parse_args().run)
