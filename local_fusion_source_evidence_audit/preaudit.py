"""Check every frozen frame before LOO/evidence; report all reproduction failures."""
from pathlib import Path
from .common import KEEP, WEATHERS, atomic_json, base_frame, contract, parser, read_json, safe_path, digest
from .extract import reference_pool, check_frame, frozen_outputs
from . import coordinates


def recheck_targets(previous, protocol):
    """All previous train failures plus one validation failure per weather.

    This engineering check cannot replace the mandatory full audit. Validation
    representatives catch the common accumulator bug before another full pass.
    """
    if previous.get('complete') is not True or previous.get('accepted') is not False or not previous.get('failures'):
        raise ValueError('Recheck requires a complete failed reproduction audit')
    targets = {}
    for row in previous['failures']:
        split, weather, index = row['split'], row['weather'], row['frame']
        if (split not in ('train', 'validation') or weather not in WEATHERS
                or index not in protocol[split+'_indices']):
            raise ValueError('Recheck frame is outside the original S0 split/weather')
        key = split+'/'+weather
        if split == 'train' or key not in targets:
            targets.setdefault(key, set()).add(index)
    return targets


def recheck_specification(path, base, base_protocol):
    """Bind an optional engineering check to its report and original base."""
    path = safe_path(path)
    previous_protocol = read_json(path.parent/'protocol.json')
    if previous_protocol['base'] != base:
        raise ValueError('Recheck report belongs to a different BASE_RUN snapshot')
    targets = recheck_targets(read_json(path), base_protocol)
    return {'path': str(path), 'sha256': digest(path), 'targets': {k: sorted(v) for k, v in targets.items()}}


def audit_frames(run, runtime, proxy_builder, to_device, input_hash, forward=None,
                 targets=None, prior=None):
    """Dependency-injected loop also exercised with pure synthetic arrays locally.

    A failed reproduction is recorded and the scan continues. Unexpected
    decoder/loader/input errors remain fatal; no invalid frame enters evidence.
    Full audits consume every entry; subset rechecks consume the unchanged
    loader prefix through the last selected frame for the weather/RNG order.
    """
    report_path = Path(run)/('REPRODUCTION_AUDIT.json' if targets is None else 'REPRODUCTION_RECHECK.json')
    summary = {'schema': 1, 'complete': False, 'accepted': False,
               'checked_frames': 0, 'failures': [], 'groups': {},
               'scope': 'frozen hooks, coordinates, candidates and output reproduction; no LOO or evidence statistics',
               'train_limit': 'original full source tensors unavailable; unproven count/geometry drift is rejected'}
    if targets is not None:
        summary.update(full_audit=False, prior_audit=prior,
                       targets={k: sorted(v) for k, v in targets.items()},
                       scope='failed-frame engineering recheck only; full reproduction-audit remains mandatory')
    atomic_json(report_path, summary)
    for split in ('train', 'validation'):
        for weather in WEATHERS:
            selected = None if targets is None else targets.get(split+'/'+weather)
            if targets is not None and not selected:
                continue
            dataset, loader = runtime.loader(split, weather)
            expected = runtime.protocol[split+'_indices']
            allowed = set(expected)
            if len(allowed) != len(expected):
                raise RuntimeError(f'Duplicate {split} frame indices in S0 protocol')
            seen, visited, checked, group_failures = [], set(), 0, 0
            for ordinal, batch in enumerate(loader):
                index = int(batch['ego']['communication_sample_index'][0])
                if index not in allowed or index in visited:
                    raise RuntimeError(f'Reproduction audit {split}/{weather} unexpected or duplicate '
                                       f'frame={index} position={ordinal}')
                # S0 train uses the original seeded RandomSampler. Its protocol
                # lists pool members, not their shuffled traversal order.
                # Validation is sequential and retains the strict order check.
                if split == 'validation' and index != expected[ordinal]:
                    raise RuntimeError(f'Reproduction audit validation/{weather} loader order differs '
                                       f'from S0: position={ordinal} expected={expected[ordinal]} actual={index}')
                seen.append(index)
                visited.add(index)
                # Consume the inherited loader up to each selected frame,
                # retaining the weather worker RNG sequence and frame order.
                if selected is not None and index not in selected:
                    continue
                cached = base_frame(run, split, weather, index)
                batch = to_device(batch, runtime.target)
                before = input_hash(batch, weather)
                predict = runtime.predict if forward is None else forward
                context, _, generated = predict(batch, weather, verify=ordinal == 0)
                pool, differences, failures = reference_pool(runtime, batch, weather, index, split, generated)
                failure = None
                if failures:
                    path = Path(run)/'diagnostics'/f'raw_reproduction_{split}_{weather}_{index:08d}.json'
                    atomic_json(path, {'split': split, 'weather': weather, 'frame': index,
                                      'checks': differences, 'reasons': failures})
                    failure = {'split': split, 'weather': weather, 'frame': index,
                               'reasons': failures, 'diagnostics': str(path)}
                else:
                    proxies = {name: proxy_builder(dataset, batch, pred) for name, pred in pool.items()}
                    _, failure = check_frame(run, split, weather, index, batch, proxies,
                                             context, runtime, cached, differences)
                    del proxies
                if before != input_hash(batch, weather):
                    raise RuntimeError('Reproduction audit mutated inference inputs')
                summary['checked_frames'] += 1
                checked += 1
                if failure:
                    summary['failures'].append(failure)
                    group_failures += 1
                    print(f'reproduction-audit FAIL {split}/{weather} frame={index}: {failure["reasons"]}', flush=True)
                    atomic_json(report_path, summary)
                if selected is not None or ordinal == 0 or (ordinal+1) % 10 == 0 or ordinal+1 == len(loader):
                    print(f'reproduction-audit {split}/{weather} {ordinal+1}/{len(loader)} '
                          f'failures={group_failures}', flush=True)
                del context, generated, pool, batch, cached
                if selected is not None and selected <= visited:
                    break
            required = allowed if selected is None else selected
            if not required <= visited:
                raise RuntimeError(f'Incomplete reproduction audit {split}/{weather}: '
                                   f'missing={sorted(required-visited)}')
            summary['groups'][split+'/'+weather] = {'frames': len(seen), 'checked': checked, 'failures': group_failures}
            atomic_json(report_path, summary)
            del dataset, loader
    summary['complete'] = True
    summary['accepted'] = not summary['failures']
    atomic_json(report_path, summary)
    if not summary['accepted']:
        raise ValueError(f'Reproduction {"audit" if targets is None else "recheck"} found {len(summary["failures"])} failed frames; '
                         f'evidence not started; all diagnostics listed in {report_path}')
    return summary


def audit(run, recheck_audit=None):
    import torch
    from local_fusion_action_utility_audit.common import Runtime
    from local_fusion_action_utility_audit.reproducibility import inference_input_hash
    from local_fusion_task_source_oracle.oracle import _proxy_cache
    from opencood.tools.train_utils import to_device
    protocol = contract(run)
    targets, prior = None, None
    if recheck_audit is not None:
        prior = recheck_specification(recheck_audit, protocol['base'], protocol['base_protocol'])
        if prior != protocol.get('repair_recheck'):
            raise ValueError('Recheck report/targets differ from new RUN protocol')
        targets = {k: set(v) for k, v in prior['targets'].items()}
    runtime = Runtime(protocol['base']['base_run'])
    def forward(batch, weather, verify):
        context, pool, processed, _, _ = frozen_outputs(runtime, batch, weather, verify)
        coordinates.validate(runtime, batch, processed, context['levels'])
        return context, pool[KEEP], pool
    with torch.no_grad():
        audit_frames(run, runtime, _proxy_cache, to_device, inference_input_hash, forward, targets, prior)


if __name__ == '__main__':
    cli = parser(__doc__)
    cli.add_argument('--recheck-audit', help='Previous failed REPRODUCTION_AUDIT.json; subset check only')
    args = cli.parse_args()
    audit(args.run, args.recheck_audit)
