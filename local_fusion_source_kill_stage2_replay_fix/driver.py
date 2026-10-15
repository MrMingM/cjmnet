"""Remote K2 repair: original AP gate, original perturbations, new provenance."""
import argparse
from pathlib import Path

from local_fusion_source_kill_stage2.common import (
    WEATHERS, atomic_json, commit_unit, external_run, load_contract, now,
    object_hash, read_json, resource_check, resume, safe_path, worker)
from .contract import initialize
from .reproduction import adapt_evaluate_frame, adapt_run, compare_trajectory


def source_gate(run, protocol, runtime):
    """Unchanged original Source FULL/AP gate, without calibration or nulls."""
    from local_fusion_source_kill_stage2.k1_source_specificity import evaluate
    with worker(run, 'source_reproduction', protocol) as folder:
        if resume(folder/'result.json', protocol['identity']) is not None:
            return
        config = {'config_hash': object_hash(dict(purpose='Source-only repair AP gate', root=protocol['identity']))}
        conditions = {}
        for weather in WEATHERS:
            for mode in ('same', 'task'):
                name = f'{weather}/Source/0/{mode}'
                conditions[name] = evaluate(folder, protocol, runtime, weather, 'Source', 0, mode, config, {})
        path = folder/'SOURCE_REPRODUCTION_GATE.json'
        atomic_json(path, dict(status='complete', root_identity=protocol['identity'], tolerance=1e-6,
            original_evaluate_function=True, nulls_run=False,
            conditions={name:dict(AP=r['AP'], Shared=r['Shared']) for name,r in conditions.items()}))
        artifacts = [path] + [p for p in (folder/'conditions').rglob('*') if p.suffix in ('.json', '.jsonl')]
        commit_unit(folder/'result.json', protocol['identity'], dict(status='complete'), artifacts)


def execute_k2(run, protocol, runtime):
    from local_fusion_source_kill_stage2 import k2_label_stability as original
    def verify(records, audits, saved, passed_protocol, weather, index, mode):
        if passed_protocol['identity'] != protocol['identity'] or saved is None:
            raise ValueError('REPRODUCTION_FAIL: missing saved trajectory or protocol mismatch')
        actual = dict(records=records, current_state_audit=audits)
        expected = dict(records=saved['records'], current_state_audit=saved['current_state_audit'])
        evidence = compare_trajectory(actual, expected)
        evidence.update(root_identity=protocol['identity'], weather=weather, mode=mode, frame=index,
            checked_at=now(), actual_trajectory=actual, historical_trajectory=expected)
        path = run/'k2'/'conditions'/weather/mode/f'{index:08d}.reproduction.json'
        atomic_json(path, evidence)
        if not evidence['accepted']:
            raise ValueError('REPRODUCTION_FAIL: decision/order/legacy numerical policy mismatch; see '+str(path))
    def commit(path, identity, payload, artifacts=(), metadata=None):
        path = Path(path)
        if path.name != 'result.json':
            evidence = path.with_name(path.stem+'.reproduction.json')
            record = read_json(evidence)
            if not record['accepted'] or record['root_identity'] != identity:
                raise ValueError('Frame missing accepted hash-protected replay evidence')
            artifacts = tuple(artifacts)+(evidence,)
        return commit_unit(path, identity, payload, artifacts, metadata)
    evaluate = adapt_evaluate_frame(original.evaluate_frame, verify)
    adapted_run = adapt_run(original.run, evaluate, runtime, commit)
    adapted_run(run)


def execute(run, parent_run, diagnosis):
    run = external_run(run)
    import fcntl
    with (run/'.repair-driver.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = initialize(run, parent_run, diagnosis)
        state = dict(status='running', root_identity=protocol['identity'], started_at=now())
        atomic_json(run/'repair_manifest.json', state)
        try:
            resource_check(protocol)
            from local_fusion_source_kill_stage1.common import guard_tree
            for root in protocol['dataset_roots'].values():
                guard_tree(root)
            if resume(run/'k2'/'result.json', protocol['identity']) is None:
                from local_fusion_source_kill_stage2.source_adapter import make_runtime
                runtime = make_runtime(protocol)
                source_gate(run, protocol, runtime)
                execute_k2(run, protocol, runtime)
            if resume(run/'source_reproduction'/'result.json', protocol['identity']) is None:
                raise ValueError('Repair K2 requires unchanged Source AP gate')
            load_contract(run)
            state.update(status='complete', completed_at=now())
        except BaseException as exc:
            from local_fusion_source_kill_stage2.reports import inconclusive_report
            folder = run/'k2'
            folder.mkdir(parents=True, exist_ok=True)
            inconclusive_report(folder, 'K2', protocol, exc)
            atomic_json(folder/'manifest.json', dict(status='failed', identity=protocol['identity'],
                error=repr(exc), completed_at=now()))
            state.update(status='failed', error=repr(exc), completed_at=now())
            raise
        finally:
            atomic_json(run/'repair_manifest.json', state)
        from .finalize import summarize
        summarize(run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--parent-run')
    parser.add_argument('--diagnosis')
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    if args.summarize_only:
        from .finalize import summarize
        summarize(safe_path(args.run))
    else:
        if not args.parent_run or not args.diagnosis:
            parser.error('--parent-run and --diagnosis are required for the worker')
        execute(args.run, args.parent_run, args.diagnosis)


if __name__ == '__main__':
    main()
