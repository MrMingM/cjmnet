"""Original K2 evaluation with a narrow proof for forbidden-E list swaps."""
import argparse
from pathlib import Path
from types import FunctionType

from local_fusion_source_kill_stage2.common import (
    atomic_json, atomic_text, external_run, load_contract, now, resource_check, resume, safe_path)
from local_fusion_source_kill_stage2_replay_fix.driver import execute_k2 as previous_execute_k2
from .contract import initialize, reference_source_gate
from .reproduction import compare_trajectory


def execute_k2(run, protocol, runtime):
    original = previous_execute_k2
    scope = dict(original.__globals__,compare_trajectory=compare_trajectory)
    adapted = FunctionType(original.__code__,scope,original.__name__,original.__defaults__,original.__closure__)
    return adapted(run,protocol,runtime)


def summarize(run):
    from local_fusion_source_kill_stage2_replay_fix.finalize import summarize as original
    result = original(run)
    if result is None:
        return None
    protocol = load_contract(run)
    result['engineering_replay_v2'] = protocol['k2_replay_repair_v2']
    atomic_json(Path(run)/'STAGE2_FINAL.json',result)
    path = Path(run)/'STAGE2_FINAL.md'
    text = path.read_text(encoding='utf-8').replace(
        '离散字段、动作和 full9 排序必须完全一致',
        '离散后果、合法动作和最终选择严格核验；全部禁用且 outcome8 等于 KEEP 的 E 候选允许保留数量的排列变化')
    text += ('\n## E 候选排列工程修复\n\n'
        '仅对全部为 E、全部 allowed=false、各自 outcome8 等于 KEEP、且最终均选 KEEP 的目标，'
        '允许相同 source-pair 多重集合改变排列。推理动作、ROI、标签及 §42 科学门槛不改。'
        'Source AP 引用前一修复 RUN 的八项完整复现凭证，全部原始哈希与历史 AP 重新核验；未重新前向。'
        '逐帧原始差异和获准例外的证明保存在 reproduction.json。\n')
    atomic_text(path,text)
    return result


def execute(run, previous):
    run = external_run(run)
    import fcntl
    with (run/'.repair-driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        protocol, conditions = initialize(run,previous)
        state = dict(status='running',root_identity=protocol['identity'],started_at=now())
        atomic_json(run/'repair_manifest.json',state)
        try:
            resource_check(protocol)
            from local_fusion_source_kill_stage1.common import guard_tree
            for root in protocol['dataset_roots'].values():
                guard_tree(root)
            reference_source_gate(run,protocol,conditions)
            if resume(run/'k2/result.json',protocol['identity']) is None:
                from local_fusion_source_kill_stage2.source_adapter import make_runtime
                runtime = make_runtime(protocol)
                execute_k2(run,protocol,runtime)
            load_contract(run)
            state.update(status='complete',completed_at=now())
        except BaseException as exc:
            from local_fusion_source_kill_stage2.reports import inconclusive_report
            folder = run/'k2'
            folder.mkdir(parents=True,exist_ok=True)
            inconclusive_report(folder,'K2',protocol,exc)
            atomic_json(folder/'manifest.json',dict(status='failed',identity=protocol['identity'],
                error=repr(exc),completed_at=now()))
            state.update(status='failed',error=repr(exc),completed_at=now())
            raise
        finally:
            atomic_json(run/'repair_manifest.json',state)
        summarize(run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    parser.add_argument('--previous-run')
    parser.add_argument('--summarize-only',action='store_true')
    args = parser.parse_args()
    if args.summarize_only:
        summarize(safe_path(args.run))
    elif args.previous_run:
        execute(args.run,args.previous_run)
    else:
        parser.error('--previous-run is required for the GPU worker')


if __name__ == '__main__':
    main()
