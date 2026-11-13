"""Orchestrator commands invoked by the POSIX entry point."""
import os
import sys
import traceback

from shared.input_audit import OUT, atomic_json, contract, mapped


def main():
    stage=sys.argv[1]
    smoke='--smoke' in sys.argv[2:]
    if stage=='contract':
        contract()
        return
    if stage=='extract':
        from shared.extract import extract
        probe = int(sys.argv[sys.argv.index('--probe')+1]) if '--probe' in sys.argv else None
        extract(smoke, probe)
        return
    if stage=='summary':
        from reports.build_summary import summarize
        summarize()
        return
    device=mapped(stage)
    if stage=='V1':
        from v1.candidate_union import analyze
    elif stage=='V2':
        from v2.region_consistency import analyze
    elif stage=='V3':
        from v3.correlated_errors import analyze
    else:
        raise ValueError(stage)
    root=OUT/(stage+'_smoke' if smoke else stage)
    root.mkdir(exist_ok=True)
    print(stage,'SMOKE' if smoke else 'FULL',device,flush=True)
    analyze(smoke)
    atomic_json(root/'STATUS.json',dict(status='complete',device=device,smoke=smoke))


if __name__=='__main__':
    try:
        main()
    except Exception:
        label=sys.argv[1]+('_smoke' if '--smoke' in sys.argv else '')
        atomic_json(OUT/(label+'_FAILURE.json'),dict(status='failed',stage=label,
                                                   traceback=traceback.format_exc()))
        raise
