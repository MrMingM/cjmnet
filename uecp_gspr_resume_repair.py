"""Executable, additive recovery repair for the existing UECP output only.

Sealed project sources and original contracts remain byte-identical. This runner
records its own ZIP identity and the human-approved comparison amendment, then
executes the original model/data/loss/training code with exact loading observers.
"""
import argparse
import ast
import inspect
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import textwrap
import traceback

from uecp_gspr_resume_diagnose import compare, diagnostic_sources, read, sha, stopped_pipeline
from uecp_gspr_resume_probe import (ARMS, STATE_FIELDS, environment, exact_live_state,
    numpy_tree, run_locals, verify_archive)
from uecp_gspr_resume_compare_v2 import POLICY, compare_states


MARKER = 'source_repair/uecp_resume_v2.json'
MID_GATE = 'smoke/resume_v2/E0_mid_epoch_complete.json'
EVIDENCE_ARCHIVE = 'c271c3aaec4fcd3b006cf72759eac16379d731c08ac649ea33a9db12bd8fd970'
NORMAL_ARMS = ARMS[:4]


def write_new(path, value):
    from v2xr_raygen_e0e3.common import atomic_bytes, plain
    path = Path(path)
    if path.exists():
        raise FileExistsError('Immutable repair artifact already exists: '+str(path))
    atomic_bytes(path, (json.dumps(plain(value), ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode())


def worker_hash_seed(command, action, seed):
    return '20261016' if command == 'worker' and action in ('dev', 'final') else str(seed)


def compile_replay_run(source, namespace):
    tree = ast.parse(textwrap.dedent(source))
    hits = [n for n in ast.walk(tree) if isinstance(n, ast.Constant) and n.value == 'smoke_replay']
    if len(hits) != 1:
        raise ValueError('Original replay directory expression differs')
    hits[0].value = 'smoke_replay_v2'
    exec(compile(ast.fix_missing_locations(tree), '<uecp-replay-directory-v2>', 'exec'), namespace)
    return namespace['run']


def compile_report_run(source, namespace):
    tree = ast.parse(textwrap.dedent(source))
    matches = [n for n in ast.walk(tree) if isinstance(n, ast.If) and isinstance(n.test, ast.Name)
               and n.test.id == 'smoke' and len(n.body) == 2 and all(isinstance(x, ast.AugAssign) for x in n.body)]
    if len(matches) != 1:
        raise ValueError('Original report gate construction differs')
    matches[0].body = ast.parse("required += ['"+MID_GATE+"']\nrequired += ['smoke/resume_v2/'+v+'_epoch2_complete.json' for v in ('E0','E2','E3_joint')]").body
    exec(compile(ast.fix_missing_locations(tree), '<uecp-report-gates-v2>', 'exec'), namespace)
    return namespace['run']


def compile_worker_command(source, namespace, archive, project):
    tree = ast.parse(textwrap.dedent(source))
    commands = [n for n in ast.walk(tree) if isinstance(n, ast.Assign) and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'cmd' and isinstance(n.value, ast.List)]
    if len(commands) != 1 or ast.unparse(commands[0].value.elts[2]) != "str(HERE / 'worker.py')":
        raise ValueError('Original worker command construction differs')
    commands[0].value.elts[2] = ast.Constant(str(archive))
    commands[0].value.elts.insert(3, ast.Constant('worker'))
    commands[0].value.elts.extend([ast.Constant('--project'), ast.Constant(str(project))])
    exec(compile(ast.fix_missing_locations(tree), '<uecp-worker-command-v2>', 'exec'), namespace)
    return namespace['run']


def gate(root, variant, epoch):
    return root / (MID_GATE if epoch == 1 else 'smoke/resume_v2/'+variant+'_epoch2_complete.json')


def file_record(path):
    from v2xr_raygen_e0e3.common import stamp
    return dict(sha256=sha(path), stamp=stamp(path))


def lineage(root, project, archive):
    from v2xr_raygen_e0e3.common import identity, stamp
    marker = read(root / MARKER)
    if marker['identity'] != identity({k: v for k, v in marker.items() if k != 'identity'}):
        raise ValueError('Repair identity changed')
    if (marker['archive_sha256'] != sha(archive) or marker['policy'] != POLICY
            or marker['output_root'] != str(root)
            or marker['request_sha256'] != sha(root / 'request.json')
            or marker['contract_sha256'] != sha(root / 'contract.json')):
        raise ValueError('Repair archive/policy/original contract changed')
    reviewed = diagnostic_sources(project, read(root / 'request.json')['sources'])
    if reviewed != marker['source_review']:
        raise ValueError('Reviewed source transition changed')
    for name, row in marker['evidence_files'].items():
        if stamp(name) != row['stamp']:
            raise ValueError('Enrolled diagnostic evidence changed: '+name)
    for name, digest in marker['original_failure_files'].items():
        if sha(name) != digest:
            raise ValueError('Preserved original failure record changed: '+name)
    return marker


def enroll(root, project, archive, evidence):
    from v2xr_raygen_e0e3.common import identity
    from uecp_gspr_v2xr.common import complete
    import torch
    torch.set_num_threads(1)
    if not (root / MARKER).exists():
        old = read(root / 'pipeline_status.json').get('smoke_E0_mid_epoch_resume_check', {})
        if old.get('status') != 'FAILED' or any((root / name).exists() for name in
                ('E0/baseline/epochs', 'E2/gspr_assisted_uecp/epochs', 'E3/dual_quality_fusion/joint/epochs')):
            raise ValueError('First enrollment requires the known failed smoke stage before formal training')
    from uecp_gspr_v2xr.runtime import bootstrap, input_stamps
    input_stamps(root)
    bootstrap(root, gpu=False)
    report = read(evidence)
    probe = evidence.parent.resolve(strict=True)
    if probe.parent != root / 'diagnostics' or not probe.name.startswith('controlled_resume_'):
        raise ValueError('Controlled evidence is outside the original diagnostic namespace')
    descriptor = read(probe / 'descriptor.json')
    if (descriptor['archive_sha256'] != EVIDENCE_ARCHIVE
            or descriptor['original_request_sha256'] != sha(root / 'request.json')
            or descriptor['original_contract_sha256'] != sha(root / 'contract.json')
            or report['status'] != 'DIAGNOSIS_FINISHED' or not report['backend_flags_equal']
            or not report['original_artifact_hashes_unchanged'] or report['formal_recovery_gate_passed']
            or report['strict_worker']['status'] != 'STRICT_OPERATION_BLOCKED'
            or 'grid_sampler_2d_backward_cuda' not in report['strict_worker'].get('error', '')):
        raise ValueError('Evidence does not match the inspected original controlled probe')
    reviewed = diagnostic_sources(project, read(root / 'request.json')['sources'])
    shared = probe / 'shared_boundary.pth'
    shared_sha = sha(shared)
    boundary = torch.load(shared, map_location='cpu', weights_only=False)
    if (shared_sha != report['common_boundary']['shared_sha256'] or not report['common_boundary']['exact']
            or boundary['position'] != 8 or boundary['global_step'] != 1
            or boundary['epoch'] != 1 or boundary['variant'] != 'E0'
            or boundary['contract_identity'] != read(root / 'contract.json')['identity']):
        raise ValueError('Common optimizer boundary differs')
    protected = {str(evidence): file_record(evidence), str(probe / 'descriptor.json'): file_record(probe / 'descriptor.json'),
                 str(shared): file_record(shared)}
    states = {}
    for arm in NORMAL_ARMS:
        ep = probe / 'arms' / arm / 'smoke/E0/baseline/epochs/01'
        record = complete(ep / 'train_complete.json')
        if record['variant'] != 'E0' or record['epoch'] != 1 or not record['smoke_only']:
            raise ValueError('Controlled checkpoint is not completed E0 smoke')
        for name in ('train_complete.json', *record['output_hashes']):
            path = ep / name
            protected[str(path)] = file_record(path)
        proof_path = probe / 'evidence' / arm / ('boundary.json' if arm == 'continuous' else 'restore.json')
        proof = read(proof_path)
        if (not proof['exact'] or proof['shared_sha256'] != shared_sha
                or proof['backend'] != report['common_boundary']['backend']):
            raise ValueError('Controlled exact-loading proof differs')
        protected[str(proof_path)] = file_record(proof_path)
        states[arm] = numpy_tree(torch.load(ep / 'resume.pth', map_location='cpu', weights_only=False), torch)
    comparisons = []
    for i, a in enumerate(NORMAL_ARMS):
        for b in NORMAL_ARMS[i+1:]:
            result = compare_states(states[a], states[b], True, True)
            if result['status'] != 'COMPLETE':
                raise ValueError('Approved comparison fails for '+a+'/'+b+': '+repr(result['failures'][:5]))
            comparisons.append(dict(left=a, right=b, comparison=result))
    backup = root / 'source_repair/uecp_resume_v2_original'
    if not (root / MARKER).exists():
        backup.mkdir(parents=True, exist_ok=True)
        for name in ('pipeline_status.json', 'failure.json', 'controller_failure.json'):
            src, dest = root / name, backup / name
            if src.exists():
                if dest.exists() and sha(src) != sha(dest):
                    raise ValueError('Original failure backup differs; preserved')
                if not dest.exists():
                    shutil.copyfile(src, dest)
    originals = {str(p): sha(p) for p in backup.glob('*.json')}
    marker = dict(repair='uecp_resume_v2', output_root=str(root), archive_sha256=sha(archive), policy=POLICY,
        authorization='User selected: 保留原网络，修复恢复检查（推荐）',
        request_sha256=sha(root / 'request.json'), contract_sha256=sha(root / 'contract.json'),
        source_review=reviewed, evidence=str(evidence), common_checkpoint_sha256=shared_sha,
        original_failure_files=originals,
        enrollment_environment=dict(python=sys.version, torch=str(torch.__version__), hip=torch.version.hip),
        evidence_files=protected,
        scope='Exact loading + common-checkpoint replay; model tolerance unchanged; Adam moments norm amendment',
        original_failure_preserved=True, original_sources_contracts_unchanged=True)
    marker['identity'] = identity(marker)
    if (root / MARKER).exists():
        if read(root / MARKER) != marker:
            raise ValueError('A different repair marker already exists; preserved')
    else:
        write_new(root / MARKER, marker)
    mid = gate(root, 'E0', 1)
    receipt = mid.parent / 'E0_mid_epoch_evidence.json'
    receipt_record = dict(source_repair_identity=marker['identity'], common_checkpoint_sha256=shared_sha,
                          comparisons=comparisons, original_gate='FAILED record preserved')
    if receipt.exists():
        if read(receipt) != receipt_record:
            raise ValueError('Mid-epoch repair evidence changed')
    else:
        write_new(receipt, receipt_record)
    from uecp_gspr_v2xr.common import commit
    commit(mid, [receipt], source_repair_identity=marker['identity'], policy=POLICY,
           same_common_checkpoint=True, exact_load_verified=True, original_model_tolerance_retained=True,
           variant='E0', epoch=1, scope='Controlled mid-epoch continuation; original failed fresh-start check preserved')
    return marker


def bind_commit(root, marker):
    from uecp_gspr_v2xr import common
    original = common.commit
    def amended(path, outputs, **fields):
        outputs = list(outputs)
        p = Path(path)
        if p.name == 'train_complete.json':
            proofs = sorted((p.parent / 'restore_checks_v2').glob('*.json'))
            outputs += [q for q in proofs if q not in outputs]
            fields['exact_loading_checks'] = [str(q.relative_to(p.parent)) for q in proofs]
        fields['execution_source_repair'] = dict(identity=marker['identity'], marker=str(root / MARKER),
                                                marker_sha256=sha(root / MARKER), archive_sha256=marker['archive_sha256'])
        return original(p, outputs, **fields)
    common.commit = amended


def patch_training(root, marker):
    import torch
    from uecp_gspr_v2xr import train
    # Change the independent replay directory only; original train arithmetic stays intact.
    train.run = compile_replay_run(inspect.getsource(train.run), dict(train.run.__globals__))
    original_restore = train.restore_rng
    def exact_restore(rng):
        original_restore(rng)
        local = run_locals(train.run)
        if local['restored']['epoch'] == local['epoch']:
            pointer = read(local['branch'] / 'recovery/latest.json')
            source = local['branch'] / 'recovery' / ('slot'+str(pointer['slot'])+'.pth')
            expected_fields = STATE_FIELDS
            actual = exact_live_state(local, train)
            reset = None
        else:
            source = local['branch'] / 'epochs' / f'{local["epoch"]-1:02d}' / 'resume.pth'
            expected_fields = ('model', 'optimizer', 'scheduler', 'criterion', 'rng', 'global_step', 'contract_identity', 'variant')
            actual = dict(model=local['model'].state_dict(), optimizer=local['opt'].state_dict(),
                scheduler=local['scheduler'].state_dict(), criterion=local['crit'].state_dict(), rng=train.rng_state(),
                global_step=local['global_step'], contract_identity=local['contract'], variant=local['variant'])
            reset = dict(position=local['position'], actual=local['actual'], gradients=local['gradients'],
                distributions=local['distributions'], elapsed=local['elapsed'], totals=local['totals'])
            fresh = dict(position=0, actual=[], gradients=[], distributions=[], elapsed=0.,
                         totals=dict(frames=0, updates=0, microsteps=0, loss=0., fused=0., single=0., auxiliary_each=0.))
            if reset != fresh:
                raise ValueError('New epoch counters or sampler progress did not reset')
        saved = torch.load(source, map_location='cpu', weights_only=False)
        result = compare(numpy_tree({k: saved[k] for k in expected_fields}, torch), numpy_tree(actual, torch), 'load', 0., 0.)
        if not result['exact']:
            raise ValueError('Checkpoint loading differs: '+repr(result['structural_or_exact_errors']+result['changed_floating_arrays'][:3]))
        proof = dict(status='COMPLETE', exact=True, source_checkpoint=str(source), source_sha256=sha(source),
            source_repair_identity=marker['identity'], current_epoch=local['epoch'], saved_epoch=saved['epoch'],
            floating_arrays=result['floating_arrays'], new_epoch_reset=reset)
        path = local['ep'] / 'restore_checks_v2' / (proof['source_sha256']+'.json')
        if path.exists():
            if read(path) != proof:
                raise ValueError('Earlier exact-load proof differs')
        else:
            write_new(path, proof)
    # The compiled train function has a copied global namespace. Bind the observer there too.
    train.restore_rng = exact_restore
    train.run.__globals__['restore_rng'] = exact_restore


def replay_check(root, variant, epoch, marker):
    from uecp_gspr_v2xr.common import complete, commit, variant_path
    if epoch == 1:
        if variant != 'E0':
            raise ValueError('Only E0 has the controlled mid-epoch gate')
        return complete(gate(root, variant, epoch))
    import torch
    torch.set_num_threads(1)
    left = variant_path(root, variant, True) / 'epochs/02'
    right = root / 'smoke_replay_v2' / variant / 'epochs/02'
    commits = [complete(p / 'train_complete.json') for p in (left, right)]
    source = variant_path(root, variant, True) / 'epochs/01/resume.pth'
    if sha(source) != sha(right.parent / '01/resume.pth'):
        raise ValueError('Epoch2 replay did not use the exact original epoch1 checkpoint')
    proofs = commits[1].get('exact_loading_checks', [])
    if not proofs or not any(read(right / p)['source_sha256'] == sha(source) and read(right / p)['exact'] for p in proofs):
        raise ValueError('Replay has no exact loading proof for the common checkpoint')
    states = [numpy_tree(torch.load(p / 'resume.pth', map_location='cpu', weights_only=False), torch) for p in (left, right)]
    result = compare_states(*states, checkpoint_verified=True, exact_load_verified=True)
    if result['status'] != 'COMPLETE':
        raise ValueError('Common-checkpoint replay exceeds the amended policy: '+repr(result['failures'][:8]))
    target = gate(root, variant, epoch)
    evidence = target.parent / (variant+'_epoch2_evidence.json')
    record = dict(comparison=result, source_sha256=sha(source),
                  train_commit_sha256=[sha(p / 'train_complete.json') for p in (left, right)])
    if evidence.exists():
        if read(evidence) != record:
            raise ValueError('Replay gate evidence differs')
    else:
        write_new(evidence, record)
    return commit(target, [evidence], source_repair_identity=marker['identity'], policy=POLICY,
                  exact_load_verified=True, same_common_checkpoint=True, variant=variant, epoch=epoch)


def amended_report(root, marker, smoke=False, partial=False):
    from uecp_gspr_v2xr import report
    target = root / ('smoke' if smoke else '')
    prefix = 'PARTIAL' if partial else 'FINAL'
    output = target / (prefix+'_RESULTS.json')
    previous = read(output) if output.exists() else None
    amended = compile_report_run(inspect.getsource(report.run), dict(report.run.__globals__))
    result = amended(root, smoke, partial)
    if not partial and previous and previous.get('status') == 'COMPLETE':
        if result['status'] != 'COMPLETE':
            raise ValueError('Previously completed report now has missing or invalid evidence; preserved')
        if previous.get('execution_source_repair', {}).get('identity') != marker['identity']:
            raise ValueError('Existing completed report belongs to another source policy; preserved')
        return previous
    result['execution_source_repair'] = dict(marker=str(root / MARKER), identity=marker['identity'],
        policy=POLICY, original_failed_check_preserved=True, actual_archive_sha256=marker['archive_sha256'])
    report.write(output, result)
    md = target / (prefix+'_RESULTS.md')
    with md.open('a', encoding='utf-8') as stream:
        stream.write('\n\n## 恢复检查修订\n\n原失败记录保留。加载状态采用零容差核验，后续重放从同一检查点开始；'
            '模型沿用原逐元素容差，仅Adam动量增加逐张量整体误差检查。'
            '本轮不声称训练逐位确定。修订记录：'+str(root / MARKER)+'；身份：'+marker['identity']+'。\n')
    return result


def worker(args, root, project, archive):
    marker = lineage(root, project, archive)
    if args.variant not in read(root / 'config.json')['variants']:
        raise ValueError('Unknown detector variant')
    from uecp_gspr_v2xr.common import lock
    with lock(root / '.worker.lock'):
        bind_commit(root, marker)
        if args.action == 'report':
            result = amended_report(root, marker, args.smoke)
            if result['status'] != 'COMPLETE':
                raise ValueError('Report still has incomplete stages')
            return
        from uecp_gspr_v2xr.runtime import bootstrap, input_stamps
        input_stamps(root)
        bootstrap(root, gpu=args.action not in ('content', 'select', 'resume-check'))
        if args.action == 'train':
            patch_training(root, marker)
            from uecp_gspr_v2xr.train import run
            run(root, args.variant, args.epoch, args.smoke, args.replay, args.interrupt_train_once)
        elif args.action == 'resume-check':
            replay_check(root, args.variant, args.epoch, marker)
        elif args.action == 'self-test':
            from uecp_gspr_v2xr.server_checks import run
            run(root)
        elif args.action == 'content':
            from uecp_gspr_v2xr.preflight import content
            content(root)
        elif args.action in ('dev', 'select', 'final'):
            from uecp_gspr_v2xr import evaluate
            if args.action == 'dev':
                evaluate.dev(root, args.variant, args.epoch, args.smoke)
            elif args.action == 'select':
                evaluate.select(root, args.variant, args.smoke)
            else:
                evaluate.final(root, args.variant, args.smoke, args.interrupt_once)
        elif args.action == 'diagnose':
            from uecp_gspr_v2xr.diagnose import run
            run(root, args.smoke)
        else:
            raise ValueError('Preflight must already be complete for this repair')


def controller(root, project, archive, mode):
    from uecp_gspr_v2xr import pipeline as original
    from uecp_gspr_v2xr.common import read, sha, variant_path, write
    from uecp_gspr_v2xr.state_logic import valid_stage
    marker = lineage(root, project, archive)
    from uecp_gspr_v2xr.common import lock
    with lock(root / '.controller.lock'):
        status = read(root / 'pipeline_status.json')
        if not valid_stage(status['preflight']):
            raise ValueError('Original preflight evidence is no longer complete')
        write(root / 'controller.json', dict(pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().split()[21],
            output=str(root), controller=str(archive), executable=sys.executable, source_repair_identity=marker['identity']))
        command = compile_worker_command(inspect.getsource(original.Pipeline.run), dict(original.Pipeline.run.__globals__), archive, project)
        class Pipeline(original.Pipeline):
            run = command
        pipe = Pipeline(root)
        try:
            pipe.run('server_self_test', 'self-test', [root / 'self_test/complete.json'])
            config = read(root / 'config.json')
            for variant in config['variants']:
                pipe.detector(variant, smoke=True)
            pipe.run('smoke_E1', 'diagnose', [root / 'smoke/E1/quality_diagnosis/complete.json'], '--smoke')
            pipe.run('smoke_E0_mid_epoch_resume_check_v2', 'resume-check', [gate(root, 'E0', 1)], '--variant', 'E0', '--epoch', 1)
            for variant in ('E0', 'E2', 'E3_joint'):
                src = variant_path(root, variant, True)
                dst = root / 'smoke_replay_v2' / variant
                for name in ('initialization.json', 'config.json', 'parameters.json'):
                    dest = dst / name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if dest.exists():
                        if sha(dest) != sha(src / name):
                            raise ValueError('Replay initialization metadata changed')
                    else:
                        shutil.copyfile(src / name, dest)
                for path in (src / 'epochs/01').rglob('*'):
                    if path.is_file():
                        dest = dst / 'epochs/01' / path.relative_to(src / 'epochs/01')
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        if dest.exists():
                            if sha(dest) != sha(path):
                                raise ValueError('Common replay checkpoint copy changed')
                        else:
                            shutil.copyfile(path, dest)
                pipe.run('smoke_'+variant+'_resume_replay_v2', 'train', [dst / 'epochs/02/train_complete.json'],
                         '--variant', variant, '--epoch', 2, '--smoke', '--replay')
                pipe.run('smoke_'+variant+'_resume_check_v2', 'resume-check', [gate(root, variant, 2)], '--variant', variant, '--epoch', 2)
            pipe.run('smoke_report_v2', 'report', [root / 'smoke/FINAL_RESULTS.json'], '--smoke')
            if mode == 'smoke':
                return
            pipe.run('accepted_training_content', 'content', [root / 'preflight/training_content.json'])
            for variant in config['variants']:
                for epoch in range(1, config['epochs']+1):
                    for part in ('train', 'dev'):
                        pipe.status.setdefault(f'{variant}_{part}_{epoch:02d}', dict(status='PENDING'))
            write(pipe.path, pipe.status)
            pipe.detector('E0')
            pipe.run('E1_quality_diagnosis', 'diagnose', [root / 'E1/quality_diagnosis/complete.json'])
            for variant in config['variants'][1:]:
                pipe.detector(variant)
            pipe.run('final_report', 'report', [root / 'FINAL_RESULTS.json', root / 'FINAL_RESULTS.md', root / 'ABLATION_RESULTS.md'])
            from uecp_gspr_v2xr.common import bound
            bound(root / 'complete.json', dict(status='COMPLETE', contract_identity=read(root / 'contract.json')['identity'],
                final_report_sha256=sha(root / 'FINAL_RESULTS.json'), source_repair_identity=marker['identity']))
        except BaseException:
            write(root / 'controller_failure.json', dict(stage=pipe.current, traceback=traceback.format_exc(), source_repair_identity=marker['identity']))
            try:
                amended_report(root, marker, partial=True)
            except Exception:
                traceback.print_exc()
            raise
        finally:
            if pipe.child is not None and pipe.child.poll() is None:
                os.killpg(pipe.child.pid, signal.SIGTERM)
                try:
                    pipe.child.wait(timeout=20)
                except __import__('subprocess').TimeoutExpired:
                    os.killpg(pipe.child.pid, signal.SIGKILL)
                    pipe.child.wait()


def main():
    if sys.platform != 'linux':
        raise RuntimeError('Repair execution is Linux-server-only')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('run', 'worker'))
    parser.add_argument('action', nargs='?')
    parser.add_argument('--output', required=True)
    parser.add_argument('--project', default='/home/cjm/OpenCOOD-main/cjmnet')
    parser.add_argument('--evidence')
    parser.add_argument('--mode', choices=('smoke', 'all'), default='all')
    parser.add_argument('--variant', default='E0')
    parser.add_argument('--epoch', type=int, default=1)
    for flag in ('smoke', 'replay', 'interrupt-once', 'interrupt-train-once'):
        parser.add_argument('--'+flag, action='store_true')
    args = parser.parse_args()
    root, project = Path(args.output).resolve(strict=True), Path(args.project).resolve(strict=True)
    archive = verify_archive()
    env = environment(root, project)
    env['PYTHONHASHSEED'] = worker_hash_seed(args.command, args.action, env['PYTHONHASHSEED'])
    if os.environ.get('PYTHONHASHSEED') != env['PYTHONHASHSEED']:
        os.execve(sys.executable, [sys.executable, '-B', str(archive), *sys.argv[1:]], env)
    os.environ.clear(); os.environ.update(env)
    sys.path.insert(0, str(project))
    from uecp_gspr_v2xr.paths import check
    check(root)
    if args.command == 'worker':
        worker(args, root, project, archive)
    else:
        with stopped_pipeline(root):
            if (root / MARKER).exists() and gate(root, 'E0', 1).exists():
                lineage(root, project, archive)
                from uecp_gspr_v2xr.common import complete
                complete(gate(root, 'E0', 1))
            else:
                evidence = args.evidence or (read(root / MARKER)['evidence'] if (root / MARKER).exists() else None)
                if not evidence:
                    raise ValueError('--evidence is required on first repair enrollment')
                enroll(root, project, archive, Path(evidence).resolve(strict=True))
        controller(root, project, archive, args.mode)


if __name__ == '__main__':
    main()
