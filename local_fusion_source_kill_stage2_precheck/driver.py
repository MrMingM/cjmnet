"""CPU-only audit of already saved JSON/cache; summary-only is explicit partial evidence."""
import argparse
import ast
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import traceback

from .common import (ROOT, PACKAGE, HISTORICAL, WEATHERS, atomic_json, atomic_text, digest,
                     output_path, read_json, scene_summary, source_hashes, timestamp, verify_snapshot, object_hash)
from .inputs import load_reports, discover_stage1, read_s0, read_oracle_details
from .source_redundancy import build_cells, summarize_groups
from .data_adequacy import h1_from_summary, h2_from_summary, proxy_cells
from .oracle_records import summarize_details
from . import recommendations, reports

OUTPUTS = ('P0_SOURCE_REDUNDANCY.json', 'P0_SOURCE_REDUNDANCY.md',
           'P0_DATA_ADEQUACY.json', 'P0_DATA_ADEQUACY.md',
           'P0_STAGE2_PRECHECK_FINAL.json', 'P0_STAGE2_PRECHECK_FINAL.md',
           'DATA_ADEQUACY_RECOMMENDATION.md')


def static_check():
    for path in PACKAGE.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    for path in PACKAGE.glob('*.sh'):
        if b'\r' in path.read_bytes():
            raise ValueError('Shell must use LF')
        subprocess.run(['sh', '-n', path.as_posix()], check=True)
    verify_snapshot()
    print('Static precheck complete; no cache/model/dataset access', flush=True)


def create_results(k0, k10, raw_rows=None, pairs=None, oracle_rows=None):
    h1 = h1_from_summary(k0)
    p0a = dict(available=False, status='unavailable_requires_original_s0_cache', cells=None,
               conclusion='现有汇总无法判断最佳离散动作是否跨 source 等价；须读取原 S0 cache。')
    if raw_rows is not None:
        p0a['cells'] = build_cells(raw_rows)
        p0a.update(available=True, status='complete_existing_evidence')
        h1 = p0a['cells']
        cls = h1['validation/adverse pooled/cls/associated']
        reg = h1['validation/adverse pooled/reg/associated']
        p0a['conclusion'] = (
            f"beneficial groups 的跨 source 等价最佳占比：cls {reports.fmt(cls['multi_source_best_fraction'], True)}，"
            f"reg {reports.fmt(reg['multi_source_best_fraction'], True)}；仅用于 N2 先验背景，不判定 K1。")
        p0a['N2_prior_by_task'] = {}
        for task, row in (('cls', cls), ('reg', reg)):
            fraction = row['multi_source_best_fraction']
            tendency = ('无 beneficial group，暂不判断先验。' if fraction is None else
                '跨 source 等价最佳占多数，N2 FAIL 的先验倾向更强；需实际随机混合对照验证。' if fraction > .5 else
                '唯一 source 最佳占多数，N2 PASS 的先验相对增强；仍可能被随机扰动解释。' if fraction < .5 else
                '两类各半，无法给出明确的 N2 先验倾向。')
            p0a['N2_prior_by_task'][task] = dict(interpretation=tendency,
                validation_multi_source_fraction=fraction,
                train_multi_source_fraction=h1[f'train/adverse pooled/{task}/associated']['multi_source_best_fraction'],
                unique_source_cross_family_only_fraction=row['family_counts']['same_source_cross_family_only'] /
                    row['beneficial_discrete_groups'] if row['beneficial_discrete_groups'] else None)
    h2 = {variant: h2_from_summary(k10, variant) for variant in ('FULL', 'A_ONLY')}
    if oracle_rows is not None:
        for variant, cells in h2.items():
            details = summarize_details([r for r in oracle_rows if r['variant'] == variant])
            for key, detail in details.items():
                cells[key].update(detail)
    counts = {split: {task: h1[f'{split}/adverse pooled/{task}/associated']['beneficial_discrete_groups']
                      for task in ('cls', 'reg')} for split in ('train', 'validation', 'train+validation')}
    extra = h2['FULL']['validation/adverse pooled']['task_extra_recovered_targets']
    conclusion = (f"adverse pooled 单任务 H1 正例：train cls/reg={counts['train']['cls']}/{counts['train']['reg']}，"
                  f"validation={counts['validation']['cls']}/{counts['validation']['reg']}；"
                  f"validation Task−Same recovered 净差={extra}。")
    conclusion += ('同状态 Task 真实正例已从原记录核对，train joint Task 仍未生成。'
                   if oracle_rows is not None else 'Task 同状态正例数与 scene 分布仍待原逐目标记录，train joint Task 未生成。')
    p0b = dict(H1=h1, H2=h2,
        task_separation_proxy=proxy_cells(pairs) if pairs is not None else None,
        true_task_positive_definition='Task selected discrete6 > best allowed Same discrete6 in the same saved Task current state',
        task_net_recovered_difference_is_positive_label_count=False,
        joint_same_s0_candidate_level={'available': False, 'positive_n': None,
            'reason': 'S0 contains cls-only and reg-only outcomes; no joint Same source counterfactuals. '
                      'Existing Same Oracle GT-target/current-state rows are reported separately.'},
        train_joint_task_oracle_available=False, conclusion=conclusion)
    if raw_rows is not None:
        partitions = {}
        for task in ('cls', 'reg'):
            for partition in ('probe_fit', 'probe_calibration'):
                selected = [r for r in raw_rows if r['split'] == 'train' and r['stratum'] == 'associated'
                            and r['task'] == task and r['weather'] in ('fog', 'rain', 'snow')
                            and r.get('train_partition') == partition]
                partitions[f'{partition}/{task}'] = summarize_groups(selected)
        p0b['training_partition_H1_adverse'] = partitions
    recommendation = recommendations.propose(p0b)
    limitations = [
        'K1-0 仅记 A-dominant descriptive decomposition，§40 没有其 PASS/FAIL 门槛。',
        'H1 cls/reg 的独立单任务正例不相加冒充 joint Same 正例。',
        '独立 greedy 路径的恢复数量差不等价于 Task-only GT 数量，也不等价于同状态 Task 正例。',
        '原 S0 没有 joint Same candidate-level outcomes；现有联合 Same 统计单位为 Oracle GT-target/current-state。',
        'train joint Task Oracle 没有生成；train 及 train+validation true Task n 保持 unavailable/null，proxy 单列。',
        '同一 frame 在多天气重复出现，pooled 仅用于样本盘点，不能当作独立泛化证据。',
        'source 唯一性、扰动稳定性和可学习性未由本轮验证；建议门槛须人工冻结。']
    if raw_rows is None:
        limitations.insert(0, '本地结果是 partial summary：P0-A、去重物理 group、fit/calibration 与 H2 同状态/scene 仍需服务器 cache/record。')
    final = dict(status='complete_existing_evidence' if raw_rows is not None else 'partial_requires_server_cache',
                 conclusion=conclusion, P0_A=p0a, P0_B=p0b, recommendations=recommendation,
                 limitations=limitations, k3_frozen=True, next_stage_auto_start=False,
                 cpu_only=True, test_data_used=False)
    return p0a, p0b, final


def write_outputs(output, p0a, p0b, final):
    atomic_json(output / OUTPUTS[0], p0a)
    atomic_text(output / OUTPUTS[1], reports.source_report(p0a))
    atomic_json(output / OUTPUTS[2], p0b)
    atomic_text(output / OUTPUTS[3], reports.adequacy_report(p0b))
    atomic_json(output / OUTPUTS[4], final)
    atomic_text(output / OUTPUTS[5], reports.final_report(final))
    atomic_text(output / OUTPUTS[6], recommendations.render(final['recommendations'], p0b))


def execute(args):
    k0, k10, final1, hashes = load_reports(args.reports)
    output = output_path(args.output or (ROOT / 'logs/k'), server=not args.summary_only)
    identity = dict(source_hashes=source_hashes(), historical_hashes=hashes,
                    mode='summary-only' if args.summary_only else 'full-cpu')
    stage1 = None
    if not args.summary_only:
        if os.name != 'posix':
            raise ValueError('Full cache statistics run on the remote POSIX server only')
        stage1 = discover_stage1(args.reports, args.stage1_run)
        identity['stage1_run'] = str(stage1)
        identity['stage1_protocol_sha256'] = digest(stage1 / 'protocol.json')
    manifest_path = output / 'P0_MANIFEST.json'
    if manifest_path.exists():
        old = read_json(manifest_path)
        if old['identity'] != identity:
            raise ValueError('Precheck output identity differs; use a new output directory')
        if old['status'] == 'complete':
            if any(digest(output / name) != value for name, value in old['outputs'].items()):
                raise ValueError('Precheck report corrupted')
            print('Reuse complete read-only precheck reports:', output, flush=True)
            return
    manifest = dict(status='running', started_at=timestamp(), completed_at=None, identity=identity,
                    project_clock=final1['project_clock'], source_commit=None)
    atomic_json(manifest_path, manifest)
    try:
        raw_rows = pairs = oracle_rows = None
        provenance = {'report_hashes': hashes}
        if stage1 is not None:
            raw_rows, pairs, scene_map, protocol = read_s0(stage1, hashes, k0)
            s0 = read_json(Path(protocol['references']['s0_run']) / 'protocol.json')
            fit, cal = set(s0['probe_fit_scenes']), set(s0['probe_calibration_scenes'])
            if fit & cal or fit | cal != set(s0['train_scenes']):
                raise ValueError('Original fit/calibration scenes invalid')
            for row in raw_rows:
                if row['split'] == 'train':
                    scene = int(row['scene_id'].split(':')[1])
                    row['train_partition'] = 'probe_fit' if scene in fit else 'probe_calibration'
            oracle_rows = read_oracle_details(stage1, k10, scene_map, protocol)
            provenance.update(stage1_run=str(stage1), s0_run=protocol['references']['s0_run'],
                s0_manifest_hash=protocol['references']['s0_manifest_hash'],
                s0_protocol_hash=protocol['references']['s0_protocol_hash'])
            # Large group records stay in the external RUN.
            for filename, values in (('p0_group_records.jsonl', raw_rows), ('p0_oracle_target_records.jsonl', oracle_rows)):
                atomic_text(output / filename, ''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in values))
        p0a, p0b, final = create_results(k0, k10, raw_rows, pairs, oracle_rows)
        final['provenance'] = provenance
        write_outputs(output, p0a, p0b, final)
        # Any mutation of a Stage 1 original file is an error.
        if any(digest(Path(args.reports) / name) != expected for name, expected in hashes.items()):
            raise ValueError('Historical Stage 1 reports changed during precheck')
        manifest.update(status='complete', evidence_status=final['status'], completed_at=timestamp(),
                        outputs={name: digest(output / name) for name in OUTPUTS})
        if raw_rows is not None:
            manifest['outputs'].update({name: digest(output / name) for name in (
                'p0_group_records.jsonl', 'p0_oracle_target_records.jsonl')})
        print('P0 statistics saved; read the UTF-8 Markdown reports for conclusions.', flush=True)
        print('REPORTS:', output, 'STATUS:', final['status'], flush=True)
    except BaseException as exc:
        manifest.update(status='failed', completed_at=timestamp(), error=repr(exc))
        raise
    finally:
        atomic_json(manifest_path, manifest)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--static', action='store_true')
    cli.add_argument('--summary-only', action='store_true')
    cli.add_argument('--reports', default=str(ROOT / 'logs/k'))
    cli.add_argument('--output')
    cli.add_argument('--stage1-run', default=os.environ.get('STAGE1_RUN'))
    args = cli.parse_args()
    if args.static:
        static_check()
        return
    verify_snapshot()
    if args.summary_only:
        execute(args)
    else:
        if not args.output:
            cli.error('--output is required for the full CPU audit')
        if os.name != 'posix':
            cli.error('Full cache statistics run on the remote POSIX server only; use --summary-only locally')
        import fcntl
        output = output_path(args.output, server=True)
        with (output / '.precheck.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            execute(args)


if __name__ == '__main__':
    main()
