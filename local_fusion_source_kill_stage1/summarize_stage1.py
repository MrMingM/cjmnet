"""Produce Stage 1 only after both independent processes complete."""
import argparse
from .common import (atomic_json, atomic_text, component_status, contract, external_run,
                     read_json, refresh_manifest)


def build(run):
    protocol = contract(run)
    for name in ('K0_PRIME', 'K1_0'):
        if component_status(run, name)['status'] != 'complete':
            raise RuntimeError('Stage 1 cannot summarize incomplete component: ' + name)
    k0 = read_json(run / 'K0_GROUP_MARGIN.json')
    k10 = read_json(run / 'K1_0_ORACLE_DECOMPOSITION.json')
    from .k0_report import conclusion, primary_tables
    from .k1_0_report import main_tables
    sentence = conclusion(k0) + ' ' + k10['interpretation']
    result = dict(stage=protocol['stage'], status='complete', conclusion=sentence,
                  K0_PRIME=k0, K1_0=k10, source_utility_status='PROBATION / 待判死',
                  k3_frozen=True, next_stage_auto_start=False, test_data_used=False,
                  project_clock=protocol['project_clock'])
    text = '# Source Utility Kill Protocol — Stage 1\n\n## 一句话结论\n\n' + sentence + '\n\n'
    text += primary_tables(k0) + '\n' + main_tables(k10)
    text += ('\n## 目前能得出什么结论\n\n' + sentence + '\n'
             f"K0 描述性信号：{k0['signal']}。细分 split、并列数量、scene 集中度见 K0_GROUP_MARGIN.md/json；"
             '完整 AP30/50/70 与动作统计见 K1_0_ORACLE_DECOMPOSITION.md/json。\n'
             '\n## 还不能得出什么结论\n\n'
             '尚不能确认 source-specificity、跨采样 label stability 或实际学习能力。'
             'A/B/E 只对应原 outcome key，不等价于逐框身份/分数完全一致。'
             'GT hindsight Oracle 是 development 诊断，不能外推部署或正式 test。'
             '三条 greedy 路径相互独立，收益没有可加要求。K3 保持冻结。\n'
             '\n## 下一阶段 K1/K2 应重点验证什么\n\n' + k10['next_stage_focus'] + '\n'
             'K1 检验来源特异性，K2 检验标签稳定性；二者 PASS 后才允许恢复 K3。'
             '当前阶段到此结束，等待人工判断，不自动启动任何下一阶段。'
             '14 天上限属于 resource stop，不构成 hypothesis rejection。\n')
    atomic_json(run / 'STAGE1_FINAL.json', result)
    atomic_text(run / 'STAGE1_FINAL.md', text)
    refresh_manifest(run, 'complete')
    print('Stage 1 complete; K3 frozen; awaiting human judgment', flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--run', required=True)
    args = cli.parse_args()
    build(external_run(args.run))


if __name__ == '__main__':
    main()
