"""Combine original K1 and repaired K2 without forging their run identities."""
from pathlib import Path
from local_fusion_source_kill_stage2.common import (
    atomic_json, atomic_text, digest, load_contract, read_json, resume, settings)
from local_fusion_source_kill_stage2.verdicts import final_verdict


def assert_inherited(parent, repaired):
    for key, value in parent.items():
        if key not in ('identity', 'input_hashes') and repaired.get(key) != value:
            raise ValueError('Parent science/provenance changed: '+key)
    if any(repaired['input_hashes'].get(p) != v for p,v in parent['input_hashes'].items()):
        raise ValueError('Parent input hashes not inherited exactly')
    if repaired['k2_replay_repair']['parent_identity'] != parent['identity']:
        raise ValueError('Parent identity mismatch')


def component(folder, protocol, component_name, report_name):
    path = folder/component_name/'manifest.json'
    if not path.exists():
        return None
    manifest = read_json(path)
    if manifest['identity'] != protocol['identity']:
        raise ValueError('Worker identity mismatch: '+str(path))
    if manifest['status'] in ('running', 'pending'):
        return None
    report = folder/component_name/(report_name+'.json')
    if manifest['status'] == 'complete':
        if resume(folder/component_name/'result.json', protocol['identity']) is None:
            raise ValueError('Complete worker lacks hash-verified result')
    row = read_json(report) if report.exists() else dict(status='failed', verdict={'SAME':'INCONCLUSIVE', 'TASK':'INCONCLUSIVE'})
    if report.exists() and row['root_identity'] != protocol['identity']:
        raise ValueError('Report identity mismatch')
    return row


def summarize(run):
    run = Path(run)
    protocol = load_contract(run)
    parent_run = Path(protocol['k2_replay_repair']['parent_run'])
    parent = load_contract(parent_run)
    assert_inherited(parent, protocol)
    k1 = component(parent_run, parent, 'k1', 'K1_SOURCE_SPECIFICITY')
    k2 = component(run, protocol, 'k2', 'K2_LABEL_STABILITY')
    if k1 is None or k2 is None:
        print('Final pending: original K1 or repaired K2 has not finished. K3 remains frozen.', flush=True)
        return None
    if k2['status'] == 'complete' and resume(run/'source_reproduction'/'result.json', protocol['identity']) is None:
        raise ValueError('Missing hash-verified original Source AP gate')
    result = final_verdict(k1['verdict']['SAME'], k1['verdict']['TASK'], k2['verdict']['SAME'], k2['verdict']['TASK'])
    references = {}
    for folder, component_name, name in ((parent_run, 'k1', 'K1_SOURCE_SPECIFICITY'), (run, 'k2', 'K2_LABEL_STABILITY')):
        for ext in ('.json', '.md'):
            path = folder/component_name/(name+ext)
            if path.exists():
                references[str(path)] = digest(path)
                # Copies retain the report's ORIGINAL root_identity.
                atomic_text(run/(name+ext), path.read_text(encoding='utf-8'))
    result.update(root_identity=protocol['identity'], parent_k1_identity=parent['identity'],
        status='complete' if k1['status']==k2['status']=='complete' else 'failed',
        K1=k1, K2=k2, report_references=references, k3_data_gate_record_only=settings()['k3_record_only'],
        resource_stop_at=protocol['resource_stop_at'], scientific_thresholds_changed=False, test_data_used=False)
    atomic_json(run/'STAGE2_FINAL.json', result)
    lines = ['# Source Utility Kill Protocol — Stage 2', '', '## 一句话结论', '', result['decision']+'。', '']
    for key in ('K1_SAME','K1_TASK','K2_SAME','K2_TASK','H1_SOURCE_SPECIFIC','H2_TASK_SPECIFIC'):
        lines.append(f'- {key} = {result[key]}')
    lines += ['', '## 复现修复与来源', '',
        'K1 引用原 RUN 的完整结果，保留原 identity；K2 来自独立修复 RUN。所有继承的科学配置、源码和输入哈希均核验一致。',
        '逐帧连续值采用原 S0 数值复现约定，离散字段、动作和 full9 排序必须完全一致；原 Source AP 门禁仍为 1e-6。', '',
        '## 触发门槛', '',
        '逐 family 的冻结阈值、置信区间和失败原因见 K1/K2 报告；缺失结果按 INCONCLUSIVE 处理。', '',
        '## 结论边界', '',
        '仅限当前 development/validation，不证明可学习、可部署或正式 test 泛化。K3 继续冻结。']
    atomic_text(run/'STAGE2_FINAL.md', '\n'.join(lines)+'\n')
    print('FINAL='+str(run/'STAGE2_FINAL.md'), flush=True)
    return result
