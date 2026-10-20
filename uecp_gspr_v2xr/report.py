"""Only committed evidence becomes a scientific result. Missing stages stay N/A."""
import csv
import io
from pathlib import Path

from .common import atomic_bytes, complete, read, sha, variant_path, write


def difference(a, b):
    return None if a is None or b is None else 100*(a-b)


def number(value, signed=False):
    return 'N/A' if value is None else (f'{value:+.4f}' if signed else f'{value:.4f}')


def group(root, variant, smoke=False):
    branch = variant_path(root, variant, smoke)
    config = read(root / 'config.json')
    n = 2 if smoke else config['epochs']
    history = []
    for e in range(1, n+1):
        ep = branch / 'epochs' / f'{e:02d}'
        train = complete(ep / 'train_complete.json'); dev = complete(ep / 'dev_complete.json')
        if train['smoke_only'] != smoke or dev['smoke_only'] != smoke:
            raise ValueError('Smoke and formal results mixed')
        history.append(dict(train=train, dev=dev))
    selected = read(branch / 'selected.json')
    if selected['candidate_count'] != n:
        raise ValueError('Checkpoint selection budget differs')
    for e, digest in selected['candidates'].items():
        if sha(branch / 'epochs' / f'{int(e):02d}' / 'dev_complete.json') != digest:
            raise ValueError('Selection candidate changed')
    if sha(selected['checkpoint']) != selected['checkpoint_sha256']:
        raise ValueError('Locked checkpoint changed')
    final = complete(branch / 'final_evaluation/complete.json')
    if (final['smoke_only'] != smoke or (not smoke and not final['full_validation'])
            or final['checkpoint_sha256'] != selected['checkpoint_sha256']):
        raise ValueError('Final evaluation does not match locked model')
    return dict(variant=variant, selected=selected, final=final,
        parameters=read(branch / 'parameters.json'),
        initialization=read(branch / 'initialization.json'),
        training_seconds=sum(r['train']['seconds'] for r in history),
        train_frames=sum(r['train']['totals']['frames'] for r in history),
        optimizer_updates=sum(r['train']['totals']['updates'] for r in history),
        max_memory_allocated=max(r['train']['max_memory_allocated'] for r in history),
        mean_forward_seconds=final['mean_forward_seconds'], timing_complete=final['timing_complete'])


def ap(groups, variant, weather, protocol, metric):
    if variant not in groups:
        return None
    return groups[variant]['final']['results'][weather]['AP'][protocol][metric]


def run(output, smoke=False, partial=False):
    root = Path(output); target = root / ('smoke' if smoke else '')
    config = read(root / 'config.json'); groups, failures = {}, []
    for v in config['variants']:
        try:
            groups[v] = group(root, v, smoke)
        except (ValueError, FileNotFoundError, KeyError) as exc:
            failures.append(dict(variant=v, reason=str(exc)))
    e1_path = target / 'E1/quality_diagnosis/complete.json'
    try:
        e1 = complete(e1_path)
    except (ValueError, FileNotFoundError) as exc:
        e1 = dict(conclusion='N/A', reason=str(exc)); failures.append(dict(variant='E1', reason=str(exc)))
    gates = {}
    required = ['self_test/complete.json']
    if smoke:
        required += ['smoke/E0_epoch1_resume_complete.json']
        required += ['smoke/'+v+'_epoch2_resume_complete.json' for v in ('E0', 'E2', 'E3_joint')]
    else:
        required += ['preflight/training_content.json', 'smoke/FINAL_RESULTS.json']
    for name in required:
        try:
            record = complete(root / name) if name != 'smoke/FINAL_RESULTS.json' else read(root / name)
            if record['status'] != 'COMPLETE':
                raise ValueError('Required gate incomplete')
            gates[name] = dict(sha256=sha(root / name), status='COMPLETE')
        except (ValueError, FileNotFoundError, KeyError) as exc:
            failures.append(dict(variant=name, reason=str(exc)))
    rows, deltas = [], []
    for v in config['variants']:
        for w in ('clean', 'fog', 'snow'):
            for p in ('O', 'O*', 'F'):
                rows.append(dict(variant=v, weather=w, protocol=p,
                    **{m: ap(groups, v, w, p, m) for m in ('ap30', 'ap50', 'ap70')}))
    for v, baseline in ((v, 'E0') for v in config['variants'][1:]):
        for w in ('clean', 'fog', 'snow'):
            for p in ('O', 'O*', 'F'):
                deltas.append(dict(comparison=v+'-'+baseline, weather=w, protocol=p,
                    **{m+'_pp': difference(ap(groups, v, w, p, m), ap(groups, baseline, w, p, m)) for m in ('ap30', 'ap50', 'ap70')}))
    for g in groups.values():
        if 'E0' in groups and (g['initialization']['shared'] != groups['E0']['initialization']['shared']
                or g['optimizer_updates'] != groups['E0']['optimizer_updates']
                or g['train_frames'] != groups['E0']['train_frames']):
            raise ValueError('Fairness audit failed: shared initialization/exposure/update budget differs')
    contrasts = {}
    for a, b in (('E2', 'E0'), ('E3_joint', 'E0'), ('E3_joint', 'E2'),
                 ('E3_joint', 'E3_no_separation'), ('E3_joint', 'E3_shuffled'), ('E3_joint', 'E3_constant_quality')):
        contrasts[a+'-'+b] = {w: difference(ap(groups, a, w, 'F', 'ap70'), ap(groups, b, w, 'F', 'ap70')) for w in ('clean', 'fog', 'snow')}
    caveats = [f'单个随机种子{config["seed"]}、预先固定{config["epochs"]}轮预算；未检验训练是否充分收敛。',
        '829帧validation已被历史实验多次查看，结果不属于独立盲测；没有访问正式test。',
        'GSPR来自额外的OPV2V天气监督训练；冻结复用，检测器更新预算相同，总历史计算量和监督信息不同。',
        'AP小幅变化只作描述，不自动声称统计显著；质量诊断的场景bootstrap区间不能转用于检测AP。',
        '原版UECP单尺度精度权重截断U梯度，池化和跨尺度路径仍可传梯度；E2_original_detach单独训练核查。',
        '原版单车与融合损失各累计一次5倍密度辅助损失，所有组相同；未增加雷达/相机模型输入。']
    if e1['conclusion'] != 'SUPPORTED':
        caveats.append('E1尚未证明密度之外的质量收益，E2/E3按探索性反证实验解释。')
    answers = [dict(question='GSPR是否提供密度之外的质量信息？', answer=e1['conclusion']),
        dict(question='加入GSPR是否改善检测？', answer='比较E2-E0的固定GT F AP70，逐天气查看，不以O单独改善作结论。', values=contrasts['E2-E0']),
        dict(question='分开建模是否优于单一质量图？', answer='比较joint与独立重训no-separation。', values=contrasts['E3_joint-E3_no_separation']),
        dict(question='收益是否来自有效质量信息？', answer='同时比较joint、shuffled与参数相同且适配器活跃的constant-quality；单独AP提升不足以确认。',
            shuffled=contrasts['E3_joint-E3_shuffled'], constant=contrasts['E3_joint-E3_constant_quality']),
        dict(question='三种天气收益是否稳定？', answer='逐天气固定GT差值如下，负值表示下降。', values=contrasts['E3_joint-E0']),
        dict(question='是否支持继续开发论文模块？', answer='需要E1有依据、joint优于单图/打乱/常数对照，Clean保持情况可接受，并经多种子与独立数据确认。当前单轮结果仅支持后续验证判断。')]
    result = dict(status='COMPLETE' if not failures else 'FAILED' if partial else 'PENDING',
        smoke_only=smoke, groups=groups, E1=e1, rows=rows, differences=deltas,
        fixed_GT_AP70_contrasts=contrasts, answers=answers, limitations=caveats, failures=failures,
        prerequisite_gates=gates,
        protocol=read(root / 'protocol.json'), stages=read(root / 'pipeline_status.json') if (root / 'pipeline_status.json').exists() else {})
    lines = ['# UECP + GSPR E0–E3', '', '状态：'+result['status'], 'Smoke：'+str(smoke), '',
        '|方法|天气|协议|AP30|AP50|AP70|', '|---|---|---|---:|---:|---:|']
    for r in rows:
        if r['variant'] in ('E0', 'E2', 'E3_joint'):
            lines.append('|'+ '|'.join([r['variant'], r['weather'], r['protocol']]+[number(r[m]) for m in ('ap30', 'ap50', 'ap70')])+'|')
    lines += ['', 'AP以0–1表示；以下差值单位为百分点（pp）。', '',
        '|比较|天气|协议|ΔAP30 pp|ΔAP50 pp|ΔAP70 pp|', '|---|---|---|---:|---:|---:|']
    for r in deltas:
        if r['comparison'].startswith(('E2-', 'E3_joint-')):
            lines.append('|'+ '|'.join([r['comparison'], r['weather'], r['protocol']]+[number(r[m+'_pp'], True) for m in ('ap30', 'ap50', 'ap70')])+'|')
    lines += ['', '|模型|可训练参数|冻结参数|训练小时|峰值显存 GiB|前向 ms|', '|---|---:|---:|---:|---:|---:|']
    for v, g in groups.items():
        ms = None if g['mean_forward_seconds'] is None else g['mean_forward_seconds']*1000
        lines.append(f'|{v}|{g["parameters"]["trainable"]}|{g["parameters"]["frozen"]}|{g["training_seconds"]/3600:.3f}|{g["max_memory_allocated"]/2**30:.3f}|{number(ms)}|')
    for answer in answers:
        lines += ['', answer['question']+' '+answer['answer']]
        for k in ('values', 'shuffled', 'constant'):
            if k in answer:
                lines.append(k+': '+', '.join(w+' '+number(v, True)+' pp' for w, v in answer[k].items()))
    lines += ['', '限制：']+['- '+v for v in caveats]
    if failures:
        lines += ['', '未完成或失败：']+['- '+r['variant']+': N/A，'+r['reason'] for r in failures]
    ablation = ['# 独立重训消融', '', '|方法|天气|协议|AP30|AP50|AP70|', '|---|---|---|---:|---:|---:|']
    for r in rows:
        ablation.append('|'+ '|'.join([r['variant'], r['weather'], r['protocol']]+[number(r[m]) for m in ('ap30', 'ap50', 'ap70')])+'|')
    ablation += ['', 'E2与E3_no_separation采用相同的单图结构，作为明确的结构对照重复训练；相同种子下可得到相同模型。',
        'Coverage-only停用质量适配器输出；constant-quality保留活跃的113参数适配器，同时计算后丢弃GSPR有效质量值。']
    stream = io.StringIO(newline=''); writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    # Partial reports have a separate name and never overwrite a committed final report.
    prefix = 'PARTIAL' if partial else 'FINAL'
    previous = target / (prefix+'_RESULTS.json')
    immutable = previous.exists() and read(previous).get('status') == 'COMPLETE' and not partial
    payloads = {prefix+'_RESULTS.json': result, prefix+'_RESULTS.md': '\n'.join(lines)+'\n',
                ('PARTIAL_ABLATION' if partial else 'ABLATION')+'_RESULTS.md': '\n'.join(ablation)+'\n',
                prefix+'_metrics.csv': stream.getvalue()}
    for name, data in payloads.items():
        path = target / name
        if immutable:
            if not path.exists():
                raise ValueError('Committed report file missing: '+str(path))
            continue
        if isinstance(data, dict):
            write(path, data)
        else:
            atomic_bytes(path, data.encode())
    return result
