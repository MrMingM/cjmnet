"""Fixed best-validation checkpoints; OPV2V-W test has no online augmentation."""
import argparse
import json
from pathlib import Path
import torch
from .network import QueryDecoder,MiniCEIF
from .runtime import rt,contract,validate


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',required=True)
    p.add_argument('--phase',choices=('development','benchmark'),default='development')
    p.add_argument('--output-dir',required=True)
    a=p.parse_args()
    root=Path(a.run)
    protocol=json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    completed=json.loads((root/'training_complete.json').read_text(encoding='utf-8'))
    identity=protocol['contract']; settings=identity['prototype']
    if completed['contract']!=identity: raise ValueError('Training completion contract mismatch')
    if a.phase=='benchmark' and settings['smoke']:
        raise ValueError('Smoke checkpoints are not eligible for benchmark evaluation')
    rt.verify_frozen(); device=rt.device()
    checkpoints={name:torch.load(root/f'{name}_best.pth',map_location=device,weights_only=True)
                 for name in ('ceif','aux_only')}
    for name,state in checkpoints.items():
        if state['contract']!=identity or state['variant']!=name:
            raise ValueError('Checkpoint identity mismatch')
    for key,value in checkpoints['ceif']['decoder'].items():
        torch.testing.assert_close(value,checkpoints['aux_only']['decoder'][key],atol=0,rtol=0)
    output=rt.new_output(a.output_dir)
    all_results={}
    for weather in ('clean','fog','rain','snow'):
        if a.phase=='benchmark':
            from gspr_evidence.benchmark import load_config
            options,hypes=load_config(root/'experiment.yaml',protocol['frontend_config'],weather)
        else:
            options,hypes=rt.load_config(root/'experiment.yaml',protocol['frontend_config'])
        rt.seed_all(options['seed'])
        model,digest=rt.load_model(hypes,options,protocol['frontend_checkpoint'],device)
        model.eval().requires_grad_(False)
        if contract(options,protocol['frontend_config'],digest,settings)!=identity:
            raise ValueError('Evaluation differs from training code/frontend/settings')
        channels=model.engine.base.cls_head.in_channels
        decoder=QueryDecoder(channels).to(device)
        decoder.load_state_dict(checkpoints['ceif']['decoder']); decoder.eval().requires_grad_(False)
        variants={name:MiniCEIF(channels,settings['hidden']).to(device) for name in checkpoints}
        for name,module in variants.items(): module.load_state_dict(checkpoints[name]['model'])
        all_results[weather]=validate(model,decoder,variants,hypes,options,
            dict(settings,_validation_weather=weather),device,settings['smoke'],
            formal_weather=weather if a.phase=='benchmark' else None,output=output/weather)
        del model,decoder,variants
        torch.cuda.empty_cache()
    record=dict(phase=a.phase,contract=identity,
        checkpoint_sha256={name:rt.sha256(root/f'{name}_best.pth') for name in checkpoints},
        chosen_epochs={name:int(state['epoch'])+1 for name,state in checkpoints.items()},
        selection='Independent best mean validation AP70; never selected using test',results=all_results)
    rt.write_json(output/'results.json',record)
    lines=['# CEIF 最小原型：实际训练结果','',f'阶段：{a.phase}；smoke={bool(settings["smoke"])}。',
        '原前端、骨干、AttFuse 和检测头冻结；两个模型仅训练相同的残差修正网络，CEIF 额外启用显式观测投影。',
        'aux_only 使用相同观测辅助损失，但不执行前向投影。全通信；额外几何载荷单独统计。','',
        '| Weather | Model | AP30 | AP50 | AP70 | ΔAP70 / pp |','|---|---|---:|---:|---:|---:|']
    for weather,row in all_results.items():
        baseline=row['results']['baseline']['ap70']
        for name,ap in row['results'].items():
            lines.append(f'| {weather} | {name} | {ap["ap30"]:.6f} | {ap["ap50"]:.6f} | {ap["ap70"]:.6f} | {100*(ap["ap70"]-baseline):+.4f} |')
    lines+=['','需分别判断：是否超过原基线、是否超过 aux_only、三种天气是否达到 +5 pp、Clean 是否受损。',
        '这是一次固定预算的最小原型，不自动按 test 修改阈值、继续训练或挑选轮次。',
        'development 是 validation 在线天气；benchmark 是真实路径中的 OPV2V-W 文件，关闭在线增强。',
        '详细输出包含查询覆盖、约束冲突比例、观测违背量、特征修正幅度和几何证据额外字节。']
    (output/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    rt.verify_frozen()
    print(f'Send back: {output}/results.md',flush=True)


if __name__=='__main__': main()
