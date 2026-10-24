#!/usr/bin/env python3
"""Read-only S4 progress/ETA viewer. Python standard library only.

Copy this file to the repository root, OUTSIDE the S4 package. It reads two
small JSON files; it never imports experiment code or opens tensor caches.
"""
import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time


WEATHERS = ('clean', 'fog', 'rain', 'snow')
STAGES = ('preflight', 'server-tests', 'repair-recheck', 'R0',
          'reproduction-audit', 'evidence-train', 'evidence-validation',
          'GEO-fit', 'GEO', 'SEM-fit', 'SEM', 'ABLATIONS', 'REPLAY',
          'FINAL', 'integrity')
TZ = dt.timezone(dt.timedelta(hours=8))


def read_json(path):
    with path.open(encoding='utf-8-sig') as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(str(path) + ': expected a JSON object')
    return value


def row_for(entries, stage, split='-', weather='-', frame='-'):
    return entries.get('/'.join(map(str, (stage, split, weather, frame))), {})


def stamp(row):
    try:
        value = float(row['updated_utc_epoch'])
        return value if math.isfinite(value) and value > 0 else None
    except (KeyError, TypeError, ValueError):
        return None


def duration(seconds):
    minutes = max(0, math.ceil(seconds / 60))
    if minutes < 60:
        return str(minutes) + ' 分钟'
    hours, minutes = divmod(minutes, 60)
    return f'{hours} 小时 {minutes} 分钟'


def clock(epoch):
    return dt.datetime.fromtimestamp(epoch, TZ).strftime('%m-%d %H:%M:%S')


def recent_rate(times, window, since=None):
    # Keep real slow frames. Do not silently discard long intervals as outliers.
    # A restarted stage has a new start timestamp, excluding the downtime gap.
    times = sorted(t for t in times if since is None or t >= since)[-window:]
    intervals = [b-a for a, b in zip(times, times[1:]) if b > a]
    if len(intervals) < 3:
        return None
    ordered = sorted(intervals)
    mean = statistics.mean(intervals)
    low = ordered[int((len(ordered)-1) * .25)]
    high = ordered[math.ceil((len(ordered)-1) * .75)]
    return {'mean': mean, 'low': min(low, mean), 'high': max(high, mean),
            'samples': len(intervals), 'last': times[-1]}


def frame_progress(entries, indices, kind, window):
    rows = []
    splits = ('train', 'validation') if kind == 'evidence' else ('validation',)
    for split in splits:
        stage = 'evidence-' + split if kind == 'evidence' else 'REPLAY'
        stage_row = row_for(entries, stage)
        since = stamp(stage_row) if stage_row.get('status') == 'in_progress' else None
        for weather in WEATHERS:
            complete = [row_for(entries, kind+'-frame', split, weather, frame)
                        for frame in indices[split]]
            complete = [row for row in complete if row.get('status') == 'complete']
            times = [stamp(row) for row in complete if stamp(row) is not None]
            rows.append({'split': split, 'weather': weather, 'done': len(complete),
                         'total': len(indices[split]),
                         'last': max(times) if times else None,
                         'rate': recent_rate(times, window, since)})
    return rows


def estimate_frames(rows):
    estimates, unknown = [], []
    for row in rows:
        remaining = row['total'] - row['done']
        if not remaining:
            continue
        rate, source = row['rate'], row
        if rate is None:
            # Only borrow from the same split and same kind of work. Never use
            # evidence throughput to predict replay, or train to predict val.
            candidates = [r for r in rows if r['split'] == row['split'] and r['rate']]
            if candidates:
                source = max(candidates, key=lambda r: r['rate']['last'])
                rate = source['rate']
        if rate is None:
            unknown.append(row['split']+'/'+row['weather'])
        else:
            estimates.append({'row': row, 'source': source, 'rate': rate,
                              'seconds': remaining * rate['mean'],
                              'low': remaining * rate['low'],
                              'high': remaining * rate['high']})
    return estimates, unknown


def print_frames(title, rows, now, active):
    done, total = sum(r['done'] for r in rows), sum(r['total'] for r in rows)
    print(f'\n{title}：{done}/{total}，剩余 {total-done} 帧')
    for row in rows:
        print(f"  {row['split']}/{row['weather']}: {row['done']}/{row['total']}")
    estimates, unknown = estimate_frames(rows)
    if done == total:
        print('  帧处理已完成；阶段收尾以 manifest 状态为准。')
        return
    for item in estimates:
        row, source, rate = item['row'], item['source'], item['rate']
        label = row['split']+'/'+row['weather']
        borrowed = '' if row is source else f"，借用 {source['split']}/{source['weather']} 速度"
        print(f"  {label}: 约 {rate['mean']:.1f} 秒/帧"
              f"（近期 {rate['samples']} 个间隔{borrowed}）"
              f"；剩余约 {duration(item['seconds'])}")
    if unknown:
        print('  无计时样本：'+', '.join(unknown))
    elif estimates:
        seconds = sum(e['seconds'] for e in estimates)
        low, high = sum(e['low'] for e in estimates), sum(e['high'] for e in estimates)
        print(f'  本项剩余处理时间：约 {duration(seconds)}；'
              f'近期速度外推范围 {duration(low)}～{duration(high)}')
        if active:
            print('  持续运行且速度相近时，本项预计完成：'+clock(now+seconds)+'（北京时间）')
    last = max((r['last'] for r in rows if r['last']), default=None)
    if active and last is not None:
        print('  最近一帧完成：'+clock(last)+f'，距今 {duration(now-last)}')


def report(run, window):
    protocol = read_json(run/'protocol.json')
    manifest = read_json(run/'manifest.json')
    entries = manifest['entries']
    if not isinstance(entries, dict):
        raise ValueError('manifest entries must be an object')
    indices = {}
    for split in ('train', 'validation'):
        values = protocol['base_protocol'][split+'_indices']
        if not isinstance(values, list) or not values:
            raise ValueError(split+'_indices must be a nonempty list')
        indices[split] = list(dict.fromkeys(int(x) for x in values))
        if len(indices[split]) != len(values):
            raise ValueError(split+'_indices contains duplicates')
    now = time.time()
    stage_rows = [(stage, row_for(entries, stage)) for stage in STAGES]
    live = [(stage, row) for stage, row in stage_rows
            if row.get('status') in ('in_progress', 'failed')]
    latest = max(live, key=lambda item: stamp(item[1]) or 0) if live else None
    failed = latest is not None and latest[1].get('status') == 'failed'
    print('RUN='+str(run.resolve()))
    print('读取时间：'+clock(now)+'（北京时间）')
    if latest:
        print('记录中的当前阶段：'+latest[0]+' / '+latest[1]['status'])
        if failed:
            print('该阶段已失败；以下耗时仅用于修复后规划，不代表仍在运行。')
            print('错误：'+str(latest[1].get('error', 'unknown')))
    print('这里只读取进度记录，不核验缓存哈希，也不检测服务器进程是否存活。')
    evidence = frame_progress(entries, indices, 'evidence', window)
    replay = frame_progress(entries, indices, 'replay', window)
    active_stage = latest[0] if latest and not failed else None
    print_frames('证据提取', evidence, now,
                 active_stage in ('evidence-train', 'evidence-validation'))
    groups = protocol['s4_config']['groups']
    if not isinstance(groups, list) or any(not isinstance(g, str) for g in groups):
        raise ValueError('s4_config.groups must be a list of group names')
    groups = [g for g in groups if g != 'output_only']
    models = [(task, group, row_for(entries, 'probe-fit', task, group))
              for task in ('reg', 'cls') for group in groups]
    fitted = sum(row.get('status') == 'complete' for _, _, row in models)
    epochs = int(protocol['base_protocol'].get('config', {}).get('probe_epochs', 30))
    print(f'\n线性模型训练：{fitted}/{len(models)} 个模型完成')
    for task, group, row in models:
        if row.get('status') == 'in_progress':
            count = sum(row_for(entries, 'probe-epoch', task, group, epoch).get('status')
                        == 'complete' for epoch in range(1, epochs+1))
            print(f'  {task}/{group}: {count}/{epochs} 轮；另含数据准备与校准耗时')
    print_frames('动作回放', replay, now, active_stage == 'REPLAY')
    pending = [stage for stage, row in stage_rows
               if stage != 'repair-recheck' and row.get('status') != 'complete']
    if protocol.get('repair_recheck') and row_for(entries, 'repair-recheck').get('status') != 'complete':
        pending.insert(0, 'repair-recheck')
    print('\n尚未标记完成的阶段：'+(', '.join(pending) if pending else '无'))
    if not pending:
        print('全流程：manifest 已记录全部阶段完成。')
    else:
        print('全流程剩余时间：暂无法可靠估计。上面的时间仅覆盖有帧计时样本的工作；')
        print('训练、评估、报告收尾和最终完整性检查的剩余耗时尚未计入。')
    print('估算假设持续运行；暂停、重启、共享 GPU 和天气差异都会影响速度。')


def main():
    parser = argparse.ArgumentParser(description='只读查看 S4 进度，按近期帧完成时间估算剩余耗时')
    parser.add_argument('--run', default=os.environ.get('RUN'), help='S4 日志目录，默认读取环境变量 RUN')
    parser.add_argument('--window', type=int, default=20, help='近期完成帧的采样数，默认 20，至少 4')
    args = parser.parse_args()
    if not args.run:
        parser.error('请通过 --run 指定正在运行的 S4 日志目录')
    if args.window < 4:
        parser.error('--window 至少为 4')
    try:
        report(Path(args.run), args.window)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print('无法读取进度：'+str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
