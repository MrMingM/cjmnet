#!/usr/bin/env python3
"""Read-only ETA monitor. Standard library only; no research modules imported.

Usage: python -B monitor_observability_eta.py --run /data/.../observability_audit_...
Timing comes from newly observed progress boundaries, never from invented log
timestamps. H3/H2 timing ratios are explicitly heuristic until H3 is measured.
The final source/data hash verification has no progress markers and no ETA.
"""
import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
import re
import statistics
import time


H2_KEYS = [f'{m}/{v}' for m in ('geo', 'gspr')
           for v in ('raw', 'count_normalized', 'shape')]
KEYS = H2_KEYS + ['H3/geo', 'H3/gspr']
MARKER = re.compile(r'^H[23] analyzed (\S+) (?:rows|pairs)=\s*(\d+)')


def read_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def duration(seconds):
    if seconds is None or not math.isfinite(seconds):
        return '未知'
    seconds = max(0, round(seconds))
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    return f'{h}小时{m:02d}分' if h else f'{m}分{s:02d}秒'


def progress(text):
    # A resume may append another complete pass. Only use its newest cohort.
    start = text.rfind('Analysis cohorts ')
    segment = text[start:] if start >= 0 else text
    done = []
    for line in segment.splitlines():
        match = MARKER.match(line)
        if match:
            key = ('H3/' if line.startswith('H3 ') else '') + match[1]
            if key in KEYS and key not in done:
                done.append(key)
    return start, done, segment


def process_sample(pid):
    """Linux /proc only. Process-local CPU usage; no subprocess or signals."""
    if pid is None or os.name != 'posix':
        return None
    try:
        text = Path(f'/proc/{pid}/stat').read_text()
        fields = text[text.rfind(')') + 2:].split()
        return {'state': fields[0],
                'cpu': (int(fields[11]) + int(fields[12])) / os.sysconf('SC_CLK_TCK'),
                'start': fields[19]}
    except FileNotFoundError:
        return {'state': 'gone'}
    except (OSError, ValueError, IndexError):
        return None


class Tracker:
    def __init__(self, poll_interval):
        self.interval = poll_interval
        self.cycle = None
        self.last_count = 0
        self.anchor_time = None
        self.anchor_count = None
        self.h2_times = []
        self.h3_times = []

    def observe(self, cycle, done, text, log_mtime, now):
        if cycle != self.cycle or len(done) < self.last_count:
            self.cycle = cycle
            self.last_count = len(done)
            self.h2_times.clear()
            self.h3_times.clear()
            self.anchor_time = None
            self.anchor_count = None
            # Existing last-line marker was written close to file mtime.
            # Earlier markers have no recoverable timestamps.
            lines = text.rstrip().splitlines()
            if lines and MARKER.match(lines[-1]) and done and log_mtime <= now:
                self.anchor_time = log_mtime
                self.anchor_count = len(done)
            return
        if len(done) > self.last_count:
            # Only a single newly completed group can supply a clean interval.
            # Multiple markers within a poll cannot be individually timed.
            if (len(done) == self.last_count + 1 and
                    self.anchor_count == self.last_count and
                    self.anchor_time is not None):
                elapsed = now - self.anchor_time
                if elapsed > 2 * self.interval:
                    key = done[-1]
                    (self.h3_times if key.startswith('H3/') else self.h2_times).append(elapsed)
            self.anchor_time = now
            self.anchor_count = len(done)
            self.last_count = len(done)

    def estimate(self, done, now):
        if not self.h2_times and not self.h3_times:
            return None
        left = [key for key in KEYS if key not in done]
        if not left:
            return None
        elapsed = max(0, now - self.anchor_time) if self.anchor_time is not None else 0
        bounds = []
        for i, key in enumerate(left):
            samples = self.h3_times if key.startswith('H3/') else self.h2_times
            if samples:
                low = min(samples) * .7
                high = max(samples) * 1.5
            elif key.startswith('H3/') and self.h2_times:
                # H3 includes 100 per-pair matrix permutations. Its cost is
                # not equal to H2; keep an explicit broad heuristic range.
                low = statistics.median(self.h2_times) * .5
                high = max(self.h2_times) * 3
            else:
                return None
            if i == 0:
                if elapsed > high:
                    return {'overdue': True, 'elapsed': elapsed}
                low, high = max(0, low-elapsed), max(0, high-elapsed)
            bounds.append((low, high))
        return {'overdue': False, 'low': sum(x[0] for x in bounds),
                'high': sum(x[1] for x in bounds), 'h3_measured': bool(self.h3_times)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--interval', type=float, default=30,
                        help='Read-only polling interval in seconds (default 30).')
    parser.add_argument('--once', action='store_true', help='One status snapshot; timing may be unavailable.')
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval < 1:
        parser.error('--interval must be finite and >= 1')
    run = args.run.expanduser().resolve()
    log = run / 'driver.log'
    if not run.is_dir() or not log.is_file():
        parser.error('RUN must be an existing audit directory containing driver.log')
    tracker = Tracker(args.interval)
    old_cpu = None
    initial_identity = None
    print('只读监控：不修改实验文件，不使用 GPU，不导入实验代码。', flush=True)
    print('ETA 为统计计算的粗略区间；结束后的源码/数据哈希校验耗时另计。', flush=True)
    print('已写入的旧日志没有逐行时间戳；需观测新的完整分组耗时才能估计。', flush=True)
    while True:
        now = time.time()
        try:
            text = log.read_text(encoding='utf-8', errors='replace')
            modified = log.stat().st_mtime
        except OSError as exc:
            print(f'日志读取失败：{exc}', flush=True)
            return 2
        manifest = read_json(run / 'manifest.json')
        cycle, done, segment = progress(text)
        tracker.observe(cycle, done, segment, modified, now)
        stamp = dt.datetime.now().astimezone().isoformat(timespec='seconds')
        print(f'\n[{stamp}] stage={manifest.get("stage", "未知")} | '
              f'H2={sum(k in done for k in H2_KEYS)}/6 | '
              f'H3={sum(k.startswith("H3/") for k in done)}/2', flush=True)
        if manifest.get('stage') == 'failed':
            failure = read_json(run / 'diagnostics' / 'failure.json')
            print('任务记录了失败：' + str(failure.get('message', '请查看 driver.log')), flush=True)
            return 2
        if manifest.get('stage') == 'complete':
            print('任务已完成。结论：' + str(manifest.get('decision', {}).get('verdict', '见 final_results.json')), flush=True)
            return 0
        try:
            pid = int((run / 'pid').read_text().strip())
        except (OSError, ValueError):
            pid = None
        sample = process_sample(pid)
        if sample and sample['state'] == 'gone':
            print('记录的进程已退出，实验尚未标记完成；检查 driver.log。', flush=True)
            return 2
        if sample and sample['state'] == 'Z':
            print('进程已成为僵尸进程，实验尚未标记完成；检查 driver.log。', flush=True)
            return 2
        if sample and 'start' in sample:
            identity = (pid, sample['start'])
            if initial_identity is not None and identity != initial_identity:
                print('进程身份发生变化；请核实是否恢复运行，再重启监控。', flush=True)
                return 2
            initial_identity = identity
            if old_cpu:
                wall = now-old_cpu[0]
                pct = 100*(sample['cpu']-old_cpu[1])/wall if wall > 0 else 0
                print(f'PID={pid} | 最近 CPU 使用率约 {pct:.1f}% | 状态={sample["state"]}', flush=True)
            old_cpu = (now, sample['cpu'])
        if cycle < 0:
            print('尚在准备或提取数据；本脚本的统计 ETA 从 Analysis cohorts 阶段开始。', flush=True)
        elif len(done) == len(KEYS):
            print('H1/H2/H3 计算已结束，正在结束校验/写报告；该阶段暂无可靠 ETA。', flush=True)
        else:
            stage = 'H1 或首个 H2 分组' if not done else next(k for k in KEYS if k not in done)
            print('当前统计阶段：' + stage, flush=True)
            eta = tracker.estimate(done, now)
            print(f'已观测完整计时间隔：H2 {len(tracker.h2_times)} 个，H3 {len(tracker.h3_times)} 个。', flush=True)
            if eta is None:
                print('统计 ETA：采集计时中，尚无法可靠估计。保持监控，出现足够的新进度后自动更新。', flush=True)
            elif eta['overdue']:
                print('当前分组已超过已有耗时范围，原 ETA 失效；等待新进度重新估计。', flush=True)
            else:
                print(f'统计剩余时间粗估：{duration(eta["low"])} ～ {duration(eta["high"])}。', flush=True)
                if not eta['h3_measured']:
                    print('H3 尚未实测，采用 H2 单组耗时的 0.5～3 倍作启发式范围；这不是置信区间。', flush=True)
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\n监控已停止。')
