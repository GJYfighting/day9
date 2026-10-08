#!/usr/bin/env python3
"""Read-only Day9 progress display; only its single-bar iframe refreshes."""
from __future__ import annotations

import html
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'results/pipeline_state.json'
HTML = ROOT / 'results/live_dashboard.html'
FRAGMENT = ROOT / 'results/live_progress_fragment.html'
DATA = ROOT / 'results/live_progress_bar.json'
STAGES = ('A_validation', 'B_pre', 'B_train', 'B_post', 'F_pre', 'F_train', 'F_post')
VALIDATION_LABELS = {
    'A_validation': ('A', 'A_once'),
    'B_pre': ('B', 'before'),
    'B_post': ('B', 'after'),
    'F_pre': ('F', 'before'),
    'F_post': ('F', 'after'),
}


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def write_atomic(path, content):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(content)
    temp.replace(path)


def latest_run(method, completed_runs=()):
    paths = [p for p in (ROOT / 'results').glob(f'*_{method}_pilot_20260909_*/events.jsonl')
             if p.parent.name not in completed_runs]
    return max(paths, key=lambda p: p.stat().st_mtime) if paths else None


def training_steps(method, state):
    item = state.get('stages', {}).get(f'{method}_train', {})
    checkpoint = read(ROOT / f'models/pilot/{method}_20260909/latest.json')
    saved = int(checkpoint.get('steps', 0))
    if item.get('status') == 'COMPLETE':
        return int(item['result']['steps']), saved, int(item['result']['updates'])
    prior = item.get('runs', [])
    path = latest_run(method, {run['run'] for run in prior})
    if not path or state.get('current_stage') != f'{method}_train' or state.get('status') != 'RUNNING':
        return saved, saved, int(checkpoint.get('updates', 0))
    baseline = int(prior[-1]['steps']) if prior else 0
    with path.open() as stream:
        transitions = sum('"kind": "transition"' in line for line in stream)
    return max(saved, baseline + transitions), saved, int(checkpoint.get('updates', 0))


def active_validation_episodes(stage, state):
    if state.get('status') != 'RUNNING' or stage not in VALIDATION_LABELS:
        return 0
    method, label = VALIDATION_LABELS[stage]
    runs = list((ROOT / 'results').glob(f'*_{method}_validation_{label}_*/declaration.json'))
    if not runs:
        return 0
    declaration = max(runs, key=lambda path: path.stat().st_mtime)
    # A completed partial run is already in pipeline_state. A new declaration
    # is written only after the controller has persisted that state.
    if declaration.stat().st_mtime <= STATE.stat().st_mtime:
        return 0
    try:
        episodes = json.loads((declaration.parent / 'episodes.json').read_text())
    except (OSError, ValueError):
        return 0
    if not isinstance(episodes, list):
        return 0
    return sum(not row.get('technical_failure')
               and (row.get('completed_episode') or row.get('task_reset_failure'))
               for row in episodes if isinstance(row, dict))


def render_once():
    state = read(STATE)
    status = state.get('status', 'UNKNOWN')
    done = sum(len(state.get('stages', {}).get(s, {}).get('episodes', []))
               for s in STAGES if not s.endswith('train'))
    done += active_validation_episodes(state.get('current_stage'), state)
    b, b_saved, b_updates = training_steps('B', state)
    f, f_saved, f_updates = training_steps('F', state)
    now = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
    units = min(6100, b + f + done)
    data = dict(updated_utc=now, pipeline_status=status, stage=state.get('current_stage'),
                progress_units=units, progress_target=6100,
                validation_done=done, validation_target=100,
                B=dict(steps=b, target=3000, checkpoint_steps=b_saved, checkpoint_updates=b_updates),
                F=dict(steps=f, target=3000, checkpoint_steps=f_saved, checkpoint_updates=f_updates),
                day10_held=read(ROOT / 'results/day10_hold.json').get('hold_after_day9') is True)
    write_atomic(DATA, json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    pct = units / 61
    refresh = '<meta http-equiv="refresh" content="10">' if status == 'RUNNING' else ''
    fragment = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">{refresh}
<style>body{{font:16px system-ui,sans-serif;margin:0;color:#eef4f8;background:#101820}}
progress{{width:100%;height:28px;accent-color:#35c4a0}}
small{{display:block;color:#a9bdc9;margin-top:8px;line-height:1.5}}</style>
<label for="day9-progress">Day9：{units}/6100 进度单位（{pct:.1f}%）</label>
<progress id="day9-progress" max="6100" value="{units}">{pct:.1f}%</progress>
<small>阶段：{html.escape(str(state.get('current_stage')))} · B {b}/3000 步 · F {f}/3000 步 · 验证 {done}/100 回合<br>
状态：{html.escape(status)} · 最近检查点：B {b_saved} 步 / F {f_saved} 步 · 更新：B {b_updates} / F {f_updates}<br>
更新时间：{now}。进度单位 = 训练步数 + 验证回合数；百分比不代表剩余墙钟时间。</small></html>\n'''
    write_atomic(FRAGMENT, fragment)
    return status


def write_parent():
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Day9 实时进度</title>
<style>body{font:16px system-ui,sans-serif;max-width:760px;margin:36px auto;padding:0 18px;background:#101820;color:#eef4f8}
h1{font-size:1.5rem}iframe{width:100%;height:170px;border:0}</style>
<h1>Day9 实时进度</h1>
<iframe title="Day9 单一进度条" src="live_progress_fragment.html"></iframe>
<p>页面本身不刷新；仅进度条区域每 10 秒读取一次最新实验记录。Day10 已按要求暂缓。</p></html>\n'''
    write_atomic(HTML, page)


if __name__ == '__main__':
    write_parent()
    while True:
        status = render_once()
        if status in ('FAILED', 'DAY9_COMPLETE_DAY10_DEFERRED', 'DAY10_HANDED_OFF'):
            break
        time.sleep(10)
