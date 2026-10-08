#!/usr/bin/env python3
"""Read-only observer for the detached Day9 workflow."""
from pathlib import Path
import json
import os
import shutil
import time

ROOT=Path(__file__).resolve().parents[1]
PROCESS=ROOT/'runtime/pipeline_process.json'
STATE=ROOT/'results/pipeline_state.json'
OUTPUT=ROOT/'results/live_progress.json'


def read(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,ValueError):return None


def identity(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
    except OSError:return None


def main():
    proc=read(PROCESS)
    if proc is None:raise SystemExit('missing owned process record')
    while True:
        state=read(STATE) or {}
        stage=state.get('current_stage')
        snap=dict(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            pipeline_status=state.get('status'),stage=stage,
            parent_process_alive=identity(proc['pid'])==proc['start_ticks'],
            disk_free_bytes=shutil.disk_usage(ROOT).free,
            stage_results={k:v.get('status') for k,v in state.get('stages',{}).items()})
        if stage and stage.endswith(('validation','pre','post')):
            method=stage[0]
            label={'A_validation':'A_once','B_pre':'before','B_post':'after',
                'F_pre':'before','F_post':'after'}[stage]
            files=sorted((ROOT/'results').glob(f'*_{method}_validation_{label}_*/episodes.json'),
                         key=lambda p:p.stat().st_mtime)
            if files:
                rows=read(files[-1]) or []
                snap['current_validation']=dict(file=str(files[-1].relative_to(ROOT)),
                    recorded=len(rows),scheduled=20,successes=sum(bool(x.get('success')) for x in rows),
                    task_reset_failures=sum(bool(x.get('task_reset_failure')) for x in rows),
                    technical_failures=sum(bool(x.get('technical_failure')) for x in rows))
        elif stage and stage.endswith('train'):
            method=stage[0]
            latest=read(ROOT/f'models/pilot/{method}_20260909/latest.json')
            files=sorted((ROOT/'results').glob(f'*_{method}_pilot_20260909_*/progress.json'),
                         key=lambda p:p.stat().st_mtime)
            snap['current_training']=dict(method=method,seed=20260909,target_steps=3000,
                checkpoint=latest,progress=read(files[-1]) if files else None)
        temp=OUTPUT.with_suffix('.json.tmp')
        temp.write_text(json.dumps(snap,ensure_ascii=False,indent=2)+'\n')
        temp.replace(OUTPUT)
        if state.get('status') in ('DAY10_HANDED_OFF','FAILED') or not snap['parent_process_alive']:
            return
        time.sleep(60)


if __name__=='__main__':main()
