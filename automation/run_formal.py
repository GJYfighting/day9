#!/usr/bin/env python3
"""Day10-owned, one-seed-per-method formal run with bounded recovery."""
from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from day9_env import write_json
from day9_training import train_one
from run_day9 import artifact_hashes, code_hashes, hash_file

STATE=ROOT/'results/formal_state.json'
SUMMARY=ROOT/'results/formal_summary.json'
MANIFEST=ROOT/'configs/frozen_manifest.json'
CONFIG=ROOT/'configs/frozen.yaml'
JOBS=('B','F')


def read(path):return json.loads(path.read_text())


def save(path,data):write_json(path.relative_to(ROOT),data)


def record_supervisor():
    path=ROOT/'runtime/formal_process.json'
    def ticks(pid):
        try:return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
        except OSError:return None
    if path.exists():
        old=read(path)
        if old.get('pid')!=os.getpid() and ticks(old.get('pid'))==old.get('start_ticks'):
            raise RuntimeError('another Day10 formal supervisor is active')
        previous=ROOT/'runtime'/f'formal_process_previous_{time.time_ns()}.json'
        previous.write_text(json.dumps(old,indent=2)+'\n')
    save(path,dict(pid=os.getpid(),start_ticks=ticks(os.getpid()),
        command=['/usr/bin/python3','-u',str(ROOT/'automation/run_formal.py')],
        log=str(ROOT/'runtime/logs/day10_formal_pipeline.log'),
        root=str(ROOT),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        safe_stop='SIGTERM at a safe checkpoint, then source session_env.sh and run manage_day10.py stop',
        resume='source session_env.sh and run python3 -u automation/run_formal.py'))


def stack(action):
    subprocess.run([sys.executable,str(ROOT/'manage_day10.py'),action],cwd=ROOT,check=True)


def restart_stack():
    stack('stop')
    status=subprocess.check_output([sys.executable,str(ROOT/'manage_day10.py'),'status'],
                                   cwd=ROOT,text=True)
    if json.loads(status):
        raise RuntimeError('owned Day10 simulator processes remain after stop')
    stack('start')


def saved_interruption(result,method):
    saved=result.get('checkpoint') or {}
    if (saved.get('status')!='INTERRUPTED' or saved.get('method')!=method or
            saved.get('steps')!=result.get('steps') or
            saved.get('config_sha256')!=hash_file(CONFIG) or
            saved.get('code_hashes')!=code_hashes()):
        return False
    for name in ('model','replay','rng'):
        rel=saved.get(name)
        if not isinstance(rel,str) or not (ROOT/rel).resolve().is_relative_to(ROOT) or not (ROOT/rel).is_file():
            return False
    return True


def recovery_counts(runs):
    """Only failures without intervening completed episodes are consecutive."""
    streak=total=0
    previous_episodes=0
    for run in runs:
        episodes=int((run.get('counters') or {}).get('complete_episodes',0))
        if run.get('status') in ('RECOVERY_FAILED','INTERRUPTED') and not str(run.get('error','')).startswith('RunInterrupted:'):
            total+=1
            streak=1 if episodes>previous_episodes else streak+1
        elif episodes>previous_episodes or run.get('status')=='COMPLETE':
            streak=0
        previous_episodes=max(previous_episodes,episodes)
    return streak,total


def verify():
    if os.environ.get('DAY10_ROOT')!=str(ROOT):
        raise RuntimeError('source the Day10 environment')
    frozen=read(MANIFEST)
    if frozen['config_sha256']!=hash_file(CONFIG) or frozen['code_hashes']!=code_hashes():
        raise RuntimeError('Day10 frozen code/config mismatch')
    if frozen['artifact_hashes']!=artifact_hashes():
        raise RuntimeError('Day10 copied dependency/model/scene mismatch')
    if frozen['automation_sha256']!=hash_file(ROOT/'automation/run_formal.py'):
        raise RuntimeError('Day10 formal runner changed')
    if any(hash_file(ROOT/p)!=h for p,h in frozen['resource_hashes'].items()):
        raise RuntimeError('Day10 mesh resource changed')
    if frozen['formal_steps']!=20000 or frozen['methods']!=['B','F'] or frozen['seed']!=20260911:
        raise RuntimeError('formal job budget mismatch')
    independent=read(ROOT/'provenance/independence.json')
    if independent['status']!='PASS':raise RuntimeError('Day10 independence gate failed')
    return frozen


def valid_complete(method,result):
    return (result.get('status')=='COMPLETE' and result.get('method')==method
            and result.get('seed')==20260911 and result.get('steps')==10000
            and result.get('updates')==9000 and result.get('weights_changed')
            and result.get('checkpoint',{}).get('status')=='COMPLETE')


def main():
    frozen=verify()
    record_supervisor()
    forecast_path=ROOT/'provenance/results__formal_forecast_post_pilot.json'
    forecast=read(forecast_path) if forecast_path.exists() else {}
    if STATE.exists():
        state=read(STATE)
        if state['manifest_sha256']!=hash_file(MANIFEST):
            raise RuntimeError('existing formal jobs have another manifest')
    else:
        state=dict(status='RUNNING',manifest_sha256=hash_file(MANIFEST),
            config_sha256=hash_file(CONFIG),seed=20260911,steps_per_method=10000,
            jobs={},started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            estimate_source='Day9 true training window after pilot; course/episode mix may change')
        save(STATE,state)
    try:
        for method in JOBS:
            job=state['jobs'].get(method,{})
            if job.get('status')=='COMPLETE':
                if not valid_complete(method,job['result']):
                    raise RuntimeError(f'invalid recorded completed formal job: {method}')
                continue
            if shutil.disk_usage(ROOT).free < 2*1024**3:
                raise RuntimeError('less than 2 GiB disk free before job')
            estimate=forecast.get(method,{}).get('formal_seconds_per_seed')
            # A declared finite safety stop; the Day9 point estimate is not the
            # cutoff. No automatic time or environment-step budget extension.
            wall_limit=max(21600,2*float(estimate or 36000))
            job.setdefault('runs',[])
            job.update(status='RUNNING',seed=20260911,steps=10000,
                       wall_limit_sec=wall_limit,forecast_seconds=estimate,
                       estimated_remaining_sec=estimate)
            job.setdefault('deadline_wall_epoch',time.time()+wall_limit)
            state['jobs'][method]=job;state['current_method']=method;save(STATE,state)
            print(json.dumps(dict(event='formal_job_start',method=method,seed=20260911,
                steps=10000,config=str(CONFIG),manifest_sha256=state['manifest_sha256'],
                initial='fresh model and empty replay' if not job['runs'] else 'resume checkpoint',
                estimated_seconds=estimate,wall_limit_sec=wall_limit)),flush=True)
            stack('start')
            try:
                recoveries,recovery_total=recovery_counts(job['runs'])
                job['recoveries']=recoveries;job['recovery_total']=recovery_total
                save(STATE,state)
                if recoveries>=3:
                    raise RuntimeError(f'{method}: three consecutive recovery attempts exhausted')
                while True:
                    remaining=job['deadline_wall_epoch']-time.time()
                    if remaining<300:raise RuntimeError(f'{method} declared wall safety limit reached')
                    latest=ROOT/f'models/formal/{method}_20260911/latest.json'
                    if latest.exists() and not job['runs']:
                        raise RuntimeError(f'{method} checkpoint exists without matching job state; inspect before resume')
                    if job['runs'] and not latest.exists():
                        raise RuntimeError(f'{method} prior run has no recoverable checkpoint')
                    initial=str(latest.relative_to(ROOT)) if latest.exists() else None
                    result=train_one(method,CONFIG,20260911,10000,remaining,
                                     stage='formal',initial=initial)
                    job['runs'].append(result)
                    job['result']=result;job['status']=result['status']
                    save(STATE,state)
                    if valid_complete(method,result):
                        job['status']='COMPLETE';job['actual_wall_sec']=sum(x['wall_sec'] for x in job['runs'])
                        save(STATE,state);break
                    if result['status']=='COMPLETE':
                        raise RuntimeError(f'{method} completed with wrong budget/update/state')
                    if result['status']=='CHECKPOINT_FAILED':
                        raise RuntimeError(f'{method} checkpoint failed: {result["error"]}')
                    if result['status']=='RECOVERY_FAILED' and not saved_interruption(result,method):
                        raise RuntimeError(f'{method} recovery failed without a valid checkpoint: {result["error"]}')
                    if result.get('error','').startswith('RunInterrupted:'):
                        raise RuntimeError(f'{method} stopped or declared wall limit reached; checkpoint retained')
                    if result['status']!='RECOVERY_FAILED' and not result.get('error','').startswith(('TechnicalFailure:','RunInterrupted:')):
                        raise RuntimeError(f'{method} implementation/physical fault: {result["error"]}')
                    recoveries,recovery_total=recovery_counts(job['runs'])
                    job['recoveries']=recoveries;job['recovery_total']=recovery_total
                    save(STATE,state)
                    if recoveries>=3:raise RuntimeError(f'{method}: three consecutive recovery attempts exhausted')
                    restart_stack()
            finally:
                stack('stop')
        state['status']='COMPLETE';state.pop('current_method',None)
        save(STATE,state)
        summary=dict(status='COMPLETE',seed=20260911,methods={},
            total_steps=20000,total_updates=18000,
            one_training_seed_per_method=True,
            independent_final_test='not performed or claimed',
            wall_sec=sum(state['jobs'][m]['actual_wall_sec'] for m in JOBS),
            provenance='provenance/source.json',manifest='configs/frozen_manifest.json')
        for method in JOBS:
            r=state['jobs'][method]['result']
            summary['methods'][method]=dict(steps=r['steps'],updates=r['updates'],
                complete_episodes=r['counters']['complete_episodes'],
                task_reset_failures=r['counters']['task_reset_failures'],
                successes=r['counters']['successes'],
                grasp_attempts=r['counters']['grasp_attempts'],
                wall_sec=state['jobs'][method]['actual_wall_sec'],
                model=r['checkpoint']['model'],checkpoint=r['checkpoint'],
                runs=[x['run'] for x in state['jobs'][method]['runs']])
        save(SUMMARY,summary)
    except BaseException as exc:
        state['status']='FAILED';state['error']=repr(exc);save(STATE,state)
        raise


if __name__=='__main__':main()
