#!/usr/bin/env python3
"""Persistent, bounded Day9 stage runner; hands formal work to Day10."""
from __future__ import annotations
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from day9_env import write_json
from day9_training import evaluate, train_one, write_pilot_analysis, freeze
from run_day9 import SpaceOnly, artifact_hashes, code_hashes, hash_file, model_hash, new_model, validate_candidate
import yaml
from stable_baselines3 import SAC

STATE = ROOT/'results/pipeline_state.json'
SUMMARY = ROOT/'results/pilot_summary.json'
SCENES = ROOT/'configs/validation_scenes.json'
STAGES = ('A_validation','B_pre','B_train','B_post','F_pre','F_train','F_post')


def read(path): return json.loads(path.read_text())


def safe_write(path, data): write_json(path.relative_to(ROOT), data)


def cleanup_day9_orphans():
    """Remove only tagged Day9 simulator/ROS daemon survivors after stack stop."""
    targets = []
    world = str(ROOT/'simulations/robot_gazebo/worlds/grasp_table.sdf')
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            env = (proc/'environ').read_bytes().split(b'\0')
            if (('DAY9_ROOT='+str(ROOT)).encode() not in env or
                    b'IGN_PARTITION=day9_ubuntu' not in env or
                    b'ROS_DOMAIN_ID=91' not in env):
                continue
            cmd = (proc/'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace').strip()
            if not (cmd == 'ign gazebo -r -s '+world or
                    ('ros2-daemon --ros-domain-id 91' in cmd and
                     'ros2cli.daemon.daemonize' in cmd)):
                continue
            stat = (proc/'stat').read_text().rsplit(')',1)[1].split()
            targets.append(dict(pid=int(proc.name), start_ticks=stat[19], command=cmd))
        except (OSError, ValueError, IndexError):
            continue

    def alive(target):
        proc = Path('/proc')/str(target['pid'])
        try:
            stat = (proc/'stat').read_text().rsplit(')',1)[1].split()
            return stat[19] == target['start_ticks'] and stat[0] != 'Z'
        except (OSError, ValueError, IndexError):
            return False

    for target in targets:
        target['signals'] = []
        for sig, wait_sec in ((signal.SIGINT,3),(signal.SIGTERM,3),(signal.SIGKILL,5)):
            if not alive(target):
                break
            try:
                os.kill(target['pid'],sig)
            except ProcessLookupError:
                break
            target['signals'].append(sig.name)
            until=time.monotonic()+wait_sec
            while alive(target) and time.monotonic()<until:
                time.sleep(.1)
        target['exited'] = not alive(target)
    if targets:
        record=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                    scope='only Day9-tagged orphan Ignition server and ROS daemon',
                    processes=targets)
        with (ROOT/'runtime/orphan_cleanup_events.jsonl').open('a') as stream:
            stream.write(json.dumps(record,ensure_ascii=False)+'\n')
    if any(not target['exited'] for target in targets):
        raise RuntimeError('Day9 orphan simulator or ROS daemon survived cleanup')


def verify_review():
    review = read(ROOT/'results/review.json')
    config = ROOT/'configs/candidate.yaml'
    validate_candidate(yaml.safe_load(config.read_text()))
    if not review['ready_for_pilot'] or review['code_hashes'] != code_hashes():
        raise RuntimeError('reviewed code differs or pilot gate is closed')
    if review['candidate_sha256'] != hash_file(config) or review['artifact_hashes'] != artifact_hashes():
        raise RuntimeError('reviewed config or assets differ')
    if review['report_sha256'] != hash_file(ROOT/'REPORT.md'):
        raise RuntimeError('human review report changed')
    if review.get('automation_hashes') != {
        str(p.relative_to(ROOT)):hash_file(p) for p in sorted((ROOT/'automation').glob('*.py'))}:
        raise RuntimeError('stage automation differs from reviewed version')
    if any(hash_file(ROOT/p) != h for p,h in review['evidence_hashes'].items()):
        raise RuntimeError('reviewed evidence changed')
    return review


def start_stack():
    status=subprocess.check_output([sys.executable,str(ROOT/'manage_day9.py'),'status'],
                                   cwd=ROOT,text=True)
    if not json.loads(status):
        cleanup_day9_orphans()
    subprocess.run([sys.executable, str(ROOT/'manage_day9.py'), 'start'],
                   cwd=ROOT, check=True)


def stop_stack():
    subprocess.run([sys.executable, str(ROOT/'manage_day9.py'), 'stop'],
                   cwd=ROOT, check=True)


def restart_stack():
    stop_stack()
    status=subprocess.check_output([sys.executable,str(ROOT/'manage_day9.py'),'status'],
                                   cwd=ROOT,text=True)
    if json.loads(status):
        raise RuntimeError('owned Day9 simulator processes remain after stop')
    start_stack()


def saved_interruption(result,method):
    saved=result.get('checkpoint') or {}
    if (saved.get('status')!='INTERRUPTED' or saved.get('method')!=method or
            saved.get('steps')!=result.get('steps') or
            saved.get('config_sha256')!=hash_file(ROOT/'configs/candidate.yaml') or
            saved.get('code_hashes')!=code_hashes()):
        return False
    for name in ('model','replay','rng'):
        rel=saved.get(name)
        if not isinstance(rel,str) or not (ROOT/rel).resolve().is_relative_to(ROOT) or not (ROOT/rel).is_file():
            return False
    return True


def recovery_counts(runs, epoch_start=0):
    """Count consecutive failed recoveries, resetting after real episode progress."""
    streak=total=0
    previous_episodes=0
    for index,run in enumerate(runs):
        if index == epoch_start:
            streak=0
        episodes=int((run.get('counters') or {}).get('complete_episodes',0))
        if run.get('status') in ('RECOVERY_FAILED','INTERRUPTED') and not str(run.get('error','')).startswith('RunInterrupted:'):
            total+=1
            streak=1 if episodes>previous_episodes else streak+1
        elif episodes>previous_episodes or run.get('status')=='COMPLETE':
            streak=0
        previous_episodes=max(previous_episodes,episodes)
    if epoch_start >= len(runs):
        streak=0
    return streak,total


def stage_result_ok(stage, result):
    if stage.endswith(('validation','pre','post')):
        return result.get('status') == 'COMPLETE' and len(result.get('episodes',[])) == 20 \
            and not any(r.get('technical_failure') for r in result['episodes']) \
            and result.get('weights_unchanged') and result.get('updates') == 0
    return result.get('status') == 'COMPLETE' and result.get('steps') == 3000 \
        and result.get('updates') == 2000 and result.get('weights_changed')


def main():
    if os.environ.get('DAY9_ROOT') != str(ROOT):
        raise RuntimeError('source the isolated Day9 session environment')
    review = verify_review()
    cfg = ROOT/'configs/candidate.yaml'
    if STATE.exists():
        state = read(STATE)
        current_review_sha=hash_file(ROOT/'results/review.json')
        if state['review_sha256'] != current_review_sha:
            accepted={a['previous_review_sha256'] for a in review.get('execution_amendments',[])}
            if state['review_sha256'] not in accepted:
                raise RuntimeError('pipeline review changed without a versioned compatible amendment')
            state['review_migration']=dict(previous=state['review_sha256'],current=current_review_sha,
                reason='versioned Day10 handoff/recovery automation only; task/config/assets unchanged')
            state['review_sha256']=current_review_sha
            safe_write(STATE,state)
    else:
        state = dict(status='RUNNING',review_sha256=hash_file(ROOT/'results/review.json'),
            config_sha256=hash_file(cfg),code_hashes=code_hashes(),stages={},
            started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            pilot_wall_limit_sec=review['proposed_pilot_wall_limit_sec'],
            deadline_wall_epoch=time.time()+float(review['proposed_pilot_wall_limit_sec']),
            planned_training_steps=6000,planned_validation_slots=100,
            formal_plan={'B':dict(seed=20260911,steps=10000),'F':dict(seed=20260911,steps=10000)})
        safe_write(STATE,state)
    if state['config_sha256'] != hash_file(cfg) or state['code_hashes'] != code_hashes():
        raise RuntimeError('pipeline source changed')
    if state['status']=='DAY10_HANDED_OFF':
        print('Day10 already owns the formal workflow; inspect its own formal_state.json',flush=True)
        return
    if state['status']=='DAY9_COMPLETE_DAY10_DEFERRED':
        print('Day9 is complete and Day10 is held by the user request; no simulator started.',flush=True)
        return
    if state['status']=='DAY9_COMPLETE':
        from automation.build_day10 import build_and_start
        build_and_start(ROOT,state)
        return
    deadline=time.monotonic()+max(0,state['deadline_wall_epoch']-time.time())
    initial={}
    for method in ('B','F'):
        path=ROOT/f'models/pilot/{method}_initial_seed20260909.zip'
        if not path.exists():
            path.parent.mkdir(parents=True,exist_ok=True)
            new_model(SpaceOnly(),20260909).save(path)
        initial[method]=path
    scenes=read(SCENES)
    if len(scenes)!=20:raise RuntimeError('fixed validation list is not 20 scenes')
    if SUMMARY.exists():
        summary=read(SUMMARY)
        if summary['config_sha256'] != hash_file(cfg) or summary['code_hashes'] != code_hashes():
            raise RuntimeError('existing pilot summary does not match reviewed source')
    else:
        summary=dict(status='RUNNING',report_sha256=state['review_sha256'],
            config_sha256=hash_file(cfg),code_hashes=code_hashes(),artifact_hashes=review['artifact_hashes'],
            training_seed=20260909,validation_seed=20260910,
            wall_limit_sec=review['proposed_pilot_wall_limit_sec'],training={},validation=[],
            plan=dict(A_validation=20,B_pre=20,B_train=3000,B_post=20,F_pre=20,F_train=3000,F_post=20))
        safe_write(SUMMARY,summary)
    start_stack()
    try:
        for stage in STAGES:
            item=state['stages'].get(stage,{})
            if item.get('status')=='COMPLETE':
                if not stage_result_ok(stage,item['result']):raise RuntimeError(f'invalid completed stage {stage}')
                continue
            if time.monotonic() >= deadline-300:raise RuntimeError('declared pilot wall limit reached')
            state['current_stage']=stage;state['status']='RUNNING';safe_write(STATE,state)
            method=stage[0]
            if stage.endswith('train'):
                latest=ROOT/f'models/pilot/{method}_20260909/latest.json'
                recoveries,recovery_total=recovery_counts(
                    item.get('runs',[]),item.get('recovery_epoch_start',0))
                item['recoveries']=recoveries;item['recovery_total']=recovery_total
                state['stages'][stage]=item;safe_write(STATE,state)
                if recoveries>=3:
                    raise RuntimeError(f'{method} three consecutive infrastructure recoveries failed')
                while True:
                    remaining=deadline-time.monotonic()
                    if remaining<300:raise RuntimeError('pilot wall limit reached before training')
                    result=train_one(method,cfg,20260909,3000,remaining,stage='pilot',
                        initial=str(latest.relative_to(ROOT)) if latest.exists() else None)
                    item.setdefault('runs',[]).append(result)
                    item['result']=result;item['status']=result['status']
                    state['stages'][stage]=item;safe_write(STATE,state)
                    if result['status']=='COMPLETE':
                        expected=model_hash(SAC.load(initial[method],device='cpu'))
                        if result['initial_policy_hash']!=expected:
                            raise RuntimeError(f'{method} training initial weights differ from pre-validation')
                        if not stage_result_ok(stage,result):raise RuntimeError(f'{method} training integrity gate failed')
                        result['run_wall_sec']=result['wall_sec']
                        result['wall_sec']=sum(run['wall_sec'] for run in item['runs'])
                        result['run_ids']=[run['run'] for run in item['runs']]
                        summary['training'][method]=result;safe_write(SUMMARY,summary)
                        break
                    if result['status']=='CHECKPOINT_FAILED':
                        raise RuntimeError(f'{method} training checkpoint failed: {result}')
                    if result['status']=='RECOVERY_FAILED' and not saved_interruption(result,method):
                        raise RuntimeError(f'{method} recovery failed without a valid checkpoint: {result}')
                    if result.get('error','').startswith('RunInterrupted:'):
                        raise RuntimeError(f'{method} stopped or declared wall deadline reached; checkpoint retained')
                    if result['status']!='RECOVERY_FAILED' and not result.get('error','').startswith(('TechnicalFailure:','RunInterrupted:')):
                        raise RuntimeError(f'{method} implementation/physical error: {result["error"]}')
                    recoveries,recovery_total=recovery_counts(
                        item['runs'],item.get('recovery_epoch_start',0))
                    item['recoveries']=recoveries;item['recovery_total']=recovery_total;safe_write(STATE,state)
                    if recoveries>=3:raise RuntimeError(f'{method} three consecutive infrastructure recoveries failed')
                    restart_stack()
            else:
                label={'A_validation':'A_once','B_pre':'before','B_post':'after',
                       'F_pre':'before','F_post':'after'}[stage]
                model=(None if method=='A' else initial[method] if label=='before'
                       else ROOT/summary['training'][method]['checkpoint']['model'])
                done=item.get('episodes',[])
                if [r['scene'] for r in done] != [s['id'] for s in scenes[:len(done)]]:
                    raise RuntimeError(f'validation progress scene mismatch: {stage}')
                retries=item.get('recoveries',0)
                while len(done)<20:
                    result=evaluate(method,cfg,scenes[len(done):],
                        str(model.relative_to(ROOT)) if model else None,label,deadline)
                    if 'RunInterrupted' in str(result.get('failure')):
                        raise RuntimeError(f'{stage}: stop requested or declared deadline reached')
                    technical=[r for r in result['episodes'] if r.get('technical_failure')]
                    valid=[r for r in result['episodes'] if not r.get('technical_failure')
                           and (r.get('completed_episode') or r.get('task_reset_failure'))]
                    done+=valid
                    item.update(status='RUNNING',episodes=done,runs=item.get('runs',[])+[result],recoveries=retries)
                    state['stages'][stage]=item;safe_write(STATE,state)
                    if technical or result['status']!='COMPLETE':
                        if len(done)==20:
                            raise RuntimeError(f'{stage}: all scenes recorded but evaluation/recovery failed')
                        retries=1 if valid else retries+1
                        item['recoveries']=retries
                        item['recovery_total']=item.get('recovery_total',0)+1
                        safe_write(STATE,state)
                        if retries>=3:raise RuntimeError(f'{stage}: three infrastructure recoveries failed')
                        stop_stack();start_stack()
                    else:break
                total=dict(status='COMPLETE',method=method,label=label,episodes=done,
                    scheduled=20,successes=sum(r['success'] for r in done),
                    wall_sec=sum(r['wall_sec'] for r in item['runs']),updates=0,replay_writes=0,
                    weights_unchanged=all(r['weights_unchanged'] for r in item['runs']),
                    run=[r['run'] for r in item['runs']])
                if not stage_result_ok(stage,total):raise RuntimeError(f'{stage} integrity gate failed')
                item.update(status='COMPLETE',result=total);safe_write(STATE,state)
                summary['validation'].append(total);safe_write(SUMMARY,summary)
            state['stages'][stage]['status']='COMPLETE';safe_write(STATE,state)
        summary['training']={m:state['stages'][m+'_train']['result'] for m in ('B','F')}
        summary['validation']=[state['stages'][s]['result'] for s in STAGES if s.endswith(('validation','pre','post'))]
        summary['status']='COMPLETE';safe_write(SUMMARY,summary)
        write_pilot_analysis(summary)
        frozen=freeze(cfg)
        state['status']='DAY9_COMPLETE';state['frozen_config']=frozen;safe_write(STATE,state)
    except BaseException as exc:
        state['status']='FAILED';state['error']=repr(exc);safe_write(STATE,state)
        summary['status']='PARTIAL';summary['error']=repr(exc);safe_write(SUMMARY,summary)
        raise
    finally:
        stop_stack()
    from automation.build_day10 import build_and_start
    build_and_start(ROOT,state)


if __name__=='__main__':main()
