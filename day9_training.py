"""Fresh SAC training with explicit physical-transition accounting.

The manual one-step collector uses SB3 policy scaling, ReplayBuffer and SAC.train.
It avoids Gym reset failures discarding the terminal transition during autoreset.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import pickle
import random
import shutil
import signal
import time

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure
from day9_env import ROOT, Day9Env, Trace, clean, local, write_json, pause_simulation, RunInterrupted
from day8_env import TechnicalFailure


def add_transition(model, obs, action, next_obs, reward, terminated, truncated):
    scaled=model.policy.scale_action(np.asarray(action,np.float32))
    if not np.isfinite(scaled).all() or np.any(np.abs(scaled)>1.000001):
        raise ValueError('invalid scaled replay action')
    model.replay_buffer.add(np.asarray(obs,np.float32)[None],np.asarray(next_obs,np.float32)[None],
        scaled[None],np.array([reward],np.float32),np.array([terminated or truncated],np.float32),
        [dict(TimeLimit_truncated=truncated and not terminated, **{'TimeLimit.truncated':truncated and not terminated})])


def update_probe():
    from run_day9 import new_model, SpaceOnly, model_hash, budget_state, hash_file, consume_budget
    b=budget_state()
    if b.get('probe_updates',0):raise RuntimeError('Probe budget already used; retain existing evidence')
    declaration=dict(purpose='isolated optimizer/replay/save-load interface probe; not pilot learning',
        environment_steps=0,updates_per_method=10,total_updates=20,batch_size=128,
        sampling='SB3 replacement sampling from measured real transitions; small buffer explicitly allowed for this interface check',
        paired_models_mutated=False,seed=20260908,wall_limit_sec=180,
        stop='nonfinite loss/parameters, missing real data, any budget exceeded')
    write_json('results/update_probe_declaration.json',declaration)
    print(json.dumps(declaration),flush=True)
    deadline=min(time.monotonic()+180,b['stop_measurement_monotonic']);results=[]
    for method in ['B','F']:
        records=[];sources=[]
        for p in sorted((ROOT/'results').glob(f'*_{method}_*/events.jsonl')):
            rows=[json.loads(s) for s in p.read_text().splitlines()]
            values=[r for r in rows if r['kind']=='transition']
            if values:records.extend(values);sources.append(dict(path=str(p.relative_to(ROOT)),sha256=hash_file(p)))
        if not records:raise RuntimeError('No real transitions for '+method)
        model=new_model(SpaceOnly(),20260908)
        model.set_logger(configure(str(ROOT/f'results/update_probe_{method}'),['csv']))
        for r in records:add_transition(model,r['observation'],r['policy_action'],r['next_observation'],r['reward'],r['terminated'],r['truncated'])
        before=model_hash(model);start=time.monotonic()
        losses=[]
        for _ in range(10):
            if time.monotonic()>=deadline:raise TechnicalFailure('probe wall budget exceeded')
            model.train(gradient_steps=1,batch_size=128)
            stats=dict(model.logger.name_to_value)
            if any(not np.isfinite(float(v)) for v in stats.values() if isinstance(v,(float,int))):
                raise TechnicalFailure('nonfinite probe metrics')
            if any(not torch.isfinite(p).all() for p in model.policy.parameters()):raise TechnicalFailure('nonfinite probe weights')
            losses.append(clean(stats));model.logger.dump(step=model._n_updates)
            consume_budget('probe')
        elapsed=time.monotonic()-start
        path=ROOT/f'models/diagnostic/probe_{method}.zip';model.save(path)
        replay=ROOT/f'models/diagnostic/probe_{method}_replay.pkl';model.save_replay_buffer(replay)
        loaded=SAC.load(path,device='cpu');loaded.load_replay_buffer(replay)
        result=dict(method=method,real_transitions=len(records),sources=sources,updates=model._n_updates,
            weights_changed=before!=model_hash(model),saved_loaded_equal=model_hash(model)==model_hash(loaded),
            replay_count=loaded.replay_buffer.size(),wall_sec=elapsed,wall_sec_per_update=elapsed/10,
            metrics=losses,training_steps=0)
        if not result['weights_changed'] or not result['saved_loaded_equal']:raise AssertionError('probe update/roundtrip failed')
        results.append(result);write_json('results/update_probe.json',dict(status='PASS' if len(results)==2 else 'PARTIAL',results=results))
    print(json.dumps(clean(results),indent=2))


def checkpoint(model, env, directory, status, counters):
    from run_day9 import code_hashes, hash_file
    directory=local(directory);directory.mkdir(parents=True,exist_ok=True)
    label=f'step_{model.num_timesteps:06d}_{time.time_ns()}'
    model_path=directory/(label+'.zip');replay_path=directory/(label+'_replay.pkl')
    start=time.monotonic();model.save(model_path);model.save_replay_buffer(replay_path)
    random_state=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                      action_space=env.action_space.np_random.bit_generator.state)
    rng_path=directory/(label+'_rng.pkl')
    with rng_path.open('xb') as stream:pickle.dump(random_state,stream)
    state=dict(status=status,steps=model.num_timesteps,updates=model._n_updates,
        method=env.method,seed=env.run_seed,config_sha256=hash_file(env.config_path),code_hashes=code_hashes(),
        model=str(model_path.relative_to(ROOT)),replay=str(replay_path.relative_to(ROOT)),
        rng=str(rng_path.relative_to(ROOT)),curriculum=env.manager.state(),next_attempt=env.attempt,
        counters=counters,initial_policy_hash=getattr(env,'initial_policy_hash',None),
        resume_semantics='continue optimizer/replay/RNG/curriculum at a new safely reset physical episode; not bitwise simulator continuation')
    write_json(directory/'latest.json',state)
    env.trace.emit('checkpoint',state=state,wall_sec=time.monotonic()-start)
    return state


def train_one(method, config, seed, steps, wall_limit, *, stage, initial=None):
    from run_day9 import new_model, resolved_model, code_hashes, model_hash, technical_error, validate_candidate, resolved_execution
    config=local(config);c=__import__('yaml').safe_load(config.read_text())
    validate_candidate(c)
    if wall_limit < 300: raise ValueError('wall limit must reserve at least 120 seconds for recovery')
    run=f'{time.strftime("%Y%m%dT%H%M%S",time.gmtime())}_{method}_{stage}_{seed}_{time.time_ns()}'
    trace=Trace(run);env=None;model=None;start=time.monotonic();deadline=start+wall_limit
    counters=dict(complete_episodes=0,successes=0,grasp_attempts=0,reset_failures=0,
                  task_reset_failures=0,technical_failures=0)
    status='RUNNING';error=None;saved=None;initial_hash=None
    declaration=dict(purpose=stage,method=method,seed=seed,steps=steps,wall_limit_sec=wall_limit,
        config=c,code_hashes=code_hashes(),fresh=initial is None,initial=initial,
        resolved_execution=resolved_execution(method,'train',c['day9']['profile'],c),
        max_task_reset_failures_consecutive=20,max_scene_attempts=2*steps,max_technical_retries=1,
        validation_steps=0,update_budget=max(0,steps-1000),
        checkpoint_every_steps=200,checkpoint_max_wall_sec=1800)
    write_json(f'results/{run}/declaration.json',declaration);print(json.dumps(declaration),flush=True)
    def stop(*_):raise RunInterrupted('training interrupted or declared wall deadline reached')
    handlers={s:signal.signal(s,stop) for s in [signal.SIGALRM,signal.SIGINT,signal.SIGTERM]}
    signal.setitimer(signal.ITIMER_REAL,max(.01,wall_limit-120))
    try:
        env=Day9Env(method,config,trace,phase='train',profile=c['day9']['profile'],seed=seed)
        model=new_model(env,seed)
        model.set_logger(configure(str(ROOT/f'results/{run}/sac'),['csv']))
        if initial is not None:
            state=json.loads(local(initial).read_text())
            from run_day9 import hash_file
            if state['method']!=method or state['seed']!=seed or state['config_sha256']!=hash_file(config) or state['code_hashes']!=code_hashes():
                raise ValueError('resume method/seed/config/code differs from saved state')
            model=SAC.load(local(state['model']),env=env,device='cpu')
            model.load_replay_buffer(local(state['replay']))
            model.set_logger(configure(str(ROOT/f'results/{run}/sac'),['csv']))
            with local(state['rng']).open('rb') as stream:rng=pickle.load(stream)
            random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch'])
            env.action_space.np_random.bit_generator.state=rng['action_space']
            env.manager.restore(state['curriculum']);env.attempt=state['next_attempt'];counters=state['counters']
        initial_hash=state.get('initial_policy_hash') if initial is not None else model_hash(model)
        if initial_hash is None: raise ValueError('checkpoint lacks original policy hash')
        env.initial_policy_hash=initial_hash
        write_json(f'results/{run}/resolved.json',resolved_model(model))
        obs=None;reset_streak=0;technical_retries=0;episode=None
        last_checkpoint_step=model.num_timesteps
        last_checkpoint_wall=time.monotonic()
        while model.num_timesteps<steps:
            if time.monotonic()>=deadline-120:raise TechnicalFailure('training wall budget reached')
            if env.attempt>=2*steps:raise TechnicalFailure('scene attempt budget reached')
            if obs is None:
                try:
                    obs,_=env.reset();reset_streak=0
                    episode=dict(attempt=env.attempt,level=env.manager.label,steps=0,return_=0.,
                        grasp_attempts=0,reward_terms={k:0. for k in ['approach_reward','success_reward','drop_penalty','collision_penalty']})
                except Exception as exc:
                    counters['reset_failures']+=1;reset_streak+=1
                    ok,detail=env.base.backend.safe_recover()
                    if not ok:raise TechnicalFailure('reset recovery failed: '+detail) from exc
                    if technical_error(exc):
                        counters['technical_failures']+=1
                        if technical_retries>=1:raise
                        technical_retries+=1;env.attempt-=1
                    else:
                        counters['task_reset_failures']=counters.get('task_reset_failures',0)+1
                        if reset_streak>=20:raise TechnicalFailure('20 consecutive task initialization failures') from exc
                    continue
            with trace.span('inference'):
                action=env.action_space.sample() if model.num_timesteps<model.learning_starts else model.predict(obs,deterministic=False)[0]
            next_obs,reward,terminated,truncated,info=env.step(action)
            add_transition(model,obs,action,next_obs,reward,terminated,truncated)
            model.num_timesteps+=1
            model._total_timesteps=steps
            model._current_progress_remaining=1-model.num_timesteps/steps
            counters['grasp_attempts']+=int(info['grasp_attempted'])
            episode['steps']+=1;episode['return_']+=reward
            episode['grasp_attempts']+=int(info['grasp_attempted'])
            for key in episode['reward_terms']:episode['reward_terms'][key]+=float(info[key])
            if model.num_timesteps>model.learning_starts:
                with trace.span('network_update'):model.train(gradient_steps=1,batch_size=128)
                metrics=dict(model.logger.name_to_value)
                if any(not np.isfinite(float(v)) for v in metrics.values() if isinstance(v,(float,int))):
                    raise TechnicalFailure('nonfinite SAC training metrics')
                if any(not torch.isfinite(p).all() for p in model.policy.parameters()):raise TechnicalFailure('nonfinite network')
                if model.log_ent_coef is not None and not torch.isfinite(model.log_ent_coef).all():
                    raise TechnicalFailure('nonfinite entropy coefficient state')
                trace.emit('network_update',steps=model.num_timesteps,updates=model._n_updates,metrics=metrics)
                model.logger.dump(step=model.num_timesteps)
            if terminated or truncated:
                counters['complete_episodes']+=1;counters['successes']+=int(info['success'])
                episode.update(success=bool(info['success']),terminated=terminated,truncated=truncated,
                    failure_reason=info['termination_reason'],level_after=env.manager.label)
                trace.emit('training_episode',**episode)
                with (trace.path.parent/'training_episodes.jsonl').open('a') as stream:
                    stream.write(json.dumps(clean(episode),ensure_ascii=False)+'\n')
                model._episode_num=counters['complete_episodes'];obs=None;model._last_obs=None
            else:obs=next_obs;model._last_obs=next_obs[None].copy()
            # Checkpoint at the next episode boundary. A partial episode remains
            # in replay if interrupted; resumption starts a new physical reset.
            checkpoint_due=(model.num_timesteps-last_checkpoint_step>=200 or
                            time.monotonic()-last_checkpoint_wall>=1800)
            if obs is None and checkpoint_due:
                free=shutil.disk_usage(ROOT).free
                trace.emit('disk_check',free_bytes=free)
                if free<2*1024**3:raise TechnicalFailure('less than 2 GiB disk free')
                with trace.span('checkpoint_write'):
                    saved=checkpoint(model,env,f'models/{stage}/{method}_{seed}','RUNNING',counters)
                elapsed=time.monotonic()-start
                write_json(f'results/{run}/progress.json',dict(status='RUNNING',method=method,
                    seed=seed,steps=model.num_timesteps,updates=model._n_updates,
                    episodes=counters['complete_episodes'],level=env.manager.label,
                    elapsed_wall_sec=elapsed,
                    estimated_remaining_sec=(steps-model.num_timesteps)*elapsed/max(1,model.num_timesteps),
                    estimate_basis='own full training window; current course and episode mix may change',
                    checkpoint=saved['model']))
                last_checkpoint_step=model.num_timesteps
                last_checkpoint_wall=time.monotonic()
        status='COMPLETE'
    except BaseException as exc:
        error=f'{type(exc).__name__}: {exc}';status='INTERRUPTED';trace.emit('training_exception',error=error)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        if model is not None and env is not None:
            try:saved=checkpoint(model,env,f'models/{stage}/{method}_{seed}',status,counters)
            except Exception as exc:trace.emit('checkpoint_failure',error=repr(exc));status='CHECKPOINT_FAILED'
        if env is not None:
            signal.setitimer(signal.ITIMER_REAL,max(.01,min(120,deadline-time.monotonic())))
            try:env.close()
            except BaseException as exc:
                trace.emit('close_failure',error=repr(exc));status='RECOVERY_FAILED'
                try:pause_simulation()
                except Exception:pass
        signal.setitimer(signal.ITIMER_REAL,0)
        for s,h in handlers.items():signal.signal(s,h)
        report=dict(run=run,status=status,error=error,method=method,seed=seed,
            steps=0 if model is None else model.num_timesteps,updates=0 if model is None else model._n_updates,
            counters=counters,wall_sec=time.monotonic()-start,timings=trace.snapshot(),checkpoint=saved,
            initial_policy_hash=initial_hash,
            weights_changed=model is not None and initial_hash is not None and model_hash(model)!=initial_hash,
            includes=['startup','reset','randomization','grasp','network_update','checkpoint','safe_close'])
        denominator=counters['complete_episodes']+counters.get('task_reset_failures',0)
        report['success_rate_including_task_reset_failures']=counters['successes']/denominator if denominator else None
        report['success_rate_denominator']=denominator
        write_json(f'results/{run}/training_summary.json',report);trace.close()
    return report


def evaluate(method, config, scenes, model_path, label, deadline):
    from run_day9 import model_hash, hash_file, technical_error, code_hashes, resolved_execution
    config=local(config);c=__import__('yaml').safe_load(config.read_text())
    run=f'{time.strftime("%Y%m%dT%H%M%S",time.gmtime())}_{method}_validation_{label}_{time.time_ns()}'
    trace=Trace(run);env=None;model=None;records=[];failure=None;before=None;updates_before=0
    declaration=dict(purpose='fixed validation, not final test',method=method,label=label,scenes=scenes,
        seed=20260910,episodes=len(scenes),max_steps=20*len(scenes),max_episode_wall_sec=600,
        wall_deadline_monotonic=deadline,updates=0,replay_writes=0,curriculum_advance=False,
        deterministic=True,config=c,code_hashes=code_hashes(),model=model_path,
        resolved_execution=resolved_execution(method,'validation',c['day9']['profile'],c),
        model_sha256=hash_file(local(model_path)) if model_path else None,
        technical_retries=0,task_reset_failure='count as failure; do not fabricate observations or transitions')
    write_json(f'results/{run}/declaration.json',declaration);print(json.dumps(clean(declaration)),flush=True)
    start=time.monotonic()
    def timeout(*_):raise RunInterrupted('validation interrupted or declared wall budget reached')
    handlers={s:signal.signal(s,timeout) for s in [signal.SIGALRM,signal.SIGINT,signal.SIGTERM]}
    try:
        env=Day9Env(method,config,trace,phase='validation',profile=c['day9']['profile'],seed=20260910)
        if model_path:
            model=SAC.load(local(model_path),device='cpu');before=model_hash(model);updates_before=model._n_updates
        for scene in scenes:
            if time.monotonic()>=deadline-120:raise TechnicalFailure('validation deadline before next scene')
            signal.setitimer(signal.ITIMER_REAL,min(600,deadline-time.monotonic()-120))
            env.set_scene(scene);row=dict(scene=scene['id'],success=False,steps=0,grasp_attempts=0,
                completed_episode=False,task_reset_failure=False,technical_failure=False,return_=0.)
            t=time.monotonic()
            try:
                obs,_=env.reset()
                for _ in range(int(c['day4']['max_steps'])):
                    with trace.span('inference'):
                        action=np.zeros(4,np.float32) if model is None else model.predict(obs,deterministic=True)[0]
                    obs,reward,term,trunc,info=env.step(action)
                    row['steps']+=1;row['return_']+=reward;row['grasp_attempts']+=int(info['grasp_attempted'])
                    row['success']=bool(info['success']);row['last_info']=clean(info)
                    if term or trunc:row['completed_episode']=True;break
            except Exception as exc:
                row.update(error=repr(exc),technical_failure=technical_error(exc),task_reset_failure=row['steps']==0 and not technical_error(exc))
                ok,detail=env.base.backend.safe_recover();row.update(recovered=ok,recovery_detail=detail)
                if not ok or row['technical_failure']:failure=repr(exc)
            finally:
                signal.setitimer(signal.ITIMER_REAL,0)
                row['wall_sec']=time.monotonic()-t;records.append(row)
                write_json(f'results/{run}/episodes.json',records)
            if failure:break
        if model is not None and (before!=model_hash(model) or model._n_updates!=updates_before):
            raise AssertionError('validation changed model')
    except BaseException as exc:failure=repr(exc)
    finally:
        signal.setitimer(signal.ITIMER_REAL,max(.01,min(120,deadline-time.monotonic())))
        if env is not None:
            try:env.close()
            except BaseException as exc:
                failure=failure or repr(exc)
                try:pause_simulation()
                except Exception:pass
        signal.setitimer(signal.ITIMER_REAL,0)
        for s,h in handlers.items():signal.signal(s,h)
        result=dict(run=run,status='COMPLETE' if len(records)==len(scenes) and failure is None else 'PARTIAL',
            method=method,label=label,episodes=records,scheduled=len(scenes),successes=sum(r['success'] for r in records),
            failure=failure,wall_sec=time.monotonic()-start,updates=0,replay_writes=0,
            weights_unchanged=model is None or model_hash(model)==before)
        write_json(f'results/{run}/validation_summary.json',result);trace.close()
    return result


def pilot(config, report_sha, wall_limit):
    """Called only by an explicit post-report command; never by diagnostics."""
    from run_day9 import hash_file,new_model,SpaceOnly,code_hashes,validate_candidate,verify_artifacts
    report_path=ROOT/'results/review.json'
    if not report_path.exists() or hash_file(report_path)!=report_sha:
        raise ValueError('exact reviewed report SHA256 is required')
    review=json.loads(report_path.read_text())
    verify_artifacts(review['artifact_hashes'])
    if review['report_sha256']!=hash_file(ROOT/'REPORT.md'):
        raise ValueError('human-readable report changed after review snapshot')
    if any(hash_file(local(p))!=value for p,value in review['evidence_hashes'].items()):
        raise ValueError('reviewed diagnostic evidence changed')
    if not review.get('ready_for_pilot'):raise ValueError('report has unresolved blockers; pilot unavailable')
    config=local(config)
    validate_candidate(__import__('yaml').safe_load(config.read_text()))
    if review['candidate_sha256']!=hash_file(config) or review['code_hashes']!=code_hashes():
        raise ValueError('code/config changed after reviewed report')
    summary_path=ROOT/'results/pilot_summary.json'
    if summary_path.exists():raise ValueError('pilot already exists; retain results and use explicit per-job resume')
    if wall_limit<300:raise ValueError('wall budget must reserve recovery time')
    deadline=time.monotonic()+wall_limit
    scenes=json.loads((ROOT/'configs/validation_scenes.json').read_text())
    summary=dict(status='RUNNING',report_sha256=report_sha,config_sha256=hash_file(config),code_hashes=code_hashes(),
        artifact_hashes=review['artifact_hashes'],
        training_seed=20260909,validation_seed=20260910,wall_limit_sec=wall_limit,training={},validation=[],
        plan=dict(A_validation=20,B_pre=20,F_pre=20,B_post=20,F_post=20,B_steps=3000,F_steps=3000))
    write_json(summary_path,summary)
    try:
        initial={}
        for method in ['B','F']:
            path=ROOT/f'models/pilot/{method}_initial_seed20260909.zip';path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():raise ValueError('initial pilot model exists; refuse overwrite')
            new_model(SpaceOnly(),20260909).save(path);initial[method]=str(path.relative_to(ROOT))
        for method in ['A','B','F']:
            result=evaluate(method,config,scenes,initial.get(method),'before' if method!='A' else 'A_once',deadline)
            summary['validation'].append(result);write_json(summary_path,summary)
            if result['status']!='COMPLETE':raise TechnicalFailure('validation interrupted')
        for method in ['B','F']:
            remaining=deadline-time.monotonic()
            if remaining<300:raise TechnicalFailure('pilot wall budget exhausted')
            result=train_one(method,config,20260909,3000,remaining,stage='pilot')
            summary['training'][method]=result;write_json(summary_path,summary)
            if result['status']!='COMPLETE':raise TechnicalFailure('pilot training interrupted')
            post=evaluate(method,config,scenes,result['checkpoint']['model'],'after',deadline)
            summary['validation'].append(post);write_json(summary_path,summary)
            if post['status']!='COMPLETE':raise TechnicalFailure('post validation interrupted')
        summary['status']='COMPLETE'
    except BaseException as exc:
        summary.update(status='PARTIAL',error=repr(exc))
    write_json(summary_path,summary)
    write_pilot_analysis(summary)
    return summary


def write_pilot_analysis(summary):
    """Replace pre-pilot throughput assumptions with actual full training windows."""
    forecasts={}
    lines=['# 先导实测与正式训练粗估','',f"先导状态：{summary['status']}。未自动启动正式训练。",'',
        '|方法|实际环境步|更新次数|完整训练墙钟 h|秒/步|10000步估算 h/种子|3种子 h|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for method,t in summary['training'].items():
        if not t['steps']:continue
        q=t['wall_sec']/t['steps']
        u=t['timings'].get('network_update',{}).get('wall',0)/max(1,t['updates'])
        expected=10000*q+(9000-10000*t['updates']/t['steps'])*u
        forecasts[method]=dict(pilot_status=t['status'],measured_steps=t['steps'],
            wall_sec_per_step=q,update_wall_sec=u,formal_seconds_per_seed=expected,
            three_seeds_seconds=3*expected,conditional_range_seconds=[.8*expected,1.8*expected],
            includes=t['includes'],assumption='same observed task/reset/grasp mix and machine; warmup update-count correction applied; curriculum and learning can change mix',
            overhead='startup/checkpoint/recovery already amortized in complete training window; do not add again; extra evaluation separate')
        lines.append(f"|{method}|{t['steps']}|{t['updates']}|{t['wall_sec']/3600:.3f}|{q:.3f}|{expected/3600:.2f}|{3*expected/3600:.2f}|")
    if len(forecasts)==2:
        total=sum(v['three_seeds_seconds'] for v in forecasts.values())
        lines+=['',f'两方法共6个作业顺序运行基准估计 {total/3600:.2f} h；条件范围 {total*.8/3600:.2f}–{total*1.8/3600:.2f} h。']
    lines+=['','估算使用完整训练窗口/实际环境步；已含复位、随机化、抓放、网络更新、启动、检查点和收尾，不另加这些分项。10000步有9000次更新，对3000步先导的2000次更新比例作修正。0.8–1.8倍是任务/课程/抓取频率变化的条件范围，不是统计置信区间；部分先导只能粗估。',
        '',f"验证已记录 {sum(len(v['episodes']) for v in summary['validation'])} / 100 个槽；先导验证墙钟 {sum(v['wall_sec'] for v in summary['validation'])/3600:.3f} h，单独列出，不计训练步数。",'',
        '不据短训无改善判算法无效，不声称参数最优或收敛。冻结仍须通过独立完整性检查。恢复能力：检查点写入成功的训练作业可由 train_one(initial=latest.json) 恢复；pilot 编排器不自动重跑已完成验证，部分先导需要明确选择未完成阶段。']
    write_json('results/formal_forecast_post_pilot.json',forecasts)
    (ROOT/'results/PILOT_REPORT.md').write_text('\n'.join(lines)+'\n')


def freeze(config):
    from run_day9 import hash_file,code_hashes,verify_artifacts
    import yaml
    p=ROOT/'results/pilot_summary.json'
    if not p.exists():raise ValueError('no pilot evidence; candidate cannot be frozen as validated')
    s=json.loads(p.read_text());config=local(config)
    verify_artifacts(s['artifact_hashes'])
    if s['status']!='COMPLETE' or s['config_sha256']!=hash_file(config) or s['code_hashes']!=code_hashes():
        raise ValueError('incomplete/stale pilot evidence')
    for m in ['B','F']:
        t=s['training'][m]
        if t['steps']!=3000 or t['updates']!=2000 or not t['weights_changed']:
            raise ValueError('pilot learning chain not verified')
    if sum(len(v['episodes']) for v in s['validation'])!=100:raise ValueError('100 validation slots not complete')
    c=yaml.safe_load(config.read_text());c['day9']['status']='PILOT_CHECKED_FROZEN'
    path=ROOT/'configs/frozen.yaml'
    if path.exists():raise ValueError('frozen config exists; refuse overwrite')
    path.write_text(yaml.safe_dump(c,sort_keys=False,allow_unicode=True))
    write_json('configs/frozen_manifest.json',dict(config_sha256=hash_file(path),code_hashes=code_hashes(),
        artifact_hashes=s['artifact_hashes'],
        automation_hashes={str(p.relative_to(ROOT)):hash_file(p) for p in sorted((ROOT/'automation').glob('*.py'))},
        pilot_sha256=hash_file(p),formal_network_and_replay='fresh per seed',formal_steps=60000,
        limitation='pilot chain verified; no optimality/convergence claim; no tuning against final test'))
    return str(path)
