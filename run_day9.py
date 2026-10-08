#!/usr/bin/env python3
"""Single Day9 entry: checks, bounded diagnostics, reports and explicit training."""
from __future__ import annotations
import argparse
import copy
import fcntl
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import signal
import shutil
import sys
import time
import traceback
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'vendor'))
import numpy as np
import gymnasium as gym
import torch
import yaml
from stable_baselines3 import SAC
from day9_env import Day9Env, Trace, clean, local, write_json, pause_simulation, RunInterrupted
from day8_env import DomainRandomizer, TechnicalFailure

P1 = dict(learning_rate=.0003, gamma=.99, tau=.005, buffer_size=100000,
          batch_size=128, learning_starts=1000, train_freq=(1, 'step'), gradient_steps=1,
          ent_coef='auto', target_entropy='auto', target_update_interval=1,
          action_noise=None, use_sde=False, optimize_memory_usage=False, device='cpu')
POLICY_KWARGS = dict(net_arch=[64, 64], activation_fn=torch.nn.ReLU, n_critics=2,
    optimizer_class=torch.optim.Adam, optimizer_kwargs=dict(betas=(.9, .999), eps=1e-8, weight_decay=0))


def hash_file(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def code_hashes():
    paths = list(ROOT.glob('*.py')) + list(ROOT.glob('*.yaml')) + list(ROOT.glob('*.sh'))
    for sub in ['generated', 'moveit_v5', 'simulations']:
        paths += [p for p in (ROOT/sub).rglob('*') if p.is_file() and p.suffix in ('.yaml','.sdf','.urdf','.srdf')]
    return {str(p.relative_to(ROOT)):hash_file(p) for p in sorted(paths)}


def artifact_hashes():
    paths=[p for p in (ROOT/'vendor').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    paths += [ROOT/'models/sac_smoke.zip', ROOT/'configs/diagnostic_scenes.json',
              ROOT/'configs/validation_scenes.json', ROOT/'configs/dds_udp.xml']
    return {str(p.relative_to(ROOT)):hash_file(p) for p in sorted(paths)}


def verify_artifacts(expected):
    if expected != artifact_hashes():
        raise ValueError('vendor/model/fixed-scene/transport artifacts differ from reviewed version')


def model_hash(model):
    h = hashlib.sha256()
    for name, value in sorted(model.policy.state_dict().items()):
        h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def new_model(env, seed):
    return SAC('MlpPolicy', env, policy_kwargs=copy.deepcopy(POLICY_KWARGS), seed=seed,
               verbose=0, **P1)


def validate_candidate(config):
    """Reject descriptive snapshots that disagree with the actual fixed P1 code."""
    policy = dict(net_arch=[64,64], activation_fn='torch.nn.ReLU', n_critics=2,
        optimizer_class='torch.optim.Adam', optimizer_kwargs=dict(betas=[.9,.999],eps=1e-8,weight_decay=0))
    if config['day9']['P1'] != clean(P1) or config['day9']['policy_kwargs'] != policy:
        raise ValueError('candidate P1 differs from the implemented and reviewed P1')
    if config['day4']['max_steps'] != 20:
        raise ValueError('20-step limit retained; no approved 20/40 evidence')
    if config['day4']['reward'] != dict(approach_scale=100.,success=50.,drop=-30.,collision=-30.):
        raise ValueError('Day8 reward changed without reviewed comparisons')


def resolved_execution(method, phase, profile, config):
    return dict(method=method,phase=phase,profile=profile,
        flags=config['day9'].get('accepted_optimizations',[]) if profile=='combined' else ([] if profile=='baseline' else [profile]),
        curriculum_mode='auto' if method=='F' and phase=='train' else 'off',
        domain_mode=('fixed at current curriculum level' if method=='F' else 'off') if phase=='train' else 'from persisted fixed scene',
        internal_smoke_model_loaded=False,source_reset_attempts=config['day4']['reset_attempts'],effective_reset_attempts=1,
        action='base only' if method=='A' else 'direct SAC physical action' if method=='B' else 'base + external SAC mapped/filtered residual',
        action_limit_m=.005,yaw_limit_rad=math.radians(5),base_gain=.8 if method!='B' else None,
        residual_scale=.2 if method=='F' else None,filter_alpha=.25 if method=='F' else None,
        confidence_fusion_gate=False,perception_validity_threshold=.66,
        closing_search_reference_rad_sec=-.10,closing_actual_hard_speed_bound_verified=False,
        closing_hold_pi=[.75,2.],hold_effort_limit_nm=.6,overclose_rad=.04,
        reward=config['day4']['reward'],max_steps=20,hold_sim_sec=5.,hold_wall_sec=5.,
        full_trajectory_completion_required=True,physics_step_sec=.002)


class SpaceOnly(gym.Env):
    """Interface checks only; never supplies simulated training transitions."""
    def __init__(self):
        self.observation_space = gym.spaces.Box(np.array([-1.]*3+[-2.14]*5+[-1.,-1.],np.float32),
            np.array([1.]*3+[2.14]*5+[1.,1.],np.float32))
        high = np.array([.005,.005,.005,math.radians(5)],np.float32)
        self.action_space = gym.spaces.Box(-high,high)


def resolved_model(model):
    import stable_baselines3, gymnasium, cv2, scipy
    defaults = {k:str(v.default) for k,v in inspect.signature(SAC).parameters.items()
                if v.default is not inspect.Parameter.empty}
    return clean(dict(P1=P1, policy='MlpPolicy', policy_kwargs={
        'net_arch':[64,64], 'activation_fn':'torch.nn.ReLU', 'n_critics':2,
        'optimizer_class':'torch.optim.Adam','optimizer_kwargs':POLICY_KWARGS['optimizer_kwargs']},
        versions=dict(sb3=stable_baselines3.__version__, gymnasium=gymnasium.__version__,
          numpy=np.__version__, opencv=cv2.__version__, torch=torch.__version__, scipy=scipy.__version__),
        imports={m.__name__:m.__file__ for m in [stable_baselines3,gymnasium,np,cv2,torch,scipy]},
        constructor_defaults=defaults, policy_constructor_defaults={k:str(v.default) for k,v in
            inspect.signature(type(model.policy)).parameters.items() if v.default is not inspect.Parameter.empty},
        actor_optimizer=model.actor.optimizer.defaults, critic_optimizer=model.critic.optimizer.defaults,
        entropy_optimizer=model.ent_coef_optimizer.defaults, target_entropy=model.target_entropy,
        initial_entropy_coefficient=float(model.log_ent_coef.detach().exp()),
        entropy_initialization='log(alpha)=0; alpha=1; Adam on log(alpha), same constant learning rate',
        replay_class=type(model.replay_buffer).__name__, replay_timeout_handling=model.replay_buffer.handle_timeout_termination,
        normalize_observation=False, normalize_reward=False, torch_threads=torch.get_num_threads(),
        torch_interop_threads=torch.get_num_interop_threads()))


def prepare_configs():
    c = yaml.safe_load((ROOT/'config.yaml').read_text())
    c['day9'] = dict(status='CANDIDATE_UNVALIDATED', profile='baseline', methods=['A','B','F'],
        seed_diagnostic=20260908, seed_pilot=20260909, seed_validation=20260910,
        formal_seeds=[20260911,20260912,20260913], pilot_steps_per_method=3000,
        formal_steps_per_method_seed=10000, task_reset_failure='count_failure_without_fake_transition',
        P1=clean(P1), policy_kwargs=resolved_model(new_model(SpaceOnly(),20260908))['policy_kwargs'])
    path = ROOT/'configs/candidate.yaml'
    if not path.exists(): path.write_text(yaml.safe_dump(c,sort_keys=False,allow_unicode=True))
    scenes_path = ROOT/'configs/diagnostic_scenes.json'
    if not scenes_path.exists():
        old = json.loads((ROOT/'provenance/day8/smoke_L0.json').read_text())
        scenes = []
        for index in [0,3,4]:
            row = next(r for r in old if r['index']==index)
            scenes.append(dict(id=f'diagnostic_L0_{index}', source='Day8 smoke_L0 point; new Day9 seed',
                position_base=row['position_base'], seed=20260908+len(scenes), level='L0', mode='fixed'))
        write_json(scenes_path, scenes)
    validation = ROOT/'configs/validation_scenes.json'
    if not validation.exists():
        r = DomainRandomizer(c); scenes=[]
        for i,lev in enumerate(['L0']*10+['L1']*4+['L2']*3+['L3']*3):
            seed=int(np.random.SeedSequence([20260910,i]).generate_state(1)[0])
            params=r.sample(seed,lev,'fixed')
            scenes.append(dict(id=f'validation_{i:02d}', seed=seed,level=lev,mode='fixed',
                               parameters=params, source='new fixed Day9 validation; not final test'))
        write_json(validation,scenes)
    return path


def offline_check():
    previous=ROOT/'results/offline_checks.json'
    if previous.exists():
        old=json.loads(previous.read_text())
        if old.get('code_hashes')==code_hashes():
            print('Reusing matching offline checks: '+str(previous));return
    config = prepare_configs()
    from residual_env import ResidualActionController, ActionGuardError
    from day4_env import Day4GraspEnv
    c = yaml.safe_load(config.read_text()); checks=[]
    def check(name, condition, detail=None):
        checks.append(dict(name=name,passed=bool(condition),detail=clean(detail)))
        if not condition: raise AssertionError(name)
    smoke=SAC.load(ROOT/'models/sac_smoke.zip',device='cpu')
    obs=np.zeros(10,np.float32);obs[9]=1
    action,_=smoke.predict(obs,deterministic=True)
    check('smoke_load_predict',action.shape==(4,) and np.isfinite(action).all(),
          dict(hash=hash_file(ROOT/'models/sac_smoke.zip'),archive_steps=smoke.num_timesteps,archive_updates=smoke._n_updates))
    ctrl=ResidualActionController(config,load_model=False)
    check('no_model_loaded_for_training',ctrl.model is None)
    raw=np.array([.005,-.005,.005,math.radians(5)],np.float32)
    d=ctrl.decide(obs,raw_action=raw)
    check('external_residual_mapping',np.allclose(d.mapped_residual,raw*.2,atol=1e-8))
    check('filter_alpha',np.allclose(d.filtered_residual,.25*d.mapped_residual))
    check('residual_not_cancelled',np.any(np.abs(d.final_action)>1e-6))
    ctrl.reset_episode();check('filter_reset',np.allclose(ctrl._filtered,0))
    for bad in [np.array([np.nan]*4),np.zeros(3),np.array([np.inf]*4)]:
        try:ctrl.decide(obs,raw_action=bad)
        except ActionGuardError: rejected=True
        else:rejected=False
        check('invalid_action_rejected',rejected)
    # Execute the actual Day4 step/reward/termination code against a deterministic
    # test backend. These are unit checks, not physical grasp evidence.
    from types import SimpleNamespace
    cases=[('approach',.02,.015,False,False,False,False,1,.5),
           ('recede',.02,.025,False,False,False,False,1,-.5),
           ('stationary',.02,.02,False,False,False,False,1,0),
           ('yaw_misaligned',.004,.004,False,False,False,False,1,0),
           ('success',.006,.004,True,True,False,False,1,50.2),
           ('ordinary_failure',.006,.004,True,False,False,False,1,.2),
           ('drop',.006,.004,True,False,True,False,1,-29.8),
           ('collision',.006,.004,True,False,False,True,1,-29.8),
           ('max_steps',.02,.02,False,False,False,False,20,0)]
    for name,prev,dist,attempt,success,drop,collision,step,expected in cases:
        env=Day4GraspEnv.__new__(Day4GraspEnv)
        env.d4=copy.deepcopy(c['day4']);env.config=c;env._steps=step-1;env._invalid_actions=0;env._episode=1
        env.selected={};env.visual={};env.reset_metrics={};env.previous_distance=prev
        env.action_space=SpaceOnly().action_space
        env.backend=SimpleNamespace(arm_q=lambda:[0]*5,state_is_valid=lambda *a,**kw:(True,''))
        env._tcp_pose=lambda:(np.zeros(3),np.eye(3))
        env._solve_increment=lambda a:([0]*5,False)
        env._publish_step_target=lambda q:{}
        env._block_displaced=lambda:False
        o=np.zeros(10,np.float32);o[0]=dist;o[9]=1
        if name=='yaw_misaligned':o[8]=math.sin(.5);o[9]=math.cos(.5)
        env._get_obs=lambda:o.copy()
        env._attempt_grasp=lambda:(success,drop,collision,success,0.)
        _,reward,terminated,truncated,info=env.step(np.zeros(4,np.float32))
        check('reward_'+name,abs(reward-expected)<1e-5 and terminated==attempt and truncated==(name=='max_steps'),info)
    model=new_model(SpaceOnly(),20260908)
    check('fresh_network_replay',model.num_timesteps==0 and model._n_updates==0 and model.replay_buffer.size()==0)
    write_json('results/resolved_P1.json',resolved_model(model))
    check('auto_entropy',model.target_entropy==-4. and float(model.log_ent_coef.detach().exp())==1.)
    roundtrip=ROOT/f'models/diagnostic/interface_roundtrip_{time.time_ns()}.zip'
    model.save(roundtrip)
    loaded=SAC.load(roundtrip,device='cpu')
    check('save_load_same_weights',model_hash(model)==model_hash(loaded))
    manifest=json.loads((ROOT/'provenance/source.json').read_text())
    altered=[name for name,h in manifest['source_hashes'].items() if hash_file(Path(manifest['source'])/name)!=h]
    check('day8_unchanged',not altered,altered)
    check('parent_index_unchanged',hash_file(ROOT.parent/'.git/index')==manifest['parent_index_sha256'])
    check('physics_step_unchanged',float(ET.parse(ROOT/'simulations/robot_gazebo/worlds/grasp_table.sdf').findtext('.//max_step_size'))==.002)
    check('hold_both_clocks_5s',c['verification']['hold_sim_sec']==5. and c['verification']['hold_wall_sec']==5.)
    report=dict(status='PASS',checks=checks,physics_proven=False,training_proven=False,code_hashes=code_hashes())
    destination='results/offline_checks.json' if not previous.exists() else f'results/offline_checks_{time.time_ns()}.json'
    write_json(destination,report)
    print(json.dumps(clean(report),ensure_ascii=False,indent=2))


def budget_state():
    return json.loads((ROOT/'runtime/diagnostic_budget.json').read_text())


def consume_budget(kind):
    with (ROOT/'runtime/budget.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        b=budget_state()
        if b.get('no_more_physical_diagnostics'):raise TechnicalFailure('diagnostic allocation closed at report checkpoint')
        if time.monotonic()>=b['stop_measurement_monotonic']:raise TechnicalFailure('diagnostic wall budget exhausted')
        field,maximum={'attempt':('attempts','max_attempts'),'step':('step_attempts','max_env_steps'),
                       'probe':('probe_updates','max_probe_updates')}[kind]
        if b.get(field,0)>=b[maximum]:raise TechnicalFailure('diagnostic '+kind+' budget exhausted')
        b[field]=b.get(field,0)+1;write_json('runtime/diagnostic_budget.json',b)


def technical_error(exc):
    return isinstance(exc,TechnicalFailure) or any(w in str(exc).upper() for w in
        ['INTERFACE','CLOCK','SERVICE','READBACK','CONTROLLER_NOT_ACTIVE','SIMULATION_FAILURE'])


def run_batch(method, profile, scenes, *, tag='paired', config=None):
    if budget_state().get('no_more_physical_diagnostics'):
        raise RuntimeError('diagnostic allocation closed at report checkpoint; no automatic extension')
    if tag=='paired' and (ROOT/'runtime/hold_comparisons.json').exists():
        raise RuntimeError('Paired comparisons held pending closing-search correctness regression')
    config=local(config or 'configs/candidate.yaml')
    run=f'{time.strftime("%Y%m%dT%H%M%S",time.gmtime())}_{method}_{profile}_{tag}_{time.time_ns()}'
    trace=Trace(run);env=None;model=None;records=[];window_start=None;window_end=None
    b=budget_state();max_steps=yaml.safe_load(config.read_text())['day4']['max_steps']
    declaration=dict(purpose=tag,method=method,profile=profile,config=yaml.safe_load(config.read_text()),
        resolved_P1=json.loads((ROOT/'results/resolved_P1.json').read_text()),code_hashes=code_hashes(),
        seed=20260908,scenes=scenes,preparation_episodes=len(scenes) if tag=='preparation' else 0,
        measured_episodes=0 if tag=='preparation' else len(scenes),max_env_steps=max_steps*len(scenes),
        max_episode_wall_sec=600,global_budget=b,training_updates=0,technical_retries_in_this_batch=0,
        recovery='physical placement before release; pause simulation if recovery unsafe',
        stop='wall/step/attempt budget, unrecovered technical failure, nonfinite transition',
        model_source='none' if method=='A' else f'models/diagnostic/{method}_P1_seed20260908.zip',
        action_mode='deterministic',curriculum_advance=False)
    declaration['config']['day9']['profile']=profile
    declaration['resolved_execution']=resolved_execution(method,'diagnostic',profile,declaration['config'])
    declaration['config_sha256']=hash_file(config)
    declaration['execution_corrections']=dict(closing_search='velocity PI reference -0.10 rad/s (not a verified hard feedback bound), integral retained into original hold servo',
        closed_hold_gains='unchanged',overclose_rad=.04,hold_sim_sec=5.,hold_wall_sec=5.,
        trajectory_completion='last actual dispatch + full duration + original tight feedback settling',
        missing_contact_vectors='explicitly flagged; never fabricated')
    write_json(f'results/{run}/declaration.json',declaration)
    for name in declaration['code_hashes']:
        destination=ROOT/f'results/{run}/source'/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,destination)
    print('DECLARED_RUN '+json.dumps(clean(declaration),ensure_ascii=False),flush=True)
    def alarm(*_):raise RunInterrupted('diagnostic interrupted or declared wall deadline reached')
    handlers={s:signal.signal(s,alarm) for s in [signal.SIGALRM,signal.SIGINT,signal.SIGTERM]}
    failure=None
    try:
        with trace.span('startup'):
            env=Day9Env(method,config,trace,profile=profile)
            if method!='A':
                model_path=ROOT/f'models/diagnostic/{method}_P1_seed20260908.zip'
                if model_path.exists(): model=SAC.load(model_path,env=env,device='cpu')
                else:
                    model=new_model(env,20260908);model.save(model_path)
                initial_hash=model_hash(model)
                trace.emit('model',source=str(model_path),file_sha256=hash_file(model_path),weights_sha256=initial_hash)
        window_start=time.monotonic()
        for scene in scenes:
            consume_budget('attempt')
            deadline=min(time.monotonic()+600,budget_state()['stop_measurement_monotonic'])
            signal.setitimer(signal.ITIMER_REAL,max(.01,deadline-time.monotonic()))
            env.set_scene(scene);start=time.monotonic();before=trace.snapshot()
            record=dict(scene=scene,steps=0,success=False,terminated=False,truncated=False,
                        completed=False,technical_failure=False,return_=0.,failure=None,grasp_attempts=0)
            failure_index=len(trace.failures)
            try:
                obs,info=env.reset()
                for _ in range(max_steps):
                    consume_budget('step')
                    with trace.span('inference'):
                        action=np.zeros(4,np.float32) if model is None else model.predict(obs,deterministic=True)[0]
                    obs,reward,term,trunc,info=env.step(action)
                    record['steps']+=1;record['return_']+=reward
                    record['grasp_attempts']+=int(info['grasp_attempted'])
                    record.update(success=bool(info['success']),terminated=term,truncated=trunc,last_info=clean(info))
                    if term or trunc:record['completed']=True;break
            except Exception as exc:
                record.update(technical_failure=technical_error(exc),failure=f'{type(exc).__name__}: {exc}')
                trace.emit('episode_exception',error=record['failure'])
                signal.setitimer(signal.ITIMER_REAL,max(.01,min(120,budget_state()['deadline_monotonic']-time.monotonic())))
                ok,detail=env.base.backend.safe_recover()
                record.update(recovered=ok,recovery_detail=detail)
                if not ok:failure=record['failure']
            finally:
                signal.setitimer(signal.ITIMER_REAL,0)
                record['elapsed_wall_sec']=time.monotonic()-start
                record['parameters']=clean(env.runtime.params)
                record['failure_events']=trace.failures[failure_index:]
                after=trace.snapshot()
                record['timings']={k:{f:v[f]-before.get(k,{}).get(f,0) for f in v} for k,v in after.items()}
                records.append(record)
                trace.emit('episode_result',record=record)
                write_json(f'results/{run}/episodes.json',records)
                print('EPISODE_RESULT '+json.dumps(clean({k:v for k,v in record.items() if k not in ('timings','parameters','last_info','failure_events')})),flush=True)
            if failure or (record['technical_failure'] and not record.get('recovered')):break
        if model is not None and model_hash(model)!=initial_hash:raise AssertionError('paired inference changed weights')
    except BaseException as exc:
        failure=f'{type(exc).__name__}: {exc}';trace.emit('run_exception',error=failure)
        traceback.print_exc()
    finally:
        signal.setitimer(signal.ITIMER_REAL,max(.01,min(120,budget_state()['deadline_monotonic']-time.monotonic())))
        if env is not None:
            try:env.close()
            except BaseException as exc:
                failure=failure or repr(exc);trace.emit('close_error',error=repr(exc))
                try:pause_simulation()
                except Exception:pass
        window_end=time.monotonic()
        signal.setitimer(signal.ITIMER_REAL,0)
        for s,h in handlers.items():signal.signal(s,h)
        summary=dict(run=run,method=method,profile=profile,tag=tag,episodes=records,
            window_wall_sec=None if window_start is None else window_end-window_start,
            environment_steps=sum(r['steps'] for r in records),completed_episodes=sum(r['completed'] for r in records),
            success_count=sum(r['success'] for r in records),requested_episodes=len(scenes),failure=failure,
            window_includes=['first_reset','inter_episode_reset','grasp','randomization','inference','final_recovery'],
            window_excludes=['startup','network_updates'],timings=trace.snapshot(),updates=0,
            weights_unchanged=model is None or model_hash(model)==initial_hash)
        write_json(f'results/{run}/summary.json',summary)
        with (ROOT/'results/runs.jsonl').open('a') as stream:
            stream.write(json.dumps(clean({k:v for k,v in summary.items() if k not in ('episodes','timings')}))+'\n')
        trace.close()
    print('RUN_SUMMARY '+json.dumps(clean({k:v for k,v in summary.items() if k not in ('episodes','timings')})),flush=True)
    return summary


def main():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('check')
    sub.add_parser('probe')
    pilot_p=sub.add_parser('pilot')
    pilot_p.add_argument('--config',default='configs/candidate.yaml')
    pilot_p.add_argument('--report-sha',required=True)
    pilot_p.add_argument('--wall-limit-sec',type=float,required=True)
    freeze_p=sub.add_parser('freeze')
    freeze_p.add_argument('--config',default='configs/candidate.yaml')
    train_p=sub.add_parser('train')
    train_p.add_argument('--method',choices=['B','F'],required=True)
    train_p.add_argument('--config',default='configs/frozen.yaml')
    train_p.add_argument('--seeds',type=int,nargs='+',required=True)
    train_p.add_argument('--steps-per-seed',type=int,default=10000)
    train_p.add_argument('--wall-limit-sec',type=float,required=True)
    train_p.add_argument('--fresh',action='store_true')
    train_p.add_argument('--resume-state')
    b=sub.add_parser('benchmark')
    b.add_argument('--method',choices=['A','B','F'],required=True)
    b.add_argument('--profile',choices=['baseline','open_servo','reset_once','no_resend','combined'],default='baseline')
    b.add_argument('--tag',default='paired')
    b.add_argument('--scene-count',type=int,choices=[1,3],default=3)
    b.add_argument('--level',choices=['L0','L1','L2','L3'])
    args=p.parse_args()
    if os.environ.get('DAY9_ROOT')!=str(ROOT):p.error('source Day9 session_env.sh first')
    if args.command=='check':offline_check();return
    if args.command=='probe':
        from day9_training import update_probe
        update_probe();return
    if args.command=='pilot':
        from day9_training import pilot
        result=pilot(args.config,args.report_sha,args.wall_limit_sec)
        print(json.dumps(clean(result),ensure_ascii=False));return
    if args.command=='freeze':
        from day9_training import freeze
        print(freeze(args.config));return
    if args.command=='train':
        from day9_training import train_one
        manifest_path=ROOT/'configs/frozen_manifest.json'
        if not manifest_path.exists():p.error('no pilot-checked frozen configuration; formal training unavailable')
        manifest=json.loads(manifest_path.read_text())
        verify_artifacts(manifest['artifact_hashes'])
        if manifest['config_sha256']!=hash_file(local(args.config)) or manifest['code_hashes']!=code_hashes():
            p.error('frozen code/config hash mismatch')
        if args.steps_per_seed!=10000 or any(s not in [20260911,20260912,20260913] for s in args.seeds):
            p.error('formal budget is 10000 steps for each of the three declared seeds')
        if len(set(args.seeds))!=len(args.seeds):p.error('duplicate formal seeds exceed the declared job allocation')
        if args.fresh==bool(args.resume_state):p.error('select exactly one of --fresh / --resume-state')
        if args.resume_state and len(args.seeds)!=1:p.error('resume exactly one job')
        for seed in args.seeds:
            result=train_one(args.method,args.config,seed,args.steps_per_seed,args.wall_limit_sec,
                             stage='formal',initial=args.resume_state)
            print(json.dumps(clean(result),ensure_ascii=False))
            if result['status']!='COMPLETE':raise SystemExit(2)
        return
    scenes=json.loads((ROOT/'configs/diagnostic_scenes.json').read_text())[:args.scene_count]
    if args.level:
        if args.scene_count!=1:p.error('domain regression uses one nominal-center scene')
        scenes[0].update(level=args.level,seed=20260908+int(args.level[1])*100,id='domain_'+args.level)
    result=run_batch(args.method,args.profile,scenes,tag=args.tag)
    if result['failure']:raise SystemExit(2)


if __name__=='__main__':main()
