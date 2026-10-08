#!/usr/bin/env python3
"""Read retained evidence only. No ROS commands, no training, no new trials."""
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def build_measurements():
    runs = []
    stages = ['startup','reset_total','randomization_physics','randomization_perception',
              'vision','ik','ik_path','ik_step','action_execution','closing','grip_settle',
              'lift','hold','place','open_motion','open_settle','retreat',
              'return_observation','recovery','inference','network_update']
    for path in sorted((ROOT/'results').glob('*/summary.json')):
        s = json.loads(path.read_text())
        d = json.loads(path.with_name('declaration.json').read_text())
        row = {k:s[k] for k in ['run','method','profile','tag','window_wall_sec',
            'environment_steps','completed_episodes','success_count','requested_episodes',
            'failure','updates','weights_unchanged']}
        row['wall_sec_per_step'] = s['window_wall_sec']/s['environment_steps'] if s['environment_steps'] else None
        row['mean_episode_steps'] = s['environment_steps']/len(s['episodes']) if s['episodes'] else None
        row['episode_wall_sec'] = [r['elapsed_wall_sec'] for r in s['episodes']]
        row['grasp_attempts'] = sum(r['grasp_attempts'] for r in s['episodes'])
        row['failure_reasons'] = [r.get('failure') or r.get('last_info',{}).get('termination_reason') for r in s['episodes']]
        row['scene_ids'] = [r['scene']['id'] for r in s['episodes']]
        row['declaration'] = str(path.with_name('declaration.json').relative_to(ROOT))
        row['declared_budget'] = {k:d[k] for k in ['max_env_steps','max_episode_wall_sec','training_updates','preparation_episodes','measured_episodes']}
        row['stages'] = {k:s['timings'].get(k,dict(wall=0.,exclusive=0.,sim=0.,count=0)) for k in stages}
        # Stage clocks: initial backend clock discovery can have no valid starting
        # clock; use only per-action/contact spans with initialized clocks.
        row['rtf'] = {}
        for name, members in [('approach',['action_execution']),('contact',['closing','grip_settle','lift','hold'])]:
            wall=sum(s['timings'].get(k,{}).get('wall',0) for k in members)
            sim=sum(s['timings'].get(k,{}).get('sim',0) for k in members)
            row['rtf'][name]=sim/wall if wall else None
        row['open_results']=[];row['contact_evidence']={};row['safe_close']=[]
        geometry=[];payloads=[];transitions=[];velocities={}
        for line in path.with_name('events.jsonl').open():
            e=json.loads(line)
            if e['kind']=='open_result':row['open_results'].append({k:v for k,v in e.items() if k not in ['run','kind','monotonic']})
            if e['kind']=='contact_geometry':geometry.append(e)
            if e['kind']=='contact':payloads.extend(e.get('payload_fields',[]))
            if e['kind']=='safe_close':row['safe_close'].append(e['ok'])
            if e['kind']=='transition':transitions.append(e)
            if e['kind']=='gripper_audit':
                for v in e['rows']:
                    velocities.setdefault(v['phase'],[]).append(v['gripper_velocity_rad_sec'])
        row['contact_evidence']=dict(geometry=geometry,payload_sample_count=len(payloads),
            real_collision_pairs=sorted(set((v['collision1'],v['collision2']) for v in payloads)),
            position_count=sum(v['position_count'] for v in payloads),
            normal_count=sum(v['normal_count'] for v in payloads),depth_count=sum(v['depth_count'] for v in payloads),
            absence_note='empty normals/depths are unavailable, never measured zero or inferred true contact')
        row['action_audit']=dict(transitions=len(transitions),
            F_steps_with_nonzero_effect=sum(any(abs(a-b)>1e-9 for a,b in zip(e['action']['final_action'],e['action']['base_action'])) for e in transitions) if row['method']=='F' else None,
            maximum_policy_absolute_components=[max((abs(e['policy_action'][i]) for e in transitions),default=0.) for i in range(4)])
        row['actual_gripper_velocity_rad_sec']={phase:dict(samples=len(v),minimum=min(v),maximum=max(v),
            samples_abs_above_model_0_10=sum(abs(x)>.1 for x in v)) for phase,v in velocities.items()}
        runs.append(row)
    out=dict(runs=runs,units='monotonic wall seconds',
        accounting='window includes first/inter-episode resets, grasp, randomization, inference and final recovery; excludes startup and network updates; parent stage wall values overlap children, never sum both',
        initial_clock_note='Ignore initial reset RTF before clock initialization in early correctness/preparation logs.')
    (ROOT/'results/measurements.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    return out


def tables(data):
    lines=['# Day9 全部短实验记录','',data['accounting'],'',
           '|实验|方法 / 配置|回合 / 成功|步数|完整窗口 s|秒 / 步|回合耗时 s|',
           '|---|---|---:|---:|---:|---:|---|']
    for r in data['runs']:
        q='—' if r['wall_sec_per_step'] is None else f"{r['wall_sec_per_step']:.3f}"
        times=', '.join(f'{v:.2f}' for v in r['episode_wall_sec'])
        lines.append(f"|[{r['tag']}]({r['run']}/declaration.json)|{r['method']} / {r['profile']}|{r['requested_episodes']} / {r['success_count']}|{r['environment_steps']}|{r['window_wall_sec']:.3f}|{q}|{times}|")
    lines+=['','旧闭合控制的 correctness / preparation 成功仅是原判据日志，不作为修复后物理回归通过。closing_correctness 保留了 NO_STALL 失败；closing_handoff_correctness 是后续修正成功。','',
            '每条声明含完整参数、代码哈希、模型来源、种子场景、回合/步数/墙钟/更新预算及恢复停止规则。原始事件、回合详情和失败均保留。','',
            '## 分项累计墙钟（父项与子项重叠，禁止逐列相加）','',
            '|方法 / 配置|复位总计|物理随机化|视觉|IK + 路径 + 步进 IK|动作执行|闭合|夹持稳定|抬升|保持|放回|松爪运动|松爪稳定|退离|回观察|最终恢复|推理|接近 / 接触 RTF|',
            '|---|'+'---:|'*17]
    for r in data['runs']:
        if r['tag']!='paired':continue
        s=r['stages'];w=lambda k:s[k]['wall']
        vals=[w('reset_total'),w('randomization_physics'),w('vision'),sum(s[k]['exclusive'] for k in ['ik','ik_path','ik_step']),
              w('action_execution'),w('closing'),w('grip_settle'),w('lift'),w('hold'),w('place'),w('open_motion'),
              w('open_settle'),w('retreat'),w('return_observation'),w('recovery'),w('inference')]
        rtf=' / '.join('—' if v is None else f'{v:.3f}' for v in r['rtf'].values())
        lines.append('|'+r['method']+' / '+r['profile']+'|'+'|'.join(f'{v:.3f}' for v in vals)+'|'+rtf+'|')
    lines+=['','启动时间、各项 exclusive 值及真实接触字段汇总另见 [measurements.json](measurements.json)。网络更新开销见 [update_probe.json](update_probe.json)，不与无学习窗口混加。']
    (ROOT/'results/MEASUREMENTS.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    data=build_measurements();tables(data)
    print(json.dumps([dict(method=r['method'],profile=r['profile'],tag=r['tag'],seconds=r['window_wall_sec'],steps=r['environment_steps'],successes=r['success_count'],failure=r['failure']) for r in data['runs']],ensure_ascii=False,indent=2))
