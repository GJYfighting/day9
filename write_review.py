#!/usr/bin/env python3
"""Produce the mandatory pre-pilot review from retained measurements only."""
import json
import math
import time
from pathlib import Path
import yaml
from analyze_day9 import build_measurements, tables
from day9_env import ROOT, write_json
from run_day9 import code_hashes, artifact_hashes, hash_file, budget_state, validate_candidate


def source_audit():
    p=json.loads((ROOT/'provenance/source.json').read_text());source=Path(p['source'])
    changed=[];missing=[];linked=[];new=[]
    for name,value in p['source_hashes'].items():
        path=source/name
        if not path.exists():missing.append(name)
        elif hash_file(path)!=value:changed.append(name)
        copy=ROOT/name
        if path.exists() and copy.is_file() and path.stat().st_dev==copy.stat().st_dev and path.stat().st_ino==copy.stat().st_ino:
            linked.append(name)
    current={str(f.relative_to(source)) for f in source.rglob('*') if f.is_file()}
    new=sorted(current-set(p['source_hashes']))
    external_links=[]
    for f in ROOT.rglob('*'):
        if f.is_symlink():
            try:f.resolve().relative_to(ROOT)
            except ValueError:external_links.append(str(f.relative_to(ROOT)))
    index_ok=hash_file(ROOT.parent/'.git/index')==p['parent_index_sha256']
    result=dict(source=str(source),source_files=len(p['source_hashes']),changed=changed,missing=missing,
        new_source_files=new,shared_inodes=linked,external_writable_links=external_links,parent_index_unchanged=index_ok,
        status='PASS' if not(changed or missing or new or linked or external_links) and index_ok else 'FAIL')
    write_json('provenance/isolation_final.json',result)
    modified=[];added=[]
    for f in sorted(ROOT.rglob('*')):
        if not f.is_file():continue
        name=str(f.relative_to(ROOT))
        if name.split('/')[0] in ['runtime','results','provenance','models']:continue
        if name not in p['source_hashes']:added.append(name)
        elif hash_file(f)!=p['source_hashes'][name]:modified.append(name)
    write_json('provenance/change_list.json',dict(modified=modified,added=added,
        output_directories=['runtime','results','models/diagnostic','provenance'],
        source_model_unchanged=hash_file(ROOT/'models/sac_smoke.zip')==p['source_hashes']['models/sac_smoke.zip']))
    return result


def main():
    if (ROOT/'results/review.json').exists():raise RuntimeError('Review already exists; preserve it and explicitly version any new report')
    c=yaml.safe_load((ROOT/'configs/candidate.yaml').read_text());validate_candidate(c)
    data=build_measurements();tables(data)
    paired={(r['method'],r['profile']):r for r in data['runs'] if r['tag']=='paired'}
    selected={m:paired[(m,c['day9']['profile'])] for m in ['B','F']}
    probe=json.loads((ROOT/'results/update_probe.json').read_text())
    u={r['method']:r['wall_sec_per_update'] for r in probe['results']}
    io=json.loads((ROOT/'results/io_probe.json').read_text())
    audit=source_audit()
    q={m:r['wall_sec_per_step'] for m,r in selected.items()}
    # Conditional task-mix range, deliberately broad; not a confidence interval.
    q_range={'B':[.85*q['B'],1.6*max(q.values())],
             'F':[.75*min(q['F'],paired[('F','open_servo')]['wall_sec_per_step']),1.6*q['F']]}
    forecast={}
    for m in ['B','F']:
        estimate=lambda n,rate:n*rate+max(0,n-1000)*u[m]
        forecast[m]=dict(measured_window_sec=selected[m]['window_wall_sec'],measured_steps=selected[m]['environment_steps'],
            observed_seconds_per_step=q[m],seconds_per_update=u[m],
            pilot_seconds=estimate(3000,q[m]),pilot_range_seconds=[estimate(3000,v) for v in q_range[m]],
            formal_seconds_per_seed=estimate(10000,q[m]),formal_range_seconds_per_seed=[estimate(10000,v) for v in q_range[m]],
            formal_three_seeds_seconds=3*estimate(10000,q[m]),
            formal_three_seeds_range_seconds=[3*estimate(10000,v) for v in q_range[m]])
    a=next(r for r in data['runs'] if r['tag']=='closing_handoff_correctness')
    eval_point=20*a['window_wall_sec']+sum(40*r['window_wall_sec']/3 for r in selected.values())
    eval_range=[.8*eval_point,1.8*eval_point]
    startups=sum(r['stages']['startup']['wall'] for r in selected.values())
    per_checkpoint=max(r['model_save_sec']+r['replay_save_sec'] for r in io['checkpoint'])
    # 4 saves for each 3000-step pilot; 11 for each 10000-step formal job.
    # Simulator cold start was not independently timed; reserve it explicitly,
    # rather than calling an inferred readiness timestamp a measurement.
    simulator_start_allowance=120.
    pilot_extra=simulator_start_allowance+4*startups+8*per_checkpoint
    formal_extra=simulator_start_allowance+3*startups+66*per_checkpoint
    pilot_point=sum(v['pilot_seconds'] for v in forecast.values())+eval_point+pilot_extra
    pilot_range=[sum(v['pilot_range_seconds'][i] for v in forecast.values())+eval_range[i]+pilot_extra for i in [0,1]]
    formal_point=sum(v['formal_three_seeds_seconds'] for v in forecast.values())+formal_extra
    formal_range=[sum(v['formal_three_seeds_range_seconds'][i] for v in forecast.values())+formal_extra for i in [0,1]]
    proposed_wall=math.ceil(pilot_range[1]/3600)*3600+120
    contract=(ROOT/'runtime/logs/contract_tests_final.log').read_text()
    ready=all(r['requested_episodes']==3 and r['completed_episodes']==3 and r['failure'] is None and
              r['weights_unchanged'] and all(r['safe_close']) for r in selected.values())
    ready=ready and selected['F']['success_count']==3 and audit['status']=='PASS' and probe['status']=='PASS' and '\nOK\n' in contract
    b=budget_state()
    lines=['# Day9：先导启动前审核报告','',
        '已停在用户要求的确认点。**先导训练0步、正式训练0步、100回合验证尚未启动；配置是待验证候选，不是已通过先导的冻结配置。**',
        '',f"本次选择的执行配置：`{c['day9']['profile']}`，包含 `{', '.join(c['day9'].get('accepted_optimizations',[]))}`。唯一网络候选为P1。",'',
        '## 正确性结果与限制','',
        '修复了驱动夹爪领先被动手指导致接触角错误锁存的问题：闭合搜索改为−0.10 rad/s速度PI，并保留搜索积分进入原保持控制。第一次不保留积分的修复触发原3 mm保护而失败，已保留。修复后A、F真实双侧接触、抬升、双时钟5秒保持和放回回归通过；L3仅有一个固定中心场景通过，不能推广为全课程通过。',
        '', '保留Day8全部奖励、20步上限、安全检查和成功判据。F实际训练接口使用外部策略输出经原残差融合链路；B不调用基础控制器。独立真实回放接口检查B/F各10次更新、权重变化、有限损失、保存/加载一致。**这20次是接口probe，不是先导学习证据。**',
        '', '未解决或未验证：闭合搜索的−0.10 rad/s是参考值，反馈仍有约−0.46 rad/s起始瞬态，不能声称实际硬限速；这需要在决定长先导前明确审阅。ROS接触消息有真实对象配对和位置，但normals/depths数组为空，不能宣称真实穿透深度为零。保留全任务区域，原近端不可见和远端IK/碰撞失败仍按任务失败计数，不伪造步数。L1/L2未追加物理回归，L3诊断未持久化逐条实际派发延迟（未来先导/验证已加入记录）。SciPy的NumPy版本范围警告仍存在。尚无3000步学习效果、课程覆盖或最终测试结论。',
        '', '调用链、关节方向/零点、动作/滤波、奖励全部系数、课程、抓放控制和恢复限制见 [CORRECTNESS.md](CORRECTNESS.md)。',
        '', '## 有限对照结果','',
        '每行B/F各3个固定场景、模型权重固定、deterministic=True、无学习/无课程推进；B每回合20步，F每回合4步。完整窗口含首次复位、连续回合复位、抓放、随机化、推理及最终恢复，未移动计时边界。每个回合最大600秒。',
        '', '|配置|B完整窗口 s / 60步|F完整窗口 s / 12步|相对基准 B / F|F成功|',
        '|---|---:|---:|---:|---:|']
    for profile in ['baseline','open_servo','reset_once','no_resend','combined']:
        if ('B',profile) not in paired or ('F',profile) not in paired:continue
        br,fr=paired[('B',profile)],paired[('F',profile)]
        changes=[100*(paired[(m,profile)]['window_wall_sec']/paired[(m,'baseline')]['window_wall_sec']-1) for m in ['B','F']]
        lines.append(f"|{profile}|{br['window_wall_sec']:.2f}|{fr['window_wall_sec']:.2f}|{changes[0]:+.1f}% / {changes[1]:+.1f}%|{fr['success_count']}/3|")
    lines += ['', '正百分比表示更慢。样本少且RTF波动，没有统计显著性结论；早期F基准窗口曾与几秒接口单元测试重叠，潜在负载扰动保留标注，后续未并行执行学习或I/O基准。修复后F基准与所有后续对照使用相同闭合控制；较早完成的B基准从未进入抓取闭合链路，故复用。早期旧闭合A/F准备回合不作为修复后通过证据。',
        '', '组合F后两回合保持阶段RTF曾降到约0.25–0.30（此前约0.8），4核系统load约13，单次采样system约49%、无交换/steal异常；原因尚未确认。完整慢回合保留，不能把差异全部归因于优化，也不按RTF倍率推算全流程。证据见 results/combined_load_observation.json、combined_cpu_sample.json 和原始阶段日志。',
        '', '参数及取舍：', '',
        '- open_servo：旧开爪目标1.30 rad、实际约1.57 rad、恒定保持1.20 Nm；候选在1.30±0.01 rad且速度≤0.01 rad/s并稳定0.2 sim s才完成，速度指令≤0.10 rad/s，PI P=.75/I=2、位置增益=.5、力矩≤1.50 Nm。修正继续推向机械限位的行为，独立对照**性能失败**，不宣传提速；重复开爪重新建立积分也占用时间。',
        '- reset_once：只在反馈证明已张开且停稳时复用同次reset第二次开爪，其他稳定、视觉、IK、碰撞和实体读回不省略。是否采纳及证据见候选decision记录。',
        '- no_resend：同轨迹最多2次发送改为1次，仍等实际派发、完整原轨迹时长、位置/速度反馈及稳定时间，不允许后续正常动作覆盖未完成轨迹。',
        '- 没有变更闭合/保持阈值、动作周期1秒（原到位稳定条件仍执行）、滤波alpha=.25、物理步长.002秒或RTF目标；没有删除RGB-D/OGRE2。GUI/RViz原已关闭，收益为0。',
        '- 每次物理随机化约秒级而非主要瓶颈；未复用实体以免未验证的速度状态或物性残留，未减小课程范围。IK缓存、从退离位直接进下回合、速度/加速度自适应时长均证据不足，未采用。网络/梯度次数/预算未为提速改变。',
        '', '全部回合、单项墙钟（松爪运动/松爪稳定/退离/回观察分别列出）、接近/接触RTF和声明预算见 [MEASUREMENTS.md](results/MEASUREMENTS.md) 与 [measurements.json](results/measurements.json)。父阶段包含子阶段，不能逐列加总；每个阶段另保留exclusive时间。准备、测量、故障恢复和更新probe分别记录。',
        '', f"全局诊断上限2小时、最多{b['max_attempts']}次尝试/{b['max_env_steps']}步/20次接口更新；实际已记{b['attempts']}次尝试、{b['step_attempts']}步、{b['probe_updates']}次接口更新，声明的诊断时钟至停机约{b['elapsed_through_stop_wall_sec']/60:.2f}分钟。预算入口已关闭，不自动追加物理实验。20/40条件对照未触发：B是逐步远离目标，缺乏持续有效接近却被截断的证据。没有追加P1以外超参数候选。",'',
        '## 耗时粗估与暂停点','',
        '估算依据选定组合的**完整窗口墙钟/累计环境步数**。物理窗口已含复位、随机化、抓放及收尾；未含网络更新，因此只另加实测单次更新×max(0,N−1000)。正式每种子10000步，每方法3种子，共60000步，A不训练。没有沿用旧43天或80000步数字。',
        '', '|方法|实测秒/步|更新秒/次|先导3000步 h|正式10000步 h/种子（条件范围）|3种子顺序 h（条件范围）|',
        '|---|---:|---:|---:|---:|---:|']
    for m,v in forecast.items():
        lo,hi=v['formal_range_seconds_per_seed'];l3,h3=v['formal_three_seeds_range_seconds']
        lines.append(f"|{m}|{q[m]:.3f}|{u[m]:.3f}|{v['pilot_seconds']/3600:.2f}|{v['formal_seconds_per_seed']/3600:.2f} ({lo/3600:.2f}–{hi/3600:.2f})|{v['formal_three_seeds_seconds']/3600:.2f} ({l3/3600:.2f}–{h3/3600:.2f})|")
    lines += ['',f"剩余先导及100回合验证：点估计 **{pilot_point/3600:.2f} h**，条件范围 **{pilot_range[0]/3600:.2f}–{pilot_range[1]/3600:.2f} h**。其中100回合验证单列 {eval_point/3600:.2f} h，范围 {eval_range[0]/3600:.2f}–{eval_range[1]/3600:.2f} h；启动/检查点另列约 {pilot_extra:.1f} s。建议确认的先导总墙钟上限为 {proposed_wall} 秒（含120秒恢复余量），超过即保存并停止，不自动加时。",'',
        f"正式两方法各3种子顺序运行：点估计 **{formal_point/3600:.2f} h（{formal_point/86400:.2f}天）**，条件范围 **{formal_range[0]/3600:.2f}–{formal_range[1]/3600:.2f} h**。启动/66次检查点额外约 {formal_extra:.1f} s 已加到这个总数。正式额外验证未指定回合预算，单独按测得每回合耗时估算，不能暗中加入60000训练步。",'',
        f"检查点每次实测模型+全容量回放写入最高 {per_checkpoint:.4f} s（OS缓冲写，不是fsync持久化保证）；启动构造分别 B={selected['B']['stages']['startup']['wall']:.3f} s、F={selected['F']['stages']['startup']['wall']:.3f} s，首回合接口/复位等待已经在完整窗口中。原始I/O/log测量见 [io_probe.json](results/io_probe.json)。",'',
        f"仿真冷启动未独立计时，以上额外开销各含一次120秒预留，不伪称实测；若每种子都重启，再单列相应冷启动。正式每追加20个验证场景，按当前测量约 A={20*a['window_wall_sec']/3600:.2f} h、B={20*selected['B']['window_wall_sec']/3/3600:.2f} h、F={20*selected['F']['window_wall_sec']/3/3600:.2f} h，学习后的抓取频率变化仍需更新。",'',
        f'这些是粗估而非承诺：仅L0三个配对场景，B平均20步、0次抓取，F平均4步、每回合1次抓取。B学会接近后抓取频率增加，不能把当前廉价失败吞吐外推全程；F课程扰动、初始化失败、回合长度和本机RTF变化也会改变秒/步。条件范围假定B物理秒/步为{q_range["B"][0]:.2f}–{q_range["B"][1]:.2f}，F为{q_range["F"][0]:.2f}–{q_range["F"][1]:.2f}；B低端取当前B的0.85倍，高端取较慢方法的1.6倍；F低端取较高RTF时独立开爪控制完整窗口与当前组合较小者的0.75倍作敏感性参照，高端为当前组合的1.6倍。这里没有按RTF比例折算流程。100验证回合为观测均值的0.8–1.8倍。这不是置信区间，也未用单个4步成功回合代表全训练。持续初始化失败会安全停止，不能保证在区间内完成指定步数。',
        '', '先导完成后自动生成 results/PILOT_REPORT.md 和 formal_forecast_post_pilot.json，按包含网络更新的真实完整训练窗口重估正式耗时，不再重复加更新；同时校正3000步与10000步预热比例。',
        '', '## 交付与下一步','',
        f"隔离审核：{audit['status']}，Day8原{audit['source_files']}文件及上级Git索引不变，无指向外部的可写副本链接；没有修改共享Python环境或提交Git。详情：[isolation_final.json](provenance/isolation_final.json)。",'',
        '主要文件：day9_env.py提供公共A/B/F执行与计时、安全恢复和夹爪修正；day9_training.py提供真实外部SAC采样/回放/更新、验证、保存和冻结；run_day9.py为统一入口和预算/配置护栏；day4_env.py增加有限动作检查、失败信息并固定副本资源根；residual_env.py/day8_env.py增加外部策略注入及实际派发完成信息；session_env.sh/manage_day9.py隔离输出及进程；generated四个模型文件仅迁移资源路径。配置、固定场景、模型、每次声明和源代码快照均在Day9内。完整修改清单：[change_list.json](provenance/change_list.json)。',
        '', '原奖励全部系数：100×距离进展、成功+50、掉落−30、碰撞−30，没有其他项。完整SAC实际默认值和自动规则见 [resolved_P1.json](results/resolved_P1.json)，执行配置见 [candidate.yaml](configs/candidate.yaml)。没有生成已验证frozen.yaml。',
        '', '运行、暂停、恢复能力以及先导确认后/冻结后命令见 [README.md](README.md)。训练入口不会自动启动仿真或正式多种子训练。先导需用户确认本报告；正式训练还需先导完整性检查与冻结，且每种子重新初始化网络和经验池。部分先导中断时会保存逐作业状态，但编排器不自动续跑已完成验证，恢复需要明确选择未完成阶段。']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    evidence=[]
    for r in data['runs']:
        evidence += [ROOT/'results'/r['run']/name for name in ['declaration.json','summary.json','episodes.json','events.jsonl']]
    evidence += [ROOT/'results'/name for name in ['offline_checks.json','update_probe.json','io_probe.json','resolved_P1.json']]
    evidence += [ROOT/'runtime/logs/contract_tests_final.log',ROOT/'provenance/source.json',ROOT/'configs/optimization_decisions.json']
    report=dict(status='AWAITING_USER_CONFIRMATION',ready_for_pilot=bool(ready),candidate_status='CANDIDATE_UNVALIDATED',
        ready_scope='interface/budget and three-scene function checks only; not full physical correctness, speed benefit, convergence or optimality; review velocity transient and missing contact fields before approving pilot',
        candidate_sha256=hash_file(ROOT/'configs/candidate.yaml'),code_hashes=code_hashes(),artifact_hashes=artifact_hashes(),
        evidence_hashes={str(p.relative_to(ROOT)):hash_file(p) for p in evidence},
        report_sha256=hash_file(ROOT/'REPORT.md'),selected_profile=c['day9']['profile'],forecast=forecast,
        pilot_total_seconds=pilot_point,pilot_total_range_seconds=pilot_range,proposed_pilot_wall_limit_sec=proposed_wall,
        simulator_cold_start_allowance_sec=simulator_start_allowance,checkpoint_write_sec=per_checkpoint,
        validation_seconds=eval_point,formal_total_seconds=formal_point,formal_total_range_seconds=formal_range,
        pilot_environment_steps=0,formal_environment_steps=0,validation_slots_executed=0,
        interface_probe_updates=20,budget=b,isolation=audit,created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    write_json('results/review.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['code_hashes','artifact_hashes','evidence_hashes','forecast','budget','isolation']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
