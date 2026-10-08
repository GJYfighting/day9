# Day9 本机实验入口

原先导前审核已完成，本轮用户授权阶段验收后自动推进至独立 Day10 的 B/F 各一个正式种子。先读 [REPORT.md](REPORT.md)、[CORRECTNESS.md](CORRECTNESS.md) 和 [短测表](results/MEASUREMENTS.md)。实际进度看 `results/pipeline_state.json`；冻结前仍为待验证候选。原 Day8 README 在 `provenance/day8/README.md`。

统一入口为 `run_day9.py`；不要用复制来的旧 manage_day8.py/run_validation.py 替代它，否则没有本次预算与 A/B/F 外部策略护栏。

## 启停隔离仿真

独立终端运行，默认无 GUI/RViz，保留 ign、OGRE2、RGB-D 和原兼容修复：

```bash
source /home/ubuntu/ros2_ws/day9/session_env.sh
python3 manage_day9.py serve
```

查看/停止：`python3 manage_day9.py status`、`python3 manage_day9.py stop`；serve 也可 Ctrl-C。只处理 PID、启动时间和 Day9 环境标识一致的进程。停止训练时先让训练进程保存并安全恢复，再停仿真。没有真机入口。

## 本轮自动阶段入口

持久入口：

```bash
cd /home/ubuntu/ros2_ws/day9
nohup /bin/bash -lc 'source ./session_env.sh && exec /usr/bin/python3 -u automation/run_pipeline.py' \
  > runtime/logs/day9_pipeline.log 2>&1 < /dev/null &
```

PID另见 `runtime/pipeline_process.json`。`results/pipeline_state.json`记录当前阶段及完整结果；中断后从同一命令恢复，已核验完成的阶段不重复执行。训练每200步或30分钟在下一安全回合边界保存。发送 `SIGTERM` 给编排器PID会让当前训练保存并安全收尾，再记录失败状态；停止仿真使用 `source session_env.sh; python3 manage_day9.py stop`。Day10建立后使用自己的 `session_env.sh`、`manage_day10.py`、`run_day10.py` 和 `automation/run_formal.py`，状态及日志都在Day10。禁止两个副本同时运行相同Gazebo世界。

## 旧单入口P1先导（保留以供核对）

**以下命令当前没有执行。** 使用审核报告SHA及其建议的总墙钟上限；代码/配置/固定场景/vendor改变后会拒绝旧报告。

```bash
source /home/ubuntu/ros2_ws/day9/session_env.sh
DAY9_REVIEW_SHA=$(sha256sum results/review.json | cut -d ' ' -f 1)
DAY9_PILOT_LIMIT=$(python3 -c 'import json; print(json.load(open("results/review.json"))["proposed_pilot_wall_limit_sec"])')
DAY9_PILOT_LOG=$(mktemp "$DAY9_ROOT/runtime/logs/pilot_XXXXXXXX.log")
python3 -u run_day9.py pilot --config configs/candidate.yaml \
  --report-sha "$DAY9_REVIEW_SHA" --wall-limit-sec "$DAY9_PILOT_LIMIT" \
  > "$DAY9_PILOT_LOG" 2>&1
```

预算：B/F各3000环境步，种子20260909，fresh网络/回放，超过1000步后每步更新一次，预期各2000次。验证种子20260910，固定20场景（10标准、10扰动），A一次、B/F训练前后各一次，共100槽；不学习、不写入训练回放、不推进课程。任务reset失败计失败分母，不造步数；技术故障最多一次重试。连续20次任务初始化失败或墙钟耗尽即保存停止，不自动加时、加种子、加候选或强制升级课程。

结果见 `results/pilot_summary.json`、`results/PILOT_REPORT.md`、`results/formal_forecast_post_pilot.json` 及各次训练/验证目录。短训无改善不等于算法无效，不要求损失单调。

## 先导完整后冻结

```bash
python3 run_day9.py freeze --config configs/candidate.yaml
```

当前没有 frozen.yaml。冻结会检查B/F各3000步/2000次更新、权重变化、100验证槽及代码/配置/模型源/依赖/场景哈希，不会把未验证候选冒充冻结配置。冻结后不根据最终测试结果调参，也不自动启动正式训练。

## 旧Day9三种子正式入口（不属于本轮授权）

B/F各3种子，每种子10000步，共60000步；A不训练。每作业 fresh，不继承先导或 sac_smoke.zip。墙钟上限取先导实测重估的条件上界并加120秒恢复余量，应结合先导报告确认。保持隔离仿真终端运行后顺序执行：

```bash
source /home/ubuntu/ros2_ws/day9/session_env.sh
set -e
DAY9_B_LIMIT=$(python3 -c 'import json,math; print(math.ceil(json.load(open("results/formal_forecast_post_pilot.json"))["B"]["conditional_range_seconds"][1])+120)')
DAY9_F_LIMIT=$(python3 -c 'import json,math; print(math.ceil(json.load(open("results/formal_forecast_post_pilot.json"))["F"]["conditional_range_seconds"][1])+120)')
DAY9_B_LOG=$(mktemp "$DAY9_ROOT/runtime/logs/formal_B_XXXXXXXX.log")
python3 -u run_day9.py train --method B --config configs/frozen.yaml \
  --seeds 20260911 20260912 20260913 --steps-per-seed 10000 \
  --fresh --wall-limit-sec "$DAY9_B_LIMIT" > "$DAY9_B_LOG" 2>&1
DAY9_F_LOG=$(mktemp "$DAY9_ROOT/runtime/logs/formal_F_XXXXXXXX.log")
python3 -u run_day9.py train --method F --config configs/frozen.yaml \
  --seeds 20260911 20260912 20260913 --steps-per-seed 10000 \
  --fresh --wall-limit-sec "$DAY9_F_LIMIT" > "$DAY9_F_LOG" 2>&1
```

wall-limit 是每种子上限；seeds 可只给三个指定种子中的一个。中途不完整会退出，避免自动继续后续种子。

## 中断与恢复

SIGINT/SIGTERM/墙钟截止保存策略及优化器、回放、Python/NumPy/Torch/动作空间RNG、课程和计数，安全放回/开爪/回观察；不能安全恢复则暂停隔离仿真。SIGKILL/主机崩溃只能用最近完整检查点，不保证OS缓冲尚未落盘的数据。达到200步或30分钟后在下一回合边界保存，阶段结束再保存；latest.json只是最新指针。

正式作业恢复示例（单种子，累计预算仍10000步，不能同时给fresh）：

```bash
python3 run_day9.py train --method B --config configs/frozen.yaml \
  --seeds 20260911 --steps-per-seed 10000 \
  --resume-state models/formal/B_20260911/latest.json \
  --wall-limit-sec "$DAY9_B_LIMIT"
```

恢复从新安全reset开始，不是物理中途逐位续接。先导逐作业也保存同样状态，可由 `day9_training.train_one(..., stage='pilot', initial=...)` 恢复；当前pilot编排器不自动合并部分先导或重复已完成验证，需要明确选择未完成阶段。此路径尚未做真实中断恢复实验。

## 文件与检查

配置及固定场景集中在configs；`optimization_decisions.json`记录功能/性能取舍。每个 `results/<run>` 保留测试前声明、预算、代码/模型哈希、事件与回合记录。早期诊断运行的源码快照已无损归档，见下文。`results/runs.jsonl`为短测索引。`models/diagnostic`与pilot模型分开。来源/修改/隔离/实际进程输出审计在provenance。

2026-09-30 完成文件数量整理：`configs/` 与根目录代码是冻结输入，`models/pilot/` 保留 B/F 最终及前一个完整检查点，`results/` 保留全部原始训练、验证与失败记录，`vendor/` 是冻结清单校验的独立依赖副本。旧恢复检查点和可重建缓存已删除；历史 ROS/Gazebo 日志与早期诊断源码快照分别无损归档为 `provenance/runtime_logs_20260930.tar.gz` 和 `provenance/diagnostic_sources_20260930.tar.gz`。逐文件哈希和清理范围在 `results/cleanup_20260930.json`、`results/structure_cleanup_20260930.json`。需要查看单个归档文件可用 `tar -tzf <归档路径>`；需要恢复原路径时，在 Day9 根目录执行 `tar -xzf provenance/<归档文件名> -C .`。保留 `runtime/logs/day9_pipeline*.log` 和 `runtime/logs/contract_tests_final.log` 供现有入口直接读取。

```bash
python3 -m unittest -v test_day9_contract.py
python3 run_day9.py check
```

以上接口检查不做物理训练，不能当学习证据。独立probe的20次更新预算已用完，再次调用会拒绝。物理benchmark受本次全局单调时钟预算约束，不得重启入口暗中扩预算。所有生成物在Day9；删除前先停副本进程，不改原Day8或共享安装。

可选单条相机信息排查（仿真运行时，保留 `--once`；此次未额外执行）：

```bash
DAY9_CAMERA_INFO=$(mktemp "$DAY9_ROOT/runtime/camera_info_XXXXXXXX.yaml")
timeout 20s ros2 topic echo /depth_cam/rgbd/camera_info --once > "$DAY9_CAMERA_INFO"
```
