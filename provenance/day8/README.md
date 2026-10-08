# DAY8：30 mm 木块域随机化与功能验收

DAY8 仅做建模、冻结策略推理和功能验收，不进行 SAC 训练。实际验收状态以
`results/day8_check.json` 为准；不能把脚本正常退出、完成一次尝试或几何检查通过解释为抓取成功。

## 来源、隔离和保持不变的内容

以 DAY7 实际生成模型为基准：桌面世界高度 0.750 m；机器人世界平移
(-0.200, 0, 0.750) m；原木块是 50 mm、30 g、mu=mu2=6。
`src/simulations` 原始相机配置为 640×400，而 DAY7 运行模型为 320×200；
原始桌面模型摩擦为 0.8，DAY8 继承的是 DAY7 的 6.0，不能混用两套基准。
运行时 CameraInfo：fx=fy≈277.191356、cx=160、cy=100，保留 ogre2。

DAY7 必需的代码、vendor、网格模型、SAC 权重、URDF/SDF、控制器和 MoveIt 配置
都实体复制到 DAY8，不复制历史日志和旧验收结果。补充复制 MoveIt 的空 sensors 配置及
Pilz 限制，入口显式使用本地文件。
MoveIt 自动构建器的已生效默认参数另存为 moveit_v5/launch_defaults.yaml，
本地入口直接构造参数，删除原包回退加载；19 组参数与原运行入口完全相同，
见 results/moveit_entry_equivalence.json。ROS/Gazebo 系统组件及原有 Python 系统依赖只读复用，
没有安装或升级共享环境。所有生成文件、缓存、日志都在 `runtime/`、`results/` 下。
独立 ROS_DOMAIN_ID=90、IGN_PARTITION=day8_ubuntu、IGN_IP=127.0.0.1。
Ignition 的应用 HOME、Ogre 日志、XDG、Torch、临时目录均隔离。

`day4_env.py` 的观测、奖励、成功判据和 `residual_env.py` 的 SAC 结构及融合保持不变；
观测 10 维、动作 4 维；保留已有置信度门控、残差限幅与低通。
课程窗口与最短驻留均为 100，晋级成功率≥0.80、降级≤0.55，技术失败不计入课程窗口。
policy_gain 固定 0.8；删除域配置、采样与应用中的增益随机化，原控制器仍存在。固定过闭合目标量按 30 mm 几何改为 0.04 rad，不属于增益修改。

## 先验与文献核查

2026-09-13 核查原始论文全文：

1. Josh Tobin 等，**Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World**，2017。
   [原文](https://arxiv.org/abs/1703.06907)，DOI: 10.48550/arXiv.1703.06907。
   第 III-A 节直接支持对象位置、成像噪声、照明与相机位姿随机化的方法。
   其相机扰动为 10×5×10 cm 区域、角度偏移至 0.1 rad、视场缩放至 5%；
   这些不是本项目相机误差的测量值，不移植为 DAY8 精确区间。
2. Xue Bin Peng 等，**Sim-to-Real Transfer of Robotic Control with Dynamics Randomization**，2018 ICRA（2017 年预印本）。
   [作者提供的论文](https://xbpeng.github.io/projects/SimToReal/SimToReal_2018.pdf)，
   [arXiv](https://arxiv.org/abs/1710.06537)。
   第 IV-C 节、表 I 直接支持质量、摩擦、观测噪声与动作时间变化的随机化类别。
   其实验对象为推动圆盘，质量 0.1–0.4 kg、摩擦 0.1–5，采用不同的时间分布；
   不能据此认定 30 g 木块及本项目延迟区间经过真机验证。

以下所有**具体数值及 L1/L2/L3 划分均为项目工程先验**，并非论文直接给出的
本机械臂标定区间；均尚待真机校准。尺寸容差用于模拟木块制造/建模差异。
30 g 和摩擦 6.0 是继承仿真基准，不声称木材实测摩擦为 6。
额外深度噪声 0.1/0.2/0.3 mm 仅为暂用工程设置，不声称覆盖真实深度相机误差。

|参数|L0 标称|L1|L2|L3|
|---|---|---|---|---|
|立方体边长 mm|30|29.4–30.6|28.95–31.05|28.5–31.5|
|质量 g|30|27–33|24–36|21–39|
|mu、mu2（分别采样）|6|5.4–6.6|4.8–7.2|4.2–7.8|
|亮度、对比度倍率（分别采样）|1|0.9–1.1|0.8–1.2|0.7–1.3|
|RGB 像素高斯噪声 σ（0–255）|0|2|5|8|
|额外深度高斯噪声 σ mm|0|0.1|0.2|0.3|
|额外深度无效比例|0|1%|3%|5%|
|外参平移向量模长上限 mm|0|2|5|8|
|外参旋转角上限 °|0|0.5|1|2|
|额外控制时延 ms|0|均匀 0–20|均匀 0–50|均匀 0–100|
|白色区域连续位置采样|全区域|全区域|全区域|全区域|

标量在区间内均匀采样；三轴共用一个尺寸系数；质量独立采样，按当前质量和边长重算
Ixx=Iyy=Izz=m·s²/6，交叉项为零。平移/旋转向量使用随机方向与均匀半径，限制的是模长，
不是每轴独立上限。噪声 σ 固定为该等级指定值，逐像素独立高斯采样；无效像素为独立
Bernoulli 掩码，单幅实现比例会有统计波动。固定 seed 决定参数及 RGB-D 随机流。
L0 除位置外没有额外扰动；本次相机修复后的静态 TF 作为新的标称外参。

## 固定观察姿态与尺寸联动、时延、观测边界

2026-09-14 按用户要求，对齐 `jetarm` 俯视形状识别抓取的初始化命令：
舵机脉冲 `[500,540,220,50,500]`，对应 J1–J5 为 `[0,9.6,-67.2,-108,0]°`，
弧度 `[0,0.16755160819145565,-1.1728612573401895,-1.8849555921538759,0]`。
[真机姿态源码](https://github.com/GJYfighting/source-code/blob/70502c8/jetarm/src/jetarm_6dof/jetarm_6dof_rgbd_cam/scripts/shape_recognition_down.py#L71)。
初次仅同步关节角；后续相机安装修复见下文，不声称已有真机逐台内参标定。

每回合通过已有 `ign service remove/create` 重建本地 SDF，visual/collision 同步修改。
中心世界 z=0.750+s/2，标称 0.765；复位和恢复同样使用当前中心高度。
生成后用 `generate_world_sdf` 读回检查质量、摩擦、两种几何尺寸和惯量。
本地实测 Gazebo 尺寸向量输出六位小数、惯性标量可输出六位有效数字；
校验仅接受原数值或对应的确定性舍入表示，visual/collision 读回仍必须完全一致。
曾修正尺寸读回的舍入校验；旧配置回合不作为当前验收依据。
名义感知高度固定 30 mm、尺寸筛选 25–40 mm。实际下降目标继续由**视觉表面**计算，
夹爪开口与接触位置继续由 URDF 和**视觉测得宽度**计算；不把每回合真实尺寸输入策略。
保留原始开口目标 1.3 rad 及至少 4 mm 桌面间隙的安全阈值。
原 10 mm 表面插入量不适合 30 mm 方块：闭合后的指尖几何最低点低于桌面约 9 mm。
诊断比较了插入量 10、0、−5、−10 mm；−5 mm 下标称过闭合状态间隙约 5.2 mm。
因此暂将固定有符号插入量设为 −7 mm（接触中心在**视觉顶面上方 7 mm**），给最小尺寸
留下间隙，并对张开至预计接触角减 0.04 rad 的闭合过程取 17 个角度检查。
这是尺寸变化所需的工程调整，不改变奖励、成功判据或安全阈值。
原 0.08 rad 过闭合量在 28.5–31.5 mm 尺寸下对应 3.78–3.94 mm 闭合行程，
超过原 3 mm 安全阈值；固定改为 0.04 rad 后为 1.86–1.95 mm，仍大于原 1.5 mm
最小接触闭合量。依据见 results/overclose_geometry.json，原控制增益保持不变。
尺寸极值安全性由网格核验和实际回合结果记录，不能仅凭标称数值推断。

感知更新只发送亮度、对比度、噪声、外参扰动及随机种子白名单；真实位置、质量、尺寸
不进入该通道。Gazebo 真值只用于复位、物理读回和继承的成功/安全检查，不作为视觉观测。

控制队列使用独立 `/clock` 订阅执行线程，按仿真秒安排发送，同步 IK/服务调用不阻塞该线程。
原先只在主线程 spin 时发出命令导致最长约 2.4 仿真秒积压，已修正并重跑受影响等级。
发送器必须收到首帧有效 `/clock` 才允许入队，防止把初始化的 0 当作当前仿真时间。
初始化门控另有确定性测试；墙钟只用于等待时钟连接的超时判断。
ROS 回调继续运行，时钟暂停时不会因墙钟等待而释放命令。每条命令记录 requested_sec、queued_sim、sent_sim、actual_sec。
actual_sec 是仿真时钟下入队至实际发布的延迟；控制器接收后的动力学响应不混入该数值。
实际延迟包含 `/clock` 接收与调度量化，可能高于抽样值；必须看记录，不能把抽样值当实效值。

## 白色识别区域与本次验收

采用 `jetarm` 配置（仓库提交 70502c8）保存的中心 x、y 和 135×175 mm 尺寸：
[配置来源](https://github.com/GJYfighting/source-code/blob/70502c8/jetarm/src/hiwonder_imgproc/color_detection/config/config.yaml#L194)。
矩形在仿真 base_link 下的中心为 **(0.1845922266, -0.0124389012, 0.0000000000) m**；
短边 135 mm（区域局部 x）、长边 175 mm（区域局部 y）。
保留真机矩阵第一轴投影得到的平面 yaw；去掉标定的 roll/pitch 及 z=0.8145 mm 偏差，
将框贴在实际仿真桌面 z_base=0 上，不倾斜桌面或木块。
机器人世界平移仍为 (-0.2,0,0.75) m，因此框中心世界坐标为
(-0.0154077734,-0.0124389012,0.75) m。
标称 30 mm 木块中心为 base_link **(0.1845922266,-0.0124389012,0.015) m**，
世界中心 z=0.765 m；随机尺寸仍按桌面+s/2 更新。

本次要求取代此前的离散网格课程：L0–L3 全部在矩形内连续均匀采样，
不读取旧有效点集合，不按抓取成功率删点。按当前尺寸和相对 yaw 内缩可采样中心范围，
确保木块完整落在矩形内部；不能把木块中心放到框边上而让半个木块越界。
矩形四角为框本身边界，不等于木块中心允许到达的极值。
感知工作区及标称中心已同步，世界模型增加只有 visual、没有 collision 的白色边框。
随机尺寸/真实位置仅用于生成和复位，未加入策略视觉观测。
其他等级扰动、奖励、成功判据、置信度门控、课程升降和固定 policy_gain=0.8 不变。
旧结果与冻结清单已清理，不适用于新姿态。
本次实际结果以 results/day8_check.json 为准。全区域可采样不代表已证明全区域可见或可抓。

## 可执行命令

所有命令先运行：

```bash
source ~/ros2_ws/day8/session_env.sh
```

三个终端分别启动：

```bash
ros2 launch ~/ros2_ws/day8/world.launch.py gui:=false
ros2 launch ~/ros2_ws/day8/moveit.launch.py
python3 ~/ros2_ws/day8/perception_v5.py --ros-args --params-file ~/ros2_ws/day8/perception_v5.yaml
```

第四个终端按序验收（不要与另一个抓取/网格检查并发控制机械臂）：

```bash
ros2 topic echo /depth_cam/rgbd/camera_info --once > runtime/camera_info.yaml
python3 day8_check.py --capture-reference
python3 white_area_check.py
python3 white_area_check.py --view
python3 day8_check.py --offline
python3 day8_check.py --nominal
python3 day8_check.py --smoke
python3 day8_check.py --finalize
python3 day8_check.py --verify-freeze
```

完整自动验收（已启动三个进程后）：`python3 run_validation.py --prepare`。新区域不会因旧网格未生成而等待。

单等级复核：`python3 day8_check.py --smoke --level L3`。
默认推理入口：`python3 day8_env.py --mode fixed --level L0 --episodes 1`。
不要将历史 DAY6/DAY7 检查脚本作为 DAY8 验收入口。

## 精简后的文件结构

- 根目录：启动、感知、环境、策略融合、几何与验收脚本；`manage_day8.py` 管理 DAY8 进程。
- `generated/`：当前 URDF/SDF；`camera_baseline/` 内两个实体模型是相机重建输入，必须保留。
- `meshes/`、`models/`、`moveit_v5/`、`simulations/`：运行模型、SAC 权重、控制器和桌面配置。
- `vendor/`：已验证兼容的私有 NumPy/OpenCV/Gymnasium/SB3 依赖，未改动库源码或二进制。
- `results/`：当前参数、视觉与抓取验收、源文件基线哈希、冻结清单和清理记录。
- `runtime/`：可再生的日志、缓存和临时模型；启动脚本自动创建所需目录。

已删除 GitHub 下载副本、旧姿态/旧网格结果、重复冒烟副本、历史日志、临时诊断和字节码缓存。
当前 `smoke_L0.json` 保留全部 6 次尝试，包括失败及同点同 seed 技术重试，没有只留成功记录。
删除旧网格绘图与一次性几何探针；`--grid`、`--visibility` 统一使用当前矩形验收。
冻结脚本直接枚举当前必需资产，不再依赖历史冻结清单。清理清单见 `results/cleanup_manifest.json`。

## 相机安装修复

相机原点从腕部轴线移到实际 CAD 外壳前方。保持现有 end_effector_link 抓取 TCP，
新增 camera_calibration_hand，只用于解释真机手眼矩阵。
真机手坐标 +X 对应仿真 link4 +Z（沿腕部向前），+Y 对应 link4 +Y，
+Z 对应 link4 -X（CAD 相机外壳所在一侧）。参考原点沿 link4 +Z 距离为
0.05945583202+0.112=0.17145583202 m，来自 jetarm_6dof_params.py，
不是将仿真的 80 mm 抓取 TCP 改成长夹爪。
真机 hand2cam_tf_matrix 的平移 [-0.101,0,0.045] m 和旋转完整应用于这个参考系。
由此得到成像点相对 link4：**[-0.045,0,0.07045583202] m**；
Gazebo X-forward 相机坐标旋转等价 rpy=[0,-pi/2,0]。
新增 depth_cam_optical_frame 明确区分光学 Z-forward 与 Gazebo X-forward；
感知仍使用原有 optical_to_camera_rotation，未重复旋转像素。

外壳重命名为 camera_housing_link，黑白 CAD 网格与光学原点分离；
黑色外壳 collision 与 visual 使用同一几何变换，白色外壳按 CAD 同坐标对齐。
MoveIt 原有两项相机邻接碰撞排除仅随外壳改名，没有增加排除对。
启动器 switch-timeout 由默认值改为 30 s，以容纳软件渲染启动；控制增益不变。
相机水平视场和分辨率仍为原仿真 1.047 rad、320×200，没有为通过检查扩大视场。
真机仓库 GEMINI 默认 RGB 640×480、depth 640×400，实际标定 CameraInfo 尚未取得；
不能把仿真内参声称为真机实测值，也未套用其他 Gemini 型号的宣传参数。

复现模型：`source ~/ros2_ws/day8/session_env.sh && python3 repair_camera.py`。
该脚本从 generated/camera_baseline/day3_v5_robot.urdf 实体基线重建，
用 ign sdf -p 得到相机转换结果，再仅将相机 visual、frame 和 sensor pose 写入原 SDF；
原 SDF 的全部 collision、joint 和 plugin 按元素字节校验保持不变，避免恢复已移除的网格碰撞或改变指尖接触名称。
URDF/SDF 实体基线均保留在 DAY8 内。
旧相机失败图像和作废的混合画面已清理；当前通过 generate_world_sdf 回读实际传感器位置。
正常重启必须等待 DAY8 的 Gazebo 子进程退出，不能仅停止外层 launch。

相机修复复核命令（已启动 DAY8 三个进程后）：

```bash
source ~/ros2_ws/day8/session_env.sh
python3 repair_camera.py --verify-live
python3 day8_check.py --capture-reference
python3 white_area_check.py --view
python3 day8_check.py --nominal --technical-retries 0
python3 day8_check.py --smoke --level L0 --technical-retries 0
python3 day8_check.py --finalize
python3 day8_check.py --verify-freeze
```

这里的 `--technical-retries 0` 仅限制本次功能检查的重复尝试，不改变课程配置。
每点均保留成败记录；不能依据一次中心成功宣布整个矩形可抓取。

## GitHub 账号交叉核查与本次最终结果

2026-09-14 核查账号的四个公开仓库：source-code、simulations、ros2_ws、day7。
检索下载副本已清理，以下保留仓库版本、来源链接与核查结论。
本次查询的树版本：simulations 8acc002ec70f6e81e12d7c52d37f618dbfc3758a、
ros2_ws b9e4d178e6e4507ae345d8ea33b67d5c7cedb917、day7 d0c57670531fedc3ab2a6d9ce991f8b9e8fa044a。

- [仿真传感器定义](https://github.com/GJYfighting/simulations/blob/main/robot_gazebo/urdf/camera.gazebo.xacro)：绑定 depth_cam_link，水平视场 1.047 rad，640×400。
- [仿真安装定义](https://github.com/GJYfighting/simulations/blob/main/jetarm_6dof_description/urdf/depth_camera.urdf.xacro)：原 depth_cam_joint 位于 link4 的 [0,0,0.014475] m；外壳 visual 位移不等于成像点位移。与本地旧模型的错误一致。
- [真机 GEMINI 启动配置](https://github.com/GJYfighting/source-code/blob/main/jetarm/src/third_party/ros_astra_camera/launch/gemini.launch)：默认 RGB 640×480、depth 640×400；启动默认值不等于真机运行时实测值。
- [真机相机内参读取](https://github.com/GJYfighting/source-code/blob/main/jetarm/src/third_party/ros_astra_camera/src/ob_camera_info.cpp)：getCameraParams 从设备属性读取参数，getColorCameraInfo 使用标定管理器或设备参数生成 CameraInfo，异常时有默认值回退。
- src/peripherals/config/camera_info.yaml 标注 camera_name=usb_cam，不能作为 GEMINI 标定。所查材料没有可确认属于目标 GEMINI 的完整实测 K/D；不使用其他相机参数代替。

当前仍保留 DAY8 的 320×200、1.047 rad；它与原仿真 640×400 宽高比相同，
单纯提高到 640×400 不会扩大几何视场。相机安装是基于源码矩阵和 CAD 坐标推导的仿真修复，尚不是经真机重新标定的外参。

实际验收：
- 相机 SDF 运行时回读、187 条机器人视觉网格射线检查通过；原物理碰撞、关节与插件保持不变。
- 5×5 木块完整可见检查 20/25 通过，靠近基座一侧 5 个点出画。保持原矩形采样范围，没有剔除失败点。
- 30 mm 标称中心抓取通过，检测中心误差约 0.35 mm，完成双指接触、抬升、保持和归还桌面。
- L0 五个不同点中，中心与 y 两侧边缘共 3 点完成抓取和归还；近侧点有效检测 0/12，远侧点预抓取 IK 因 link3/servo_link2 自碰撞被拒绝。
- 共 6 次尝试：5 次首轮，y 正向点遇到 Gazebo 回读超时、时钟停止后以相同点位和 seed 重试一次并通过。原技术失败记录保留；归还确认共 5/6，不能把该次历史故障说成恢复成功。
- 合并记录 results/smoke_L0.json；首轮及重试均已合并，冗余副本已清理。
- 本次相机修复功能通过，但全矩形视觉/抓取验收 FAIL。L1–L3 未在本次相机配置下重新验收，旧结果不得用于宣布通过。未运行正式 SAC 训练。

本次保留 repair_camera.py、manage_day8.py 和 analyze_self_occlusion.py；修改本地相机 URDF/SDF、config.yaml 相机元数据、SRDF 外壳名称、world.launch.py 启动等待，以及验收脚本。策略、控制器固定增益、指定预观察关节角和矩形保持不变。
可用 `source ~/ros2_ws/day8/session_env.sh && python3 manage_day8.py restart` 启动或重新启动 DAY8 三个进程，再执行上一节复核命令。
冻结的是当前可复现修复快照，未通过项保留在 results/day8_check.json，不代表 DAY8 全部验收通过。

## 精简后复核

清空旧缓存后重新启动 Gazebo、MoveIt 和感知，实际相机模型回读通过；
4000 组区域采样与各等级 100 组参数验证通过。相机模型可从保留基线重新生成，
106 处绝对模型资源引用有效，私有依赖、权重和网格等必需资产哈希未变。
SAC 权重加载通过，观测 10 维、动作 4 维；一次新的标称视觉抓取完成夹持、抬升、保持与归还。
当前结果替代旧标称记录；L0 所有边缘成败记录仍保留。
清理只改变文件组织与旧网格验收分支，不修复现有近侧视野和远侧 IK 限制，因此整体验收仍为 FAIL。
