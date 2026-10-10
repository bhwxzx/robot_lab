# 评估批次 latest49999-analysis-20261010-003

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-leg-v0`；训练：`2026-10-10_01-25-39`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| zero-delay-a01 | completed | zero-delay，2000 / 1 / 42 | complete | [结果](<../latest49999-reference1009-20261010-001/raw/zero-delay-a01/result.json>) / [日志](<../latest49999-reference1009-20261010-001/raw/zero-delay-a01/console.log>) |
| play-equivalent-a02 | completed | play-equivalent，2000 / 1 / 42 | complete | [结果](<../latest49999-reference1009-retry-20261010-002/raw/play-equivalent-a02/result.json>) / [日志](<../latest49999-reference1009-retry-20261010-002/raw/play-equivalent-a02/console.log>) |
| fixed-15ms-a02 | completed | fixed-15ms，2000 / 1 / 42 | complete | [结果](<../latest49999-reference1009-retry-20261010-002/raw/fixed-15ms-a02/result.json>) / [日志](<../latest49999-reference1009-retry-20261010-002/raw/fixed-15ms-a02/console.log>) |
| training-randomized-a01 | completed | training-randomized，2000 / 1 / 42 | complete | [结果](<../latest49999-reference1009-20261010-001/raw/training-randomized-a01/result.json>) / [日志](<../latest49999-reference1009-20261010-001/raw/training-randomized-a01/console.log>) |
| play-equivalent-a01 | failed | play-equivalent，2000 / 1 / 42 | — | — / [日志](<../latest49999-reference1009-20261010-001/raw/play-equivalent-a01/console.log>) |
| fixed-15ms-a01 | failed | fixed-15ms，2000 / 1 / 42 | — | — / [日志](<../latest49999-reference1009-20261010-001/raw/fixed-15ms-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`b46799d67a3d0ad5e686257fe3832a0c41c5af29dc9a8e3cb1cef41ffc1c3d12`；runner：`OnPolicyRunner`。
- [场景来源 scenario-9555e3b5fdedd5186f0c120b3e2d2209218f92539009c542cf0cdf359da3df70.json](<../../provenance/scenario-9555e3b5fdedd5186f0c120b3e2d2209218f92539009c542cf0cdf359da3df70.json>)
- [场景来源 scenario-aa6be8ecd36bc39815d81966fe24db3c65d63811200047e3efbb348b431de05c.json](<../../provenance/scenario-aa6be8ecd36bc39815d81966fe24db3c65d63811200047e3efbb348b431de05c.json>)
- [场景来源 scenario-c6ebce903fb79668afa7dc6a7e2b9a84f1bba201cd2fd3a7b448939417eea6a6.json](<../../provenance/scenario-c6ebce903fb79668afa7dc6a7e2b9a84f1bba201cd2fd3a7b448939417eea6a6.json>)
- [场景来源 scenario-fc714f4520ddd77d83c2bbfd61318eec8a2c6db92647109dbe7720f10b096bb1.json](<../../provenance/scenario-fc714f4520ddd77d83c2bbfd61318eec8a2c6db92647109dbe7720f10b096bb1.json>)
- [训练上下文 context-76e09091681988239986b9e1ed36574d9892693c8042bcb3297115b7e1778442.json](<../../provenance/context-76e09091681988239986b9e1ed36574d9892693c8042bcb3297115b7e1778442.json>)
- [训练有效配置 config-af14ff75f436ff28e8c227255ee5ea4a03ef2c5f8c4f2c6e9d820b11cc19b5a8.json](<../../provenance/config-af14ff75f436ff28e8c227255ee5ea4a03ef2c5f8c4f2c6e9d820b11cc19b5a8.json>)

- `zero-delay-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `zero-delay-a01` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.motion_file": "source/robot_lab/robot_lab/datasets/LW/motion_beyondmimic/se3_trajopt/2026-10-09/leg_to_wheel_transform_50hz.npz", "evaluation.motion_mode": "nominal_start", "evaluation.motion_physics_window": [0, 166], "evaluation.motion_sha256": "47805c271159df64c6ebc0a40e283bfa2361333ee7de6d0698e33f371bd02293"}`。
- `play-equivalent-a02` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `play-equivalent-a02` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.joint_position_range": [0, 0], "commands.motion.motion_file": "source/robot_lab/robot_lab/datasets/LW/motion_beyondmimic/se3_trajopt/2026-10-09/leg_to_wheel_transform_50hz.npz", "commands.motion.pose_range": {}, "commands.motion.velocity_range": {}, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 166], "evaluation.motion_sha256": "47805c271159df64c6ebc0a40e283bfa2361333ee7de6d0698e33f371bd02293", "events.add_joint_default_pos": null, "events.base_com": null, "events.physics_material": null, "events.push_robot": null, "events.randomize_actuator_gains": null, "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "observations.policy.enable_corruption": false}`。
- `fixed-15ms-a02` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `fixed-15ms-a02` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.joint_position_range": [0, 0], "commands.motion.motion_file": "source/robot_lab/robot_lab/datasets/LW/motion_beyondmimic/se3_trajopt/2026-10-09/leg_to_wheel_transform_50hz.npz", "commands.motion.pose_range": {}, "commands.motion.velocity_range": {}, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 166], "evaluation.motion_sha256": "47805c271159df64c6ebc0a40e283bfa2361333ee7de6d0698e33f371bd02293", "events.add_joint_default_pos": null, "events.base_com": null, "events.physics_material": null, "events.push_robot": null, "events.randomize_actuator_gains": null, "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "observations.policy.enable_corruption": false, "scene.robot.actuators.foots.max_delay": 3, "scene.robot.actuators.foots.min_delay": 3, "scene.robot.actuators.legs.max_delay": 3, "scene.robot.actuators.legs.min_delay": 3, "scene.robot.actuators.wheels.max_delay": 3, "scene.robot.actuators.wheels.min_delay": 3}`。
- `training-randomized-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `training-randomized-a01` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.motion_file": "source/robot_lab/robot_lab/datasets/LW/motion_beyondmimic/se3_trajopt/2026-10-09/leg_to_wheel_transform_50hz.npz", "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 166], "evaluation.motion_sha256": "47805c271159df64c6ebc0a40e283bfa2361333ee7de6d0698e33f371bd02293"}`。
- `play-equivalent-a01` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.joint_position_range": [0, 0], "commands.motion.motion_file": "source/robot_lab/robot_lab/datasets/LW/motion_beyondmimic/se3_trajopt/2026-10-09/leg_to_wheel_transform_50hz.npz", "commands.motion.pose_range": {}, "commands.motion.velocity_range": {}, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 166], "evaluation.motion_sha256": "47805c271159df64c6ebc0a40e283bfa2361333ee7de6d0698e33f371bd02293", "events.add_joint_default_pos": null, "events.base_com": null, "events.physics_material": null, "events.push_robot": null, "events.randomize_actuator_gains": null, "events.randomize_apply_external_force_torque": null, "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "observations.policy.enable_corruption": false}`。
- `play-equivalent-a01` 未完成原因：evaluation exit=1; see console log
- `fixed-15ms-a01` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.joint_position_range": [0, 0], "commands.motion.motion_file": "source/robot_lab/robot_lab/datasets/LW/motion_beyondmimic/se3_trajopt/2026-10-09/leg_to_wheel_transform_50hz.npz", "commands.motion.pose_range": {}, "commands.motion.velocity_range": {}, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 166], "evaluation.motion_sha256": "47805c271159df64c6ebc0a40e283bfa2361333ee7de6d0698e33f371bd02293", "events.add_joint_default_pos": null, "events.base_com": null, "events.physics_material": null, "events.push_robot": null, "events.randomize_actuator_gains": null, "events.randomize_apply_external_force_torque": null, "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "observations.policy.enable_corruption": false, "scene.robot.actuators.foots.max_delay": 3, "scene.robot.actuators.foots.min_delay": 3, "scene.robot.actuators.legs.max_delay": 3, "scene.robot.actuators.legs.min_delay": 3, "scene.robot.actuators.wheels.max_delay": 3, "scene.robot.actuators.wheels.min_delay": 3}`。
- `fixed-15ms-a01` 未完成原因：evaluation exit=1; see console log

## 观察与限制

- 结论：该本机最新 checkpoint 可以在本次四个场景中完整完成 leg-to-wheel 变形。每场景 Native、1 环境、seed=42、2000 控制步（40 s），控制周期20ms，物理采样5ms；均有11次第0帧起始且motion_finished无其他失败终止的完整成功，1个未结束尾段未计成功。
- 检查对象：训练2026-10-10_01-25-39/model_49999.pt，internal iter=49999；checkpoint SHA256 b46799d67a3d0ad5e686257fe3832a0c41c5af29dc9a8e3cb1cef41ffc1c3d12，参考10-09 NPZ SHA256 47805c271159df64c6ebc0a40e283bfa2361333ee7de6d0698e33f371bd02293。
- 启动命令由用户确认 bash scripts/start_beyondmimic.sh --type leg；原训练提交2b2bb9e7e37b54652ca60164e948e33a9bf4b3d2，与本次采集提交8e369d5b779e31099ade30e16480ee057b3e02ec的相关训练源码一致，均保留配置中10-09参考路径。
- 训练最后1000轮：平均reward 7.3443、motion_finished 99.8335%、body position error 0.01766m、body rotation error 0.09766rad，训练步数到49999且所检查指标有限。训练采用随机阶段起点；本次完整轨迹统计独立验证，未有批准的收敛判据。
- 场景：zero-delay关闭观测噪声、环境/初始化随机化及驱动延迟；play-equivalent关闭噪声和环境/初始化随机化，保留训练的0-3物理子步驱动延迟；fixed-15ms在Play条件下固定3子步延迟；training-randomized保留训练随机化/噪声/0-3子步延迟，但固定从参考第0帧开始。
- RMSE使用全部2000控制样本（包含未结束尾段）；body位置采用机器人anchor XY和yaw对齐的参考，anchor位置误差表示未对齐的位置跟踪；腾空统计仅采用11次完整回合。腾空定义为1.0-1.9s内最长连续四个足/轮接触力范数均<1N的区间，检测范围仅这四个连杆。
- 代表回合的接触力、俯仰角与力矩见相关PNG；随机化场景首个回合仅为示例，最大接触峰值来自其他回合。接触力为仿真5ms物理步平均法向力，驱动力矩为PD量，未包含被动冲击载荷。
- 四场景指标：

|场景|完整成功|body RMSE cm|anchor RMSE cm|非轮关节 RMSE rad|腾空 min/median/max s|双轮Fz最大 kN|5ms最大力矩利用率|
|---|---|---|---|---|---|---|---|
|zero-delay|11/11|2.76|3.46|0.118|0.220/0.220/0.220|2.44|77.3%|
|play-equivalent|11/11|2.71|4.52|0.116|0.220/0.225/0.230|2.65|79.1%|
|fixed-15ms|11/11|2.60|4.23|0.110|0.230/0.230/0.230|2.44|78.7%|
|training-randomized|11/11|4.02|15.43|0.142|0.205/0.225/0.260|3.74|100.0%|

- zero-delay：1.0-1.8s起跳/腾空/落地窗口实际pitch范围 -12.46° 至 8.51°，最大绝对pitch在参考控制步末 1.36s、回合0。
- play-equivalent：1.0-1.8s起跳/腾空/落地窗口实际pitch范围 -16.18° 至 8.47°，最大绝对pitch在参考控制步末 1.36s、回合10。
- fixed-15ms：1.0-1.8s起跳/腾空/落地窗口实际pitch范围 -16.18° 至 7.50°，最大绝对pitch在参考控制步末 1.36s、回合1。
- training-randomized：1.0-1.8s起跳/腾空/落地窗口实际pitch范围 -19.83° 至 12.98°，最大绝对pitch在参考控制步末 1.36s、回合0。
- 腾空与参考：零延迟实际1.32-1.54s、0.22s；Play 0.220-0.230s；固定15ms 0.230s；训练随机化0.205-0.260s，参考计划1.30-1.55s、0.25s。1/5/10N检测灵敏度已记录于phase-contact JSON，不属于成功验收阈值。
- 轮式阶段1.8-3.34s的已采集物理步：四场景双轮接触比例100%（每轮接触范数>5N），足部接触比例0%；非轮关节RMSE依次0.049、0.068、0.069、0.078rad；轮速RMSE依次0.559、1.342、0.447、1.825rad/s。测试只覆盖参考末尾，未评价随后长时间轮式稳定性。
- 随机化场景PD限幅：32个关节-物理子步样本，左小腿12、左足14、右足6，发生于参考0.09-1.175s准备阶段；最大计算驱动力矩141.24Nm，左小腿应用上限120Nm、足关节27Nm，应用力矩超限0。落地峰值约3.74kN，发生于1.560-1.565s物理区间；落地阶段PD最大利用率43.8%，冲击接触载荷需独立评估。
- 首批Play和固定延迟的2次失败是评估配置中不存在events.randomize_apply_external_force_torque字段导致的启动失败；修正场景覆盖后原2场景成功重测，所有失败日志保留。此采纳报告合并原证据，不重复运行、不复制遥测、不追加重复经验事件。
- 统计范围：单seed单环境；零延迟/固定延迟重复回合基本确定性。训练随机化仅覆盖该进程一个startup质量/摩擦/COM样本与多次reset扰动，不能外推随机化分布成功率，也未建立硬件载荷承受或总体收敛结论。
- 仅可进入受监督实物测试；未经实物验证，不代表 hardware-ready。
- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据

- [manifest.json](<../latest49999-reference1009-20261010-001/manifest.json>)
- [manifest.json](<../latest49999-reference1009-retry-20261010-002/manifest.json>)
- [phase-contact-20261010-001.json](<../../evidence/motion_phase/phase-contact-20261010-001.json>)
- [contacts-pitch-effort-20261010-001.png](<../../evidence/motion_phase/contacts-pitch-effort-20261010-001.png>)
- [summary-latest49999-20261010-001.json](<../../evidence/training/summary-latest49999-20261010-001.json>)

## 建议与待授权事项

- 后续诊断优先关注起跳俯仰偏差、提前/延后接触与落地载荷；任何奖励、轨迹、PD或延迟参数修改先提出方案并获用户批准。
- 若要评估统计鲁棒性，应另行约定多seed/多startup载荷预算及接触/姿态/跟踪验收指标；当前测试不构成随机化分布保证。
- 本次仅新增评估证据和分析图，保留现有10-09 motion_file修改，未修改项目源码、训练参数或安装软件。
