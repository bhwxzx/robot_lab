# 评估批次 wheel-to-leg-wheel-linear-20261110-001

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-wheel-v0`；训练：`2026-11-09_13-08-33`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| fixed-start-nominal-a01 | completed | fixed-start-nominal，1000 / 1 / 42 | complete | [结果](<raw/fixed-start-nominal-a01/result.json>) / [日志](<raw/fixed-start-nominal-a01/console.log>) |
| fixed-start-randomized-a01 | completed | fixed-start-randomized，1000 / 1 / 42 | complete | [结果](<raw/fixed-start-randomized-a01/result.json>) / [日志](<raw/fixed-start-randomized-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`235464c2ff1da90f610085b1670950c53ba74d1cda40013ffb82f9956fa3a16d`；runner：`OnPolicyRunner`。
- [场景来源 scenario-aca67b316f45851bafb0b305f67593e10ce69b4648ad8c92e32ec65c2c4fb9e8.json](<../../provenance/scenario-aca67b316f45851bafb0b305f67593e10ce69b4648ad8c92e32ec65c2c4fb9e8.json>)
- [场景来源 scenario-daa7ecd17b55c57281536be411550435c7f92f26e643d44964234a5629ce0880.json](<../../provenance/scenario-daa7ecd17b55c57281536be411550435c7f92f26e643d44964234a5629ce0880.json>)
- [训练上下文 context-5bb56f8ad0dd069723a7af1c7f2c711d960f8f5eac8fee9f3f230859dc3d253f.json](<../../provenance/context-5bb56f8ad0dd069723a7af1c7f2c711d960f8f5eac8fee9f3f230859dc3d253f.json>)
- [训练有效配置 config-48bc78af51bddd5cff843fdcc193f7d35afbc5c024d27452fa58b6f1b5757027.json](<../../provenance/config-48bc78af51bddd5cff843fdcc193f7d35afbc5c024d27452fa58b6f1b5757027.json>)

- `fixed-start-nominal-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `fixed-start-nominal-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "nominal_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03", "rewards.joint_acc_wheel_l2.params.ramp_time_s": 0.0, "rewards.joint_acc_wheel_l2.params.start_time_s": 0.0, "rewards.joint_acc_wheel_l2.weight": -0.002, "rewards.joint_vel_wheel_l2.params.ramp_time_s": 0.0, "rewards.joint_vel_wheel_l2.params.start_time_s": 0.0, "rewards.joint_vel_wheel_l2.weight": -0.1, "rewards.motion_body_ang_vel.params": {"body_names": ["base_link", "right_hip_link", "left_hip_link", "right_thigh_link", "left_thigh_link", "right_shank_link", "left_shank_link", "right_foot_link", "left_foot_link"], "command_name": "motion", "std": 3.14}, "rewards.motion_body_ori.params": {"body_names": ["base_link", "right_hip_link", "left_hip_link", "right_thigh_link", "left_thigh_link", "right_shank_link", "left_shank_link", "right_foot_link", "left_foot_link"], "command_name": "motion", "std": 0.4}}`。
- `fixed-start-randomized-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `fixed-start-randomized-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03", "rewards.joint_acc_wheel_l2.params.ramp_time_s": 0.0, "rewards.joint_acc_wheel_l2.params.start_time_s": 0.0, "rewards.joint_acc_wheel_l2.weight": -0.002, "rewards.joint_vel_wheel_l2.params.ramp_time_s": 0.0, "rewards.joint_vel_wheel_l2.params.start_time_s": 0.0, "rewards.joint_vel_wheel_l2.weight": -0.1, "rewards.motion_body_ang_vel.params": {"body_names": ["base_link", "right_hip_link", "left_hip_link", "right_thigh_link", "left_thigh_link", "right_shank_link", "left_shank_link", "right_foot_link", "left_foot_link"], "command_name": "motion", "std": 3.14}, "rewards.motion_body_ori.params": {"body_names": ["base_link", "right_hip_link", "left_hip_link", "right_thigh_link", "left_thigh_link", "right_shank_link", "left_shank_link", "right_foot_link", "left_foot_link"], "command_name": "motion", "std": 0.4}}`。

## 观察与限制

- 结论：本版标称场位置跟踪及落地后轮速有所改善，但随机场全程位置跟踪和足关节冲击没有整体改善；不能判为全面优于上一版本机策略。
- 来源：younghit，2026-11-09_13-08-33/model_49999.pt，SHA256 235464c2ff1da90f610085b1670950c53ba74d1cda40013ffb82f9956fa3a16d；4096环境、seed42、PPO/OnPolicyRunner。用户确认 bash scripts/start_beyondmimic.sh --type wheel，无额外CLI/Hydra参数。W&B缺少argv，本次用户确认已单独保存。训练HEAD ea876d372c3a04f6c68de505dfd3f7b1f8e8f98c，训练时flat_env_cfg.py为dirty；已从保存的原始diff重建并核对git blob，不将当前HEAD冒充训练来源。
- 当前评估HEAD 555c8bee543c127be75a5cf40a2c35cddad5863e 的轮配置与训练快照不同；本批使用8项显式运行时覆盖恢复训练奖励，未修改代码。训练为11个连杆的位置/线速度跟踪，姿态/角速度仅9个非轮连杆；轮速weight=-0.10、轮加速度weight=-0.002，start/ramp均0；高度weight=0，全局anchor位置weight=0.5/std=0.3。
- 有限Native测试：标称初态和随机化初态各1环境、1000控制步、seed42、无视频，完整记录每5ms物理子步。两场均5个完整成功周期、0失败、1个预算截断尾段；遥测完整、必需信号无缺失，原始bundle验证通过。完成周期不等于收敛或硬件可用。
- 对比基线：本机2026-11-08_10-58-49/model_49999.pt，SHA256 bd4d7eef0c02b1356598d1b5f3949f1f395d352a27b8e5c533fbb726bfd8c92e，wheel-to-leg-position-20261109-001。场景、seed、预算、环境数、motion SHA一致，原始bundle已重新验证。可观测的初始质量、COM、材质及PhysX刚度/阻尼一致；软件PD和延迟等随机量未独立完整捕获，不能认定全部随机实现完全相同。
- 配置差异共31项语义变化、agent配置无变化，同时包含连杆范围、轮奖励时段/权重、全局位置std 0.15→0.30、轮持续接触奖励0.5→0以及critic输入146→164。因此这里只比较配置组合效果，不声称某一奖励项导致改善。
- 位置口径：基座原点世界坐标，post-physics状态对齐下一参考帧(+20ms)，末帧参考保持；3D RMSE先按完整周期计算再取5周期均值。标称3D RMSE 4.742→3.306 cm；首次足触地附近X偏差-4.881→-1.807 cm；终点X偏差-4.425→-2.176 cm。负值表示相对参考落后。
- 随机场3D RMSE 15.858→17.003 cm，触地X绝对偏差均值15.531→14.142 cm，但新策略逐周期触地X偏差为[-3.52,-7.24,+30.31,+26.58,+3.06] cm；终点为[-0.94,-5.37,+28.17,+36.03,+8.13] cm。触地局部指标改善不能代表全程前向漂移改善。
- 起跳速度为基座COM世界线速度，不是整机COM；在参考阶段1.1–1.7s的控制采样中，标称vz峰值1.7407→1.7412 m/s，vx峰值1.9918→2.0162 m/s，未显示标称动作更偏竖直。随机各周期vz峰值均值1.8271→1.8911 m/s，新版范围1.6104–2.1408。标称基座高度峰值0.67823→0.67943 m，随机均值0.67910→0.68685 m。
- 参考轮支撑/腾空/足支撑分界来自原始motion audit：离地1.300s、触地1.600s。双轮接触法向力模长均<10N连续>=15ms定义离地，5ms分辨率；标称两版均1.325s(+25ms)，随机旧[1.380,1.370,1.350,1.370,1.335]、新[1.375,1.365,1.330,1.370,1.335]s，平均延迟61→55ms。1/5/10/20N敏感性结果保存在liftoff-and-impact-metrics.json。
- 主腾空段定义为最长的全部监测接触离地段：标称240→250ms，随机新版235–250ms。随机episode4先有1.335–1.345s短暂15ms无接触，随后1.355s开始240ms主腾空；forward-jump-metrics中的first-gap=15ms只表示首次短段，不能解释为整个跳跃腾空时间。
- 双轮速度RMS，完整周期、两轮合并、物理子步：1.25–1.65s窗口标称5.260→4.263 rad/s、随机6.939→6.270；1.85–3.4s窗口标称0.4267→0.3937、随机0.4664→0.4433 rad/s。落地后轮加速度RMS标称0.926→0.416 rad/s²，但随机1.687→2.595；轮加速度并非全面下降。全程轮速绝对峰值新版左/右为标称10.157/8.054、随机14.282/11.512 rad/s，均低于33 rad/s配置限值。
- 关节负载使用所有1000控制步的4000个物理子步，包含截断尾段，报实际applied_torque。新版随机左小腿峰速16.864 rad/s，力矩111.474/120 N·m；左足峰速10.015 rad/s(尾段一个5ms采样)，力矩26.087/27 N·m；右足峰速23.074 rad/s(配置速度限值10)，实际力矩27.000/27 N·m，99%限幅占5ms，未裁剪computed_torque峰值34.725 N·m。新版标称左足峰速14.949 rad/s，超10共30ms(6次重复落地各1子步)，力矩22.734 N·m。所有10关节数据和峰值时间分别在joint-loads.csv及forward-jump-metrics.json。
- 落地冲击诊断：随机episode4右足速度峰在1.595s，力矩限幅在1.600s；接触和关节邻近原始子步一并保存。足部最大接触法向力模长标称1261→1578N、随机1211→1590N；轮部随机峰951→1042N。没有检测到监测的大腿/小腿对地>10N接触。接触瞬态可使实际速度超过配置速度限值，该限值不是物理状态的严格夹紧。此处为数值仿真观测，不是实物力测量。
- 训练TensorBoard最终step49999；末段47500–49999 anchor位置误差均值0.1350m、motion_finished均值0.9955、noise均值0.1772，完整55标量的分窗口统计已保存。奖励定义发生变化，训练reward不能直接跨版排序；未提供批准的收敛验收标准，收敛结论为indeterminate/insufficient_evidence。
- 保全：1799个既有证据、相关源码、模型/配置/导出受保护文件hash全部一致；Git HEAD未变，唯一新增为本run证据目录；无训练启停、参数修改、导出或Git提交推送；评估结束GPU无计算进程。本批没有视频，动作判断来自遥测曲线。补充离地分析首次仅发生NumPy整型JSON序列化失败，未发布输出，a02按原口径修复后成功，原方法记录保留。
- 范围限制：单seed、每场5个完整周期，标称重复不是独立试验；未进行多seed、扰动强度扫描或实物测试，不能据此确认泛化更好。仅可进入受监督实物测试；未经实物验证，不代表 hardware-ready。
- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据

- [analysis-method.json](<derived/analysis-method.json>)
- [cases-ready.json](<derived/cases-ready.json>)
- [comparison-metrics.json](<derived/comparison-metrics.json>)
- [cycle-metrics.csv](<derived/cycle-metrics.csv>)
- [execution-receipt.json](<derived/execution-receipt.json>)
- [forward-analysis-method.json](<derived/forward-analysis-method.json>)
- [forward-jump-comparison.png](<derived/forward-jump-comparison.png>)
- [forward-jump-metrics.json](<derived/forward-jump-metrics.json>)
- [joint-loads.csv](<derived/joint-loads.csv>)
- [liftoff-and-impact-method-a02.json](<derived/liftoff-and-impact-method-a02.json>)
- [liftoff-and-impact-method.json](<derived/liftoff-and-impact-method.json>)
- [liftoff-and-impact-metrics.json](<derived/liftoff-and-impact-metrics.json>)
- [phase-comparison.png](<derived/phase-comparison.png>)
- [effective-config-diff-20261110-001.json](<../../evidence/training/effective-config-diff-20261110-001.json>)
- [evaluation-request-20261110-001.json](<../../evidence/training/evaluation-request-20261110-001.json>)
- [protection-after-20261110-001.json](<../../evidence/training/protection-after-20261110-001.json>)
- [protection-before-20261110-001.json](<../../evidence/training/protection-before-20261110-001.json>)
- [tensorboard-windows-20261110-001.json](<../../evidence/training/tensorboard-windows-20261110-001.json>)
- [training-origin-20261110-001.json](<../../evidence/training/training-origin-20261110-001.json>)
- [source-training-20261110-001.patch](<../../evidence/source/source-training-20261110-001.patch>)
- [manifest.json](<../../../2026-11-08_10-58-49/evaluations/wheel-to-leg-position-20261109-001/manifest.json>)
- [report.md](<../../../2026-11-08_10-58-49/evaluations/wheel-to-leg-position-20261109-001/report.md>)

## 建议与待授权事项

- 优先保留两版作对照，下一步可做匹配的多seed有限Native测试，重点统计落地足关节速度/力矩限幅、接触冲击、全程基座X误差与离地延迟；当前批次不自动扩展测试预算。
- 在解释奖励效果前，先复核1.57–1.67s足部着地姿态、左右触地顺序与执行器响应；本批证据不足以支持直接指定新的奖励权重，direct_parameter_change_supported=false。
