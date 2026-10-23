# 评估批次 wheel-to-leg-mechanics-20261023-002

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-wheel-v0`；训练：`2026-10-22_17-40-29`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| play-ground-a01 | completed | play-ground，1000 / 1 / 42 | complete | [结果](<../wheel-to-leg-latest-20261023-001/raw/play-ground-a01/result.json>) |
| training-random-phase-a01 | completed | training-random-phase，1000 / 1 / 42 | complete | [结果](<../wheel-to-leg-latest-20261023-001/raw/training-random-phase-a01/result.json>) |

## 策略与场景

- checkpoint SHA-256：`9b42606f73222fd18d17949be0aa829fde37758ec2c99f23774722bd8cf0b952`；runner：`OnPolicyRunner`。
- [场景来源 scenario-82f5ae48c3949ffe984156750a659482caa34c7777e2cbb575dda0721da4cf62.json](<../../provenance/scenario-82f5ae48c3949ffe984156750a659482caa34c7777e2cbb575dda0721da4cf62.json>)
- [场景来源 scenario-84f30ca999aeae8839d5db6e3657e31b6c280aad4393d8ded73690496c03851b.json](<../../provenance/scenario-84f30ca999aeae8839d5db6e3657e31b6c280aad4393d8ded73690496c03851b.json>)
- [训练上下文 context-4642d359340c5b75a11c4e1a35489f11aed8711d6946d77e24490829c794656a.json](<../../provenance/context-4642d359340c5b75a11c4e1a35489f11aed8711d6946d77e24490829c794656a.json>)
- [训练有效配置 config-139b35afadcf0265f8b113ccd586f88cf1a212d7adf644125b614c524989be61.json](<../../provenance/config-139b35afadcf0265f8b113ccd586f88cf1a212d7adf644125b614c524989be61.json>)

- `play-ground-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `play-ground-a01` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.joint_position_range": [0, 0], "commands.motion.pose_range": {}, "commands.motion.velocity_range": {}, "evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "4d17c9a09f1fd6201bf14d72a468eec9c8a58000b6f67777d93b964c387f5d62", "events.add_joint_default_pos": null, "events.base_com": null, "events.physics_material": null, "events.push_robot": null, "events.randomize_actuator_gains": null, "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "observations.policy.enable_corruption": false, "rewards.wheel_contact_continuous.weight": 0.5}`。
- `training-random-phase-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `training-random-phase-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "randomized_phase", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "4d17c9a09f1fd6201bf14d72a468eec9c8a58000b6f67777d93b964c387f5d62"}`。

## 观察与限制

- 本批次回答新的动作与载荷问题，复用既有的两组 Native 结果进行离线分析；新增仿真步数为 0。原执行批次、原报告及原始遥测保持不变。
- 阶段姿态与双轮接触主统计使用固定回放的 5 个完整 frame-zero 周期；逐关节全程峰值/RMS 保留全部 1000 控制步（含末尾未完成片段）。随机组为不同起始相位的 12 个完成片段及 1 个未完成片段，不能当作 12 次完整起跳。
- 准备阶段定义为参考时间 0–1.10 s，蹬伸为 1.10–1.36 s；这些仅为报告窗口。力、力矩、速度采样间隔 5 ms，姿态和角度采样间隔 20 ms。
- 准备阶段 5/5 完整周期均出现单轮短暂失载：法向力 <1 N 的最长连续段，右轮 75 ms、左轮 65 ms；未记录到双轮同时 <1 N。排除最初 20 ms 后，双轮同时 >=1 N 的时间比例均值 85.2%，同时 >=奖励力阈值 10 N 的比例均值 75.4%。接触间隙和 PhysX 烘焙后碰撞形状未记录，不能把这些失载片段等同于已测得明显抬轮高度。
- 准备阶段机身绝对俯仰最大 11.52°（参考最大 1.65°），横滚最大 2.64°；蹬伸阶段横滚最大 5.40°。最大值可能发生在不同相位，逐时刻姿态与参考均见曲线及 JSON。
- 髋关节两侧原始轴方向均为 -X，镜像对称应比较 q_left 与 -q_right，并减去参考的不对称量。准备阶段镜像残差峰值 5.39°、RMS 2.38°；蹬伸峰值 16.92°、RMS 9.56°。参考 1.22 s 的一个完整周期，左/右髋为 +17.85°/-0.91°，存在明显非镜像姿态。
- 固定回放完整周期的主要驱动力矩峰值发生于蹬伸左小腿：108.30 N·m，参考时间约 1.265 s，占 120 N·m 上限 90.3%；落地右小腿峰值 92.41 N·m。全部记录中各关节均未达到驱动力矩限幅。
- 固定回放全部记录中，小腿驱动力矩 RMS 左/右为 28.12/24.52 N·m。右足关节末尾未完成片段出现 10.88 rad/s 的单个 5 ms 速度样本，高于保存配置中的 10 rad/s；5 个完整周期中足关节峰值左/右为 9.70/9.55 rad/s。
- 随机组左小腿达到 120 N·m 限幅，累计 15 个物理采样（75 ms）；右足关节在触地的 1.605 s 达到 -21.80 rad/s，下一物理步驱动力矩达到 +27 N·m 上限。其余每个关节峰值、RMS、超限样本及相位见 CSV/JSON。
- 固定回放的首次落地在参考时间 1.600–1.620 s。单足法向接触力峰值 1.521 kN，双足同时合力峰值 2.064 kN，约为该组 26.73 kg 质量对应重力的 7.87 倍；5 个周期首次触地后约 100 ms 的法向接触冲量为 59.45–62.36 N·s。
- 随机组足端同时法向合力峰值至少 2.448 kN；右大腿在 1.670 s 出现 5.014 kN 对地法向力，约为该组 27.89 kg 重力的 18.33 倍。2/12 个完成片段出现大腿触地，未完成末尾片段也出现一次。
- 这里的冲击峰值是 5 ms 物理步平均法向力，排除摩擦；不是瞬时峰值。遥测中的 applied_torque 为驱动力矩，不包含碰撞引起的被动关节反力矩，因此不能用驱动力矩限幅代替实际机械冲击载荷判断。
- 仅可进入受监督实物测试；未经实物验证，不代表 hardware-ready。
- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据

- [mechanics-analysis-v2.json](<derived/mechanics-analysis-v2.json>)
- [mechanics-analysis.json](<derived/mechanics-analysis.json>)
- [manifest.json](<../wheel-to-leg-latest-20261023-001/manifest.json>)
- [landing-contact-detail-20261023-001.json](<../../evidence/training/landing-contact-detail-20261023-001.json>)
- [preparation-support-pose.png](<derived/preparation-support-pose.png>)
- [joint-torque-speed-play-ground-a01.png](<derived/joint-torque-speed-play-ground-a01.png>)
- [joint-torque-speed-training-random-phase-a01.png](<derived/joint-torque-speed-training-random-phase-a01.png>)
- [landing-loads.png](<derived/landing-loads.png>)
- [joint-peak-rms.csv](<derived/joint-peak-rms.csv>)

## 建议与待授权事项

- 后续验证应同时覆盖准备阶段俯仰、髋关节镜像残差、双轮承载，以及落地足端/大腿冲击、足关节瞬态速度和执行器限幅。
- 优先检查准备姿态及双侧支撑的跟踪、落地缓冲和随机相位恢复；当前证据不足以把某个奖励权重或延迟设置认定为唯一原因。
- 本批次仅生成离线分析证据，未修改源代码或训练参数，不启动训练。
