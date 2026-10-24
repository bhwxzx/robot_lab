# 评估批次 wheel-to-leg-local-20261024-001

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-wheel-v0`；训练：`2026-10-23_19-58-38`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| play-ground-a01 | completed | play-ground，1000 / 1 / 42 | complete | [结果](<raw/play-ground-a01/result.json>) / [日志](<raw/play-ground-a01/console.log>) |
| training-random-phase-a01 | completed | training-random-phase，1000 / 1 / 42 | complete | [结果](<raw/training-random-phase-a01/result.json>) / [日志](<raw/training-random-phase-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`ced8afebad41a9acb25e30338bd01ae341d08025353a0a1ec676e0419a2d62af`；runner：`OnPolicyRunner`。
- [场景来源 scenario-c2c3419d743386e9253d7174318e94da8afbe646074b056deeaa2dad6f389d67.json](<../../provenance/scenario-c2c3419d743386e9253d7174318e94da8afbe646074b056deeaa2dad6f389d67.json>)
- [场景来源 scenario-d557ac9a6eb465b257d69588c62c0ec39ad4a78106494fc4dc901609c26b1a70.json](<../../provenance/scenario-d557ac9a6eb465b257d69588c62c0ec39ad4a78106494fc4dc901609c26b1a70.json>)
- [训练上下文 context-bcfa7dba8d28196867492e42ecb501ae8598929a040ef62b7917b9f61282a58b.json](<../../provenance/context-bcfa7dba8d28196867492e42ecb501ae8598929a040ef62b7917b9f61282a58b.json>)
- [训练有效配置 config-bfa90867d72bcd8875064e7e627f40086df11ddeb29385d0f2d659af9a2a2c94.json](<../../provenance/config-bfa90867d72bcd8875064e7e627f40086df11ddeb29385d0f2d659af9a2a2c94.json>)

- `play-ground-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `play-ground-a01` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.joint_position_range": [0, 0], "commands.motion.pose_range": {}, "commands.motion.velocity_range": {}, "evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03", "events.add_joint_default_pos": null, "events.base_com": null, "events.physics_material": null, "events.push_robot": null, "events.randomize_actuator_gains": null, "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "observations.policy.enable_corruption": false, "rewards.wheel_contact_continuous.weight": 0.5}`。
- `training-random-phase-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `training-random-phase-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "randomized_phase", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03"}`。

## 观察与限制

- 本机 younghit 场次 2026-10-23_19-58-38，用户确认 bash scripts/start_beyondmimic.sh --type wheel、无额外覆盖；检查点 ced8afebad41a9acb25e30338bd01ae341d08025353a0a1ec676e0419a2d62af。使用 10/23 参考；action_rate_l2=-0.15，action_smoothness=-0.075；腿/轮/足执行延迟均为 0..3 物理步。
- 通用 PPO profile 需补充 BeyondMimic 契约：环境展开的分项 10 帧历史、590 维策略输入、10 维动作；每次 reset 历史一致性由运行时断言核验。
- play-ground-a01：1000 控制步、4000 物理步，遥测 complete；motion_finished 片段 5，其中从 frame 0 完整周期 5；非 motion_finished 终止 0，尾段 1。大腿触地的已完成片段 5/5。随机相位片段不能算完整跳跃次数。
- play-ground-a01，统计范围为 frame-zero complete cycles、n=5：起跳窗口 1.10–1.36 s 竖直速度峰值均值 0.587 m/s（参考 1.967）；1.20–1.30 s 速度 RMSE 0.991 m/s；1.20–1.70 s 高度 RMSE 9.97 cm；1 N 下记录八连杆最长连续失载均值 167.0 ms（参考计划飞行 300 ms）。
- play-ground-a01，准备阶段 0.02–1.30 s 双轮同时 ≥1 N 比例均值 79.8%；机身俯仰/横滚绝对峰值 7.18/4.18°；髋镜像残差峰值 6.65°。
- play-ground-a01 全程：单大腿/小腿对地力峰值 4.270 kN，right_thigh_link，参考 1.620 s；1.40–2.00 s 记录对地 Fz 合力峰值 4.282 kN。力矩到限样本：无；速度超名义上限：无。
- training-random-phase-a01：1000 控制步、4000 物理步，遥测 complete；motion_finished 片段 12，其中从 frame 0 完整周期 0；非 motion_finished 终止 0，尾段 1。大腿触地的已完成片段 4/12。随机相位片段不能算完整跳跃次数。
- training-random-phase-a01，统计范围为 completed fragments covering reference 1.10-1.80 s、n=2：起跳窗口 1.10–1.36 s 竖直速度峰值均值 0.487 m/s（参考 1.967）；1.20–1.30 s 速度 RMSE 1.280 m/s；1.20–1.70 s 高度 RMSE 10.04 cm；1 N 下记录八连杆最长连续失载均值 165.0 ms（参考计划飞行 300 ms）。
- training-random-phase-a01，准备阶段 0.02–1.30 s 双轮同时 ≥1 N 比例均值 76.3%；机身俯仰/横滚绝对峰值 8.89/5.84°；髋镜像残差峰值 11.56°。
- training-random-phase-a01 全程：单大腿/小腿对地力峰值 4.599 kN，left_thigh_link，参考 1.590 s；1.40–2.00 s 记录对地 Fz 合力峰值 5.916 kN。力矩到限样本：无；速度超名义上限：left_foot_joint:10.17/10, right_foot_joint:10.05/10。
- 参考轨迹源规划记录 IPOPT=-5（600 s 超时），严格 1e-6 节点可行性验收失败；它是用户指定比较基线。该问题不能单独证明本策略失败的原因。
- 5 ms 力为物理步平均法向力；未记录 base/hip 对地力，八连杆失载不等于全身腾空证明。驱动力矩不包含碰撞被动反力矩。
- 检查点、保存 YAML、既有导出、配置代码、参考轨迹和 Isaac Sim 持久用户设置的字节及 mtime 均未变化。评估 CLI 显式 --/app/settings/persistent=false。
- 单 seed 的有限仿真回放，未提供批准的收敛标准，不作收敛或实物成功结论。与异机历史结果仅作描述比较，不归因于单项奖励。
- 起跳结论采用 analysis-v2.json：1.10–1.36 s 为起跳峰速窗口，1.10–1.80 s 的较大峰值可能属于触地后恢复，不能作为起跳速度。固定 5 周期起跳峰速 0.559–0.611 m/s；1.30 s 速度均值 0.315 m/s（参考 1.827）。
- 固定周期下蹲最低基座高度均值 0.541 m，参考 0.490 m，偏高约 5.1 cm；最长八连杆失载段从参考 1.425–1.445 s 才开始，晚于计划离地 1.30 s。随后大腿首次触地 1.610–1.630 s。失载区间不是正确完成参考跳跃的证明。
- 1.10–1.30 s 固定完整周期，左/右大腿峰速 4.24/3.36 rad/s（参考 6.75/6.75），左/右小腿 3.67/4.65（参考 7.22/7.22）；存在蹬伸幅度/速度及左右时序偏差。全程无驱动力矩限幅，不能据此归因于电机额定能力不足。
- 训练 TensorBoard 确认至 step 49999；最后 1000 轮 motion_finished 均值 0.997443，mean_reward 6.9978（前 1000 轮 7.0740），anchor 线速度误差 0.12959（前段 0.12646）。指标未显示末段继续改善，且 motion_finished 不等于跳跃成功。
- 最后 1000 轮 sampling_top1_bin 均为 0.25。源代码该指标为 bin_index/bin_count，170 帧、50 Hz 对应 4 bins，故主采样为索引 1，起始相位大约 0.84–1.68 s；不是最后一段。其采样概率均值约 46.8%，与起跳附近仍困难一致，但不能定位单一原因。
- 奖励机制存在允许小幅变形、低跳后站起的可能：global anchor 位置权重 0.5、std=0.3 m；单独 10 cm 位置误差时该指数项仍约为 0.895（忽略其他轴）。速度奖励对多连杆误差平均，专门起跳/高度项已移除。该机制是待验证解释，不是单项奖励的因果结论；本轮不改奖励。
- 两组 Native 子进程均成功完成并通过 bundle 校验；封装器在等待报告审阅文本时遇到 PTY 单行截断。随后用相同 contract、原始结果和原始 console 完成首次封存，未重跑用例、未改写原始证据。
- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据

- [jump-contact-posture.png](<derived/jump-contact-posture.png>)
- [joint-torque-speed-play-ground-a01.png](<derived/joint-torque-speed-play-ground-a01.png>)
- [joint-torque-speed-training-random-phase-a01.png](<derived/joint-torque-speed-training-random-phase-a01.png>)
- [analysis.json](<derived/analysis.json>)
- [joint-peak-rms.csv](<derived/joint-peak-rms.csv>)
- [protection-after-20261024-001.json](<../../evidence/training/protection-after-20261024-001.json>)
- [training-origin-20261024-001.json](<../../evidence/training/training-origin-20261024-001.json>)
- [tensorboard-tail-20261024-001.json](<../../evidence/training/tensorboard-tail-20261024-001.json>)
- [analysis-v2.json](<derived/analysis-v2.json>)

## 建议与待授权事项

- 先按完整起跳的竖直速度、失载时长、准备支撑、左右姿态及大腿触地共同判断；本轮不修改配置或启动训练。
- 后续若需扩大验证，使用额外 seed 的有界新批次，保留全部旧证据。
