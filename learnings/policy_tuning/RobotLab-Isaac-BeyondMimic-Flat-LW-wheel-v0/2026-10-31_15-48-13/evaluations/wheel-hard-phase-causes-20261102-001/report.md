# 评估批次 wheel-hard-phase-causes-20261102-001

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-wheel-v0`；训练：`2026-10-31_15-48-13`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| random-phase-nominal-a01 | failed | random-phase-nominal，5000 / 1 / 42 | — | — / [日志](<raw/random-phase-nominal-a01/console.log>) |
| random-phase-randomized-a01 | completed | random-phase-randomized，5000 / 1 / 42 | complete | [结果](<raw/random-phase-randomized-a01/result.json>) / [日志](<raw/random-phase-randomized-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`d5d75f2da8d11057f00d65f4f73f13dcee9dfb4dec312d362622fc1b0a058059`；runner：`OnPolicyRunner`。
- [场景来源 scenario-d5eef33827088ce19e3d8501bef593fcfa26f5989a3c9c3206e501f36c3d4cef.json](<../../provenance/scenario-d5eef33827088ce19e3d8501bef593fcfa26f5989a3c9c3206e501f36c3d4cef.json>)
- [训练上下文 context-60e0625421acf2b392e2fb4747819ca55c61cfdd47f159030da914d0575159d4.json](<../../provenance/context-60e0625421acf2b392e2fb4747819ca55c61cfdd47f159030da914d0575159d4.json>)
- [训练有效配置 config-fec6608572a1beb4da6cba32099c69d5c38db56618098433f023566a1a54ab7f.json](<../../provenance/config-fec6608572a1beb4da6cba32099c69d5c38db56618098433f023566a1a54ab7f.json>)

- `random-phase-nominal-a01` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.joint_position_range": [0.0, 0.0], "commands.motion.pose_range": {}, "commands.motion.sampling_diagnostics_dir": "/home/server/liufengrong/robot_lab/learnings/policy_tuning/RobotLab-Isaac-BeyondMimic-Flat-LW-wheel-v0/2026-10-31_15-48-13/evaluations/wheel-hard-phase-causes-20261102-001/raw/random-phase-nominal-a01/sampling-diagnostics", "commands.motion.sampling_diagnostics_window_steps": 1000, "commands.motion.velocity_range": {}, "evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "randomized_phase", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03", "events.add_joint_default_pos": null, "events.base_com": null, "events.physics_material": null, "events.push_robot": null, "events.randomize_actuator_gains": null, "events.randomize_apply_external_force_torque": null, "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "observations.policy.enable_corruption": false, "rewards.hip_pos_zero_l2.weight": -2.0, "scene.robot.actuators.foots.max_delay": 0, "scene.robot.actuators.foots.min_delay": 0, "scene.robot.actuators.legs.max_delay": 0, "scene.robot.actuators.legs.min_delay": 0, "scene.robot.actuators.wheels.max_delay": 0, "scene.robot.actuators.wheels.min_delay": 0}`。
- `random-phase-nominal-a01` 未完成原因：evaluation exit=1; see console log
- `random-phase-randomized-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `random-phase-randomized-a01` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.sampling_diagnostics_dir": "/home/server/liufengrong/robot_lab/learnings/policy_tuning/RobotLab-Isaac-BeyondMimic-Flat-LW-wheel-v0/2026-10-31_15-48-13/evaluations/wheel-hard-phase-causes-20261102-001/raw/random-phase-randomized-a01/sampling-diagnostics", "commands.motion.sampling_diagnostics_window_steps": 1000, "evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "randomized_phase", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03", "rewards.hip_pos_zero_l2.weight": -2.0}`。

## 观察与限制

- 最新 server 2026-10-31_15-48-13/model_49999.pt，SHA256 d5d75f2da8d11057f00d65f4f73f13dcee9dfb4dec312d362622fc1b0a058059；沿用已确认训练来源与保存配置。
- 本批次为随机相位名义/训练随机化两组，每组1环境、5000控制步、seed42，Native均值动作，无视频。固定第0帧既有评估直接复用，未重跑。
- 随机相位名义组关闭事件随机化、初始姿态/速度/关节噪声、观测噪声和执行器延迟；随机化组保留保存训练配置。
- 当前诊断代码尚未提交；精确运行源码（含未跟踪helper）另行快照，训练来源仍为原始879357b。
- 诊断每1000控制步保存计数，所有阶段暴露数及终止原因与独立遥测逐帧交叉核验；失效率为每万环境控制步失败次数，非每回合失败概率。
- 失败区间按floor(frame*4/170)定义；第二段为43..84帧(0.86..1.68s)。采样初始帧分桶用169而非170，边界略有差异。
- 该批次从空的困难采样计数开始，使用确定性均值动作，没有训练探索噪声；单个环境/seed不能还原8192环境训练期失败比例或确证历史采样因果。
- 各终止原因可同时成立，分别保留；motion_finished与time_out不单独算失败。起始帧为评估实际首帧，诊断reset_starts另保存采样选定帧。
- 仅可进入受监督实物测试；未经实物验证，不代表 hardware-ready。
- random-phase-randomized-a01: ended=58, failures=0, motion_finished_without_failure=58, censored=1
- random-phase-randomized-a01 终止原因={"anchor_ori": 0, "anchor_pos": 0, "ee_body_pos": 0, "motion_finished": 58, "thigh_shank_ground_contact": 0, "time_out": 0}
- random-phase-randomized-a01 分段暴露/失败/每万步率=[{"bin": 0, "exposure_steps": 249, "failures": 0, "failures_per_10000_steps": 0.0, "termination_counts": {"anchor_ori": 0, "anchor_pos": 0, "ee_body_pos": 0, "motion_finished": 0, "thigh_shank_ground_contact": 0, "time_out": 0}}, {"bin": 1, "exposure_steps": 926, "failures": 0, "failures_per_10000_steps": 0.0, "termination_counts": {"anchor_ori": 0, "anchor_pos": 0, "ee_body_pos": 0, "motion_finished": 0, "thigh_shank_ground_contact": 0, "time_out": 0}}, {"bin": 2, "exposure_steps": 1619, "failures": 0, "failures_per_10000_steps": 0.0, "termination_counts": {"anchor_ori": 0, "anchor_pos": 0, "ee_body_pos": 0, "motion_finished": 0, "thigh_shank_ground_contact": 0, "time_out": 0}}, {"bin": 3, "exposure_steps": 2206, "failures": 0, "failures_per_10000_steps": 0.0, "termination_counts": {"anchor_ori": 0, "anchor_pos": 0, "ee_body_pos": 0, "motion_finished": 58, "thigh_shank_ground_contact": 0, "time_out": 0}}]
- random-phase-randomized-a01 失败发生于重置后<=5/10/25步=(0, 0, 0)
- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据

- [manifest.json](<../wheel-to-leg-fixed-start-20261102-001/manifest.json>)
- [motion_sampling_diagnostics.jsonl](<raw/random-phase-randomized-a01/sampling-diagnostics/motion_sampling_diagnostics.jsonl>)
- [failure-phase-analysis.json](<derived/failure-phase-analysis.json>)
- [phase-failure-diagnosis.png](<derived/phase-failure-diagnosis.png>)
- [protection.json](<derived/protection.json>)

## 建议与待授权事项

- 以真实失败的时间、触发原因和相位暴露归一化率判断困难片段；不以采样占比单独代表失败率。
- 若失败主要在随机中途起点后短时间出现，应优先分析重置状态与闭环历史的差异；不能直接通过加大奖励惩罚认定根因。
- 本批次只评估冻结策略，没有训练、修改奖励/持久配置或Git提交；后续修改先提出方案并获得批准。
