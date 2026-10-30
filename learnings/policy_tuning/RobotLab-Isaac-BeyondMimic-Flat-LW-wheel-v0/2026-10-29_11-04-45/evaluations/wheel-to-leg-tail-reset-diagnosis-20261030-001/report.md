# 评估批次 wheel-to-leg-tail-reset-diagnosis-20261030-001

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-wheel-v0`；训练：`2026-10-29_11-04-45`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| random-phase-training-a01 | completed | random-phase-training，1000 / 1 / 42 | complete | [结果](<raw/random-phase-training-a01/result.json>) / [日志](<raw/random-phase-training-a01/console.log>) |
| random-phase-nominal-a01 | completed | random-phase-nominal，1000 / 1 / 42 | complete | [结果](<raw/random-phase-nominal-a01/result.json>) / [日志](<raw/random-phase-nominal-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`47a797c57620e761af2118a38daf838ae4e0a14ac6800faa7f0ed0d392be7650`；runner：`OnPolicyRunner`。
- [场景来源 scenario-423012755c6e2eb3f81079b92d7de25f424fc8a013c80baa598432060a798272.json](<../../provenance/scenario-423012755c6e2eb3f81079b92d7de25f424fc8a013c80baa598432060a798272.json>)
- [场景来源 scenario-a59ebffb3053a47f8c36923db1774cb93c13669273bd25b3296a72a02cda4ca4.json](<../../provenance/scenario-a59ebffb3053a47f8c36923db1774cb93c13669273bd25b3296a72a02cda4ca4.json>)
- [训练上下文 context-512a7180ffb510b698f2e91b0e30dc43d44165974a3df7dad2f7a3d9d9fc9d1a.json](<../../provenance/context-512a7180ffb510b698f2e91b0e30dc43d44165974a3df7dad2f7a3d9d9fc9d1a.json>)
- [训练有效配置 config-3d5f5125c913763bc6bba441103ba4e859f58b6c1b0eceffe70c276542660583.json](<../../provenance/config-3d5f5125c913763bc6bba441103ba4e859f58b6c1b0eceffe70c276542660583.json>)

- `random-phase-training-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `random-phase-training-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "randomized_phase", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03", "rewards.wheel_contact_continuous.weight": 0.5}`。
- `random-phase-nominal-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `random-phase-nominal-a01` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.joint_position_range": [0.0, 0.0], "commands.motion.pose_range": {}, "commands.motion.velocity_range": {}, "evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "randomized_phase", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03", "events.add_joint_default_pos": null, "events.base_com": null, "events.physics_material": null, "events.push_robot": null, "events.randomize_actuator_gains": null, "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "observations.policy.enable_corruption": false, "rewards.wheel_contact_continuous.weight": 0.5, "scene.robot.actuators.foots.max_delay": 0, "scene.robot.actuators.foots.min_delay": 0, "scene.robot.actuators.legs.max_delay": 0, "scene.robot.actuators.legs.min_delay": 0, "scene.robot.actuators.wheels.max_delay": 0, "scene.robot.actuators.wheels.min_delay": 0}`。

## 观察与限制

- random-phase-training-a01全部起始相位：{"episodes": 46, "observed_terminations": 46, "successful_motion_finished": 2, "failed": 44, "censored": 0, "failure_term_counts": {"anchor_pos": 19, "ee_body_pos": 30}, "control_steps_mean": 21.73913043478261, "start_phase_range_s": [0.06, 3.36]}
- random-phase-training-a01末段起始：{"episodes": 19, "observed_terminations": 19, "successful_motion_finished": 1, "failed": 18, "censored": 0, "failure_term_counts": {"ee_body_pos": 18}, "control_steps_mean": 3.9473684210526314, "start_phase_range_s": [2.64, 3.36]}
- random-phase-nominal-a01全部起始相位：{"episodes": 31, "observed_terminations": 30, "successful_motion_finished": 0, "failed": 30, "censored": 1, "failure_term_counts": {"anchor_pos": 16, "ee_body_pos": 16}, "control_steps_mean": 32.56666666666667, "start_phase_range_s": [0.08, 3.24]}
- random-phase-nominal-a01末段起始：{"episodes": 4, "observed_terminations": 4, "successful_motion_finished": 0, "failed": 4, "censored": 0, "failure_term_counts": {"ee_body_pos": 4}, "control_steps_mean": 4, "start_phase_range_s": [2.86, 3.24]}
- 与已封存固定第0帧批次分开；本批次运行现有randomized_phase模式，训练随机化组与关闭随机化/噪声/延迟组，各1000控制步。
- 使用新建单环境采样器；没有保存和恢复训练时4096环境的失败计数，采样分布及相位序列不等于训练历史。按实际起始帧>=126统计末段，并单独保留终止帧。
- 参考末段没有超出URDF软关节限位；170帧正运动学位置与NPZ最大差4.657e-7m。这说明参考几何自洽，不能单独证明动力学稳定。
- 首次位置动作目标按默认关节角+0.25*动作计算；名义组没有默认角随机化，训练随机化组该重建是近似值。
- 已知训练时末段采样概率约99.96%，末期ee_body_pos指标约0.911；本批次用于定位直接重置后的失败，不能宣称重现训练全部因果。
- 所有既有模型、训练配置、源码、持久设置、固定起始批次及其证据的字节/mtime保护：通过。
- 仅可进入受监督实物测试；未经实物验证，不代表 hardware-ready。
- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据

- [manifest.json](<../wheel-to-leg-fixed-start-20261030-001/manifest.json>)
- [analysis.json](<derived/analysis.json>)
- [tail-diagnosis-protection-before-20261030-001.json](<../../evidence/training/tail-diagnosis-protection-before-20261030-001.json>)
- [tail-diagnosis-protection-after-20261030-001.json](<../../evidence/training/tail-diagnosis-protection-after-20261030-001.json>)

## 建议与待授权事项

- 结合末段首个动作目标偏差、失败前关节运动和关闭扰动组对照判断原因；需要修改采样器或动作重置逻辑时先提方案并获得批准。
