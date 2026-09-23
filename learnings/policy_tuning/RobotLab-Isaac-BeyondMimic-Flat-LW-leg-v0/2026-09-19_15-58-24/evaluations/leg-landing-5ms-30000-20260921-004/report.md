# 评估批次 leg-landing-5ms-30000-20260921-004

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-leg-v0`；训练：`2026-09-19_15-58-24`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| nominal-start-30000-5ms-a01 | completed | nominal-start，2000 / 1 / 42 | complete | [结果](<raw/nominal-start-30000-5ms-a01/result.json>) / [日志](<raw/nominal-start-30000-5ms-a01/console.log>) |
| randomized-start-30000-5ms-a01 | completed | randomized-start，2000 / 1 / 42 | complete | [结果](<raw/randomized-start-30000-5ms-a01/result.json>) / [日志](<raw/randomized-start-30000-5ms-a01/console.log>) |
| randomized-phase-30000-5ms-a01 | completed | randomized-phase，2000 / 1 / 42 | complete | [结果](<raw/randomized-phase-30000-5ms-a01/result.json>) / [日志](<raw/randomized-phase-30000-5ms-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`6924df9721d7676efe6f424c89655313c15afc949b289085da8c03fbc8b4792b`；runner：`OnPolicyRunner`。
- [场景来源 scenario-6b8f48c4174fde5703fee9f6ef16690d3a59fdcf2c2f1f431bd7d35b32822c87.json](<../../provenance/scenario-6b8f48c4174fde5703fee9f6ef16690d3a59fdcf2c2f1f431bd7d35b32822c87.json>)
- [场景来源 scenario-9595d858857d36f29ffd06aad63f036c2a80ef9966910b544afe6140c472f81e.json](<../../provenance/scenario-9595d858857d36f29ffd06aad63f036c2a80ef9966910b544afe6140c472f81e.json>)
- [场景来源 scenario-fe4b70abce81a4f8899553cf7cb6e0246dab2f3494e238de2ac0efdcf4a5aed0.json](<../../provenance/scenario-fe4b70abce81a4f8899553cf7cb6e0246dab2f3494e238de2ac0efdcf4a5aed0.json>)
- [训练上下文 context-e0894100a88eb499433cf09bf722da3fa9fbfb2a9e3ff7bf00a0a0af0c4dbb3d.json](<../../provenance/context-e0894100a88eb499433cf09bf722da3fa9fbfb2a9e3ff7bf00a0a0af0c4dbb3d.json>)
- [训练有效配置 config-7d4e603b42fa6e9aa9d1ffc8a61b4311bbbc55bf67dec4de3e0563efb562e20c.json](<../../provenance/config-7d4e603b42fa6e9aa9d1ffc8a61b4311bbbc55bf67dec4de3e0563efb562e20c.json>)

- `nominal-start-30000-5ms-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `nominal-start-30000-5ms-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_mode": "nominal_start", "evaluation.motion_physics_window": [46, 91], "evaluation.motion_sha256": "85e53dc37459b295a4a3f06dd56044eb96b661caf01461ccceb1908cf52fa75d", "rewards.joint_acc_wheel_l2": null, "rewards.joint_vel_wheel_l2": null}`。
- `randomized-start-30000-5ms-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `randomized-start-30000-5ms-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [46, 91], "evaluation.motion_sha256": "85e53dc37459b295a4a3f06dd56044eb96b661caf01461ccceb1908cf52fa75d", "rewards.joint_acc_wheel_l2": null, "rewards.joint_vel_wheel_l2": null}`。
- `randomized-phase-30000-5ms-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `randomized-phase-30000-5ms-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_mode": "randomized_phase", "evaluation.motion_physics_window": [46, 91], "evaluation.motion_sha256": "85e53dc37459b295a4a3f06dd56044eb96b661caf01461ccceb1908cf52fa75d", "rewards.joint_acc_wheel_l2": null, "rewards.joint_vel_wheel_l2": null}`。

## 观察与限制

- 200 Hz capture at actual 5 ms physics updates; policy remains 50 Hz. Frames 46–91 inclusive; all four substeps retained before resets.
- Current source contains user wheel reward changes. Both wheel penalties are overridden to null in evaluation to reproduce checkpoint training rewards; user files were preserved.
- Fresh source context identifies this evaluation checkout; params/env.yaml and params/agent.yaml remain the original training configuration authority.
- Contact values are net normal forces averaged over each physics step, excluding tangential friction. Link-origin velocities avoid COM-offset/spin contamination.
- One environment, seed 42, 2000 control steps per scenario; startup randomization is one sample. These results are not convergence or hardware load certification.
- 仅可进入受监督实物测试；未经实物验证，不代表 hardware-ready。
- nominal-start: 2608 physics samples; summed wheel peak 3375.66 N (12.87 BW), vs 20 ms endpoints 737.63 N; computed torque over-limit joint samples 0, applied over-limit 0. Peak episode 0, frame 76, substep 2 (zero-based).
- nominal-start old/new control endpoint parity: {"frame_episode_done_match": true, "max_absolute_difference": {"action": 0.0, "applied_torque": 0.0, "contact_forces_w": 0.0, "joint_position": 0.0, "joint_velocity": 0.0, "reward": 0.0, "root_position_w": 0.0}, "sample_count_match": true}
- randomized-start: 2608 physics samples; summed wheel peak 4834.56 N (17.87 BW), vs 20 ms endpoints 4834.56 N; computed torque over-limit joint samples 0, applied over-limit 0. Peak episode 13, frame 76, substep 3 (zero-based).
- randomized-start old/new control endpoint parity: {"frame_episode_done_match": true, "max_absolute_difference": {"action": 0.0, "applied_torque": 0.0, "contact_forces_w": 0.0, "joint_position": 0.0, "joint_velocity": 0.0, "reward": 0.0, "root_position_w": 0.0}, "sample_count_match": true}
- randomized-phase: 2908 physics samples; summed wheel peak 3816.58 N (14.11 BW), vs 20 ms endpoints 3263.80 N; computed torque over-limit joint samples 1, applied over-limit 0. Peak episode 6, frame 78, substep 1 (zero-based).
- randomized-phase old/new control endpoint parity: {"frame_episode_done_match": true, "max_absolute_difference": {"action": 0.0, "applied_torque": 0.0, "contact_forces_w": 0.0, "joint_position": 0.0, "joint_velocity": 0.0, "reward": 0.0, "root_position_w": 0.0}, "sample_count_match": true}
- Paired checkpoint nominal-start: model_49999 summed wheel peak 3254.66 N; model_30000 3375.66 N. Same scenario/seed/duration/overrides; source/runtime matching must accompany interpretation.
- Paired checkpoint randomized-start: model_49999 summed wheel peak 3337.77 N; model_30000 4834.56 N. Same scenario/seed/duration/overrides; source/runtime matching must accompany interpretation.
- Paired checkpoint randomized-phase: model_49999 summed wheel peak 5387.73 N; model_30000 3816.58 N. Same scenario/seed/duration/overrides; source/runtime matching must accompany interpretation.
- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据

- [manifest.json](<../leg-to-wheel-30000-20260921-002/manifest.json>)
- [manifest.json](<../leg-landing-5ms-49999-20260921-003/manifest.json>)
- [landing-analysis.json](<landing-analysis.json>)
- [landing-window.png](<landing-window.png>)

## 建议与待授权事项

- Interpret pre-contact downward speed, force transient and torque together; do not infer joint structural loads from motor clipping.
- Keep reward choices separate from recording; no training or automatic checkpoint selection is performed.
