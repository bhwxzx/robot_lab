# 评估批次 wheel-to-leg-local-20261023-001

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-wheel-v0`；训练：`2026-10-22_20-03-44`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| play-ground-a01 | completed | play-ground，1000 / 1 / 42 | complete | [结果](<raw/play-ground-a01/result.json>) / [日志](<raw/play-ground-a01/console.log>) |
| training-random-phase-a01 | completed | training-random-phase，1000 / 1 / 42 | complete | [结果](<raw/training-random-phase-a01/result.json>) / [日志](<raw/training-random-phase-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`ef63b2dfa8107b9fa1d8f136f2c9031fca34f609d3e85ca83386f01effe4e818`；runner：`OnPolicyRunner`。
- [场景来源 scenario-72a76854d566f9d41cf9d58dcb21890e954f70c3e1ea8520573b95de4cb85f9a.json](<../../provenance/scenario-72a76854d566f9d41cf9d58dcb21890e954f70c3e1ea8520573b95de4cb85f9a.json>)
- [场景来源 scenario-7beffd8299f97e4d531a638e649107d205fbe592b21ed02206283373a37831cb.json](<../../provenance/scenario-7beffd8299f97e4d531a638e649107d205fbe592b21ed02206283373a37831cb.json>)
- [训练上下文 context-b57f3ef581bf6add8d4ae117fa6fd76b72264f52cc0ab616f7e6c1f35c0f6cd4.json](<../../provenance/context-b57f3ef581bf6add8d4ae117fa6fd76b72264f52cc0ab616f7e6c1f35c0f6cd4.json>)
- [训练有效配置 config-86f6bcd7ea2a94251ce0592e189f98531a8d4b2c53bb4ccc86f9219bb7d66e32.json](<../../provenance/config-86f6bcd7ea2a94251ce0592e189f98531a8d4b2c53bb4ccc86f9219bb7d66e32.json>)

- `play-ground-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `play-ground-a01` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.joint_position_range": [0, 0], "commands.motion.pose_range": {}, "commands.motion.velocity_range": {}, "evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "4d17c9a09f1fd6201bf14d72a468eec9c8a58000b6f67777d93b964c387f5d62", "events.add_joint_default_pos": null, "events.base_com": null, "events.physics_material": null, "events.push_robot": null, "events.randomize_actuator_gains": null, "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "observations.policy.enable_corruption": false, "rewards.wheel_contact_continuous.weight": 0.5}`。
- `training-random-phase-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `training-random-phase-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "randomized_phase", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "4d17c9a09f1fd6201bf14d72a468eec9c8a58000b6f67777d93b964c387f5d62"}`。

## 观察与限制

- 本机 younghit 的 2026-10-22_20-03-44/model_49999.pt；与服务器 17-40-29 策略不同。
- 本次训练启动命令已由用户确认，无额外参数或配置覆盖；有效配置使用本场次保存的 env.yaml 和 agent.yaml。
- 独立起跳速度与高度奖励为 0，位置/连杆位置/线速度跟踪权重为 2/2/3；两项动作平滑权重保持 -0.2/-0.075。
- 两个 Native 场景各 1000 步、1 环境、seed 42；不录制视频，不启动训练，不导出策略。
- 全部评估进程显式传入 --/app/settings/persistent=false；未设置持久物理线程参数。
- completed 仅表示结果和遥测成功封存；随机相位片段不能当作完整 frame-zero 跳跃周期；未提供批准的收敛判据。
- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据


## 建议与待授权事项

- 按批次报告检查失败项和动作表现；复测使用新的 batch_id 与 attempt_id。
