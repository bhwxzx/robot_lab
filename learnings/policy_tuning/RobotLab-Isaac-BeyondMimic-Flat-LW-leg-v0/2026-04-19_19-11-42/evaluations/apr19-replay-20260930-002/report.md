# 评估批次 apr19-replay-20260930-002

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-leg-v0`；训练：`2026-04-19_19-11-42`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| apr19-nominal-start-a02 | completed | apr19-nominal-start，280 / 1 / 42 | complete | [结果](<raw/apr19-nominal-start-a02/result.json>) / [日志](<raw/apr19-nominal-start-a02/console.log>) |
| apr19-randomized-start-a02 | completed | apr19-randomized-start，280 / 1 / 42 | complete | [结果](<raw/apr19-randomized-start-a02/result.json>) / [日志](<raw/apr19-randomized-start-a02/console.log>) |

## 策略与场景

- checkpoint SHA-256：`59c7317a082740fe957974e581958864c1dc1eaa384714a2ca5915aae444df71`；runner：`OnPolicyRunner`。
- [场景来源 scenario-2eb37ee5087f69bc0f68250a4fd0cf2eaf11caaf497b62ea979e3b047283fe9c.json](<../../provenance/scenario-2eb37ee5087f69bc0f68250a4fd0cf2eaf11caaf497b62ea979e3b047283fe9c.json>)
- [场景来源 scenario-c08148defe399ba3195ddd8a14e6016d47d7303bb50b4c7bbbfcd8a8039e632c.json](<../../provenance/scenario-c08148defe399ba3195ddd8a14e6016d47d7303bb50b4c7bbbfcd8a8039e632c.json>)
- [训练上下文 context-5928b893e4759a83e2b93fbc99f8bd54e8d9d3018ebca9d855f1c26e55f2a194.json](<../../provenance/context-5928b893e4759a83e2b93fbc99f8bd54e8d9d3018ebca9d855f1c26e55f2a194.json>)
- [训练有效配置 config-bdfb51e3dc8e2ae406f5b0e2bf11b0864acf69445bd5f6c1d768ec61baee32f4.json](<../../provenance/config-bdfb51e3dc8e2ae406f5b0e2bf11b0864acf69445bd5f6c1d768ec61baee32f4.json>)

- `apr19-nominal-start-a02` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `apr19-nominal-start-a02` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.motion_file": "/home/young/liufengrong/robot_lab/learnings/policy_tuning/RobotLab-Isaac-BeyondMimic-Flat-LW-leg-v0/2026-04-19_19-11-42/evidence/source/leg_to_wheel_apr19_reconstructed_50hz.npz", "evaluation.motion_mode": "nominal_start", "evaluation.motion_physics_window": [0, 138], "evaluation.motion_sha256": "85e53dc37459b295a4a3f06dd56044eb96b661caf01461ccceb1908cf52fa75d", "events.base_com.params.com_range.x": [-0.025, 0.025], "events.base_com.params.com_range.y": [-0.05, 0.05], "events.base_com.params.com_range.z": [-0.05, 0.05], "events.physics_material.params.make_consistent": false, "events.randomize_actuator_gains": null, "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "observations.critic.joint_pos.func": "lambda env, asset_cfg, wheel_asset_cfg: (lambda q: q.index_fill(1, __import__('torch').as_tensor(wheel_asset_cfg.joint_ids, device=q.device), 0))(env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids] - env.scene[asset_cfg.name].data.default_joint_pos[:, asset_cfg.joint_ids])", "observations.policy.joint_pos.func": "lambda env, asset_cfg, wheel_asset_cfg: (lambda q: q.index_fill(1, __import__('torch').as_tensor(wheel_asset_cfg.joint_ids, device=q.device), 0))(env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids] - env.scene[asset_cfg.name].data.default_joint_pos[:, asset_cfg.joint_ids])", "rewards.action_rate_l2.weight": -0.02, "rewards.action_smoothness.weight": -0.02, "rewards.joint_acc_wheel_l2": null, "rewards.joint_vel_wheel_l2": null, "rewards.torque_limit.weight": -1.0, "scene.robot.actuators.foots.damping": 1.8, "scene.robot.actuators.foots.max_delay": 6, "scene.robot.actuators.foots.stiffness": 36.0, "scene.robot.actuators.legs.max_delay": 6, "scene.robot.actuators.wheels.max_delay": 6}`。
- `apr19-randomized-start-a02` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `apr19-randomized-start-a02` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.motion_file": "/home/young/liufengrong/robot_lab/learnings/policy_tuning/RobotLab-Isaac-BeyondMimic-Flat-LW-leg-v0/2026-04-19_19-11-42/evidence/source/leg_to_wheel_apr19_reconstructed_50hz.npz", "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 138], "evaluation.motion_sha256": "85e53dc37459b295a4a3f06dd56044eb96b661caf01461ccceb1908cf52fa75d", "events.base_com.params.com_range.x": [-0.025, 0.025], "events.base_com.params.com_range.y": [-0.05, 0.05], "events.base_com.params.com_range.z": [-0.05, 0.05], "events.physics_material.params.make_consistent": false, "events.randomize_actuator_gains": null, "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "observations.critic.joint_pos.func": "lambda env, asset_cfg, wheel_asset_cfg: (lambda q: q.index_fill(1, __import__('torch').as_tensor(wheel_asset_cfg.joint_ids, device=q.device), 0))(env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids] - env.scene[asset_cfg.name].data.default_joint_pos[:, asset_cfg.joint_ids])", "observations.policy.joint_pos.func": "lambda env, asset_cfg, wheel_asset_cfg: (lambda q: q.index_fill(1, __import__('torch').as_tensor(wheel_asset_cfg.joint_ids, device=q.device), 0))(env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids] - env.scene[asset_cfg.name].data.default_joint_pos[:, asset_cfg.joint_ids])", "rewards.action_rate_l2.weight": -0.02, "rewards.action_smoothness.weight": -0.02, "rewards.joint_acc_wheel_l2": null, "rewards.joint_vel_wheel_l2": null, "rewards.torque_limit.weight": -1.0, "scene.robot.actuators.foots.damping": 1.8, "scene.robot.actuators.foots.max_delay": 6, "scene.robot.actuators.foots.stiffness": 36.0, "scene.robot.actuators.legs.max_delay": 6, "scene.robot.actuators.wheels.max_delay": 6}`。

## 观察与限制

- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据


## 建议与待授权事项

- 按批次报告检查失败项和动作表现；复测使用新的 batch_id 与 attempt_id。
