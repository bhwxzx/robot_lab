# 评估批次 tail-phase-failure-baseline-20261012-001

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-leg-v0`；训练：`2026-10-09_12-49-52`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| uniform-clean-a01 | completed | uniform-clean，1200 / 1 / 42 | complete | [结果](<raw/uniform-clean-a01/result.json>) / [日志](<raw/uniform-clean-a01/console.log>) |
| uniform-training-randomized-a01 | completed | uniform-training-randomized，1200 / 1 / 42 | complete | [结果](<raw/uniform-training-randomized-a01/result.json>) / [日志](<raw/uniform-training-randomized-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`8909d6a6c2db883fc3dd37843053fdb887f90b92ead76cff036f8341b5cdd2b1`；runner：`OnPolicyRunner`。
- [场景来源 scenario-1f16c4f7fdeb6ef6c38c715291fe6dce984200df814e0a2d0e14e1f8d751830c.json](<../../provenance/scenario-1f16c4f7fdeb6ef6c38c715291fe6dce984200df814e0a2d0e14e1f8d751830c.json>)
- [场景来源 scenario-5e1fcd7295dd03242a042583fc46f8dbde5e02265c019c347d300189a385ac6d.json](<../../provenance/scenario-5e1fcd7295dd03242a042583fc46f8dbde5e02265c019c347d300189a385ac6d.json>)
- [训练上下文 context-3448189a06f73418e46904e71ee1838c028485b8608b22a3f698b0471ccc24f3.json](<../../provenance/context-3448189a06f73418e46904e71ee1838c028485b8608b22a3f698b0471ccc24f3.json>)
- [训练有效配置 config-ed83bdbc225d9af2a6db3b732873a090767a80ec6db51bff78a6a5895739bfad.json](<../../provenance/config-ed83bdbc225d9af2a6db3b732873a090767a80ec6db51bff78a6a5895739bfad.json>)

- `uniform-clean-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `uniform-clean-a01` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.adaptive_uniform_ratio": 1000000, "commands.motion.joint_position_range": [0, 0], "commands.motion.pose_range": {}, "commands.motion.velocity_range": {}, "evaluation.motion_mode": "randomized_phase", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8f5dec53148b5229b28055b8af968d18dc757e873f1a04bd423b0e8bfc6b2a6b", "events.add_joint_default_pos": null, "events.base_com": null, "events.physics_material": null, "events.push_robot": null, "events.randomize_actuator_gains": null, "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "observations.policy.enable_corruption": false, "rewards.joint_acc_wheel_l2.weight": 0, "rewards.joint_vel_wheel_l2.weight": 0, "scene.robot.actuators.foots.max_delay": 0, "scene.robot.actuators.foots.min_delay": 0, "scene.robot.actuators.legs.max_delay": 0, "scene.robot.actuators.legs.min_delay": 0, "scene.robot.actuators.wheels.max_delay": 0, "scene.robot.actuators.wheels.min_delay": 0}`。
- `uniform-training-randomized-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `uniform-training-randomized-a01` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.adaptive_uniform_ratio": 1000000, "evaluation.motion_mode": "randomized_phase", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8f5dec53148b5229b28055b8af968d18dc757e873f1a04bd423b0e8bfc6b2a6b", "rewards.joint_acc_wheel_l2.weight": 0, "rewards.joint_vel_wheel_l2.weight": 0}`。

## 观察与限制

- User-authorized four-case offline tail-failure reproduction; each case 1 env,1200 control steps,seed42,no video,no retries. Near-uniform reference start sampling is an explicit evaluation-only override; it does not reproduce the collapsed training distribution.
- Clean disables randomization/noise/push and uses zero actuator delay. Training-randomized uses the latest training randomization for both policies; baseline wheel-penalty weights are restored to zero. No Wheel-policy handoff is simulated.
- The 10/11 exact launch script and absence of extra overrides were confirmed by the user; current source context is recorded independently of saved training YAML.
- Tail episode details and first-failure snapshots are in tail_summary.json. Contact force is PhysX substep-averaged normal force, not a hardware impulse peak.
- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据

- [tail_summary.json](<tail_summary.json>)

## 建议与待授权事项

- 按批次报告检查失败项和动作表现；复测使用新的 batch_id 与 attempt_id。
