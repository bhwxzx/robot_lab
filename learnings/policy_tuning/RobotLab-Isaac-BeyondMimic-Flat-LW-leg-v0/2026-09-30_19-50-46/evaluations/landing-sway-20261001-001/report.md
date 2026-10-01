# 评估批次 landing-sway-20261001-001

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-leg-v0`；训练：`2026-09-30_19-50-46`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| play-equivalent-a01 | completed | play-equivalent，1000 / 1 / 42 | complete | [结果](<raw/play-equivalent-a01/result.json>) / [日志](<raw/play-equivalent-a01/console.log>) |
| zero-delay-a01 | completed | zero-delay，1000 / 1 / 42 | complete | [结果](<raw/zero-delay-a01/result.json>) / [日志](<raw/zero-delay-a01/console.log>) |
| fixed-15ms-a01 | completed | fixed-15ms，1000 / 1 / 42 | complete | [结果](<raw/fixed-15ms-a01/result.json>) / [日志](<raw/fixed-15ms-a01/console.log>) |
| randomized-no-push-a01 | completed | randomized-no-push，1000 / 1 / 42 | complete | [结果](<raw/randomized-no-push-a01/result.json>) / [日志](<raw/randomized-no-push-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`782f9cedbe27a68fcfda54b9a47801250d585a17ca42cb08d509cf9c3c02db85`；runner：`OnPolicyRunner`。
- [场景来源 scenario-138b6ab0517e857bf9d71c4ee987db94794ab95a1a04419f6169a6a0b4413e1a.json](<../../provenance/scenario-138b6ab0517e857bf9d71c4ee987db94794ab95a1a04419f6169a6a0b4413e1a.json>)
- [场景来源 scenario-3ed51190ae3a19cf417ac9b4211563ca900cbd2edd43125da22e18eed90b735e.json](<../../provenance/scenario-3ed51190ae3a19cf417ac9b4211563ca900cbd2edd43125da22e18eed90b735e.json>)
- [场景来源 scenario-7f384d06ecc06a2b77344ec5955dec7c1c7da66dc29d8eb90d285139b5c3cfa5.json](<../../provenance/scenario-7f384d06ecc06a2b77344ec5955dec7c1c7da66dc29d8eb90d285139b5c3cfa5.json>)
- [场景来源 scenario-f9f99e040d0995dd30f884c1d7a9190dfae49e1ce0da860b51c84ff7d46bf09e.json](<../../provenance/scenario-f9f99e040d0995dd30f884c1d7a9190dfae49e1ce0da860b51c84ff7d46bf09e.json>)
- [训练上下文 context-31cea88133b2c9234f7461519fb0559bbb559956dd38801cb28364ed738191a0.json](<../../provenance/context-31cea88133b2c9234f7461519fb0559bbb559956dd38801cb28364ed738191a0.json>)
- [训练有效配置 config-64f55c031b41134030731554a055f85ce0e20f55596632a891a7d39dbda13f5b.json](<../../provenance/config-64f55c031b41134030731554a055f85ce0e20f55596632a891a7d39dbda13f5b.json>)

- `play-equivalent-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `play-equivalent-a01` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.joint_position_range": [0.0, 0.0], "commands.motion.pose_range": {}, "commands.motion.velocity_range": {}, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 141], "evaluation.motion_sha256": "48dd4ee594e132f447ee86cfda1c7b3d4fe2d87eb20e2c37a1ca84c24de31bda", "events.add_joint_default_pos": null, "events.base_com": null, "events.physics_material": null, "events.push_robot": null, "events.randomize_actuator_gains": null, "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "observations.policy.enable_corruption": false}`。
- `zero-delay-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `zero-delay-a01` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.joint_position_range": [0.0, 0.0], "commands.motion.pose_range": {}, "commands.motion.velocity_range": {}, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 141], "evaluation.motion_sha256": "48dd4ee594e132f447ee86cfda1c7b3d4fe2d87eb20e2c37a1ca84c24de31bda", "events.add_joint_default_pos": null, "events.base_com": null, "events.physics_material": null, "events.push_robot": null, "events.randomize_actuator_gains": null, "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "observations.policy.enable_corruption": false, "scene.robot.actuators.foots.max_delay": 0, "scene.robot.actuators.foots.min_delay": 0, "scene.robot.actuators.legs.max_delay": 0, "scene.robot.actuators.legs.min_delay": 0, "scene.robot.actuators.wheels.max_delay": 0, "scene.robot.actuators.wheels.min_delay": 0}`。
- `fixed-15ms-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `fixed-15ms-a01` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.joint_position_range": [0.0, 0.0], "commands.motion.pose_range": {}, "commands.motion.velocity_range": {}, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 141], "evaluation.motion_sha256": "48dd4ee594e132f447ee86cfda1c7b3d4fe2d87eb20e2c37a1ca84c24de31bda", "events.add_joint_default_pos": null, "events.base_com": null, "events.physics_material": null, "events.push_robot": null, "events.randomize_actuator_gains": null, "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "observations.policy.enable_corruption": false, "scene.robot.actuators.foots.max_delay": 3, "scene.robot.actuators.foots.min_delay": 3, "scene.robot.actuators.legs.max_delay": 3, "scene.robot.actuators.legs.min_delay": 3, "scene.robot.actuators.wheels.max_delay": 3, "scene.robot.actuators.wheels.min_delay": 3}`。
- `randomized-no-push-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `randomized-no-push-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 141], "evaluation.motion_sha256": "48dd4ee594e132f447ee86cfda1c7b3d4fe2d87eb20e2c37a1ca84c24de31bda", "events.push_robot": null}`。

## 观察与限制

- 使用用户确认的标准训练启动命令；本轮只作描述性评估，未提供本运行的批准收敛判据。
- 播放脚本关闭推扰、观测噪声和材料/质量随机化，但保留 0–15 ms 执行器延迟；测试使用 seed=42。
- 评估记录每一步的重置前状态，自动重置后明确回到参考帧 0；播放脚本自动重置后的下一观测可能为帧 1，这个 20 ms 差异未单独消融。
- 没有调用会覆盖 exported 策略的播放脚本，也没有视频证据；机身摇晃依据姿态与角速度遥测。
- play-equivalent-a01：完成动作 7/7，失败 0，截尾 1；末 0.6 s 俯仰峰峰值平均 5.43 deg，俯仰角速度 RMS 0.345 rad/s；单轮 5 ms 法向力峰值 2338.1 N。
- zero-delay-a01：完成动作 7/7，失败 0，截尾 1；末 0.6 s 俯仰峰峰值平均 4.02 deg，俯仰角速度 RMS 0.314 rad/s；单轮 5 ms 法向力峰值 2387.5 N。
- fixed-15ms-a01：完成动作 7/7，失败 0，截尾 1；末 0.6 s 俯仰峰峰值平均 4.51 deg，俯仰角速度 RMS 0.312 rad/s；单轮 5 ms 法向力峰值 2246.2 N。
- randomized-no-push-a01：完成动作 7/7，失败 0，截尾 1；末 0.6 s 俯仰峰峰值平均 7.02 deg，俯仰角速度 RMS 0.476 rad/s；单轮 5 ms 法向力峰值 2427.1 N。
- 参考仅 142 帧 / 50 Hz；末帧俯仰角速度仍 0.205 rad/s，末 0.6 s 参考俯仰峰峰值 0.88 deg；环境在末帧结束，没有静止保持尾段。
- 零延迟对照：末段俯仰峰峰值由播放等效 5.43 deg 变为 4.02 deg。此为同一策略的延迟敏感性，不能直接推导重训效果。
- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据

- [analysis.json](<../../evidence/analysis/landing-sway-20261001-001/analysis.json>)
- [summary-last1000-20261001-001.json](<../../evidence/training/summary-last1000-20261001-001.json>)
- [landing-sway-comparison.png](<../../evidence/analysis/landing-sway-20261001-001/landing-sway-comparison.png>)

## 建议与待授权事项

- 先查看 analysis.json 的完整周期统计与 5 ms 时序图，区分冲击激发和控制持续振荡；本批次未修改奖励或训练参数。
- 优先核验参考末端的静止保持阶段和播放执行器延迟；任何参考或配置修改另行提出具体方案。
- 仅可进入受监督实物测试；未经实物验证，不代表 hardware-ready。
