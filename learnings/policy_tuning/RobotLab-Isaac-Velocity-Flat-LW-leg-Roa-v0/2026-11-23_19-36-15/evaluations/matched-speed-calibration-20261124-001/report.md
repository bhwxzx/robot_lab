# 评估批次 matched-speed-calibration-20261124-001

任务：`RobotLab-Isaac-Velocity-Flat-LW-leg-Roa-v0`；训练：`2026-11-23_19-36-15`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| forward-028-a01 | completed | forward-028，1000 / 1 / 42 | complete | [结果](<raw/forward-028-a01/result.json>) / [日志](<raw/forward-028-a01/console.log>) |
| forward-030-a01 | completed | forward-030，1000 / 1 / 42 | complete | [结果](<raw/forward-030-a01/result.json>) / [日志](<raw/forward-030-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`ca0269c71f3b983efb7fb8f3643d037c18d36b44afa557da7f09cff4d682f217`；runner：`OnPolicyRunnerROA`。
- [场景来源 scenario-e8cf429f406b3368fb0aa3bed16521b282222349f7aea948eabf124a02d8e181.json](<../../provenance/scenario-e8cf429f406b3368fb0aa3bed16521b282222349f7aea948eabf124a02d8e181.json>)
- [场景来源 scenario-efec726e0865779f8106afe7a65c7fbcc4b928e729f9831df8c903c87c7b36ad.json](<../../provenance/scenario-efec726e0865779f8106afe7a65c7fbcc4b928e729f9831df8c903c87c7b36ad.json>)
- [训练上下文 context-ff5942de4a7185cc2ff082e70aeb491f32fedf81b3af4aec9882f2201791e38a.json](<../../provenance/context-ff5942de4a7185cc2ff082e70aeb491f32fedf81b3af4aec9882f2201791e38a.json>)
- [训练有效配置 config-dd9a8f90e060a9c634d8995a65ecced6b5f7d59b70465a2e692f333e3a83bd45.json](<../../provenance/config-dd9a8f90e060a9c634d8995a65ecced6b5f7d59b70465a2e692f333e3a83bd45.json>)

- `forward-028-a01` 指标：`{"max_joint_velocity_utilization": 1.000238800048828, "max_tilt": 0.09520130604505539, "termination_rate": 0.0, "tracking_xy_rmse": 0.09554028289812973, "tracking_yaw_rmse": 0.08497074775457647}`
- `forward-028-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0.28, 0, 0], "end_step": 749, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 750}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 65.0, "evaluation.roa_mode": "student", "events.randomize_push_robot": null, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane"}`。
- `forward-030-a01` 指标：`{"max_joint_velocity_utilization": 1.0174944877624512, "max_tilt": 0.10039983689785004, "termination_rate": 0.0, "tracking_xy_rmse": 0.09875963413410752, "tracking_yaw_rmse": 0.08225199752780793}`
- `forward-030-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0.3, 0, 0], "end_step": 749, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 750}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 65.0, "evaluation.roa_mode": "student", "events.randomize_push_robot": null, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane"}`。

## 观察与限制

- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据


## 建议与待授权事项

- 按批次报告检查失败项和动作表现；复测使用新的 batch_id 与 attempt_id。
