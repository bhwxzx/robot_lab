# 评估批次 foot-probe-smoke-20261124-001

任务：`RobotLab-Isaac-Velocity-Flat-LW-leg-Roa-v0`；训练：`2026-11-23_19-36-15`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| stand-a01 | completed | stand，50 / 1 / 42 | complete | [结果](<raw/stand-a01/result.json>) / [日志](<raw/stand-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`ca0269c71f3b983efb7fb8f3643d037c18d36b44afa557da7f09cff4d682f217`；runner：`OnPolicyRunnerROA`。
- [场景来源 scenario-34d4bca743a3a5efa84db76d80a68abe21d52327511d259b832dd326d4b5991e.json](<../../provenance/scenario-34d4bca743a3a5efa84db76d80a68abe21d52327511d259b832dd326d4b5991e.json>)
- [训练上下文 context-ff5942de4a7185cc2ff082e70aeb491f32fedf81b3af4aec9882f2201791e38a.json](<../../provenance/context-ff5942de4a7185cc2ff082e70aeb491f32fedf81b3af4aec9882f2201791e38a.json>)
- [训练有效配置 config-dd9a8f90e060a9c634d8995a65ecced6b5f7d59b70465a2e692f333e3a83bd45.json](<../../provenance/config-dd9a8f90e060a9c634d8995a65ecced6b5f7d59b70465a2e692f333e3a83bd45.json>)

- `stand-a01` 指标：`{"max_joint_velocity_utilization": 1.0000004768371582, "max_tilt": 0.0884898453950882, "termination_rate": 0.0, "tracking_xy_rmse": 0.2071658755527139, "tracking_yaw_rmse": 0.21966798273212745}`
- `stand-a01` 命令调度：`[{"start_step": 0, "end_step": 49, "command": [0.0, 0.0, 0.0]}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 65.0, "evaluation.roa_foot_ground": true, "evaluation.roa_mode": "student", "events.randomize_push_robot": null, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane"}`。

## 观察与限制

- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据


## 建议与待授权事项

- 按批次报告检查失败项和动作表现；复测使用新的 batch_id 与 attempt_id。
