# 评估批次 turn-factor-ablation-20260914-001

任务：`RobotLab-Isaac-Velocity-Flat-LW-leg-Amp-Roa-v0`；训练：`2026-09-12_22-03-53`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| turn-left-com-zero-offset-a01 | completed | turn-left-com-zero-offset，1000 / 1 / 42 | complete | [结果](<raw/turn-left-com-zero-offset-a01/result.json>) / [视频](<raw/turn-left-com-zero-offset-a01/video.mp4>) / [日志](<raw/turn-left-com-zero-offset-a01/console.log>) |
| turn-right-com-zero-offset-a01 | completed | turn-right-com-zero-offset，1000 / 1 / 42 | complete | [结果](<raw/turn-right-com-zero-offset-a01/result.json>) / [视频](<raw/turn-right-com-zero-offset-a01/video.mp4>) / [日志](<raw/turn-right-com-zero-offset-a01/console.log>) |
| turn-left-contact-fixed-a01 | completed | turn-left-contact-fixed，1000 / 1 / 42 | complete | [结果](<raw/turn-left-contact-fixed-a01/result.json>) / [视频](<raw/turn-left-contact-fixed-a01/video.mp4>) / [日志](<raw/turn-left-contact-fixed-a01/console.log>) |
| turn-right-contact-fixed-a01 | completed | turn-right-contact-fixed，1000 / 1 / 42 | complete | [结果](<raw/turn-right-contact-fixed-a01/result.json>) / [视频](<raw/turn-right-contact-fixed-a01/video.mp4>) / [日志](<raw/turn-right-contact-fixed-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`c085221208af7328c74da8c260f198822b604c0388a9609c13eb0ec515971536`；runner：`OnPolicyRunnerAmpROA`。
- [场景来源 scenario-0dae12683e573fe4d4fc24c925f3f0a51a141ee9cac20cc200c6c765ddb96824.json](<../../provenance/scenario-0dae12683e573fe4d4fc24c925f3f0a51a141ee9cac20cc200c6c765ddb96824.json>)
- [场景来源 scenario-3d9d9ae70dd1994d2976e82955d3e07cf4e4daf5704765121903043372a3ce17.json](<../../provenance/scenario-3d9d9ae70dd1994d2976e82955d3e07cf4e4daf5704765121903043372a3ce17.json>)
- [场景来源 scenario-7f8fb6ddee2e00ad258be7fa86835d29dd5b4109e62538ea63318069a3f50ab2.json](<../../provenance/scenario-7f8fb6ddee2e00ad258be7fa86835d29dd5b4109e62538ea63318069a3f50ab2.json>)
- [场景来源 scenario-870418722b4a021129dc88d52df0e572df50253d5ebe95a0931063adb5661029.json](<../../provenance/scenario-870418722b4a021129dc88d52df0e572df50253d5ebe95a0931063adb5661029.json>)
- [训练上下文 context-23d26bb9c9cb723f9847ef908f7d24a54ba254da12fd648da317ec9dc73549e5.json](<../../provenance/context-23d26bb9c9cb723f9847ef908f7d24a54ba254da12fd648da317ec9dc73549e5.json>)
- [训练有效配置 config-4dcda3327a07455d3febf0ff28b54880e56ee007150dfb322674284ee62c0343.json](<../../provenance/config-4dcda3327a07455d3febf0ff28b54880e56ee007150dfb322674284ee62c0343.json>)

- `turn-left-com-zero-offset-a01` 指标：`{"max_joint_velocity_utilization": 1.018086814880371, "max_tilt": 0.12159661203622818, "termination_rate": 0.0, "tracking_xy_rmse": 0.07264162684641295, "tracking_yaw_rmse": 0.23155946720266454}`
- `turn-left-com-zero-offset-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0, 0, 0.5], "end_step": 749, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 750}]`；训练配置覆盖：`{"episode_length_s": 65.0, "events.randomize_com_positions.params.com_range": {"x": [0.0, 0.0], "y": [0.0, 0.0], "z": [0.0, 0.0]}, "events.randomize_push_robot": null}`。
- `turn-right-com-zero-offset-a01` 指标：`{"max_joint_velocity_utilization": 1.018086814880371, "max_tilt": 0.11427231878042221, "termination_rate": 0.0, "tracking_xy_rmse": 0.07540772141803981, "tracking_yaw_rmse": 0.19628706674282223}`
- `turn-right-com-zero-offset-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0, 0, -0.5], "end_step": 749, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 750}]`；训练配置覆盖：`{"episode_length_s": 65.0, "events.randomize_com_positions.params.com_range": {"x": [0.0, 0.0], "y": [0.0, 0.0], "z": [0.0, 0.0]}, "events.randomize_push_robot": null}`。
- `turn-left-contact-fixed-a01` 指标：`{"max_joint_velocity_utilization": 0.8305233955383301, "max_tilt": 0.1949966549873352, "termination_rate": 0.0, "tracking_xy_rmse": 0.08559440934576641, "tracking_yaw_rmse": 0.2938598496283586}`
- `turn-left-contact-fixed-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0, 0, 0.5], "end_step": 749, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 750}]`；训练配置覆盖：`{"episode_length_s": 65.0, "events.randomize_push_robot": null, "events.randomize_rigid_body_material.params.dynamic_friction_range": [1.0, 1.0], "events.randomize_rigid_body_material.params.restitution_range": [0.0, 0.0], "events.randomize_rigid_body_material.params.static_friction_range": [1.0, 1.0]}`。
- `turn-right-contact-fixed-a01` 指标：`{"max_joint_velocity_utilization": 0.9309533119201661, "max_tilt": 0.17799469828605652, "termination_rate": 0.0, "tracking_xy_rmse": 0.06504263230661413, "tracking_yaw_rmse": 0.3367451087303031}`
- `turn-right-contact-fixed-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0, 0, -0.5], "end_step": 749, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 750}]`；训练配置覆盖：`{"episode_length_s": 65.0, "events.randomize_push_robot": null, "events.randomize_rigid_body_material.params.dynamic_friction_range": [1.0, 1.0], "events.randomize_rigid_body_material.params.restitution_range": [0.0, 0.0], "events.randomize_rigid_body_material.params.static_friction_range": [1.0, 1.0]}`。

## 观察与限制

- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据


## 建议与待授权事项

- 按批次报告检查失败项和动作表现；复测使用新的 batch_id 与 attempt_id。
