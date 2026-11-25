# 评估批次 flat-leg-roa-49999-plane-20261125-001

任务：`RobotLab-Isaac-Velocity-Flat-LW-leg-Roa-v0`；训练：`2026-11-24_14-41-24`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| stand-a01 | completed | stand，1000 / 1 / 42 | complete | [结果](<raw/stand-a01/result.json>) / [日志](<raw/stand-a01/console.log>) |
| forward-a01 | completed | forward，1000 / 1 / 42 | complete | [结果](<raw/forward-a01/result.json>) / [日志](<raw/forward-a01/console.log>) |
| backward-a01 | completed | backward，1000 / 1 / 42 | complete | [结果](<raw/backward-a01/result.json>) / [日志](<raw/backward-a01/console.log>) |
| turn-left-a01 | completed | turn-left，1000 / 1 / 42 | complete | [结果](<raw/turn-left-a01/result.json>) / [日志](<raw/turn-left-a01/console.log>) |
| turn-right-a01 | completed | turn-right，1000 / 1 / 42 | complete | [结果](<raw/turn-right-a01/result.json>) / [日志](<raw/turn-right-a01/console.log>) |
| forward-left-a01 | completed | forward-left，1000 / 1 / 42 | complete | [结果](<raw/forward-left-a01/result.json>) / [日志](<raw/forward-left-a01/console.log>) |
| forward-right-a01 | completed | forward-right，1000 / 1 / 42 | complete | [结果](<raw/forward-right-a01/result.json>) / [日志](<raw/forward-right-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`be1cdbe764c71a69fd5877a3ec0d301be7d7f1bb88225a1fb2ac45ac845e8ece`；runner：`OnPolicyRunnerROA`。
- [场景来源 scenario-2036fa10309cf369f98ee7af3b0407e9717d40fdcda93dd046969a0284071aaa.json](<../../provenance/scenario-2036fa10309cf369f98ee7af3b0407e9717d40fdcda93dd046969a0284071aaa.json>)
- [场景来源 scenario-3b3e7465d0b14d120e4d28258b03346cfc62fc032c8aa54fa9bd75ea10596239.json](<../../provenance/scenario-3b3e7465d0b14d120e4d28258b03346cfc62fc032c8aa54fa9bd75ea10596239.json>)
- [场景来源 scenario-427ce65f84946a7bfb6ceb8d5ec37df42211bc11793e9de34b616382c2c9787e.json](<../../provenance/scenario-427ce65f84946a7bfb6ceb8d5ec37df42211bc11793e9de34b616382c2c9787e.json>)
- [场景来源 scenario-6416953c4ca286ea1fcc5d0b1531b735e3628bd4dc6b2f36694e1bd5a144b45e.json](<../../provenance/scenario-6416953c4ca286ea1fcc5d0b1531b735e3628bd4dc6b2f36694e1bd5a144b45e.json>)
- [场景来源 scenario-bd80b3737b646c277683348f93d531ea1e2bc31d9165b6fd4a82cf47e4f2f573.json](<../../provenance/scenario-bd80b3737b646c277683348f93d531ea1e2bc31d9165b6fd4a82cf47e4f2f573.json>)
- [场景来源 scenario-c90df12ff192725b9ff5825193095850983ec2f450f0cd682844f8f728614c11.json](<../../provenance/scenario-c90df12ff192725b9ff5825193095850983ec2f450f0cd682844f8f728614c11.json>)
- [场景来源 scenario-ff5833a07ec0297aed6fb093cd6955da989251dfb276e65f188386a164845488.json](<../../provenance/scenario-ff5833a07ec0297aed6fb093cd6955da989251dfb276e65f188386a164845488.json>)
- [训练上下文 context-de97fb3fc021b957963b1b2c2bccbbcf51551f530b1258b6adb8b8d6d624bba9.json](<../../provenance/context-de97fb3fc021b957963b1b2c2bccbbcf51551f530b1258b6adb8b8d6d624bba9.json>)
- [训练有效配置 config-dda41db5786ef0d7987922b3aa9bee3df3096af742d3ee622b0bcc792649ebde.json](<../../provenance/config-dda41db5786ef0d7987922b3aa9bee3df3096af742d3ee622b0bcc792649ebde.json>)

- `stand-a01` 指标：`{"max_joint_velocity_utilization": 1.0000000953674317, "max_tilt": 0.17805446684360504, "termination_rate": 0.0, "tracking_xy_rmse": 0.06304121056068802, "tracking_yaw_rmse": 0.053622641211332}`
- `stand-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 999, "start_step": 0}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 65.0, "evaluation.roa_mode": "student", "events.randomize_push_robot": null, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane"}`。
- `forward-a01` 指标：`{"max_joint_velocity_utilization": 1.1436073303222656, "max_tilt": 0.20997120440006256, "termination_rate": 0.0, "tracking_xy_rmse": 0.18084190815334839, "tracking_yaw_rmse": 0.11607696395383303}`
- `forward-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0.4, 0, 0], "end_step": 749, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 750}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 65.0, "evaluation.roa_foot_ground": true, "evaluation.roa_mode": "student", "events.randomize_push_robot": null, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane"}`。
- `backward-a01` 指标：`{"max_joint_velocity_utilization": 1.0324577331542968, "max_tilt": 0.19431094825267792, "termination_rate": 0.0, "tracking_xy_rmse": 0.18974912900961624, "tracking_yaw_rmse": 0.1632977268658292}`
- `backward-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [-0.4, 0, 0], "end_step": 749, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 750}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 65.0, "evaluation.roa_foot_ground": true, "evaluation.roa_mode": "student", "events.randomize_push_robot": null, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane"}`。
- `turn-left-a01` 指标：`{"max_joint_velocity_utilization": 1.2826967239379883, "max_tilt": 0.8361077308654785, "termination_rate": 0.001, "tracking_xy_rmse": 0.258348599100139, "tracking_yaw_rmse": 0.5088440648563675}`
- `turn-left-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0, 0, 0.5], "end_step": 749, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 750}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 65.0, "evaluation.roa_mode": "student", "events.randomize_push_robot": null, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane"}`。
- `turn-right-a01` 指标：`{"max_joint_velocity_utilization": 1.2825800895690918, "max_tilt": 0.27124419808387756, "termination_rate": 0.0, "tracking_xy_rmse": 0.23299554547342088, "tracking_yaw_rmse": 0.3516131728876927}`
- `turn-right-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0, 0, -0.5], "end_step": 749, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 750}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 65.0, "evaluation.roa_mode": "student", "events.randomize_push_robot": null, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane"}`。
- `forward-left-a01` 指标：`{"max_joint_velocity_utilization": 1.2706422805786133, "max_tilt": 0.27894726395606995, "termination_rate": 0.0, "tracking_xy_rmse": 0.18073690168256976, "tracking_yaw_rmse": 0.3554627728418044}`
- `forward-left-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0.4, 0, 0.5], "end_step": 749, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 750}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 65.0, "evaluation.roa_mode": "student", "events.randomize_push_robot": null, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane"}`。
- `forward-right-a01` 指标：`{"max_joint_velocity_utilization": 1.0748414039611816, "max_tilt": 0.2424369603395462, "termination_rate": 0.0, "tracking_xy_rmse": 0.1830015223585523, "tracking_yaw_rmse": 0.35330792387403265}`
- `forward-right-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0.4, 0, -0.5], "end_step": 749, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 750}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 65.0, "evaluation.roa_mode": "student", "events.randomize_push_robot": null, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane"}`。

## 观察与限制

- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据


## 建议与待授权事项

- 按批次报告检查失败项和动作表现；复测使用新的 batch_id 与 attempt_id。
