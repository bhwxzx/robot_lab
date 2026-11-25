# 评估批次 flat-leg-roa-49999-gait-turn-check-20261125-001

任务：`RobotLab-Isaac-Velocity-Flat-LW-leg-Roa-v0`；训练：`2026-11-24_14-41-24`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| forward-030-a01 | completed | forward-030，1000 / 1 / 42 | complete | [结果](<raw/forward-030-a01/result.json>) / [日志](<raw/forward-030-a01/console.log>) |
| turn-left-extended-a01 | completed | turn-left-extended，1000 / 1 / 42 | complete | [结果](<raw/turn-left-extended-a01/result.json>) / [日志](<raw/turn-left-extended-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`be1cdbe764c71a69fd5877a3ec0d301be7d7f1bb88225a1fb2ac45ac845e8ece`；runner：`OnPolicyRunnerROA`。
- [场景来源 scenario-63ece1f73c29e7737382b6b95b710708b9758c24eb8a1dd60ae8e309608e493c.json](<../../provenance/scenario-63ece1f73c29e7737382b6b95b710708b9758c24eb8a1dd60ae8e309608e493c.json>)
- [场景来源 scenario-de71a8a259a01cb5a70b12751b2d60ba4a7bdf1adc30e4a3402e341983861bc5.json](<../../provenance/scenario-de71a8a259a01cb5a70b12751b2d60ba4a7bdf1adc30e4a3402e341983861bc5.json>)
- [训练上下文 context-de97fb3fc021b957963b1b2c2bccbbcf51551f530b1258b6adb8b8d6d624bba9.json](<../../provenance/context-de97fb3fc021b957963b1b2c2bccbbcf51551f530b1258b6adb8b8d6d624bba9.json>)
- [训练有效配置 config-dda41db5786ef0d7987922b3aa9bee3df3096af742d3ee622b0bcc792649ebde.json](<../../provenance/config-dda41db5786ef0d7987922b3aa9bee3df3096af742d3ee622b0bcc792649ebde.json>)

- `forward-030-a01` 指标：`{"max_joint_velocity_utilization": 1.1415823936462401, "max_tilt": 0.2119901180267334, "termination_rate": 0.0, "tracking_xy_rmse": 0.1869780712463751, "tracking_yaw_rmse": 0.1158218944760581}`
- `forward-030-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0.3, 0, 0], "end_step": 749, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 750}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 65.0, "evaluation.roa_foot_ground": true, "evaluation.roa_mode": "student", "events.randomize_push_robot": null, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane"}`。
- `turn-left-extended-a01` 指标：`{"max_joint_velocity_utilization": 1.0597393989562989, "max_tilt": 0.8361077308654785, "termination_rate": 0.001, "tracking_xy_rmse": 0.27050321653887105, "tracking_yaw_rmse": 0.4639657528485019}`
- `turn-left-extended-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0, 0, 0.5], "end_step": 849, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 850}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 65.0, "evaluation.roa_mode": "student", "events.randomize_push_robot": null, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane"}`。

## 观察与限制

- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据


## 建议与待授权事项

- 按批次报告检查失败项和动作表现；复测使用新的 batch_id 与 attempt_id。
