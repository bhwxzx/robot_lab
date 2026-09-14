# 评估批次 native-assess-20260914-001

任务：`RobotLab-Isaac-Velocity-Flat-LW-wheel-Dwaq-v0`；训练：`2026-09-12_22-05-27`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| baseline-no-push-a01 | completed | baseline-no-push，2100 / 4 / 42 | complete | [结果](<raw/baseline-no-push-a01/result.json>) / [日志](<raw/baseline-no-push-a01/console.log>) |
| legacy-push-a01 | completed | legacy-push，2100 / 4 / 42 | complete | [结果](<raw/legacy-push-a01/result.json>) / [日志](<raw/legacy-push-a01/console.log>) |
| current-push-a01 | completed | current-push，2100 / 4 / 42 | complete | [结果](<raw/current-push-a01/result.json>) / [日志](<raw/current-push-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`4007440e1dac0e48396aa87cc2454202ddb66de730c25790170b6d4c4bcf5652`；runner：`OnPolicyRunnerDwaq`。
- [场景来源 scenario-49a67f1967f2f1fa2fe398ac0b051a69d01f9afcd41ea532ea66a988c75b0ac0.json](<../../provenance/scenario-49a67f1967f2f1fa2fe398ac0b051a69d01f9afcd41ea532ea66a988c75b0ac0.json>)
- [场景来源 scenario-6f96f390c6d2d9d0a6f4114efb0240014c44046ba2eae47341ef527e7f282604.json](<../../provenance/scenario-6f96f390c6d2d9d0a6f4114efb0240014c44046ba2eae47341ef527e7f282604.json>)
- [场景来源 scenario-dbba4acb3f9ac4b7263f274c3e0b3b772a7bedc160d67c252dad1a9055e41184.json](<../../provenance/scenario-dbba4acb3f9ac4b7263f274c3e0b3b772a7bedc160d67c252dad1a9055e41184.json>)
- [训练上下文 context-54eedfcce866adfff0f4ca26b6663ebb25b3f777b62c6149451d7e772a92f74a.json](<../../provenance/context-54eedfcce866adfff0f4ca26b6663ebb25b3f777b62c6149451d7e772a92f74a.json>)
- [训练有效配置 config-f46d43603f25a340da12ac319f6fd9a594315e59c6bc510c30a5574a910f700f.json](<../../provenance/config-f46d43603f25a340da12ac319f6fd9a594315e59c6bc510c30a5574a910f700f.json>)

- `baseline-no-push-a01` 指标：`{"max_joint_velocity_utilization": 0.586325385353782, "max_tilt": 0.182007297873497, "termination_rate": 0.0, "tracking_xy_rmse": 0.15455523669272966, "tracking_yaw_rmse": 0.20543047610818616}`
- `baseline-no-push-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 299, "start_step": 0}, {"command": [0.5, 0, 0], "end_step": 599, "start_step": 300}, {"command": [1.0, 0, 0], "end_step": 899, "start_step": 600}, {"command": [-0.5, 0, 0], "end_step": 1199, "start_step": 900}, {"command": [0.5, 0, 0.6], "end_step": 1499, "start_step": 1200}, {"command": [0.5, 0, -0.6], "end_step": 1799, "start_step": 1500}, {"command": [0, 0, 0], "end_step": 2099, "start_step": 1800}]`；训练配置覆盖：`{"episode_length_s": 60.0, "events.randomize_push_robot": null}`。
- `legacy-push-a01` 指标：`{"max_joint_velocity_utilization": 0.805986346620502, "max_tilt": 0.2594101130962372, "termination_rate": 0.0, "tracking_xy_rmse": 0.23025694386863454, "tracking_yaw_rmse": 0.24329924060269068}`
- `legacy-push-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 299, "start_step": 0}, {"command": [0.5, 0, 0], "end_step": 599, "start_step": 300}, {"command": [1.0, 0, 0], "end_step": 899, "start_step": 600}, {"command": [-0.5, 0, 0], "end_step": 1199, "start_step": 900}, {"command": [0.5, 0, 0.6], "end_step": 1499, "start_step": 1200}, {"command": [0.5, 0, -0.6], "end_step": 1799, "start_step": 1500}, {"command": [0, 0, 0], "end_step": 2099, "start_step": 1800}]`；训练配置覆盖：`{"episode_length_s": 60.0, "events.randomize_push_robot.params.velocity_range": {"x": [-1.0, 1.0], "y": [1.0, 1.0]}}`。
- `current-push-a01` 指标：`{"max_joint_velocity_utilization": 0.8011798858642578, "max_tilt": 0.18535882234573364, "termination_rate": 0.0, "tracking_xy_rmse": 0.19993258714284518, "tracking_yaw_rmse": 0.22307350865865555}`
- `current-push-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 299, "start_step": 0}, {"command": [0.5, 0, 0], "end_step": 599, "start_step": 300}, {"command": [1.0, 0, 0], "end_step": 899, "start_step": 600}, {"command": [-0.5, 0, 0], "end_step": 1199, "start_step": 900}, {"command": [0.5, 0, 0.6], "end_step": 1499, "start_step": 1200}, {"command": [0.5, 0, -0.6], "end_step": 1799, "start_step": 1500}, {"command": [0, 0, 0], "end_step": 2099, "start_step": 1800}]`；训练配置覆盖：`{"episode_length_s": 60.0, "events.randomize_push_robot.params.velocity_range": {"x": [-1.0, 1.0], "y": [-1.0, 1.0]}}`。

## 观察与限制

- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据


## 建议与待授权事项

- 按批次报告检查失败项和动作表现；复测使用新的 batch_id 与 attempt_id。
