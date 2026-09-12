# 评估批次 native-assess-20260912-001

任务：`RobotLab-Isaac-Velocity-Flat-LW-wheel-Dwaq-v0`；训练：`2026-09-11_17-04-10`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| baseline-no-push-a01 | completed | baseline-no-push，2100 / 4 / 42 | complete | [结果](<raw/baseline-no-push-a01/result.json>) / [日志](<raw/baseline-no-push-a01/console.log>) |
| legacy-push-a01 | completed | legacy-push，2100 / 4 / 42 | complete | [结果](<raw/legacy-push-a01/result.json>) / [日志](<raw/legacy-push-a01/console.log>) |
| current-push-a01 | completed | current-push，2100 / 4 / 42 | complete | [结果](<raw/current-push-a01/result.json>) / [日志](<raw/current-push-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`8798c9d1f744f5795ff7c9234147b7cab033295005056c18857fe83fbdbc22b5`；runner：`OnPolicyRunnerDwaq`。
- [场景来源 scenario-547f340e5994b2b451a4b3dd465962109e9b16b2546c15a6abb98d98a26a20af.json](<../../provenance/scenario-547f340e5994b2b451a4b3dd465962109e9b16b2546c15a6abb98d98a26a20af.json>)
- [场景来源 scenario-79015ad277de1ad58a1f3a8fce1089d0bd93c6ae9219e66459f44d3d3ef0259f.json](<../../provenance/scenario-79015ad277de1ad58a1f3a8fce1089d0bd93c6ae9219e66459f44d3d3ef0259f.json>)
- [场景来源 scenario-89059d3adbdba97fd375930c875a196d87435b2396d0072eef0ce527fd35c703.json](<../../provenance/scenario-89059d3adbdba97fd375930c875a196d87435b2396d0072eef0ce527fd35c703.json>)
- [训练上下文 context-2686b80cb207919f0fac07b73d48c355ae739c87b36f2a77b365851c006123ef.json](<../../provenance/context-2686b80cb207919f0fac07b73d48c355ae739c87b36f2a77b365851c006123ef.json>)
- [训练有效配置 config-5b31a80d98c92183b81556df703062b454a62b7df4091becebb2ffcb97a93730.json](<../../provenance/config-5b31a80d98c92183b81556df703062b454a62b7df4091becebb2ffcb97a93730.json>)

- `baseline-no-push-a01` 指标：`{"max_joint_velocity_utilization": 0.5898014415394176, "max_tilt": 0.1827106922864914, "termination_rate": 0.0, "tracking_xy_rmse": 0.17145124056659897, "tracking_yaw_rmse": 0.2046880816883155}`
- `baseline-no-push-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 299, "start_step": 0}, {"command": [0.5, 0, 0], "end_step": 599, "start_step": 300}, {"command": [1.0, 0, 0], "end_step": 899, "start_step": 600}, {"command": [-0.5, 0, 0], "end_step": 1199, "start_step": 900}, {"command": [0.5, 0, 0.6], "end_step": 1499, "start_step": 1200}, {"command": [0.5, 0, -0.6], "end_step": 1799, "start_step": 1500}, {"command": [0, 0, 0], "end_step": 2099, "start_step": 1800}]`；训练配置覆盖：`{"episode_length_s": 60.0, "events.randomize_push_robot": null}`。
- `legacy-push-a01` 指标：`{"max_joint_velocity_utilization": 0.7855234435110381, "max_tilt": 0.9999255537986755, "termination_rate": 0.00011904761904761905, "tracking_xy_rmse": 0.3245625036233137, "tracking_yaw_rmse": 0.2793007058177612}`
- `legacy-push-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 299, "start_step": 0}, {"command": [0.5, 0, 0], "end_step": 599, "start_step": 300}, {"command": [1.0, 0, 0], "end_step": 899, "start_step": 600}, {"command": [-0.5, 0, 0], "end_step": 1199, "start_step": 900}, {"command": [0.5, 0, 0.6], "end_step": 1499, "start_step": 1200}, {"command": [0.5, 0, -0.6], "end_step": 1799, "start_step": 1500}, {"command": [0, 0, 0], "end_step": 2099, "start_step": 1800}]`；训练配置覆盖：`{"episode_length_s": 60.0, "events.randomize_push_robot.params.velocity_range": {"x": [-1.0, 1.0], "y": [1.0, 1.0]}}`。
- `current-push-a01` 指标：`{"max_joint_velocity_utilization": 0.8724938016949277, "max_tilt": 0.2934057116508484, "termination_rate": 0.0, "tracking_xy_rmse": 0.24134384386882637, "tracking_yaw_rmse": 0.22930047862610412}`
- `current-push-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 299, "start_step": 0}, {"command": [0.5, 0, 0], "end_step": 599, "start_step": 300}, {"command": [1.0, 0, 0], "end_step": 899, "start_step": 600}, {"command": [-0.5, 0, 0], "end_step": 1199, "start_step": 900}, {"command": [0.5, 0, 0.6], "end_step": 1499, "start_step": 1200}, {"command": [0.5, 0, -0.6], "end_step": 1799, "start_step": 1500}, {"command": [0, 0, 0], "end_step": 2099, "start_step": 1800}]`；训练配置覆盖：`{"episode_length_s": 60.0, "events.randomize_push_robot.params.velocity_range": {"x": [-1.0, 1.0], "y": [-1.0, 1.0]}}`。

## 观察与限制

- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据


## 建议与待授权事项

- 按批次报告检查失败项和动作表现；复测使用新的 batch_id 与 attempt_id。
