# 评估批次 native-assess-20260910-001

任务：`RobotLab-Isaac-Velocity-Flat-LW-wheel-Dwaq-v0`；训练：`2026-09-04_11-24-29`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| baseline-no-push-a01 | completed | baseline-no-push，2100 / 4 / 42 | complete | [结果](<raw/baseline-no-push-a01/result.json>) / [日志](<raw/baseline-no-push-a01/console.log>) |
| training-push-a01 | completed | training-push，2100 / 4 / 42 | complete | [结果](<raw/training-push-a01/result.json>) / [日志](<raw/training-push-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`ce093b030001c77d53e3809d0e85a3e152f953aa6cc0a6a04be247f71ac2c799`；runner：`OnPolicyRunnerDwaq`。
- [场景来源 scenario-1c901959b13d5be0edd9eb749b570e435849f7941a1f0c059e5f410a7187db42.json](<../../provenance/scenario-1c901959b13d5be0edd9eb749b570e435849f7941a1f0c059e5f410a7187db42.json>)
- [场景来源 scenario-5fbf2ddd96227f41d2cb89973ca1d2d9ddc8d48f5b3b58f3686c575673abb2ec.json](<../../provenance/scenario-5fbf2ddd96227f41d2cb89973ca1d2d9ddc8d48f5b3b58f3686c575673abb2ec.json>)
- [训练上下文 context-6fa1839a572eb2d281403a64179235c1339830a1f5dd1ddc145b671e512a22a9.json](<../../provenance/context-6fa1839a572eb2d281403a64179235c1339830a1f5dd1ddc145b671e512a22a9.json>)
- [训练有效配置 config-2c9ee4205b19a7b45ab62c2de2b58252ab60ee5706804f8b832fa1767d25f96a.json](<../../provenance/config-2c9ee4205b19a7b45ab62c2de2b58252ab60ee5706804f8b832fa1767d25f96a.json>)

- `baseline-no-push-a01` 指标：`{"max_joint_velocity_utilization": 0.5463266083688447, "max_tilt": 0.1554565131664276, "termination_rate": 0.0, "tracking_xy_rmse": 0.1763441279410229, "tracking_yaw_rmse": 0.20617079305982322}`
- `baseline-no-push-a01` 命令调度：`[{"start_step": 0, "end_step": 299, "command": [0, 0, 0]}, {"start_step": 300, "end_step": 599, "command": [0.5, 0, 0]}, {"start_step": 600, "end_step": 899, "command": [1.0, 0, 0]}, {"start_step": 900, "end_step": 1199, "command": [-0.5, 0, 0]}, {"start_step": 1200, "end_step": 1499, "command": [0.5, 0, 0.6]}, {"start_step": 1500, "end_step": 1799, "command": [0.5, 0, -0.6]}, {"start_step": 1800, "end_step": 2099, "command": [0, 0, 0]}]`；训练配置覆盖：`{"episode_length_s": 60.0, "events.randomize_push_robot": null}`。
- `training-push-a01` 指标：`{"max_joint_velocity_utilization": 0.8052614385431464, "max_tilt": 0.3203573226928711, "termination_rate": 0.0, "tracking_xy_rmse": 0.32417319673414824, "tracking_yaw_rmse": 0.25473383764517943}`
- `training-push-a01` 命令调度：`[{"start_step": 0, "end_step": 299, "command": [0, 0, 0]}, {"start_step": 300, "end_step": 599, "command": [0.5, 0, 0]}, {"start_step": 600, "end_step": 899, "command": [1.0, 0, 0]}, {"start_step": 900, "end_step": 1199, "command": [-0.5, 0, 0]}, {"start_step": 1200, "end_step": 1499, "command": [0.5, 0, 0.6]}, {"start_step": 1500, "end_step": 1799, "command": [0.5, 0, -0.6]}, {"start_step": 1800, "end_step": 2099, "command": [0, 0, 0]}]`；训练配置覆盖：`{"episode_length_s": 60.0}`。

## 观察与限制

- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据


## 建议与待授权事项

- 按批次报告检查失败项和动作表现；复测使用新的 batch_id 与 attempt_id。
