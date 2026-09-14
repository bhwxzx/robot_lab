# 评估批次 flat-review-20260912-001

任务：`RobotLab-Isaac-Velocity-Flat-LW-leg-Amp-Roa-v0`；训练：`2026-09-11_16-56-27`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| stand-a01 | completed | stand，1000 / 1 / 42 | complete | [结果](<raw/stand-a01/result.json>) / [视频](<raw/stand-a01/video.mp4>) / [日志](<raw/stand-a01/console.log>) |
| forward-stop-a01 | completed | forward-stop，1000 / 1 / 42 | complete | [结果](<raw/forward-stop-a01/result.json>) / [视频](<raw/forward-stop-a01/video.mp4>) / [日志](<raw/forward-stop-a01/console.log>) |
| backward-stop-a01 | completed | backward-stop，1000 / 1 / 42 | complete | [结果](<raw/backward-stop-a01/result.json>) / [视频](<raw/backward-stop-a01/video.mp4>) / [日志](<raw/backward-stop-a01/console.log>) |
| turn-left-a01 | completed | turn-left，1000 / 1 / 42 | complete | [结果](<raw/turn-left-a01/result.json>) / [视频](<raw/turn-left-a01/video.mp4>) / [日志](<raw/turn-left-a01/console.log>) |
| turn-right-a01 | completed | turn-right，1000 / 1 / 42 | complete | [结果](<raw/turn-right-a01/result.json>) / [视频](<raw/turn-right-a01/video.mp4>) / [日志](<raw/turn-right-a01/console.log>) |
| moving-turn-a01 | completed | moving-turn，1000 / 1 / 42 | complete | [结果](<raw/moving-turn-a01/result.json>) / [视频](<raw/moving-turn-a01/video.mp4>) / [日志](<raw/moving-turn-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`cf0ae2f701c0b2ec15f5a43bfadb61425b879f68e425667a41d67a0b673d4d93`；runner：`OnPolicyRunnerAmpROA`。
- [场景来源 scenario-781a02e854bac25cfc0c456c5a46a981ec953fe192e25002d7fb1ed3d8403e8d.json](<../../provenance/scenario-781a02e854bac25cfc0c456c5a46a981ec953fe192e25002d7fb1ed3d8403e8d.json>)
- [场景来源 scenario-7dd2fd554b4401e97558afcc1fc311703455eac58bddfa611c0347e7c06da9aa.json](<../../provenance/scenario-7dd2fd554b4401e97558afcc1fc311703455eac58bddfa611c0347e7c06da9aa.json>)
- [场景来源 scenario-bf0788e5059f1ad5f8ae9f1f4e52d72fbcd73eab65d7eec08fac2869f0e9e58e.json](<../../provenance/scenario-bf0788e5059f1ad5f8ae9f1f4e52d72fbcd73eab65d7eec08fac2869f0e9e58e.json>)
- [场景来源 scenario-cac76334de98399409a1d6b5675e74ddc71b7b7dcc65f7d136a8be4755ecd8be.json](<../../provenance/scenario-cac76334de98399409a1d6b5675e74ddc71b7b7dcc65f7d136a8be4755ecd8be.json>)
- [场景来源 scenario-cff6bfd96c4e0780a11c120048bf21be3818c4c69ebcb95d3c29db8c96abbaad.json](<../../provenance/scenario-cff6bfd96c4e0780a11c120048bf21be3818c4c69ebcb95d3c29db8c96abbaad.json>)
- [场景来源 scenario-eea6404416cf88982926d7acbfe410b5f6ba6f69266060b57f663538b8de2728.json](<../../provenance/scenario-eea6404416cf88982926d7acbfe410b5f6ba6f69266060b57f663538b8de2728.json>)
- [训练上下文 context-81acb5359f1d1d0c02f1c56d9ffecd2ca49944775fde3d764257e5d17c073b3e.json](<../../provenance/context-81acb5359f1d1d0c02f1c56d9ffecd2ca49944775fde3d764257e5d17c073b3e.json>)
- [训练有效配置 config-7db9a64a6a06a89fe1ebe97abae3bf4d468aa6f8277752ade4caf4d198232053.json](<../../provenance/config-7db9a64a6a06a89fe1ebe97abae3bf4d468aa6f8277752ade4caf4d198232053.json>)

- `stand-a01` 指标：`{"max_joint_velocity_utilization": 1.0761926651000977, "max_tilt": 0.22647535800933838, "termination_rate": 0.0, "tracking_xy_rmse": 0.14625772644300633, "tracking_yaw_rmse": 0.16877291990451612}`
- `stand-a01` 命令调度：`[{"start_step": 0, "end_step": 999, "command": [0, 0, 0]}]`；训练配置覆盖：`{"episode_length_s": 65.0, "events.randomize_push_robot": null}`。
- `forward-stop-a01` 指标：`{"max_joint_velocity_utilization": 0.8900698661804199, "max_tilt": 0.17007936537265778, "termination_rate": 0.0, "tracking_xy_rmse": 0.14500354812943458, "tracking_yaw_rmse": 0.13655247804220158}`
- `forward-stop-a01` 命令调度：`[{"start_step": 0, "end_step": 99, "command": [0, 0, 0]}, {"start_step": 100, "end_step": 749, "command": [0.4, 0, 0]}, {"start_step": 750, "end_step": 999, "command": [0, 0, 0]}]`；训练配置覆盖：`{"episode_length_s": 65.0, "events.randomize_push_robot": null}`。
- `backward-stop-a01` 指标：`{"max_joint_velocity_utilization": 1.1488786697387696, "max_tilt": 0.21258483827114105, "termination_rate": 0.0, "tracking_xy_rmse": 0.1760138558550174, "tracking_yaw_rmse": 0.13150748941394058}`
- `backward-stop-a01` 命令调度：`[{"start_step": 0, "end_step": 99, "command": [0, 0, 0]}, {"start_step": 100, "end_step": 749, "command": [-0.4, 0, 0]}, {"start_step": 750, "end_step": 999, "command": [0, 0, 0]}]`；训练配置覆盖：`{"episode_length_s": 65.0, "events.randomize_push_robot": null}`。
- `turn-left-a01` 指标：`{"max_joint_velocity_utilization": 1.0139134407043457, "max_tilt": 0.17772167921066284, "termination_rate": 0.0, "tracking_xy_rmse": 0.07450210693278976, "tracking_yaw_rmse": 0.42470401926958307}`
- `turn-left-a01` 命令调度：`[{"start_step": 0, "end_step": 99, "command": [0, 0, 0]}, {"start_step": 100, "end_step": 749, "command": [0, 0, 0.5]}, {"start_step": 750, "end_step": 999, "command": [0, 0, 0]}]`；训练配置覆盖：`{"episode_length_s": 65.0, "events.randomize_push_robot": null}`。
- `turn-right-a01` 指标：`{"max_joint_velocity_utilization": 0.9773658752441406, "max_tilt": 0.21218016743659973, "termination_rate": 0.0, "tracking_xy_rmse": 0.08464879647484504, "tracking_yaw_rmse": 0.44200770779541104}`
- `turn-right-a01` 命令调度：`[{"start_step": 0, "end_step": 99, "command": [0, 0, 0]}, {"start_step": 100, "end_step": 749, "command": [0, 0, -0.5]}, {"start_step": 750, "end_step": 999, "command": [0, 0, 0]}]`；训练配置覆盖：`{"episode_length_s": 65.0, "events.randomize_push_robot": null}`。
- `moving-turn-a01` 指标：`{"max_joint_velocity_utilization": 1.029372787475586, "max_tilt": 0.18136359751224518, "termination_rate": 0.0, "tracking_xy_rmse": 0.15595103376592984, "tracking_yaw_rmse": 0.39029089596891103}`
- `moving-turn-a01` 命令调度：`[{"start_step": 0, "end_step": 99, "command": [0, 0, 0]}, {"start_step": 100, "end_step": 749, "command": [0.4, 0, 0.5]}, {"start_step": 750, "end_step": 999, "command": [0, 0, 0]}]`；训练配置覆盖：`{"episode_length_s": 65.0, "events.randomize_push_robot": null}`。

## 观察与限制

- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据


## 建议与待授权事项

- 按批次报告检查失败项和动作表现；复测使用新的 batch_id 与 attempt_id。
