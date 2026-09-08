# 评估批次 turning-review-20260908-001

任务：`RobotLab-Isaac-Velocity-Flat-LW-leg-Amp-Roa-v0`；训练：`2026-09-04_11-16-35`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| turn-followup-left-direct-20260907-001 | completed | turn-followup-left-direct-20260907-001，1000 / 1 / 42 | complete | [结果](<../../evidence/play/turn-followup-left-direct-20260907-001/result.json>) / [视频](<../../evidence/play/turn-followup-left-direct-20260907-001/video.mp4>) |
| turn-followup-left-ramp-20260907-001 | completed | turn-followup-left-ramp-20260907-001，2500 / 1 / 42 | complete | [结果](<../../evidence/play/turn-followup-left-ramp-20260907-001/result.json>) / [视频](<../../evidence/play/turn-followup-left-ramp-20260907-001/video.mp4>) |
| turn-followup-left-walking-20260907-001 | completed | turn-followup-left-walking-20260907-001，1500 / 1 / 42 | complete | [结果](<../../evidence/play/turn-followup-left-walking-20260907-001/result.json>) / [视频](<../../evidence/play/turn-followup-left-walking-20260907-001/video.mp4>) |
| turn-followup-right-direct-20260907-001 | completed | turn-followup-right-direct-20260907-001，1000 / 1 / 42 | complete | [结果](<../../evidence/play/turn-followup-right-direct-20260907-001/result.json>) / [视频](<../../evidence/play/turn-followup-right-direct-20260907-001/video.mp4>) |
| turn-followup-right-ramp-20260907-001 | completed | turn-followup-right-ramp-20260907-001，2500 / 1 / 42 | complete | [结果](<../../evidence/play/turn-followup-right-ramp-20260907-001/result.json>) / [视频](<../../evidence/play/turn-followup-right-ramp-20260907-001/video.mp4>) |
| turn-followup-right-walking-20260907-001 | completed | turn-followup-right-walking-20260907-001，1500 / 1 / 42 | complete | [结果](<../../evidence/play/turn-followup-right-walking-20260907-001/result.json>) / [视频](<../../evidence/play/turn-followup-right-walking-20260907-001/video.mp4>) |

## 策略与场景

- checkpoint SHA-256：`59ab5245acb17de8ea1b79e924c2f0189765be3c51c85f437b22b8e2423b0e19`；runner：`OnPolicyRunnerAmpROA`。
- [训练上下文 identity-turn-followup-left-direct-20260907-001.json](<../../evidence/source/identity-turn-followup-left-direct-20260907-001.json>)
- [训练上下文 identity-turn-followup-left-ramp-20260907-001.json](<../../evidence/source/identity-turn-followup-left-ramp-20260907-001.json>)
- [训练上下文 identity-turn-followup-left-walking-20260907-001.json](<../../evidence/source/identity-turn-followup-left-walking-20260907-001.json>)
- [训练上下文 identity-turn-followup-right-direct-20260907-001.json](<../../evidence/source/identity-turn-followup-right-direct-20260907-001.json>)
- [训练上下文 identity-turn-followup-right-ramp-20260907-001.json](<../../evidence/source/identity-turn-followup-right-ramp-20260907-001.json>)
- [训练上下文 identity-turn-followup-right-walking-20260907-001.json](<../../evidence/source/identity-turn-followup-right-walking-20260907-001.json>)

- `turn-followup-left-direct-20260907-001` 指标：`{"max_joint_velocity_utilization": 0.9597506523132324, "max_tilt": 0.09370432794094086, "termination_rate": 0.0, "tracking_xy_rmse": 0.061332283643953336, "tracking_yaw_rmse": 0.36546041439407767}`
- `turn-followup-left-direct-20260907-001` 命令调度：`[{"command": [0, 0, 0], "end_step": 499, "start_step": 0}, {"command": [0, 0, 0.5], "end_step": 999, "start_step": 500}]`；训练配置覆盖：`{"episode_length_s": 65.0, "events.randomize_push_robot": null}`。
- `turn-followup-left-ramp-20260907-001` 指标：`{"max_joint_velocity_utilization": 0.9597506523132324, "max_tilt": 0.17069269716739655, "termination_rate": 0.0, "tracking_xy_rmse": 0.12922303330919208, "tracking_yaw_rmse": 0.45946074347730875}`
- `turn-followup-left-ramp-20260907-001` 命令调度：`[{"command": [0, 0, 0], "end_step": 499, "start_step": 0}, {"command": [0.4, 0, 0.5], "end_step": 999, "start_step": 500}, {"command": [0.2, 0, 0.5], "end_step": 1499, "start_step": 1000}, {"command": [0.1, 0, 0.5], "end_step": 1999, "start_step": 1500}, {"command": [0, 0, 0.5], "end_step": 2499, "start_step": 2000}]`；训练配置覆盖：`{"episode_length_s": 65.0, "events.randomize_push_robot": null}`。
- `turn-followup-left-walking-20260907-001` 指标：`{"max_joint_velocity_utilization": 0.9597506523132324, "max_tilt": 0.17069269716739655, "termination_rate": 0.0, "tracking_xy_rmse": 0.13986212580695306, "tracking_yaw_rmse": 0.422238497795652}`
- `turn-followup-left-walking-20260907-001` 命令调度：`[{"command": [0, 0, 0], "end_step": 499, "start_step": 0}, {"command": [0.4, 0, 0.5], "end_step": 999, "start_step": 500}, {"command": [0, 0, 0.5], "end_step": 1499, "start_step": 1000}]`；训练配置覆盖：`{"episode_length_s": 65.0, "events.randomize_push_robot": null}`。
- `turn-followup-right-direct-20260907-001` 指标：`{"max_joint_velocity_utilization": 0.9597506523132324, "max_tilt": 0.10373694449663162, "termination_rate": 0.0, "tracking_xy_rmse": 0.06118129455185702, "tracking_yaw_rmse": 0.35924127040347265}`
- `turn-followup-right-direct-20260907-001` 命令调度：`[{"command": [0, 0, 0], "end_step": 499, "start_step": 0}, {"command": [0, 0, -0.5], "end_step": 999, "start_step": 500}]`；训练配置覆盖：`{"episode_length_s": 65.0, "events.randomize_push_robot": null}`。
- `turn-followup-right-ramp-20260907-001` 指标：`{"max_joint_velocity_utilization": 0.9747695922851562, "max_tilt": 0.16490136086940765, "termination_rate": 0.0, "tracking_xy_rmse": 0.1436067618808395, "tracking_yaw_rmse": 0.44175369856313396}`
- `turn-followup-right-ramp-20260907-001` 命令调度：`[{"command": [0, 0, 0], "end_step": 499, "start_step": 0}, {"command": [0.4, 0, -0.5], "end_step": 999, "start_step": 500}, {"command": [0.2, 0, -0.5], "end_step": 1499, "start_step": 1000}, {"command": [0.1, 0, -0.5], "end_step": 1999, "start_step": 1500}, {"command": [0, 0, -0.5], "end_step": 2499, "start_step": 2000}]`；训练配置覆盖：`{"episode_length_s": 65.0, "events.randomize_push_robot": null}`。
- `turn-followup-right-walking-20260907-001` 指标：`{"max_joint_velocity_utilization": 0.9597506523132324, "max_tilt": 0.16490136086940765, "termination_rate": 0.0, "tracking_xy_rmse": 0.1402562267838323, "tracking_yaw_rmse": 0.39905196828464784}`
- `turn-followup-right-walking-20260907-001` 命令调度：`[{"command": [0, 0, 0], "end_step": 499, "start_step": 0}, {"command": [0.4, 0, -0.5], "end_step": 999, "start_step": 500}, {"command": [0, 0, -0.5], "end_step": 1499, "start_step": 1000}]`；训练配置覆盖：`{"episode_length_s": 65.0, "events.randomize_push_robot": null}`。

## 观察与限制

- 本批次仅整理 2026-09-07 已完成的 6 个转向补测；没有重新运行或改写这些结果。
- 原补测报告末 5 秒世界航向变化：纯转左约 +0.226°、纯转右约 +0.037°；vx=0.4 的组合命令左约 +8.36°、右约 -18.93°。结合视频，组合命令确实能转向，纯转响应很弱。
- 由行走切回纯转后航向变化再次接近零。右转在 vx=0.2 阶段仍有转向、vx=0.1 时基本消失；只覆盖 1 个 seed，不能据此确定通用速度阈值。
- 6 个补测均无 termination，初始站立阶段一致；未采集脚接触信号，因此未用脚接触解释控制器内部机制。
- 本报告的整体 RMSE 覆盖整段调度；稳态转向结论采用原补测报告明确的分段窗口。
- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据

- [turn-followup-report-20260907-001.md](<../../evidence/training/turn-followup-report-20260907-001.md>)
- [summary-turn-followup-summary-20260907-001.json](<../../evidence/training/summary-turn-followup-summary-20260907-001.json>)
- [turn-followup-heading-20260907-001.png](<../../evidence/training/turn-followup-heading-20260907-001.png>)

## 建议与待授权事项

- 后续参数建议应针对纯转和低速转向覆盖，并用相同分段调度检查速度跟踪、姿态、关节速度和终止情况。
- 本批次未建立用户批准的收敛判据，也不自动授权修改奖励或重新训练。
