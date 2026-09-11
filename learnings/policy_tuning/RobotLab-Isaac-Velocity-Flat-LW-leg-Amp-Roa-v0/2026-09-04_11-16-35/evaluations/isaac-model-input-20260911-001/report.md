# 评估批次 isaac-model-input-20260911-001

任务：`RobotLab-Isaac-Velocity-Flat-LW-leg-Amp-Roa-v0`；训练：`2026-09-04_11-16-35`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| stand-trace-a01 | completed | stand-model-input-trace，500 / 1 / 42 | complete | [结果](<raw/stand-trace-a01/result.json>) / [日志](<raw/stand-trace-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`59ab5245acb17de8ea1b79e924c2f0189765be3c51c85f437b22b8e2423b0e19`；runner：`OnPolicyRunnerAmpROA`。
- [场景来源 scenario-a22cf1efe130d7439070f2acdf2ca9a008ab886633a31a57615a06d1d3815554.json](<../../provenance/scenario-a22cf1efe130d7439070f2acdf2ca9a008ab886633a31a57615a06d1d3815554.json>)
- [训练上下文 context-09384ce54946455e5b0d358798a4a4ecffd22ed40dcfb90259037663e8388882.json](<../../provenance/context-09384ce54946455e5b0d358798a4a4ecffd22ed40dcfb90259037663e8388882.json>)
- [训练有效配置 config-60b130fed4816625417deec5d99ba9ddd74eb87f4922dbcbe81bd84bc7920a74.json](<../../provenance/config-60b130fed4816625417deec5d99ba9ddd74eb87f4922dbcbe81bd84bc7920a74.json>)

- `stand-trace-a01` 指标：`{"max_joint_velocity_utilization": 0.9597506523132324, "max_tilt": 0.08984678983688354, "termination_rate": 0.0, "tracking_xy_rmse": 0.0793526608989271, "tracking_yaw_rmse": 0.06699987027339803}`
- `stand-trace-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 499, "start_step": 0}]`；训练配置覆盖：`{"episode_length_s": 65.0, "events.randomize_push_robot": null}`。

## 观察与限制

- Native 仿真已完成；额外模型输入采集未发布（启动时安装的 profiler 未留下有效追踪，退出清理未执行到），本批次不能确认两列模型输入。使用新批次重新采集。
- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据

- [trace_launch.json](<raw/stand-trace-a01/trace_launch.json>)

## 建议与待授权事项

