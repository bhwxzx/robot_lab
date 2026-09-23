# 评估批次 leg-to-wheel-native-20260921-001

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-leg-v0`；训练：`2026-09-19_15-58-24`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| nominal-start-a01 | completed | nominal-start，2000 / 1 / 42 | complete | [结果](<raw/nominal-start-a01/result.json>) / [日志](<raw/nominal-start-a01/console.log>) |
| randomized-start-a01 | completed | randomized-start，2000 / 1 / 42 | complete | [结果](<raw/randomized-start-a01/result.json>) / [日志](<raw/randomized-start-a01/console.log>) |
| randomized-phase-a01 | completed | randomized-phase，2000 / 1 / 42 | complete | [结果](<raw/randomized-phase-a01/result.json>) / [日志](<raw/randomized-phase-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`aa31a4d3af8a8dd27894dffdc3a2edf7b8e7b8056854dae7be990c98d566dccb`；runner：`OnPolicyRunner`。
- [场景来源 scenario-1538944a88e8f135a1131471cccfff50a7b38c5ecfc5687470ef6df5d70351cd.json](<../../provenance/scenario-1538944a88e8f135a1131471cccfff50a7b38c5ecfc5687470ef6df5d70351cd.json>)
- [场景来源 scenario-7bfffc40678f528bde8ccbe50a83a77f6ea6788dcc9ce9bc085606e2347e8865.json](<../../provenance/scenario-7bfffc40678f528bde8ccbe50a83a77f6ea6788dcc9ce9bc085606e2347e8865.json>)
- [场景来源 scenario-7d77570c0b7037bc3737d755f24892322333b8933f35e058d6aba8dab0e37890.json](<../../provenance/scenario-7d77570c0b7037bc3737d755f24892322333b8933f35e058d6aba8dab0e37890.json>)
- [训练上下文 context-e76f4ecba9b16797a2d0f69bdf9d7cfa9977f89f4f5dca3a13f577cb7e6b9729.json](<../../provenance/context-e76f4ecba9b16797a2d0f69bdf9d7cfa9977f89f4f5dca3a13f577cb7e6b9729.json>)
- [训练有效配置 config-7d4e603b42fa6e9aa9d1ffc8a61b4311bbbc55bf67dec4de3e0563efb562e20c.json](<../../provenance/config-7d4e603b42fa6e9aa9d1ffc8a61b4311bbbc55bf67dec4de3e0563efb562e20c.json>)

- `nominal-start-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `nominal-start-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_mode": "nominal_start", "evaluation.motion_sha256": "85e53dc37459b295a4a3f06dd56044eb96b661caf01461ccceb1908cf52fa75d"}`。
- `randomized-start-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `randomized-start-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_mode": "randomized_start", "evaluation.motion_sha256": "85e53dc37459b295a4a3f06dd56044eb96b661caf01461ccceb1908cf52fa75d"}`。
- `randomized-phase-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `randomized-phase-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_mode": "randomized_phase", "evaluation.motion_sha256": "85e53dc37459b295a4a3f06dd56044eb96b661caf01461ccceb1908cf52fa75d"}`。

## 观察与限制

- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据


## 建议与待授权事项

- 按批次报告检查失败项和动作表现；复测使用新的 batch_id 与 attempt_id。
