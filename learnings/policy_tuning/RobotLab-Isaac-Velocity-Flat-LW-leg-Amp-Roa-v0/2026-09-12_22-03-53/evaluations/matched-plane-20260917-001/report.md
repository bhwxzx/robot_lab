# 评估批次 matched-plane-20260917-001

任务：`RobotLab-Isaac-Velocity-Flat-LW-leg-Amp-Roa-v0`；训练：`2026-09-12_22-03-53`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| turn-right-a01 | completed | turn-right，1000 / 1 / 42 | complete | [结果](<raw/turn-right-a01/result.json>) / [日志](<raw/turn-right-a01/console.log>) |
| moving-turn-a01 | completed | moving-turn，1000 / 1 / 42 | complete | [结果](<raw/moving-turn-a01/result.json>) / [日志](<raw/moving-turn-a01/console.log>) |
| moving-turn-right-a01 | completed | moving-turn-right，1000 / 1 / 42 | complete | [结果](<raw/moving-turn-right-a01/result.json>) / [日志](<raw/moving-turn-right-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`c085221208af7328c74da8c260f198822b604c0388a9609c13eb0ec515971536`；runner：`OnPolicyRunnerAmpROA`。
- [场景来源 scenario-15839f27a4810cbef437db12aec87908dd701b9d7afc8d0683074c6b4dc418e9.json](<../../provenance/scenario-15839f27a4810cbef437db12aec87908dd701b9d7afc8d0683074c6b4dc418e9.json>)
- [场景来源 scenario-2466eda407db26afc526b192f61ba53bcf9a63eab65089727d3a413e657c03dc.json](<../../provenance/scenario-2466eda407db26afc526b192f61ba53bcf9a63eab65089727d3a413e657c03dc.json>)
- [场景来源 scenario-24cf517c40ff3c02b64d44b09d88376876ed6b2daf80a815e1f69d44bae990f1.json](<../../provenance/scenario-24cf517c40ff3c02b64d44b09d88376876ed6b2daf80a815e1f69d44bae990f1.json>)
- [训练上下文 context-46ffefbbfa7dd84492d2e9fb0f750d776019691dbff8e8b47d7bdd5795c8a9b1.json](<../../provenance/context-46ffefbbfa7dd84492d2e9fb0f750d776019691dbff8e8b47d7bdd5795c8a9b1.json>)
- [训练有效配置 config-4dcda3327a07455d3febf0ff28b54880e56ee007150dfb322674284ee62c0343.json](<../../provenance/config-4dcda3327a07455d3febf0ff28b54880e56ee007150dfb322674284ee62c0343.json>)

- `turn-right-a01` 指标：`{"max_joint_velocity_utilization": 1.2396879196166992, "max_tilt": 0.23029839992523193, "termination_rate": 0.0, "tracking_xy_rmse": 0.09109656710657899, "tracking_yaw_rmse": 0.2443213873112025}`
- `turn-right-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0, 0, -0.5], "end_step": 749, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 750}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 65.0, "events.randomize_push_robot": null, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane"}`。
- `moving-turn-a01` 指标：`{"max_joint_velocity_utilization": 1.343833827972412, "max_tilt": 0.2695205509662628, "termination_rate": 0.0, "tracking_xy_rmse": 0.15024389322264847, "tracking_yaw_rmse": 0.3458647352042577}`
- `moving-turn-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0.4, 0, 0.5], "end_step": 749, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 750}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 65.0, "events.randomize_push_robot": null, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane"}`。
- `moving-turn-right-a01` 指标：`{"max_joint_velocity_utilization": 1.8772773742675781, "max_tilt": 0.23912331461906433, "termination_rate": 0.0, "tracking_xy_rmse": 0.15296974454309936, "tracking_yaw_rmse": 0.34390748839007795}`
- `moving-turn-right-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0.4, 0, -0.5], "end_step": 749, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 750}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 65.0, "events.randomize_push_robot": null, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane"}`。

## 观察与限制

- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据


## 建议与待授权事项

- 按批次报告检查失败项和动作表现；复测使用新的 batch_id 与 attempt_id。
