# 评估批次 wheel-to-leg-fixed-start-20261026-001

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-wheel-v0`；训练：`2026-10-25_12-18-23`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| fixed-start-nominal-a01 | completed | fixed-start-nominal，1000 / 1 / 42 | complete | [结果](<raw/fixed-start-nominal-a01/result.json>) / [日志](<raw/fixed-start-nominal-a01/console.log>) |
| fixed-start-randomized-a01 | completed | fixed-start-randomized，1000 / 1 / 42 | complete | [结果](<raw/fixed-start-randomized-a01/result.json>) / [日志](<raw/fixed-start-randomized-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`390e2f15bf848295d2aee8ef7b316d5a0f7b7539b853cbba4290ba33ad897139`；runner：`OnPolicyRunner`。
- [场景来源 scenario-0d79bb6e8fc928a2c07aa5a8bb22a969f88263dc9362c20138bd9427b51a09f8.json](<../../provenance/scenario-0d79bb6e8fc928a2c07aa5a8bb22a969f88263dc9362c20138bd9427b51a09f8.json>)
- [场景来源 scenario-a1b11cbd592ccf9bb6eb105a8c0a16c156bd5bd30c493f99b61448cc1433db14.json](<../../provenance/scenario-a1b11cbd592ccf9bb6eb105a8c0a16c156bd5bd30c493f99b61448cc1433db14.json>)
- [训练上下文 context-e61a9b51242daa3d6fe90a75a3377fb35f20d42eb6fd646520786c28cf57b38d.json](<../../provenance/context-e61a9b51242daa3d6fe90a75a3377fb35f20d42eb6fd646520786c28cf57b38d.json>)
- [训练有效配置 config-eedda8943ee21dc322141b5e843536e01482bbe596e182a7add6476b08889902.json](<../../provenance/config-eedda8943ee21dc322141b5e843536e01482bbe596e182a7add6476b08889902.json>)

- `fixed-start-nominal-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `fixed-start-nominal-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "nominal_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03"}`。
- `fixed-start-randomized-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `fixed-start-randomized-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03"}`。

## 观察与限制

- fixed-start-nominal-a01：1000控制步、4000物理步；5个完整参考周期、0个失败终止、另有1个未完成尾段。所有重置均从frame0开始，遥测complete；终止项计数{"motion_finished": 5}。
- fixed-start-nominal-a01已到达起跳窗口的6段：竖直速度峰值1.293–1.293m/s；参考同窗口峰值1.967m/s。分段指标见analysis.json，提前终止后的窗口不是零误差。
- fixed-start-nominal-a01：接触视图交叉误差最大0N；全程驱动力矩最大限幅比0.8763、饱和物理样本0；关节速度最大名义限值比1.0545。
- fixed-start-randomized-a01：1000控制步、4000物理步；5个完整参考周期、0个失败终止、另有1个未完成尾段。所有重置均从frame0开始，遥测complete；终止项计数{"motion_finished": 5}。
- fixed-start-randomized-a01已到达起跳窗口的6段：竖直速度峰值1.224–1.643m/s；参考同窗口峰值1.967m/s。分段指标见analysis.json，提前终止后的窗口不是零误差。
- fixed-start-randomized-a01：接触视图交叉误差最大0N；全程驱动力矩最大限幅比1.0000、饱和物理样本1；关节速度最大名义限值比1.0042。
- 名义组nominal_start关闭噪声、重置扰动、材质/质量/COM/增益随机化并将执行延迟设为0。随机化组randomized_start仅固定参考起始相位为0，保留训练随机化和0..3步延迟；两组均为确定性策略推理。
- 测试使用训练保存配置对应源码：基座位置/连杆位置/线速度权重2/2/3、undesired_contacts=-0.1、anchor_pos高度阈值0.15m，以及大腿/小腿触地>1N失败终止。未关闭新触地终止项。随机起始相位不参与本批次。
- 5ms物理遥测包含4个足/轮连杆净法向力和4个大腿/小腿对地法向力。八连杆失载并非全机器人腾空的直接证明，base/hip未单独测量；净力不包含摩擦。applied_torque仅为驱动输出，不包含接触被动反力矩。
- 失败终止会截断后续落地过程：不得把未观察到的高度、冲击或接触按零处理，也不能直接以较低冲击证明训练改善。名义重复不是独立种子试验，单环境startup随机化也不覆盖全部物理参数组合。
- 所有模型、训练YAML、既有导出、USD、参考NPZ、相关源码及Isaac Sim持久配置字节与mtime均未变动。实际评估argv含进程内--/app/settings/persistent=false。未启动训练、导出、部署、安装、删除或Git提交。
- 没有用户批准的收敛判据，收敛结论为indeterminate。本批次只支持有限仿真观察；仅可进入受监督实物测试；未经实物验证，不代表 hardware-ready。
- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据

- [analysis.json](<derived/analysis.json>)
- [joint-peak-rms.csv](<derived/joint-peak-rms.csv>)
- [fixed-start-motion-contact.png](<derived/fixed-start-motion-contact.png>)
- [protection-before-20261026-001.json](<../../evidence/training/protection-before-20261026-001.json>)
- [protection-after-20261026-001.json](<../../evidence/training/protection-after-20261026-001.json>)
- [training-origin-20261026-001.json](<../../evidence/training/training-origin-20261026-001.json>)

## 建议与待授权事项

- 按分段数据审查真实起跳、准备轮子支撑、姿态与落地触地终止；保持现有配置和证据，不自动调整训练参数。
