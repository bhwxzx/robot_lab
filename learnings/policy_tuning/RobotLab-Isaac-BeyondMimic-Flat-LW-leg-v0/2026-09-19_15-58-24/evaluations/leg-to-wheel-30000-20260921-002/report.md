# 评估批次 leg-to-wheel-30000-20260921-002

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-leg-v0`；训练：`2026-09-19_15-58-24`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| nominal-start-30000-a01 | completed | nominal-start，2000 / 1 / 42 | complete | [结果](<raw/nominal-start-30000-a01/result.json>) / [日志](<raw/nominal-start-30000-a01/console.log>) |
| randomized-start-30000-a01 | completed | randomized-start，2000 / 1 / 42 | complete | [结果](<raw/randomized-start-30000-a01/result.json>) / [日志](<raw/randomized-start-30000-a01/console.log>) |
| randomized-phase-30000-a01 | completed | randomized-phase，2000 / 1 / 42 | complete | [结果](<raw/randomized-phase-30000-a01/result.json>) / [日志](<raw/randomized-phase-30000-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`6924df9721d7676efe6f424c89655313c15afc949b289085da8c03fbc8b4792b`；runner：`OnPolicyRunner`。
- [场景来源 scenario-1538944a88e8f135a1131471cccfff50a7b38c5ecfc5687470ef6df5d70351cd.json](<../../provenance/scenario-1538944a88e8f135a1131471cccfff50a7b38c5ecfc5687470ef6df5d70351cd.json>)
- [场景来源 scenario-7bfffc40678f528bde8ccbe50a83a77f6ea6788dcc9ce9bc085606e2347e8865.json](<../../provenance/scenario-7bfffc40678f528bde8ccbe50a83a77f6ea6788dcc9ce9bc085606e2347e8865.json>)
- [场景来源 scenario-7d77570c0b7037bc3737d755f24892322333b8933f35e058d6aba8dab0e37890.json](<../../provenance/scenario-7d77570c0b7037bc3737d755f24892322333b8933f35e058d6aba8dab0e37890.json>)
- [训练上下文 context-e76f4ecba9b16797a2d0f69bdf9d7cfa9977f89f4f5dca3a13f577cb7e6b9729.json](<../../provenance/context-e76f4ecba9b16797a2d0f69bdf9d7cfa9977f89f4f5dca3a13f577cb7e6b9729.json>)
- [训练有效配置 config-7d4e603b42fa6e9aa9d1ffc8a61b4311bbbc55bf67dec4de3e0563efb562e20c.json](<../../provenance/config-7d4e603b42fa6e9aa9d1ffc8a61b4311bbbc55bf67dec4de3e0563efb562e20c.json>)

- `nominal-start-30000-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `nominal-start-30000-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_mode": "nominal_start", "evaluation.motion_sha256": "85e53dc37459b295a4a3f06dd56044eb96b661caf01461ccceb1908cf52fa75d"}`。
- `randomized-start-30000-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `randomized-start-30000-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_mode": "randomized_start", "evaluation.motion_sha256": "85e53dc37459b295a4a3f06dd56044eb96b661caf01461ccceb1908cf52fa75d"}`。
- `randomized-phase-30000-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `randomized-phase-30000-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_mode": "randomized_phase", "evaluation.motion_sha256": "85e53dc37459b295a4a3f06dd56044eb96b661caf01461ccceb1908cf52fa75d"}`。

## 观察与限制

- 对比对象为同一训练 run 的 model_30000.pt 与 model_49999.pt；每个场景同为 Native、seed42、1环境、2000控制步、无视频。
- 空转口径：对应轮接触力向量范数恰为零的控制采样点，按右/左轮分别筛选后合并计算速度误差 RMS。它不证明控制步内所有物理子步均无接触。
- 转换窗口为参考帧46–91；末段固定为112–138（最后约20%）。误差为逐控制步、逐身体/关节等权统计，包含末尾被预算截断片段已观测到的采样，不将截断片段计作成功。
- 接触力最大值取每个控制步末物理子步的单轮接触力范数；转换P95取窗口内两轮较大值的95分位。它不是完整物理频率峰值，也不是冲量。
- nominal-start：30k→49999，已结束片段正常结束 14/14→14/14；无接触轮速误差 5.9698→8.0748 rad/s；单轮最大接触力 399.7→2812.0 N；末段身体位置 RMSE 1.002→2.983 cm；末段身体姿态 RMSE 0.0397→0.0985 rad。
- randomized-start：30k→49999，已结束片段正常结束 14/14→14/14；无接触轮速误差 6.1797→7.7274 rad/s；单轮最大接触力 3119.1→2916.5 N；末段身体位置 RMSE 2.188→3.016 cm；末段身体姿态 RMSE 0.1090→0.1273 rad。
- randomized-phase：30k→49999，已结束片段正常结束 29/29→29/29；无接触轮速误差 7.1833→8.7566 rad/s；单轮最大接触力 2986.1→2670.3 N；末段身体位置 RMSE 2.455→2.954 cm；末段身体姿态 RMSE 0.1068→0.1281 rad。
- 相同seed和相同初始物性已核对，但随机相位采样/失败重置会导致后续随机数及状态轨迹分岔；这不是逐步完全配对的反事实试验。单环境的startup质量/材质仍只有一个样本。
- 奖励与参数在两个checkpoint之间保持一致；checkpoint差异不能证明某项奖励调整有效。direct_parameter_change_supported=false；本次没有修改奖励或启动训练。
- 仅可进入受监督实物测试；未经实物验证，不代表 hardware-ready。
- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据

- [manifest.json](<../leg-to-wheel-native-20260921-001/manifest.json>)
- [comparison-metrics.json](<comparison-metrics.json>)

## 建议与待授权事项

- 先依据空转、接触力、末段误差的共同趋势决定下一项单变量奖励实验；不自动选择或导出checkpoint。
- 当前LW Leg的joint_vel_wheel_l2与joint_acc_wheel_l2均关闭，body跟踪列表不包含轮link。若两checkpoint均出现空转，可先考虑弱轮速参考跟踪项；新增权重没有因果验证，需要另行批准训练对照。
- 冲击和末段误差需联合查看逐帧曲线；不得为降低冲击而直接强制全程足/轮接触，先核对参考动作是否含预期腾空。
