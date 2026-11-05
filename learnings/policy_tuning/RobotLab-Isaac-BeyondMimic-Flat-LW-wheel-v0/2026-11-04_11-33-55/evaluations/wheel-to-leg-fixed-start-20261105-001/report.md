# 评估批次 wheel-to-leg-fixed-start-20261105-001

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-wheel-v0`；训练：`2026-11-04_11-33-55`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| fixed-start-nominal-a01 | completed | fixed-start-nominal，1000 / 1 / 42 | complete | [结果](<raw/fixed-start-nominal-a01/result.json>) / [日志](<raw/fixed-start-nominal-a01/console.log>) |
| fixed-start-randomized-a01 | completed | fixed-start-randomized，1000 / 1 / 42 | complete | [结果](<raw/fixed-start-randomized-a01/result.json>) / [日志](<raw/fixed-start-randomized-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`facc6f107806efbf1ca9e488b9498108fd8a0f80e8e4f289b917d3983f9f2373`；runner：`OnPolicyRunner`。
- [场景来源 scenario-20960dc9f23b91f7c9d1774a98ae7e7a8a9bb27936580410e348fe5278a78667.json](<../../provenance/scenario-20960dc9f23b91f7c9d1774a98ae7e7a8a9bb27936580410e348fe5278a78667.json>)
- [场景来源 scenario-ba97223353c9e978395dc76f9f2ee676ecbeba041992f59d9dfb604aae20b31b.json](<../../provenance/scenario-ba97223353c9e978395dc76f9f2ee676ecbeba041992f59d9dfb604aae20b31b.json>)
- [训练上下文 context-e4d02f485b16ecb51ad9b6d9359d58540acec0d7801acf1ccc6c0d088822e8dc.json](<../../provenance/context-e4d02f485b16ecb51ad9b6d9359d58540acec0d7801acf1ccc6c0d088822e8dc.json>)
- [训练有效配置 config-1f5227e89a34dcbc5f56911f10e043fbf5df7b966aeb8c90fd7b4aa2adb8a893.json](<../../provenance/config-1f5227e89a34dcbc5f56911f10e043fbf5df7b966aeb8c90fd7b4aa2adb8a893.json>)

- `fixed-start-nominal-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `fixed-start-nominal-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "nominal_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03", "rewards.motion_anchor_roll_horizontal.weight": 0.0}`。
- `fixed-start-randomized-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `fixed-start-randomized-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03", "rewards.motion_anchor_roll_horizontal.weight": 0.0}`。

## 观察与限制

- 训练来源由用户于 2026-11-05 确认：bash scripts/start_beyondmimic.sh --type wheel，无额外 CLI/Hydra 覆盖。8192 环境，50000 轮，PPO/OnPolicyRunner，seed 42。
- 训练 HEAD 7f17bed；保存的唯一源码差异为 motion_anchor_roll_horizontal.weight 0.5→0；已在内存重建并校验 Git blob。评估通过 scenario override 恢复 0，未修改工作区源码。
- 本次为两组 Native mean-action 固定 frame 0 测试，各 1000 控制步、1 环境、seed 42、无视频。nominal 去掉随机化；randomized 保留训练随机化。非随机动作分布测试。
- 训练有效值：anchor z-only 终止阈值 0.25 m；undesired_contacts=-0.1；大小腿触地 >1 N 终止启用；joint_symmetry=-0.5, scales=(0.1,0.2,0.2,0.2) rad；双轮阶段 leg_symmetry=0.5、轮距惩罚=-20、轮距范围0.496–0.536 m。
- 末尾1000轮：motion_finished均值0.9958217；采样归一化熵0.8528834、最高概率0.4083198。无完整实际采样直方图，不能称均匀采样或据此证明收敛。
- 髋突变分析覆盖1.0–1.8s；同时输出相对参考修正的镜像残差。phase 为动作前参考帧/50Hz，状态为动作后采样，极值时刻沿用该标签。
- 单 seed、每组单环境；nominal 周期重复不是独立成功率试验，startup 随机化只采到一组物性。无批准的收敛/硬件验收标准；无视频不作视觉判断。
- 对照为本机最近已有评估的11月2日策略，不是11月3日训练；多项奖励和终止设置不同，只作描述性比较，不推断单个权重的因果效果。
- 仅可进入受监督实物测试；未经实物验证，不代表 hardware-ready。
- fixed-start-nominal-a01: 完整动作=5, 失败=0, 截尾=1；完整遥测=1000控制+4000物理样本。髋镜像残差(1–1.8s)峰值=3.936°，右髋单20ms变化=1.399°；俯仰RMSE=1.696°，最大|俯仰|=10.216°；最大anchor z误差=0.03325m，>0.15m样本=0/1000；大小腿地面接触峰值=0.000N，触地episode数=0；起跳窗口峰值向上速度=1.147m/s，最高基座=0.7083m，参考=0.7135m。
- fixed-start-nominal-a01: 双轮阶段轮心水平距离 0.5157–0.5173m，超0.496–0.536范围的物理样本占比=0.00%。此范围是训练软惩罚边界，不是硬件验收阈值。
- fixed-start-randomized-a01: 完整动作=5, 失败=0, 截尾=1；完整遥测=1000控制+4000物理样本。髋镜像残差(1–1.8s)峰值=13.480°，右髋单20ms变化=2.580°；俯仰RMSE=4.563°，最大|俯仰|=18.122°；最大anchor z误差=0.09280m，>0.15m样本=0/1000；大小腿地面接触峰值=0.000N，触地episode数=0；起跳窗口峰值向上速度=1.431m/s，最高基座=0.7276m，参考=0.7135m。
- fixed-start-randomized-a01: 双轮阶段轮心水平距离 0.4613–0.5192m，超0.496–0.536范围的物理样本占比=54.23%。此范围是训练软惩罚边界，不是硬件验收阈值。
- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据

- [analysis.json](<derived/analysis.json>)
- [phase-diagnostics.png](<derived/phase-diagnostics.png>)
- [joint-loads.csv](<derived/joint-loads.csv>)
- [training-origin-20261105-001.json](<../../evidence/training/training-origin-20261105-001.json>)
- [training-windows-20261105-001.json](<../../evidence/training/training-windows-20261105-001.json>)
- [training-source-20261105-001.json](<../../evidence/source/training-source-20261105-001.json>)
- [training-source-20261105-001.patch](<../../evidence/source/training-source-20261105-001.patch>)
- [manifest.json](<../../../2026-11-02_14-06-00/evaluations/wheel-to-leg-fixed-start-20261103-001/manifest.json>)

## 建议与待授权事项

- 根据分阶段姿态、接触和关节负载审阅动作质量；进一步优化先提出具体方案等待批准。此次未修改源码、启动训练、导出部署或提交推送。
