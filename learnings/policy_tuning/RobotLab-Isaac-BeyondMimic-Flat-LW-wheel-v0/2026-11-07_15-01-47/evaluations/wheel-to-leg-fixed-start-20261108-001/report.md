# 评估批次 wheel-to-leg-fixed-start-20261108-001

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-wheel-v0`；训练：`2026-11-07_15-01-47`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| fixed-start-nominal-a01 | completed | fixed-start-nominal，1000 / 1 / 42 | complete | [结果](<raw/fixed-start-nominal-a01/result.json>) / [日志](<raw/fixed-start-nominal-a01/console.log>) |
| fixed-start-randomized-a01 | completed | fixed-start-randomized，1000 / 1 / 42 | complete | [结果](<raw/fixed-start-randomized-a01/result.json>) / [日志](<raw/fixed-start-randomized-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`920426b2d8eecdf1dcd763b7c2dedd3f7cb978946b558261a1886fbd8c6623f8`；runner：`OnPolicyRunner`。
- [场景来源 scenario-4804be402ab9d368da58c58371ee29c62aa4ec1bad8859ae9d2b71efc34f724a.json](<../../provenance/scenario-4804be402ab9d368da58c58371ee29c62aa4ec1bad8859ae9d2b71efc34f724a.json>)
- [场景来源 scenario-cc3f0121bd995d27c6f0baece4474df2f965e87cb50eb97bc2a564137c0f0007.json](<../../provenance/scenario-cc3f0121bd995d27c6f0baece4474df2f965e87cb50eb97bc2a564137c0f0007.json>)
- [训练上下文 context-98f3a0dbed37669f4ac02c675931ad3d52278ef32700819157a1c5a6fdb91920.json](<../../provenance/context-98f3a0dbed37669f4ac02c675931ad3d52278ef32700819157a1c5a6fdb91920.json>)
- [训练有效配置 config-a7701e8c299ea6a6def7e23a7e0a587c091febc86f9bb58aaa817e5fd7cea25b.json](<../../provenance/config-a7701e8c299ea6a6def7e23a7e0a587c091febc86f9bb58aaa817e5fd7cea25b.json>)

- `fixed-start-nominal-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `fixed-start-nominal-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "nominal_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03", "rewards.hip_pos_zero_l2.weight": -2.0}`。
- `fixed-start-randomized-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `fixed-start-randomized-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03", "rewards.hip_pos_zero_l2.weight": -2.0}`。

## 观察与限制

- 结论：当前全程轮速约束有效，尤其降低了标称和落地后的轮速；但随机化推蹬峰值、小腿过伸和落地高度误差仍需优化。这不是整体运动质量全面改善的版本。
- 评估对象：2026-11-07_15-01-47/model_49999.pt，checkpoint SHA256 920426b2d8eecdf1dcd763b7c2dedd3f7cb978946b558261a1886fbd8c6623f8；训练和评估 HEAD e4317a70b221940a0a2a9f4e74547204dd950fe7。用户确认启动命令 bash scripts/start_beyondmimic.sh --type wheel --num_envs 8192，无额外 Hydra 覆盖。PPO/OnPolicyRunner，8192 环境，seed42，24步 rollout，完成50000轮；55项训练标量末段均有限。
- 保存配置：joint_vel_wheel_l2 weight=-0.1、velocity_scale=10 rad/s；joint_acc_wheel_l2 weight=-0.002、acceleration_scale=50 rad/s²；两项 start=0、ramp=0；motion_flight_height weight=0。腿/足/轮执行器延迟均0–3步。与Nov5无高度版相比，agent完全一致，env主要增加全程轮速/轮加速度惩罚（另有零权重高度项及日志路径）；与Nov6高度版相比还关闭了原weight=2的高度奖励，不能把所有变化归因于轮速约束。
- 测试范围：fixed-start-nominal 与 fixed-start-randomized，各1环境、seed42、1000控制步、4000个5ms物理样本，无视频；两组均5个完整成功、0失败、1截尾周期，遥测完整，bundle验证通过。历史Nov5/Nov6只读取既有证据，没有重跑。标称重复周期是确定性重复，单seed随机化也不能代表普遍成功率。
- 标称对照（峰值/RMS包含截尾周期已观察样本，高度范围仅完整周期；落地窗口按命令时间1.45–1.90s）：

| 指标 | Nov5无高度、无轮惩罚 | Nov6高度weight2、无轮惩罚 | 当前无高度、全程轮惩罚 |
|---|---:|---:|---:|
| 轮速峰值 rad/s | 11.205 | 10.953 | 5.977 |
| 轮速RMS rad/s | 4.125 | 3.478 | 1.785 |
| 轮加速度峰值 rad/s² | 901.3 | 806.9 | 847.8 |
| 轮加速度RMS rad/s² | 45.99 | 34.73 | 36.53 |
| 小腿速度峰值 rad/s | 15.097 | 16.172 | 14.984 |
| 全部关节驱动力矩峰值 N·m | 82.502 | 79.073 | 79.152 |
| 完整周期最高基座Z范围 m | 0.6735–0.6735 | 0.7044–0.7044 | 0.6816–0.6816 |
| 落地高度RMSE cm | 6.02 | 2.87 | 6.46 |
| 完整周期最低落地基座Z m | 0.4439 | 0.4738 | 0.4235 |
| 全程横滚峰值 deg | 0.591 | 3.423 | 1.385 |
| 落地后轮速峰值 rad/s | 5.952 | 7.035 | 2.319 |
- 随机化对照（峰值/RMS包含截尾周期已观察样本，高度范围仅完整周期；落地窗口按命令时间1.45–1.90s）：

| 指标 | Nov5无高度、无轮惩罚 | Nov6高度weight2、无轮惩罚 | 当前无高度、全程轮惩罚 |
|---|---:|---:|---:|
| 轮速峰值 rad/s | 15.177 | 16.279 | 13.728 |
| 轮速RMS rad/s | 5.208 | 5.042 | 3.433 |
| 轮加速度峰值 rad/s² | 2388.2 | 1513.1 | 1292.8 |
| 轮加速度RMS rad/s² | 77.99 | 77.33 | 79.62 |
| 小腿速度峰值 rad/s | 16.710 | 17.495 | 17.622 |
| 全部关节驱动力矩峰值 N·m | 93.294 | 99.598 | 106.983 |
| 完整周期最高基座Z范围 m | 0.6524–0.6878 | 0.6950–0.7169 | 0.6589–0.6975 |
| 落地高度RMSE cm | 4.42 | 2.09 | 5.56 |
| 完整周期最低落地基座Z m | 0.4239 | 0.4627 | 0.4047 |
| 全程横滚峰值 deg | 6.912 | 6.588 | 6.086 |
| 落地后轮速峰值 rad/s | 9.767 | 10.381 | 3.369 |
- 当前关节负载（全部5ms采样的绝对峰值；上方自动通用结果中的力矩指标仅控制步末采样，物理峰值以本表为准）：

| 当前关节（左/右） | 标称速度 rad/s | 随机化速度 rad/s | 标称力矩 N·m | 随机化力矩 N·m |
|---|---:|---:|---:|---:|
| 髋 | 1.358 / 1.765 | 2.719 / 3.056 | 24.228 / 26.023 | 38.900 / 34.216 |
| 大腿 | 10.195 / 10.224 | 14.606 / 12.685 | 33.031 / 24.335 | 41.614 / 40.656 |
| 小腿 | 14.362 / 14.984 | 17.622 / 16.206 | 73.117 / 79.152 | 106.983 / 76.053 |
| 足 | 4.410 / 4.542 | 10.054 / 10.033 | 16.508 / 14.881 | 23.028 / 21.890 |
| 轮 | 5.977 / 5.702 | 13.728 / 11.962 | 2.772 / 3.935 | 6.994 / 5.638 |
- 轮速阶段：标称左轮5.977 rad/s在1.130s、右轮5.702在1.200s，均处推蹬阶段；离地1.325s、足落地1.570s。随机化左轮13.728在第3周期1.300s，早于该周期离地1.330s；右轮11.962在第1周期0.290s，属于准备阶段。随机化起跳峰值仍突出。落地后轮峰值已降到标称2.319、随机化3.369 rad/s。
- 小腿过伸未解决：标称左小腿最小屈曲角0.140rad，参考同期最低0.694rad；原无高度版为0.182rad、高度版为0.097rad。当前标称小腿速度峰值14.984rad/s，与Nov5的15.097接近；随机化17.622高于Nov5的16.710，左小腿力矩峰值106.983N·m。参考小腿速度峰约7.22rad/s。不能因轮速下降就判断蹬伸动作已经接近参考。
- 足关节：随机化左/右速度峰值10.054/10.033rad/s，各有1个5ms样本超过配置10rad/s，发生在足落地后；标称无超速。全场均无驱动力矩触及配置限幅。轮关节未超过33rad/s配置值，但跟踪目标远小于这一上限。腿部触地传感器峰值0N；最大单足竖直法向力标称1226.6N、随机化1400.9N，为5ms平均仿真接触力，不是实物瞬时冲击或关节结构负载。
- 起跳与落地：当前标称最高基座0.6816m，较Nov5的0.6735m略高，低于高度版0.7044m及参考0.7135m。落地高度RMSE由Nov5的6.02cm变为6.46cm（高度版2.87cm）；最低基座0.4235m，低于Nov5的0.4439m和高度版0.4738m。随机化完整周期最高0.6589–0.6975m，最低落地基座最低0.4047m，落地RMSE5.56cm，说明落地支撑/姿态恢复有退步。
- 推蹬关联：标称轮轴中点世界X后退量（1.10→1.30s）Nov5 6.16cm→当前2.61cm，高度版4.03cm；随机化当前五周期均值8.86cm，Nov5为7.37cm、高度版9.47cm，并未一致改善。标称估算载荷加权滑移均值Nov5 0.03657→当前0.01813m/s；随机化仍可出现约0.696m/s的20ms采样滑移峰值。该滑移为理想轮半径0.1095m的接触切点代理，不是实际接触斑或摩擦能量损失。
- 有效推蹬：标称1.10s到离地总竖直支撑冲量/净冲量/全身质心离地速度，Nov5为94.80N·s / 34.49N·s / 0.966m/s，当前94.11 / 35.11 / 0.950，高度版98.02 / 39.02 / 1.093。当前随机化全身质心离地速度0.963–1.326m/s。轮轴后退减少没有带来相应的质心起跳速度提高，小腿过伸仍存在，因此不支持“后退减少即可解决过伸”的充分因果解释。全身质心速度由刚体运动学与质量重建，轮轴速度交叉误差<6.1e-6m/s，冲量–速度闭合误差<2.5e-6m/s。基座COM向上速度峰值1.464m/s是另一指标，不能当成全身质心离地速度。
- 轮加速度约束只部分有效：相对Nov5，标称RMS45.99→36.53rad/s²；随机化77.99→79.62，未降低。标称5ms峰值847.76rad/s²发生在落地1.570s，而控制步末子步最大仅330.53；当前奖励每个控制步读取一次joint_acc，未汇总4个物理子步，部分冲击峰不会直接进入该项的同一步采样。不能仅根据训练奖励均值判定物理加速度峰值已受控。
- 比较边界：三个版本同场景/seed/环境数/时长/参考动作，初始质量、COM、材质及PhysX增益逐项相同；没有独立记录所有软件PD增益/延迟抽样，因此不宣称随机条件完全同轨。每版独立训练、单种子，不构成严格因果消融。没有用户批准的收敛判据，因此收敛判定为indeterminate；没有实物验证。
- 复现证据：analysis.json与push-off-dynamics.json内嵌完整离线源码及输入哈希，sampling-audit.json记录子步采样核验，CSV保存关节/周期明细，phase-comparison.png已检查。训练有效配置全量差异及55项标量窗口在training证据中。110个保护文件核验未变，HEAD和两个stash保持；新增内容仅本次评估证据。
- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据

- [manifest.json](<../../../2026-11-05_16-40-14/evaluations/wheel-to-leg-fixed-start-20261106-001/manifest.json>)
- [manifest.json](<../../../2026-11-06_11-39-51/evaluations/wheel-to-leg-fixed-start-20261107-001/manifest.json>)
- [analysis.json](<derived/analysis.json>)
- [cycle-metrics.csv](<derived/cycle-metrics.csv>)
- [joint-loads.csv](<derived/joint-loads.csv>)
- [phase-comparison.png](<derived/phase-comparison.png>)
- [push-off-dynamics.json](<derived/push-off-dynamics.json>)
- [sampling-audit.json](<derived/sampling-audit.json>)
- [effective-config-diff-20261108-001.json](<../../evidence/training/effective-config-diff-20261108-001.json>)
- [evaluation-request-20261108-001.json](<../../evidence/training/evaluation-request-20261108-001.json>)
- [protection-after-20261108-001.json](<../../evidence/training/protection-after-20261108-001.json>)
- [protection-before-20261108-001.json](<../../evidence/training/protection-before-20261108-001.json>)
- [tensorboard-summary-20261108-001.json](<../../evidence/training/tensorboard-summary-20261108-001.json>)
- [training-origin-20261108-001.json](<../../evidence/training/training-origin-20261108-001.json>)
- [report-input.json](<derived/report-input.json>)

## 建议与待授权事项

- 建议保留当前全程轮速weight=-0.1、轮加速度weight=-0.002作为下一次对照基线，暂不同时加重两项；随机化小腿峰速/力矩与落地最低高度作为反向指标。
- 若继续优化，优先提出单变量高度恢复对照：motion_flight_height weight 0→1，保持std=0.1、窗口1.1–1.9s、ramp0.05及两项轮约束不变。预期改善跳跃与落地高度，风险是小腿速度/力矩和侧倾回升；历史weight2版本仅提供方向性支持，weight1尚未验证。本轮未修改参数或启动训练。
- 另一独立方向是针对推蹬阶段小腿位置/速度跟踪及轮地滑移诊断，不能用继续增大全程轮惩罚替代；物理子步加速度聚合也需单独设计与批准，不与高度项一起改。direct_parameter_change_supported=false。
- 需要更强鲁棒性结论时，再单独定义有限多seed/随机参数测试预算；本次不自动扩展评估，不导出或部署。
