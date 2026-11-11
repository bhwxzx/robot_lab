# 评估批次 wheel-to-leg-landing-std030-20261111-001

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-wheel-v0`；训练：`2026-11-10_14-03-57`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| fixed-start-nominal-a01 | completed | fixed-start-nominal，1000 / 1 / 42 | complete | [结果](<raw/fixed-start-nominal-a01/result.json>) / [日志](<raw/fixed-start-nominal-a01/console.log>) |
| fixed-start-randomized-a01 | completed | fixed-start-randomized，1000 / 1 / 42 | complete | [结果](<raw/fixed-start-randomized-a01/result.json>) / [日志](<raw/fixed-start-randomized-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`68ad2c214c1677ebc5d6c23587ccd11d4dacd40d757c129c0406c0c3a858c782`；runner：`OnPolicyRunner`。
- [场景来源 scenario-1948284ab8508eacbae636bc9757d5f35ebb0818218ce1a3a97d801eb1c14f96.json](<../../provenance/scenario-1948284ab8508eacbae636bc9757d5f35ebb0818218ce1a3a97d801eb1c14f96.json>)
- [场景来源 scenario-be8f31388683b3928c910b7314248686f9762551368df2a6637b5dddec8e46ce.json](<../../provenance/scenario-be8f31388683b3928c910b7314248686f9762551368df2a6637b5dddec8e46ce.json>)
- [训练上下文 context-99a56dadbb1c76fd28a7c05a5fabf89ae21e9da083c9efe13e3631f988c89a79.json](<../../provenance/context-99a56dadbb1c76fd28a7c05a5fabf89ae21e9da083c9efe13e3631f988c89a79.json>)
- [训练有效配置 config-24665cfbd02bd38fa2df134a5a565924037b4d6b268cf4550c99ffdaa548d040.json](<../../provenance/config-24665cfbd02bd38fa2df134a5a565924037b4d6b268cf4550c99ffdaa548d040.json>)

- `fixed-start-nominal-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `fixed-start-nominal-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "nominal_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03"}`。
- `fixed-start-randomized-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `fixed-start-randomized-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03"}`。

## 观察与限制

- 结论：本版在本次有限测试中完成动作，落地后轮速明显降低；但标称动作向前偏移增加、竖直起跳减弱、基座高度和位置跟踪变差，随机场仍有足关节速度尖峰及力矩限幅。若以向上跳跃和贴合参考落地为主，现有证据不支持认定本版全面优于已归档的11月8日本机策略。
- 来源：younghit，本机2026-11-10_14-03-57/model_49999.pt，SHA256 68ad2c214c1677ebc5d6c23587ccd11d4dacd40d757c129c0406c0c3a858c782；4096环境、seed42、PPO/OnPolicyRunner、50000迭代、actor输入590、critic输入146。用户已确认 bash scripts/start_beyondmimic.sh --type wheel，无额外CLI/Hydra参数；W&B本身未记录argv，已保存用户确认。
- 训练源码HEAD ffecc6a56ef912db64a51f6eed0579a0dbcc45bb，flat_env_cfg.py为已批准但未提交的修改。训练git快照中的patch与当前该文件的diff一致，其余相关文件与训练HEAD/已核验motion资源一致。评估HEAD为4734fbbf5ce07efc3e087e6cd461ed26428c3c33。训练来源与评估checkout分别记录，本批未以当前HEAD替代训练HEAD。
- 有效配置：独立高度奖励0，数据增强和镜像损失开启(coeff0.1)，全局位置weight0.5/std0.30，9个非轮连杆跟踪；轮速weight-0.5，参考相位1.65s开始、ramp0.10s、velocity_scale10；轮加速度weight-0.1，1.65s开始、ramp0.20s、acceleration_scale50。落地惩罚由参考时间触发，不是实际触地事件。关闭独立高度奖励不等于不约束Z，全局XYZ位置仍包含高度。
- 测试合同：标称初态/随机化初态各1环境、seed42、1000控制步、无视频，控制周期20ms、物理5ms，物理遥测窗口覆盖全部参考frame0..169。两场均5个完整成功周期、0失败、1个因步数预算截断的尾段；遥测complete，必需信号无缺失，原始bundle已重新验证。所有主动作统计使用5个完整周期；关节峰值另统计全部4000个物理子步以保留尾段风险。
- 基线为本机2026-11-08_10-58-49/model_49999.pt，SHA256 bd4d7eef0c02b1356598d1b5f3949f1f395d352a27b8e5c533fbb726bfd8c92e。两版测试条件、seed、预算、环境数、motion SHA一致；基线bundle重新验证通过。完整有效配置diff仅3项：log_dir、global位置std0.15→0.30、wheel_contact_continuous权重0.5→0；agent及其余环境配置无语义差异。由于实际有两项奖励变化，不能将表现仅归因于std。
- 遥测记录的初始质量、COM、材质、PhysX关节刚度/阻尼在两版对应场景中一致；软件PD、延迟等随机量没有独立完整证据，不将同seed等同于所有随机实现完全一致。
- 基座位置采用世界坐标原点，将post-physics状态对齐下一参考帧(+20ms)，末帧参考保持。标称3D RMSE(每周期计算再平均)4.742→8.128cm；首次足触地附近X误差-4.881→+8.653cm，终点-4.425→+10.353cm。正值表示超前参考。标称不再是落后，而是落地后持续约10cm前偏。
- 随机场3D RMSE 15.858→15.970cm，触地X绝对偏差均值15.531→15.688cm，未见整体位置改善。新策略逐周期触地X误差[-15.32,-7.56,+29.74,+14.88,+10.94]cm，终点[-16.66,-8.01,+30.13,+18.43,+10.18]cm；既有前偏也有后偏，不能将所有随机周期概括为前冲。
- 基座速度为基座COM线速度由body旋转至world，不是整机COM；控制步取样不是连续极值。起跳参考窗口1.10–1.70s，标称vz峰值1.7407→1.6433m/s、vx峰值1.9918→2.2084m/s；随机各周期vz峰值均值1.8271→1.7309m/s，新版范围1.4977–1.9437。标称同一物理相位1.30s，vx/vz=1.8278/1.6022m/s，对齐参考1.2220/1.8274m/s。向前速度增大而竖直速度降低支持标称动作更偏前跳的判断。
- 高度：标称基座峰值0.67823→0.65878m，随机峰值均值0.67910→0.66179m。对齐后全周期Z RMSE标称1.626→2.566cm、随机2.371→2.611cm；沿用未对齐command参考的落地窗口[1.46,1.90]s高度RMSE为标称4.690→6.077cm、随机3.350→4.904cm，两种时间口径分开列示，不能混用。
- 接触事件：双轮法向力模长均<10N持续>=15ms定义离地，5ms采样；参考离地1.300s、足触地1.600s来自原始motion audit。标称两版离地均1.325s(+25ms)。随机新离地[1.370,1.325,1.340,1.380,1.330]s，平均延迟49ms，旧版61ms；各阈值1/5/10/20N敏感性已保存。
- 主腾空段为最长的全部监测接触离地段：标称240→245ms；随机均值240→228ms，新版范围215–250ms。随机episode1在1.325s出现15ms短段，1.345s重新开始230ms主段；forward-jump-metrics中的first-gap=15ms不能解释为整个腾空时长。新标称首次足触地1.570s，比参考早30ms；随机首次触地[1.620,1.575,1.560,1.605,1.545]s。
- 落地后轮速：完整周期两轮物理子步合并，在参考1.85–3.40s窗口，速度RMS标称0.4267→0.1109rad/s(-74.0%)、随机0.4664→0.1983rad/s(-57.5%)；随机轮加速度RMS1.6865→1.4024rad/s²。这个改善仅对应落地后窗口，不意味着全程峰值下降；随机全程左轮峰速14.896→16.465rad/s，仍低于33rad/s配置限值。
- 关节：标称场全部关节速度均低于配置限值；左右足峰速7.152/7.155rad/s，实际峰值力矩12.656/12.641N·m。随机右足峰速20.033rad/s(配置限10，超限累计10ms)，实际力矩27.000N·m触顶5ms；未裁剪computed_torque峰38.022N·m。该速度峰发生episode1/1.580s，力矩峰在1.585s，与落地时段相邻。随机左足10.463rad/s超限5ms，力矩20.069N·m；左小腿16.877rad/s、105.779N·m(限20rad/s、120N·m)。其他关节未观察到速度超限或99%力矩限幅；详细10关节数据及时间在joint-loads.csv、forward-jump-metrics.json。
- 接触冲击：足部法向力模长峰值标称1261→1792N，随机1211→1406N；新版轮部峰值标称390N、随机937N。标称足关节速度未超限不代表落地更柔和；足部冲击反而增加。本批监测的大腿/小腿对地力未超过10N。接触力为5ms仿真法向力模长，不包含摩擦力，不是实物测量。
- 训练进度：checkpoint内部iter49999，TensorBoard55个标量全部到49999、值有限。末段47500–49999 anchor位置误差均值0.138774m、motion_finished均值0.995076、noise均值0.178788；此前两个窗口anchor位置误差0.141024/0.140063m。只能说明训练完成及后期统计，奖励定义不同不能跨版直接排序；无批准收敛标准，收敛为indeterminate/insufficient_evidence。
- 保全：1891个既有证据、模型、配置和相关源码文件hash一致；保留用户尚未提交的flat_env_cfg.py，SHA256 d0c957152a53cc1e2ef185fcc6dd6550083011aaa21ef0268c6aca2bd1727494不变。无代码修改、训练启停、导出、归档或Git操作；评估结束GPU无计算进程。只有本run新增证据。
- 局限：单seed、每场5完整周期，标称重复不是独立样本；无视频、多seed扰动扫描和实物验证。仅可进入受监督实物测试；未经实物验证，不代表 hardware-ready。
- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据

- [analysis-method.json](<derived/analysis-method.json>)
- [cases-ready.json](<derived/cases-ready.json>)
- [comparison-metrics.json](<derived/comparison-metrics.json>)
- [cycle-metrics.csv](<derived/cycle-metrics.csv>)
- [forward-analysis-method.json](<derived/forward-analysis-method.json>)
- [forward-jump-comparison.png](<derived/forward-jump-comparison.png>)
- [forward-jump-metrics.json](<derived/forward-jump-metrics.json>)
- [joint-loads.csv](<derived/joint-loads.csv>)
- [liftoff-and-impact-method-a02.json](<derived/liftoff-and-impact-method-a02.json>)
- [liftoff-and-impact-metrics.json](<derived/liftoff-and-impact-metrics.json>)
- [phase-comparison.png](<derived/phase-comparison.png>)
- [batch-driver-method-20261111-001.json](<../../evidence/training/batch-driver-method-20261111-001.json>)
- [effective-config-diff-20261111-001.json](<../../evidence/training/effective-config-diff-20261111-001.json>)
- [evaluation-request-20261111-001.json](<../../evidence/training/evaluation-request-20261111-001.json>)
- [protection-after-20261111-001.json](<../../evidence/training/protection-after-20261111-001.json>)
- [protection-before-20261111-001.json](<../../evidence/training/protection-before-20261111-001.json>)
- [tensorboard-windows-20261111-001.json](<../../evidence/training/tensorboard-windows-20261111-001.json>)
- [training-origin-20261111-001.json](<../../evidence/training/training-origin-20261111-001.json>)
- [source-training-20261111-001.patch](<../../evidence/source/source-training-20261111-001.patch>)
- [manifest.json](<../../../2026-11-08_10-58-49/evaluations/wheel-to-leg-position-20261109-001/manifest.json>)
- [report.md](<../../../2026-11-08_10-58-49/evaluations/wheel-to-leg-position-20261109-001/report.md>)
- [report-input.json](<derived/report-input.json>)

## 建议与待授权事项

- 建议保留11月8日版本作为位置跟踪对照，本版作为落地后轮速较低的候选；不要仅因5周期完成或轮速下降就替换当前基线。
- 下一轮有限验证应优先复查落地足关节冲击和前向位置误差，并增加匹配多seed测试。若要研究参数因果，应分别隔离位置std和轮持续接触奖励的影响；本次不直接修改权重或启动训练，direct_parameter_change_supported=false。
