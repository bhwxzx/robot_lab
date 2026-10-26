# 评估批次 wheel-to-leg-fixed-start-20261026-002

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-wheel-v0`；训练：`2026-10-25_11-55-09`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| fixed-start-nominal-a01 | completed | fixed-start-nominal，1000 / 1 / 42 | complete | [结果](<raw/fixed-start-nominal-a01/result.json>) / [日志](<raw/fixed-start-nominal-a01/console.log>) |
| fixed-start-randomized-a01 | completed | fixed-start-randomized，1000 / 1 / 42 | complete | [结果](<raw/fixed-start-randomized-a01/result.json>) / [日志](<raw/fixed-start-randomized-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`8512cbffcace36bc1512682d6e9c1b5cd9eaf158b4dd394f924733bb85ad607c`；runner：`OnPolicyRunner`。
- [场景来源 scenario-4cf9cb83e2d4989d19689a4472d140584eae081006e6feee7cef6dd9b2f9311f.json](<../../provenance/scenario-4cf9cb83e2d4989d19689a4472d140584eae081006e6feee7cef6dd9b2f9311f.json>)
- [场景来源 scenario-ab95131e11df1925a6fecf4683c8b5ea2cafa1f18871d7aada132da641f5b2dc.json](<../../provenance/scenario-ab95131e11df1925a6fecf4683c8b5ea2cafa1f18871d7aada132da641f5b2dc.json>)
- [训练上下文 context-e3ab53e9cb1fc4f9bd11b805863ce793569977ec5c6978110de661388bc36b42.json](<../../provenance/context-e3ab53e9cb1fc4f9bd11b805863ce793569977ec5c6978110de661388bc36b42.json>)
- [训练有效配置 config-5c021199b9f352c34bfa9f640935d99d73d1fc1d84da691cd519c3ffe7c19fcf.json](<../../provenance/config-5c021199b9f352c34bfa9f640935d99d73d1fc1d84da691cd519c3ffe7c19fcf.json>)

- `fixed-start-nominal-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `fixed-start-nominal-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "nominal_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03"}`。
- `fixed-start-randomized-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `fixed-start-randomized-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03"}`。

## 观察与限制

- 问题：同一模型2026-10-25_11-55-09/model_49999.pt、同一10/23参考轨迹，两组均固定从frame0开始，比较关闭随机化与保留全部训练随机化。模型SHA-256 8512cbffcace36bc1512682d6e9c1b5cd9eaf158b4dd394f924733bb85ad607c。
- 两组各1000控制步、4000物理步、单环境、seed42、Native、无视频、300s上限。均完成5个从frame0到169的完整周期，另有1个未完成尾段；无非motion_finished终止。遥测complete，无必需信号缺失，接触视图交叉校验0N，reset历史一致性断言通过。
- 无随机化fixed-start-nominal-a01采用nominal_start：关闭观测噪声、材质、质量、COM、增益及关节默认位姿随机化；重置pose/velocity/joint perturbation归零；腿/轮/足执行延迟均置0。策略推理确定性。原训练push_robot已关闭，两组均没有外部推扰。
- 有随机化fixed-start-randomized-a01采用randomized_start：仅将自适应起始相位采样固定为0；保留训练的观测噪声、重置pose/velocity/joint扰动、材质/质量/COM/增益和0..3步执行延迟随机化。不是随机起始相位，也不是上一轮randomized_phase。
- 模型奖励仍为基座位置/连杆位置/连杆线速度2/2/3、undesired_contacts=-1、action_rate=-0.2、smoothness=-0.075；anchor_pos高度阈值0.15m、左右foot_link的ee_body_pos高度阈值0.25m。
- 无随机化5周期关键指标完全一致：起跳1.10–1.36s峰速1.334m/s；1.20–1.70s高度RMSE3.666cm；八连杆连续失载245ms；准备双轮同时≥1N比例99.22%；落地1.60–2.00s最低基座高度0.443m。参考峰速1.967m/s、计划飞行300ms、落地最低高度0.473m。
- 无随机化组5个完整周期及尾段均未记录大腿/小腿对地接触；记录Fz合力峰值0.957kN。全程驱动力矩未达到配置限幅、关节速度未超过名义上限。5次指标相同的名义重复不是5个独立随机种子试验。
- 有随机化5周期起跳峰速1.161–1.439m/s、均值1.321；高度RMSE2.52–4.53cm、均值3.588cm；八连杆连续失载250–265ms、均值258ms。平均跟踪指标与名义组接近，不能代替接触异常检查。
- 有随机化组5个完整周期中2个大腿触地（episode0、3），另有未完成尾段episode5触地；首次触地参考时间分别1.660、1.685、1.670s。2/5是本次观察计数，不是总体失败概率。
- 有随机化全程单大腿/小腿对地峰值及记录Fz合力峰值7.176kN，left_thigh_link，参考1.660s。四个大腿/小腿对地接触时长完整周期均值17ms，最大45ms；正法向冲量均值16.45 N·s。未覆盖base/hip接触。
- 有随机化落地最低基座高度均值0.419m，最差0.378m，低于参考0.473m；准备俯仰峰值均值11.74°、最多16.91°，髋镜像残差峰值均值14.43°、最多17.42°。名义组相应准备均值6.55°和9.70°。飞行阶段姿态指标详见analysis.json。
- 有随机化全程左右小腿驱动力矩峰值118.15/106.69Nm，未达到120Nm限幅；左右足关节峰速10.263/10.009rad/s，分别2/1个样本超过10rad/s名义上限。接触被动反力矩不在applied_torque中。
- 有随机化一阶动作差分平方和周期均值0.860，名义组0.467；二阶分别0.369/0.151。指标按片段内部计算且排除reset跨界，仍包含初始化后的恢复过程。
- 有随机化ee_body_pos最大Z误差12.66cm，left_foot_link，参考1.60s；名义组最大6.54cm，right_foot_link，1.30s。均低于25cm阈值；两组没有触发ee_body_pos或anchor_pos终止。motion_finished不能等同正确落地。
- 两组记录初始总质量分别26.731/27.886kg；这是运行时记录的差异之一。单环境的startup随机化只覆盖一组启动参数，不能据此定位究竟是质量、材质、COM、增益、执行延迟、噪声还是重置扰动导致异常。
- 固定相位后保留随机化仍能复现大腿触地，故异常不能完全归因于中途相位初始化；本批次只比较随机化组合，未隔离各随机化分量的因果作用。
- 与上一轮play-ground不同：上一轮关闭主要噪声/事件但保留0..3步延迟，本批次无随机化组将执行延迟设为0；不要把两个名义定义混用。此批次未重跑或覆盖旧证据。
- 接触力为5ms物理步平均法向力；八连杆失载不等同全身腾空证明。单seed、单环境、无视频、有限样本不能证明普遍成功率或实物效果；缺少批准的收敛判据，收敛结论indeterminate。
- 全部模型、保存YAML、既有导出、参考轨迹、源码与Isaac Sim持久设置字节和mtime保持不变；CLI显式 --/app/settings/persistent=false。仅新增此批次与保护核验记录，未修改代码、训练、重新导出、部署、安装删除或提交推送。
- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据

- [jump-contact-fixed-start-nominal-a01.png](<derived/jump-contact-fixed-start-nominal-a01.png>)
- [jump-contact-fixed-start-randomized-a01.png](<derived/jump-contact-fixed-start-randomized-a01.png>)
- [analysis.json](<derived/analysis.json>)
- [joint-peak-rms.csv](<derived/joint-peak-rms.csv>)
- [protection-after-20261026-002.json](<../../evidence/training/protection-after-20261026-002.json>)
- [training-origin-20261026-001.json](<../../evidence/training/training-origin-20261026-001.json>)
- [training-tail-20261026-001.json](<../../evidence/training/training-tail-20261026-001.json>)
- [manifest.json](<../wheel-to-leg-local-20261026-001/manifest.json>)

## 建议与待授权事项

- 固定相位下随机化组合已足以复现异常，后续优先分组排查重置姿态/速度扰动、观测噪声、执行延迟和物理参数；每组使用新的有界测试批次。
- 先保留本轮配置与证据，不自动缩小训练随机化范围或增加奖励。额外seed用于扩大覆盖，不能从当前2/5计数推断总体失败率。
