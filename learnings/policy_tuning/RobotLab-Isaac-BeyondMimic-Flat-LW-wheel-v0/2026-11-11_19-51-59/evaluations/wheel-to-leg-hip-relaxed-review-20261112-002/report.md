# 评估批次 wheel-to-leg-hip-relaxed-review-20261112-002

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-wheel-v0`；训练：`2026-11-11_19-51-59`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| fixed-start-nominal-a01 | completed | fixed-start-nominal，1000 / 1 / 42 | complete | [结果](<../wheel-to-leg-hip-relaxed-20261112-001/raw/fixed-start-nominal-a01/result.json>) / [日志](<../wheel-to-leg-hip-relaxed-20261112-001/raw/fixed-start-nominal-a01/console.log>) |
| fixed-start-randomized-a01 | completed | fixed-start-randomized，1000 / 1 / 42 | complete | [结果](<../wheel-to-leg-hip-relaxed-20261112-001/raw/fixed-start-randomized-a01/result.json>) / [日志](<../wheel-to-leg-hip-relaxed-20261112-001/raw/fixed-start-randomized-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`674287619ce107af262bdc84c7a1b870ec2815fb8f92757da92b2bef51c1c146`；runner：`OnPolicyRunner`。
- [场景来源 scenario-e671c49911faf17d2a950e26d0714d748a83e0c1578dc83e3a622d23cffc6c9d.json](<../../provenance/scenario-e671c49911faf17d2a950e26d0714d748a83e0c1578dc83e3a622d23cffc6c9d.json>)
- [场景来源 scenario-ff26085a318d284a935f51fd891e3a6e69f5b1aa7ea99a7f045bae54405ec19d.json](<../../provenance/scenario-ff26085a318d284a935f51fd891e3a6e69f5b1aa7ea99a7f045bae54405ec19d.json>)
- [训练上下文 context-1d89ec39cdebf1a338331864a4be61d9db15725f3ee3196f9a19015e88941999.json](<../../provenance/context-1d89ec39cdebf1a338331864a4be61d9db15725f3ee3196f9a19015e88941999.json>)
- [训练有效配置 config-bad80bee30b8f43d909493c28b2f55dd750d678b96acd204c31c392acfe460c0.json](<../../provenance/config-bad80bee30b8f43d909493c28b2f55dd750d678b96acd204c31c392acfe460c0.json>)

- `fixed-start-nominal-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `fixed-start-nominal-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "nominal_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03"}`。
- `fixed-start-randomized-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `fixed-start-randomized-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03"}`。

## 观察与限制

- 结论：本版标称动作的前冲、位置误差和足部接触峰值改善；随机化条件的位置跟踪、双髋姿态和起跳前轮距没有同步改善，不能判定全面优于11月10日版本。
- 本机 younghit，2026-11-11_19-51-59/model_49999.pt，SHA256 674287619ce107af262bdc84c7a1b870ec2815fb8f92757da92b2bef51c1c146。训练及评估HEAD均为bca81448975fba069befc8cf6f33bb0a497e0307；训练保存diff与当前未提交flat_env_cfg.py完全一致。用户确认 bash scripts/start_beyondmimic.sh --type wheel，无额外CLI/Hydra；W&B未记录argv，此部分来源为用户确认。
- 训练4096环境、seed42、PPO/OnPolicyRunner，actor输入590、critic146，10帧逐项历史。model内部iter49999，55个TensorBoard标量到49999且全部有限。末段47500–49999平均motion_finished为0.995044、anchor位置误差0.138944m、noise0.180846；与旧版0.995076/0.138774m相近。训练完成不等于收敛，未提供批准的收敛标准，结论indeterminate。
- 与2026-11-10_14-03-57有效配置完整比较：除log_dir外只有feet_distance_penalize权重-20→-0、hip_pos_zero_l2权重-2→-1；agent配置无差异。其余保持：独立高度奖励0、增强+镜像损失开启、全局XYZ位置std0.30/weight0.5、无轮连杆跟踪、参考相位1.65s后轮速/轮加速度惩罚。不能将结果单独归因于某一个权重变化。
- 沿用旧版合同：标称初态与随机化初态各1环境、1000控制步、seed42、无视频。两场各5完整成功周期、0失败、1预算截断尾段；遥测complete，required缺失项为空，每场1000控制记录和4000物理记录，20ms控制、5ms物理。主动作统计用5完整周期，关节极值用全部记录以保留截断尾段风险。
- 标称场关闭观测噪声、质量/材质/COM/PD增益等随机化并设延迟0；随机场保留随机化并从参考frame0开始。标称重复不是独立样本。历史bundle和旧manifest均重新校验；motion哈希、预算、场景、seed一致。遥测可见的初始质量、COM、材质、PhysX关节刚度/阻尼一致；软件PD、默认位置偏移和延迟抽样没有完整独立遥测，不能把同seed等同于所有随机数完全一致。
- 位置按基座原点世界坐标，post-physics状态对齐下一参考帧(+20ms)，末帧参考保持。标称每周期3D RMSE均值8.128→5.283cm；首次足触地附近X误差+8.653→-2.811cm；终点+10.349→+3.660cm。正号为前于参考。本版标称前冲减轻，落地略后于参考、终点仍略前。
- 随机化3D RMSE均值15.970→18.119cm，触地X绝对误差均值15.688→17.182cm。新版五周期触地X误差[-20.88,-13.87,+33.90,+12.00,+5.26]cm，终点[-22.04,-16.72,+36.09,+18.95,+9.19]cm；存在正负抵消，不能用有符号均值变小判定改善。
- 基座COM线速度由body旋转至world，不是整机COM速度。起跳窗口1.10–1.72s：标称vx峰值2.2084→1.9502m/s，vz峰值1.6433→1.7195m/s；基座最高点0.65878→0.66252m，向上峰速略增、向前峰速降低。随机逐周期vz峰值均值1.7309→1.7185m/s，范围1.5420–1.8746；vx峰值均值1.7114→1.7257m/s。随机最高点均值0.66179→0.67417m，不能仅以高度判定竖直起跳更强。
- 离地定义为双轮法向力模长均<10N并持续至少15ms，5ms取样；参考离地1.300s、足触地1.600s。标称双轮离地1.325→1.360s，较参考延迟25→60ms；足首次触地1.570→1.575s。最长全监测接触腾空段245→215ms。随机离地平均延迟49→48ms、主腾空均值228→246ms；阈值1/5/10/20N敏感性已保留，短离地段不能当作主腾空段。
- 标称起跳窗口最大侧倾绝对值0.431→3.500°，落地窗口最大俯仰绝对值6.262→10.015°；随机分别7.677→8.431°和9.096→12.597°。前冲减轻不代表姿态稳定性同步改善。
- 足部法向接触力峰值标称1791.55→923.35N（-48.5%），随机1406.33→1155.93N（-17.8%）。新版轮部接触峰值标称397.23N、随机882.42N；监测的大腿/小腿对地法向力未见>10N。以上为5ms仿真法向力模长，非实物传感器力，且不包含摩擦力。
- 关节极值包含截断尾段。标称右足峰速10.347rad/s，超过10rad/s配置参考值，全部6次落地合计30ms；峰值力矩22.666N·m，未触27N·m力矩上限。左足5.838rad/s、18.634N·m。旧版标称左右足均约7.15rad/s、12.65N·m；足部总冲击减小不意味着每个关节载荷都减小。
- 随机右足峰速20.033→10.024rad/s，峰值力矩27.000→19.373N·m；左足10.463→10.008rad/s、20.069→20.285N·m。新版左右足分别仅单个5ms样本略高于10rad/s；不应与旧版20rad/s尖峰等量看待。所有关节未出现>=99%配置力矩上限的样本，左小腿峰值106.214N·m/限120。详细10关节速度、力矩、超限时长与发生时刻见joint-loads.csv和forward-jump-metrics.json。
- 双髋：标称左髋峰速0.732→1.976rad/s，右髋0.662→0.809；峰值力矩左36.211→41.756N·m、右35.581→41.618N·m。随机左髋2.931rad/s/43.258N·m、右髋3.300rad/s/29.591N·m；右髋峰值来自预算截断尾段，单独标注在极值记录。
- 双髋实际角度采用控制步反馈。随机完整周期最大绝对偏角右8.154→11.173°、左4.977→10.581°；标称右2.023→1.783°、左1.792→2.414°。标称落地后[2.0,3.4)s平均偏角绝对值由右1.168/左1.112°降到0.806/0.682°。符号只是关节坐标，不把绝对角度一律称为向外偏。
- feet_distance_penalize虽然名称含feet，实际约束双轮轮心世界XY距离，参考0–1.25s、原区间[0.496,0.536]m。该窗口随机完整周期轮距均值49.804→46.893cm，样本范围旧46.13–51.92cm、新39.50–51.92cm；落在原区间外的样本比例38.7%→98.6%。标称轮距仍约51.64cm、全部位于原区间。支持随机条件轮距收窄的观察，不是因果鉴定或新的验收阈值。
- 落地后参考1.85–3.40s两轮物理速度合并RMS：标称0.1109→0.2077rad/s，随机0.1983→0.1409rad/s。随机全程左轮峰速16.465→12.790rad/s，右轮11.875→12.641rad/s。轮速改善依赖场景，不能概括为全程均更平滑。
- 实机抖动尚未复测。标称0–1.1s右/左髋目标角变化范围由11.433/10.835°降为9.454/8.822°，随机范围变化不一致；这里只是action×0.25的变化范围，常量default offset不影响该统计，不是执行器延迟后的实际目标。不能由Native结果断言11月11日实测的高频振荡已消除。
- 保全检查1956个既有证据、模型、配置和源码文件哈希一致，原未提交flat_env_cfg.py未改动；没有训练启停、导出、归档或Git修改操作。测试后GPU无计算进程。历史查询因观测/部署fingerprint未知返回query_context_incomplete，不支持自动参数变更；本报告依靠显式匹配场景与重新校验的原始结果作描述性比较。
- 局限：单seed、每场5完整周期，无视频或实物验证，标称重复不代表5次独立随机试验。仅可进入受监督实物测试；未经实物验证，不代表 hardware-ready。
- 封存说明：原有限测试批次的两场结果完整且校验通过，封存进程退出143，原因未确定；原目录和.summary-claim保留。此新编号为adopted汇总，只引用原始结果，不代表又运行了两场测试。
- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据

- [analysis-method.json](<../wheel-to-leg-hip-relaxed-20261112-001/derived/analysis-method.json>)
- [cases-ready.json](<../wheel-to-leg-hip-relaxed-20261112-001/derived/cases-ready.json>)
- [comparison-metrics.json](<../wheel-to-leg-hip-relaxed-20261112-001/derived/comparison-metrics.json>)
- [cycle-metrics.csv](<../wheel-to-leg-hip-relaxed-20261112-001/derived/cycle-metrics.csv>)
- [forward-analysis-method.json](<../wheel-to-leg-hip-relaxed-20261112-001/derived/forward-analysis-method.json>)
- [forward-jump-comparison.png](<../wheel-to-leg-hip-relaxed-20261112-001/derived/forward-jump-comparison.png>)
- [forward-jump-metrics.json](<../wheel-to-leg-hip-relaxed-20261112-001/derived/forward-jump-metrics.json>)
- [hip-and-stance-comparison.png](<../wheel-to-leg-hip-relaxed-20261112-001/derived/hip-and-stance-comparison.png>)
- [hip-and-stance-method.json](<../wheel-to-leg-hip-relaxed-20261112-001/derived/hip-and-stance-method.json>)
- [hip-and-stance-metrics.json](<../wheel-to-leg-hip-relaxed-20261112-001/derived/hip-and-stance-metrics.json>)
- [joint-loads.csv](<../wheel-to-leg-hip-relaxed-20261112-001/derived/joint-loads.csv>)
- [liftoff-and-impact-method-a02.json](<../wheel-to-leg-hip-relaxed-20261112-001/derived/liftoff-and-impact-method-a02.json>)
- [liftoff-and-impact-metrics.json](<../wheel-to-leg-hip-relaxed-20261112-001/derived/liftoff-and-impact-metrics.json>)
- [phase-comparison.png](<../wheel-to-leg-hip-relaxed-20261112-001/derived/phase-comparison.png>)
- [wheel-spacing-metrics.json](<../wheel-to-leg-hip-relaxed-20261112-001/derived/wheel-spacing-metrics.json>)
- [batch-driver-method-20261112-001.json](<../../evidence/training/batch-driver-method-20261112-001.json>)
- [effective-config-diff-20261112-001.json](<../../evidence/training/effective-config-diff-20261112-001.json>)
- [evaluation-request-20261112-001.json](<../../evidence/training/evaluation-request-20261112-001.json>)
- [history-query-20261112-001.json](<../../evidence/training/history-query-20261112-001.json>)
- [protection-after-20261112-001.json](<../../evidence/training/protection-after-20261112-001.json>)
- [protection-before-20261112-001.json](<../../evidence/training/protection-before-20261112-001.json>)
- [tensorboard-windows-20261112-001.json](<../../evidence/training/tensorboard-windows-20261112-001.json>)
- [training-origin-20261112-001.json](<../../evidence/training/training-origin-20261112-001.json>)
- [manifest.json](<../../../2026-11-10_14-03-57/evaluations/wheel-to-leg-landing-std030-20261111-001/manifest.json>)
- [report-input.json](<../wheel-to-leg-hip-relaxed-20261112-001/derived/report-input.json>)
- [contract.json](<../wheel-to-leg-hip-relaxed-20261112-001/contract.json>)
- [summary-recovery-20261112-002.json](<../../evidence/training/summary-recovery-20261112-002.json>)

## 建议与待授权事项

- 保留11月10日基线，本版作为标称前冲和冲击较低的候选；目前不建议以“整体更好”替换基线。
- 下一步有限验证优先增加匹配多seed，检查起跳前轮距收窄、双髋大角度及侧倾。若继续调参，应将轮距惩罚与髋归零惩罚分别隔离比较，先提交具体方案再修改；direct_parameter_change_supported=false。
- 实机抖动需要匹配状态的策略敏感度或新的受监督硬件证据确认，仿真周期成功不能替代该验证。
