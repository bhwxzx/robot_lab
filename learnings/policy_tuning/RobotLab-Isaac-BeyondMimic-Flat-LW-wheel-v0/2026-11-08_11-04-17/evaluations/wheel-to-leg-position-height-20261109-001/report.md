# 评估批次 wheel-to-leg-position-height-20261109-001

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-wheel-v0`；训练：`2026-11-08_11-04-17`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| fixed-start-nominal-a01 | completed | fixed-start-nominal，1000 / 1 / 42 | complete | [结果](<raw/fixed-start-nominal-a01/result.json>) / [日志](<raw/fixed-start-nominal-a01/console.log>) |
| fixed-start-randomized-a01 | completed | fixed-start-randomized，1000 / 1 / 42 | complete | [结果](<raw/fixed-start-randomized-a01/result.json>) / [日志](<raw/fixed-start-randomized-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`d3cb45353eea83c8f34ee36422cd97df8e54cd5f7a3a7386a432cc8f989eabc3`；runner：`OnPolicyRunner`。
- [场景来源 scenario-092689db458d832357ac71cf4418b90bf034966489574535b61e5fac2076b54a.json](<../../provenance/scenario-092689db458d832357ac71cf4418b90bf034966489574535b61e5fac2076b54a.json>)
- [场景来源 scenario-7accd151d81c847de360e25fe091c74782fd0fc28469092f8688059fb2ae1014.json](<../../provenance/scenario-7accd151d81c847de360e25fe091c74782fd0fc28469092f8688059fb2ae1014.json>)
- [训练上下文 context-7aea0ff73aa827b0a5e8812afd051f89b2535ed631abe48372c3a65c95afc43b.json](<../../provenance/context-7aea0ff73aa827b0a5e8812afd051f89b2535ed631abe48372c3a65c95afc43b.json>)
- [训练有效配置 config-56705e963c290216c0086763063644d933192bf8c33a343e29a7198c6276f788.json](<../../provenance/config-56705e963c290216c0086763063644d933192bf8c33a343e29a7198c6276f788.json>)

- `fixed-start-nominal-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `fixed-start-nominal-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "nominal_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03", "rewards.hip_pos_zero_l2.weight": -2.0}`。
- `fixed-start-randomized-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `fixed-start-randomized-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03", "rewards.hip_pos_zero_l2.weight": -2.0}`。

## 观察与限制

- 结论：相对server上一版无高度奖励/全程轮约束策略，当前组合明显恢复了基座高度、标称位置跟踪和有效推蹬；落地后轮速仍低。但小腿更接近完全伸直、随机化小腿峰速提高、标称侧倾加重，随机位置离散仍大，不能判定为全面改善。
- 对象：server训练2026-11-08_11-04-17/model_49999.pt，内部iter49999，SHA256 d3cb45353eea83c8f34ee36422cd97df8e54cd5f7a3a7386a432cc8f989eabc3。8192环境、PPO/OnPolicyRunner、seed42、24steps/env、50000轮；用户确认bash scripts/start_beyondmimic.sh --type wheel --num_envs 8192且无额外覆盖。不是younghit的4096环境2026-11-08_10-58-49，本批不把跨机策略当受控对照。
- 来源：训练HEAD d11e72b5fba4a380f83a0b0929bdb76c67d4a4ff，加未提交height weight0→2；评估HEAD 3d2d9baca45498cc02ca1c0a500160581a9b6d28，保留同一源码修改。两HEAD之间scripts/source/.agents无变更，训练diff与当前参数一致。上下文v2绑定当前dirty source及差异hash，effective-config v2保存训练YAML，未临时修改源码恢复参数。
- 有效奖励：motion_global_anchor_pos weight0.5、std0.15；motion_flight_height weight2、std0.1、窗口1.1–1.9s、ramp0.05；轮速weight-0.5、scale10、start1.65、ramp0.1；轮加速度weight-0.1、scale50、start1.65、ramp0.2。三类执行器delay0–3；agent与本机Nov6/Nov7比较完全一致。相对Nov7，同时改了高度项、位置std、轮惩罚强度和阶段，不能单项归因。
- 有限Native测试：nominal_start和randomized_start，各1环境/1000控制步/seed42、frame0开始、mean action、无视频；每场5完整成功、0失败、1截尾。每场1000控制和4000物理样本，遥测完整。历史Nov6/Nov7复用已有结果并验证，未重跑。标称五次为确定性重复，单seed不能证明普遍鲁棒性。
- 比较标签：图和机器数据no_height=server 2026-11-07_15-01-47（std0.30、height0、全程轮约束）；height=server 2026-11-06_11-39-51（std0.30、height2、无轮约束）；current=本次（std0.15、height2、落地后轮约束）。
- 统计口径：关节速度、加速度、applied驱动力矩峰值用全部4000个5ms子步，含截尾观察段；基座高度与速度峰值是20ms控制采样，完整周期统计只含前5周期。旧口径落地高度RMSE按命令帧1.45–1.90s、不补偿20ms；另给出按物理时间插值参考的对齐指标。基座Z是link-origin世界高度，竖直速度是基座COM，不能与全身COM离地速度混用。
- 标称对照：

| 指标 | Nov7无高度/全程轮约束 | Nov6高度2/无轮约束 | 当前 |
|---|---:|---:|---:|
| 完整周期最高基座Z范围 m | 0.6816–0.6816 | 0.7044–0.7044 | 0.7071–0.7071 |
| 完整周期基座向上速度峰范围 m/s | 1.464–1.464 | 1.837–1.837 | 1.766–1.766 |
| 旧口径落地高度RMSE cm | 6.46 | 2.87 | 2.82 |
| 完整周期最低落地基座Z m | 0.4235 | 0.4738 | 0.4751 |
| 全程roll峰 deg | 1.385 | 3.423 | 3.975 |
| 轮速峰 rad/s | 5.977 | 10.953 | 10.043 |
| 小腿速度峰 rad/s | 14.984 | 16.172 | 16.252 |
| 全部关节驱动力矩峰 N·m | 79.152 | 79.073 | 78.993 |
| 单足竖直法向力峰 N | 1226.6 | 1215.0 | 1102.8 |
| 对齐3D位置RMSE逐周期均值 cm | 9.223 | 4.395 | 2.491 |
| 对齐落地高度RMSE逐周期均值 cm | 5.355 | 1.621 | 1.447 |
| 落地点X误差平均绝对值 cm | 8.695 | 0.272 | 2.481 |
| 渐入结束后轮速RMS rad/s | 0.882 | 2.280 | 0.399 |
- 随机化对照：

| 指标 | Nov7无高度/全程轮约束 | Nov6高度2/无轮约束 | 当前 |
|---|---:|---:|---:|
| 完整周期最高基座Z范围 m | 0.6589–0.6975 | 0.6950–0.7169 | 0.6830–0.7304 |
| 完整周期基座向上速度峰范围 m/s | 1.502–2.024 | 1.757–2.155 | 1.775–2.130 |
| 旧口径落地高度RMSE cm | 5.56 | 2.09 | 2.75 |
| 完整周期最低落地基座Z m | 0.4047 | 0.4627 | 0.4512 |
| 全程roll峰 deg | 6.086 | 6.588 | 6.353 |
| 轮速峰 rad/s | 13.728 | 16.279 | 13.100 |
| 小腿速度峰 rad/s | 17.622 | 17.495 | 18.409 |
| 全部关节驱动力矩峰 N·m | 106.983 | 99.598 | 104.831 |
| 单足竖直法向力峰 N | 1400.9 | 1508.1 | 1495.9 |
| 对齐3D位置RMSE逐周期均值 cm | 14.674 | 15.816 | 14.469 |
| 对齐落地高度RMSE逐周期均值 cm | 4.908 | 2.244 | 2.534 |
| 落地点X误差平均绝对值 cm | 13.987 | 13.997 | 13.381 |
| 渐入结束后轮速RMS rad/s | 0.709 | 3.122 | 0.656 |
- 当前关节负载，峰值未必同一时刻发生；通用result中的max torque是控制步末采样，物理峰值以此表为准：

| 关节（左/右） | 标称峰速 rad/s | 随机化峰速 rad/s | 标称峰矩 N·m | 随机化峰矩 N·m |
|---|---:|---:|---:|---:|
| 髋 | 1.577 / 2.796 | 2.517 / 3.212 | 44.825 / 45.609 | 53.470 / 38.451 |
| 大腿 | 11.954 / 11.351 | 13.529 / 13.046 | 32.211 / 23.618 | 41.806 / 48.613 |
| 小腿 | 16.252 / 15.252 | 18.409 / 17.051 | 78.993 / 77.885 | 104.831 / 85.666 |
| 足 | 5.232 / 5.429 | 10.077 / 10.095 | 17.523 / 14.937 | 18.111 / 18.830 |
| 轮 | 10.043 / 8.968 | 13.100 / 11.996 | 2.306 / 1.519 | 6.609 / 5.891 |
- 高度恢复：标称最高基座0.70711m，接近参考0.71351m；相对Nov7的0.68158m增加2.55cm，最低落地基座0.42349→0.47511m。基座向上速度峰1.46446→1.76617m/s，仍低于参考1.96683m/s。实际离地1.325s、持续足接触1.575s；离地瞬间基座vz由20ms样本插值约1.60271m/s。随机化完整周期最高基座0.68299–0.73039m，基座vz峰1.77478–2.13007m/s，最低落地基座最小0.45122m。
- 位置与前冲：标称物理时间对齐3D位置RMSE9.223→2.491cm，落地点X误差+8.695→-2.481cm，空中基座前移26.02→18.04cm；标称起跳窗口基座vx峰2.214→1.920m/s，仍高于参考1.222m/s。随机化对齐3D位置RMSE逐周期均值14.674→14.469cm，仅小幅变化；落地X误差[-9.60,-12.48,+26.07,+14.77,+3.99]cm，平均绝对值13.381cm，仍有明显位置离散。不能将标称改善推广为随机条件位置问题已解决。
- 小腿过伸仍突出：标称左/右小腿最小屈曲角0.05687/0.04470rad，前一版为0.14022/0.12347rad；参考左侧最低0.69407rad。随机化左小腿最小接近0（-0.000040rad），代表几乎伸直，不据此声称越过机械限位。标称小腿峰速16.252、随机化18.409rad/s；随机峰发生第1完整周期1.345s离地附近，约为配置20rad/s的92.0%。最大驱动力矩104.831N·m发生左小腿1.280s推蹬，约为120N·m的87.4%。
- 轮速：标称左/右峰10.043/8.968rad/s，在1.385s腾空阶段；随机化左峰13.100在第2周期1.245s推蹬，右峰11.996在截尾第6周期1.440s腾空。若只看全程标称峰值，较Nov7全程轮约束版5.977更高；但参考1.85s后两项ramp均结束，轮速RMS标称约0.399rad/s、随机约0.656rad/s，落地后抑制有效。门控按参考时间，不按实际接触。
- 侧倾与足部：标称roll峰3.9746度，发生物理1.580s（首次持续足接触1.575s之后5ms），此前腾空末段已持续增长；随机roll峰6.3531度发生1.560s，比该周期持续足接触早5ms。不能简单归因于落地冲击。随机左/右足速度10.077/10.095rad/s，各超10rad/s累计5/10ms，均在落地后。标称无超速，两组所有驱动力矩均未触及限幅。足竖直法向力峰1102.8/1495.9N，为5ms平均仿真法向力，不是实物瞬时冲击或结构关节负载；所监测大腿/小腿地面力为0。
- 推蹬：标称总竖直支撑冲量97.726N·s、扣重力后净冲量38.723N·s、全身COM离地vz1.12365m/s，Nov7为94.109/35.106/0.94976；对应有效竖直推蹬恢复。Nov6高度版为98.021/39.019/1.09311，当前与其接近。随机化全身COM离地vz1.0018–1.3542m/s。标称轮轴1.10→1.30s后退4.84cm（Nov7 2.61cm），再次说明后退量和竖直起跳能力并非简单单调关系。
- 动力学验证：URDF关节原点重建误差<4.4e-7m、轮轴速度交叉误差<6.1e-6m/s、冲量–速度闭合误差<2.5e-6m/s。载荷加权轮地滑移使用20ms采样、理想轮半径0.1095m和完整世界角速度估算，不含实际接触斑或摩擦能量，不能据其声称因果损失。
- 可比性：本批三策略同server、8192环境训练、同seed、同参考、同评估场景/时长；初始质量/COM/材质/PhysX增益逐项相同，软件PD和延迟每次抽样未独立捕获，不能称所有随机量完全同轨。多项奖励同时变化，且仅各一次训练，不能把高度或位置改善单独归因于某一个奖励。
- 训练完成但不等于收敛：checkpoint和55项TensorBoard标量均至49999，末5000选定窗口均有限。末1000 motion_finished均值0.996465，是训练自适应随机起点统计，不是本次完整动作成功率。缺少已批准收敛判据，结论indeterminate。仅可进入受监督实物测试；未经实物验证，不代表 hardware-ready。
- 证据与保护：分析JSON内嵌源码/输入hash，保存完整遥测、CSV和两张已检查对照图；原有1380个证据/模型/exports/源码文件hash不变，HEAD、用户height2未提交差异及两个stash保持。此次不训练、导出、修改代码、删除或Git提交。
- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据

- [manifest.json](<../../../2026-11-06_11-39-51/evaluations/wheel-to-leg-fixed-start-20261107-001/manifest.json>)
- [manifest.json](<../../../2026-11-07_15-01-47/evaluations/wheel-to-leg-fixed-start-20261108-001/manifest.json>)
- [analysis.json](<derived/analysis.json>)
- [cycle-metrics.csv](<derived/cycle-metrics.csv>)
- [joint-loads.csv](<derived/joint-loads.csv>)
- [phase-comparison.png](<derived/phase-comparison.png>)
- [position-velocity-analysis.json](<derived/position-velocity-analysis.json>)
- [position-velocity-comparison.png](<derived/position-velocity-comparison.png>)
- [push-off-dynamics.json](<derived/push-off-dynamics.json>)
- [effective-config-diff-20261109-001.json](<../../evidence/training/effective-config-diff-20261109-001.json>)
- [evaluation-request-20261109-001.json](<../../evidence/training/evaluation-request-20261109-001.json>)
- [protection-after-20261109-001.json](<../../evidence/training/protection-after-20261109-001.json>)
- [tensorboard-summary-20261109-001.json](<../../evidence/training/tensorboard-summary-20261109-001.json>)
- [training-origin-20261109-001.json](<../../evidence/training/training-origin-20261109-001.json>)
- [report-input.json](<derived/report-input.json>)

## 建议与待授权事项

- 保留此版作为高度与标称位置改善的候选基线；当前主要剩余问题是小腿过伸/峰速、腾空末段侧倾和随机化位置离散，暂不进一步收紧全局位置std或加大落地轮惩罚。
- 若下一步调整，先在相同8192环境和同机评估条件下设计单变量对照：保持std0.15与轮约束不变，只比较height weight2→1。预期减轻对高度的激励压力，但可能降低起跳速度和落地支撑；需同时检查小腿最小屈曲角、峰速/力矩、基座高度和roll。该值未验证，不能保证改善，本轮不执行。
- 也可先用现有遥测复核推蹬PD位置目标、腿部制动时序与横滚增长，确定是否需要独立的小腿参考速度/位置约束；任何新增奖励先给方案并获批准。direct_parameter_change_supported=false。
- 需要鲁棒性结论时，再明确有限多seed/多参数测试范围；当前5个完整周期/场不是统计意义上的普遍成功率。
