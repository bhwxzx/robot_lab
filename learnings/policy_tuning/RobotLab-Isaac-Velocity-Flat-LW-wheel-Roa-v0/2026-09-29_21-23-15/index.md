# Wheel-flat ROA model_29000 有界评估（2026-09-30）

**结论：**当前中期策略在固定的 0.5 m/s 直行仿真里跟踪明显改善，但零速前向漂移仍在，且在历史双髋抖动数据的增长段表现出较高的髋动作敏感度。现有结果不足以判断它在实机右转是否会抖动，也不足以判定训练收敛。

## 快照与方法

- 训练运行：`2026-09-29_21-23-15`，`OnPolicyRunnerROA` / `ROAPPO`，seed 42；固定 checkpoint `model_29000.pt`，SHA-256 `44edf7046a865fbb56376b59fe6aaf788c9a3c8609b5d4c7df797dfa9bb59261`。此检查点是 50000 轮训练中的中间快照。
- 本次配置：启用显式速度估计，估计速度使用概率自起始即为 1；关节速度观测噪声为均匀分布 `[-0.5, 0.5] rad/s`；髋动作比例为 0.125。训练启动时 HEAD 为 `175206032beca49f7f3e2e5588f387d3a2ad7494` 并有两处受控未提交改动；源码重建及当前 HEAD 校验见[启动验证](evidence/source/launch-verification-model29000-overlap-20260930-001.json)。
- Native Play：与历史 `right-delay15-noise` 相同的 seed、场景覆盖及前 2000 步指令；0–499 步零速，500–1999 步前进 0.5 m/s，单环境、15 ms 固定执行器延迟、有观测噪声、无视频。当前检查点只运行这一次 2000 步；旧策略不重新运行。完整遥测无缺失，无终止。
- 训练健康：评估前后可比的 TensorBoard 步数从 30104 增至 30174，状态 `healthy`；截至独立标量快照记录为 30425 轮，学生估计速度使用比例为 100%。这只证明训练持续推进，未建立评估期间与非评估期间的吞吐对照或收敛阈值。

## 匹配的零速与直行仿真

站立窗口为步骤 50–499（9 秒），直行稳态窗口为步骤 550–1999（29 秒）。历史 ROA 数值直接读取已归档的同名窗口；DWAQ 没有对应的同场景闭环结果，因此在下一节仅作静态比较。

| 策略 | 零速平均 vx (m/s) | 零速净位移 (m) | 直行平均 vx (m/s) | 直行 XY 跟踪 RMSE (m/s) | 直行双髋目标步差 RMS (rad) | 直行双髋速度 5–25 Hz RMS (rad/s) |
|---|---:|---:|---:|---:|---:|---:|
| ROA 9/23（有速度估计） | +0.033 | 0.289 | 0.416 | 0.079 | 0.0310 | 0.198 |
| ROA 9/26（有速度估计、髋比例 0.125） | +0.045 | 0.404 | 0.431 | 0.073 | 0.0164 | 0.211 |
| ROA 9/28（无速度估计） | -0.024 | 0.215 | 0.401 | 0.087 | 0.0176 | 0.207 |
| 本次 ROA 9/29 @ 29000 | +0.058 | 0.521 | 0.491 | 0.059 | 0.0157 | 0.201 |

本次零速段约 0.058 m/s 前向漂移、9 秒净位移 0.521 m；9/28 无速度估计版分别为 -0.024 m/s 与 0.215 m。直行 0.5 m/s 指令下，本次平均 vx 0.491 m/s、XY 跟踪 RMSE 0.059 m/s，优于 9/26 的 0.431/0.073 与 9/28 的 0.401/0.087。直行双髋 5–25 Hz 速度 RMS 0.201 rad/s，与旧版 0.207–0.211 rad/s 接近；这个直行指标不覆盖右转抖动。仿真学生速度估计 x 轴 RMSE 为零速 0.040、直行 0.035 m/s。

[闭环原始结果](evidence/play/model29000-standing-forward-a01/result.json) · [窗口比较与旧证据引用](evidence/analysis/closed-loop-prefix-model29000-20260930-001/metrics.json) · [修正的对照图](evidence/analysis/closed-loop-prefix-model29000-20260930-001/comparison-edge-corrected.png)（[修正说明](evidence/analysis/closed-loop-prefix-model29000-20260930-001/plot-correction.json)）

## 历史实机抖动输入上的静态敏感度

使用 9/24 实机右转、GetDown 之前的 505 帧，主对齐为状态延迟 1 帧/前次动作延迟 1 帧；另检验三个相邻对齐。只对本次 checkpoint 新做推理（49,490 个模型输入向量，CPU 单线程），DWAQ 与 9/26、9/28 ROA 直接复用归档结果。下表是增长段 20 帧，Gq/Gdq 为双髋**物理目标**对髋位置/速度观测的局部 2×2 雅可比最大奇异值中位数。

| 策略 | Gq (rad/rad) | Gdq (rad/(rad/s)) | 髋目标峰值 (rad) | 髋目标逐步变化 RMS (rad) |
|---|---:|---:|---:|---:|
| 旧 DWAQ | 2.070 | 0.123 | 1.810 | 1.071 |
| ROA 9/26 | 2.818 | 0.030 | 0.604 | 0.437 |
| ROA 9/28 无速度 | 2.102 | 0.057 | 0.853 | 0.563 |
| 本次 ROA @ 29000 | 2.338 | 0.132 | 1.575 | 1.074 |

本次 Gdq 是 9/26 的 4.38 倍、9/28 的 2.32 倍，已接近旧 DWAQ（1.07 倍）。相同实机输入下，本次增长段髋目标峰值为 1.575 rad，9/26 为 0.604 rad。冻结当前帧髋观测的离线消融使本次增长段目标步差 RMS 降低约 85%；把估计速度码置零只降低约 0.9%。这提示直接髋状态通路值得优先排查，但消融不是稳定性证明，不能据此判定速度估计或噪声改动单独造成风险。

[静态指标与原始数值](evidence/analysis/static-rightturn-model29000-20260930-001/metrics.json) · [静态对照图](evidence/analysis/static-rightturn-model29000-20260930-001/comparison.png)

## 解释范围与下一步

静态分析输入来自旧实机策略的记录，含旧策略生成的前次动作；其原髋比例为 0.25，而近两版及当前策略为 0.125。DWAQ 历史长 5 帧、ROA 长 10 帧。录制时真正的 actor 输入时刻与传感器龄期也未完全恢复。局部目标雅可比、冻结输入与短窗口目标跳变均不能代替当前策略的闭环右转或新实机测试。当前短仿真只有一个 seed、一个环境，尚未测当前策略的右转/停车；各策略训练配置同时变化，无法作单因素因果归因。

建议让训练继续按现有计划进行；待训练结束且 GPU 空闲后，对选定稳定 checkpoint 做与旧版完全一致的 4500 步零速—直行—右转—停车 Native 闭环测试，重点看零速净位移、转弯双髋目标步差和实测髋速度 5–25 Hz。现阶段不据此改训练参数或作实机部署决定。仅可进入受监督实物测试；未经实物验证，不代表 hardware-ready。

[训练健康前](evidence/health/health-model29000-overlap-20260930-001.json) · [训练健康后](evidence/health/health-model29000-overlap-post-20260930-001.json) · [训练标量快照](evidence/training/summary-model29000-overlap-20260930-001.json) · [运行身份](evidence/source/identity-model29000-overlap-20260930-001.json)

## 最终模型 model_49999（2026-09-30）

- [六场景批次报告](evaluations/wheel-flat-final-estvel-20260930-001/report.md)与[封存清单](evaluations/wheel-flat-final-estvel-20260930-001/manifest.json)：6 组完成，0 组失败。
- [闭环窗口指标与历史同协议对照](evidence/analysis/final-model49999-batch-20260930-001/metrics.json)及[转弯对比图](evidence/analysis/final-model49999-batch-20260930-001/comparison.png)。
- [历史实机输入的静态双髋敏感度](evidence/analysis/static-rightturn-model49999-20260930-001/metrics.json)及[静态对比图](evidence/analysis/static-rightturn-model49999-20260930-001/comparison.png)。

旧版数值均直接读取归档结果；本次新运行的只有最终模型。静态实机输入回放不代表最终模型的实机闭环测试。
