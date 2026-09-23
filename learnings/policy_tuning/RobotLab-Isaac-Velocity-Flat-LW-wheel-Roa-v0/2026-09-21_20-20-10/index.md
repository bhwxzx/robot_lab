# 2026-09-21_20-20-10

此页可重新生成，仅用于导航；证据以各批次 manifest 及其校验链为准。

## 评估批次

- [delay30-assess-20260922-001](<evaluations/delay30-assess-20260922-001/report.md>)：6 完成 / 0 失败 / 0 未运行；[manifest](<evaluations/delay30-assess-20260922-001/manifest.json>)。

## 原始记录与生命周期

- [provenance](<provenance>)

旧版证据保持原位；本索引不会移动、压缩或重写已有结果。

## 本轮对比与训练记录

- [分阶段指标、匹配条件及基线引用](evidence/analysis/delay30-assess-20260922-001/metrics.json)
- [抖动、偏航跟踪和站立漂移对比图](evidence/analysis/delay30-assess-20260922-001/comparison.png)
- [完整性与引用校验](evidence/analysis/delay30-assess-20260922-001/verification.json)
- [授权范围及启动来源](evidence/analysis/delay30-assess-20260922-001/preflight.json)
- [训练标量检查与末段统计](evidence/training/tensorboard-review-20260922-001.json)
- [相对 9 月 16 日及 19 日的有效配置完整差异](evidence/training/config-comparison-20260922-001.json)

6 项共 30,000 步，单环境 seed 42，必需遥测完整，无 done/timeout。42–70 秒转弯窗口相对 9 月 19 日：机身横滚/俯仰角速度 RMS 下降约 12%–26%，实际髋角速度 RMS 下降约 21%–36%，但偏航跟踪 RMSE 增加约 19%–24%；实际平均偏航右转约 −0.16、左转约 +0.20 rad/s，指令为 ±0.6 rad/s。不能将抖动指标降低直接解释为对同等实际转向的抗抖能力改善。

固定零执行器延迟、40–120 秒站立窗口：无噪声净 XY 位移 0.00992 m，有噪声 0.25708 m。两者均小于 9 月 19 日版本；有噪声时大于 9 月 16 日版本的 0.04181 m。本轮相对 9 月 19 日同时取消髋动作惩罚并增大训练延迟，不能分离两项改动的因果贡献。

站立场景保留历史默认 Native student 推理路径，未请求单独的 ROA 诊断张量，相应估计误差为 null；不得解释为零误差。转弯场景使用显式 student 分支，执行动作与 student 诊断动作、teacher 速度标签与同帧真值均一致。左右转前 40 秒逐元素匹配。

保持暂停实机复现；未经实物验证，不代表 hardware-ready。本次未修改代码或训练参数，未启动训练，未导出部署、提交或推送；新增证据保留在工作区。

## 原实机数据双髋敏感度回放

- [三版 ROA 与历史 DWAQ 对比指标及方法](evidence/analysis/roa-delay30-same-observation-20260922/metrics.json)
- [逐帧目标和双髋敏感度曲线](evidence/analysis/roa-delay30-same-observation-20260922/comparison.png)
- [完整 2×2 矩阵与双扰动幅度 CSV](evidence/analysis/roa-delay30-same-observation-20260922/replay_and_sensitivity.csv)
- [引用与重算校验](evidence/analysis/roa-delay30-same-observation-20260922/verification.json)

使用 9 月 18 日启动振荡的原始 29 帧（CSV 行 3200–3228，约 0–560 ms），不是后来无日志的右转事故。三版 ROA 接收相同记录观测和原动作历史，CPU 确定性回放；DWAQ 复用已有同观测分析。输出不回灌，不运行物理动力学。

历史填满后的 200–560 ms：本轮双髋位置敏感度最大奇异值的中位数 3.235 rad/rad、峰值 6.276，速度敏感度中位数 0.10126 rad/(rad/s)。相对 9 月 16 日位置中位数下降约 17.7%，相对 9 月 19 日反而增加约 12.3%；位置峰值相比两版约下降 40%。但本轮同输入髋目标绝对峰值增至 3.990 rad（9 月 16 日 3.678，9 月 19 日 3.413，历史 DWAQ 2.901）。不能以单个敏感度峰值降低认定实机振荡已解决。

200–300 ms 早期填满历史窗口，本轮位置敏感度中位数 4.483，高于两版旧 ROA（4.208/4.146）和历史 DWAQ（2.746）。完整模型与固定编码器的 actor 通道量级接近；这是局部输入输出导数，不能解释为物理闭环增益、阻尼或根因贡献率。新策略扰动幅度加倍后的矩阵相对变化最大约 0.26%；旧策略目标回放与历史分析最大差约 1.2e-6 rad。

继续暂停实机复现，保留未提交证据；本轮未修改代码或训练参数。
