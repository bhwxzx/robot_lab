# 2026-09-23_17-23-24

此页可重新生成，仅用于导航；证据以各批次 manifest 及其校验链为准。

## 评估批次

- [flat-roa-final-assess-20260924-001](<evaluations/flat-roa-final-assess-20260924-001/report.md>)：6 完成 / 0 失败 / 0 未运行；[manifest](<evaluations/flat-roa-final-assess-20260924-001/manifest.json>)。

## 原始记录与生命周期

- [provenance](<provenance>)

旧版证据保持原位；本索引不会移动、压缩或重写已有结果。

## 本次分析

- [闭环同场景比较与逐关节指标](<evidence/analysis/flat-roa-final-assess-20260924-001/metrics.json>)；[对照图](<evidence/analysis/flat-roa-final-assess-20260924-001/comparison.png>)；[校验记录](<evidence/analysis/flat-roa-final-assess-20260924-001/verification-final.json>)。
- [实机历史数据静态回放与关节敏感度](<evidence/analysis/static-realdata-final-20260924-001/metrics.json>)；[回放图](<evidence/analysis/static-realdata-final-20260924-001/comparison.png>)；[校验记录](<evidence/analysis/static-realdata-final-20260924-001/verification-final.json>)。
- [训练末段与旧版训练指标比较](<evidence/training/tensorboard-review-20260924-001.json>)。
- [9 月 24 日实机右转失稳重算与奖励优化提案](<evidence/analysis/hardware-rightturn-20260926-001/metrics.json>)；[起振时序与双髋贡献图](<evidence/analysis/hardware-rightturn-20260926-001/onset-and-joint-contributions.png>)。来源为 `sim2real_test/b89df0a`；仅分析已录制数据，未运行新增测试。
- [9 月 24 日数据：旧 DWAQ 与 7 个 ROA 策略的双髋静态敏感度比较](<evidence/analysis/static-rightturn-policy-comparison-20260926-001/metrics.json>)；[比较图](<evidence/analysis/static-rightturn-policy-comparison-20260926-001/comparison.png>)；[逐帧数据](<evidence/analysis/static-rightturn-policy-comparison-20260926-001/per-frame-comparison.csv>)；[自动求导核验](<evidence/analysis/static-rightturn-policy-comparison-20260926-001/autograd-verification.json>)。505 帧冻结输入，旧 DWAQ 基线为 `2026-06-03-17-15`，不包含 GetDown 或任何物理闭环。

## 导出与归档

- [检查点选择记录](<evidence/checkpoint_selection/selection-deploy-20260924-001.json>)；[JIT/ONNX 导出校验](<evidence/export/deploy-20260924-001/receipt.json>)；[归档记录](<evidence/export/deploy-20260924-001/archive-receipt.json>)。
- `policy_storage/LW/wheel_loco/2026-09-23-17-23-24`；`policy_storage/master` 提交 `900772b390092eff15d2888f9890f8204cd7ac33`。
