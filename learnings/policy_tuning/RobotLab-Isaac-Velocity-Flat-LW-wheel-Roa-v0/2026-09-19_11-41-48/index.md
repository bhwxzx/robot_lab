# 2026-09-19_11-41-48

此页可重新生成，仅用于导航；证据以各批次 manifest 及其校验链为准。

## 评估批次

- [native-assess-20260921-001](<evaluations/native-assess-20260921-001/report.md>)：4 完成 / 0 失败 / 0 未运行；[manifest](<evaluations/native-assess-20260921-001/manifest.json>)。

## 原始记录与生命周期

- [provenance](<provenance>)
- [evidence/checkpoint_selection](<evidence/checkpoint_selection>)
- [evidence/export](<evidence/export>)
- [evidence/source](<evidence/source>)

旧版证据保持原位；本索引不会移动、压缩或重写已有结果。

## 本次对比分析

- [双髋惩罚对比指标、方法和证据哈希](evidence/analysis/hip-penalty-comparison-20260921-001.json)
- [髋目标、机身抖动与站立漂移对比图](evidence/analysis/hip-penalty-comparison-20260921-001.png)
- [训练 TensorBoard 全标量有限性检查与末两段 5000 迭代统计](evidence/training/tensorboard-review-20260921-001.json)
- [训练来源与当前评估源码逐文件核对](evidence/training/preflight-20260921-001.json)
- [训练日志摘要](evidence/training/summary-20260921-001.json)；[收敛判定限制](evidence/training/assessment-20260921-001.json)

四个固定 seed 42 场景完成，无跌倒或重置。新增双髋原始动作 L1 惩罚 -0.5 压低目标幅度，但 40–120 秒站立窗口的漂移和机身抖动退步；不能据此认定已解决实机启动振荡。下一步建议在既有实机观测上回放新旧策略，比较启动输出及髋反馈敏感度，再决定奖励调整；本次未修改参数或启动训练。

仅可进入受监督实物测试；未经实物验证，不代表 hardware-ready。

## 新旧 ROA 同实机观测回放

- [启动髋输出、反馈敏感度、方法与输入校验](evidence/analysis/roa-hip-penalty-same-observation-20260921/metrics.json)
- [对比曲线](evidence/analysis/roa-hip-penalty-same-observation-20260921/comparison.png)
- [逐帧输出与 2×2 雅可比](evidence/analysis/roa-hip-penalty-same-observation-20260921/replay_and_sensitivity.csv)

固定使用旧策略实机事故的 29 帧观测与记录动作历史，CPU 离线推理；不是新版的实机闭环轨迹。新版早期髋目标有所降低，但历史填满后的 200–300 ms 位置敏感度接近旧版，整段目标仍被放大至约 3.41 rad；不能判定实机振荡已解决。
