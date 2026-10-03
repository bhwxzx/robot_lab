# wheel-flat ROA 2026-10-01_00-04-08 评估索引

本页只提供可重新生成的导航；原始证据及哈希以批次 manifest 和分析指标文件为准。本次评估最终 `model_49999.pt`（内部迭代 50000，SHA-256 `6177c222df1a283e535d9543360737af2f1fd272bcd0ef4bef1027d81662a94c`），使用 isaacsim-5.1。

本次新测六个 Native 单环境场景，共 30000 控制步；静态分析新测最终模型在 9/24 右转事故的 505 帧上，共 49490 个输入向量。旧 DWAQ/ROA 与本轮中期 checkpoint 的测量均复用归档结果。

## 最终 checkpoint 的闭环测试

- [六场景批次报告](evaluations/wheel-flat-final-noise15-20261002-001/report.md) · [密封 manifest](evaluations/wheel-flat-final-noise15-20261002-001/manifest.json)
- [完整分阶段指标、历史比较及可核验分析源码](evidence/analysis/final-model49999-batch-20261002-001/metrics.json)
- [转弯段与旧 ROA 比较图](evidence/analysis/final-model49999-batch-20261002-001/comparison.png)
- [右转 15/30 ms 全时序曲线](evidence/analysis/final-model49999-batch-20261002-001/timeline.png)
- [120 秒零指令站定曲线](evidence/analysis/final-model49999-batch-20261002-001/standing.png)
- [最后训练记录的有界扫描及校验](evidence/training/final-tail-20261002-001.json)

## 最终 checkpoint 的静态双髋敏感度

- [敏感度指标、有限差分/autograd 复核、历史对齐范围](evidence/analysis/static-rightturn-model49999-20261002-001/metrics.json)
- [旧 DWAQ、其他 ROA、33000 轮与最终模型比较图](evidence/analysis/static-rightturn-model49999-20261002-001/comparison.png)
- [逐帧敏感度与物理目标 CSV](evidence/analysis/static-rightturn-model49999-20261002-001/per-frame-comparison.csv)

## 先前训练轮次比较

- [10000–33000 轮趋势指标](evidence/analysis/static-rightturn-training-trend-20261001-001/metrics.json) · [趋势图](evidence/analysis/static-rightturn-training-trend-20261001-001/checkpoint-trend.png) · [逐 checkpoint/阶段表](evidence/analysis/static-rightturn-training-trend-20261001-001/checkpoint-comparison.csv)
- [10000 轮](evidence/analysis/static-rightturn-model10000-20261001-001/metrics.json)
- [20000 轮](evidence/analysis/static-rightturn-model20000-20261001-001/metrics.json)
- [30000 轮](evidence/analysis/static-rightturn-model30000-20261001-001/metrics.json)
- [32000 轮](evidence/analysis/static-rightturn-model32000-20261001-001/metrics.json)
- [33000 轮](evidence/analysis/static-rightturn-model33000-20261001-001/metrics.json)

## 来源与执行范围

- [原训练源码捕获](provenance/context-6056aac32c5d9db4a06f528e80fd9d52f5f12067c146323127e1a05ec90a28fe.json)
- [本次评估上下文](provenance/context-50737a142e11a9a1096d8729617c8e16ddea450a1d6db4864ff0a1b7ce0ffef7.json)
- [有效训练配置](provenance/config-e8e4b3da5a8ac9435411385ee2c48eb554823bd48d8030b013aa976540481952.json)
- [完整启动参数、源码一致性与测试预算](evidence/source/launch-and-budget-final-20261002-001.json) · [有限批次输入](evidence/source/wheel-flat-final-noise15-batch-input-20261002-001.json)
- [评估来源](provenance/) · [批次事件](events/)

相同指令、延迟、seed 的历史对照保留各自训练噪声与奖励配置，不能归因于单项参数。每个场景只测一个环境和一个 seed；30 ms 延迟超过训练的 0–15 ms 范围。静态输入来自旧策略，未覆盖 9/30 减速事故；局部雅可比不构成闭环稳定性证明。仅可进入受监督实物测试；未经实物验证，不代表 hardware-ready。

## 当前策略导出与归档

按原训练启动时间归档至 policy_storage 的 `LW/wheel_loco/2026-10-01-00-04-08`，GitHub master 提交 `cdfce8d66dca9b26c283508de73a8369c0a7247b` 已与远端核对一致。

- [用户选择与 checkpoint 校验回执](evidence/checkpoint_selection/selection-final-storage-20261002-001.json)
- [Native/JIT/ONNX 时序及 reset 一致性回执](evidence/export/final-storage-20261002-001/receipt.json)
- [导出 JIT](evidence/export/final-storage-20261002-001/policy.pt) · [导出 ONNX](evidence/export/final-storage-20261002-001/policy.onnx)
- [归档输入](evidence/source/archive-input-final-storage-20261002-001.json) · [ff-only pull 与归档前检查](evidence/source/archive-preflight-final-storage-20261002-001.json)
- [归档回执](evidence/source/archive-receipt-final-storage-20261002-001.json) · [Git 提交推送和远端 SHA 核验](evidence/source/storage-git-publication-final-storage-20261002-001.json)
- [导出启动参数与有界校验预算](evidence/source/export-launch-final-storage-20261002-001.json) · [导出日志](evidence/source/export-console-final-storage-20261002-001.log)
- [GitHub 归档目录](https://github.com/bhwxzx/policy_storage/tree/master/LW/wheel_loco/2026-10-01-00-04-08)
