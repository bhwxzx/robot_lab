# LW 关节位置观测修复：最终代码验证证据

入口：[最终汇总](lw-unified-joint-position-summary-20260911-001.json)。这是代码与观测契约验证，不是策略性能评估。

## 保留范围

- `baseline`：修改前配置基线，用于核对除位置噪声外的配置保持原样。
- `leg`、`wheel`、`wheel-beyondmimic` JSON：通过的实际观测、轮位置置零和历史布局证据。
- `leg-teacher`、`leg-beyondmimic` JSON 与日志：已有启动错误的证据，保留验证缺口。
- `summary`：统一结论和上述文件的 SHA-256 引用。原文保持不变。
- [validation-context](lw-unified-joint-position-validation-context-20260911.json)：生成历史报告时的完整验证脚本，以及当时已有的本地 LW_Leg Flat 奖励配置快照。后者仅说明验证上下文，其奖励修改未纳入本次生产代码提交。历史脚本已从日常测试精简；需复核旧结果时，可从 `sources[].content_utf8` 还原原脚本并核对 SHA-256。

42 个配置中 38 个完成实例化与配置一致性核对；另外 4 个存在已有缺失导入。3 个代表环境实测通过；Leg Teacher 和 Leg BeyondMimic 的启动缺口仍未解决，详见最终汇总。

## 日常核心回归

```bash
conda run --no-capture-output -n isaacsim-5.1 python scripts/tests/test_joint_position_observation.py
```

2026-09-11，主机 younghit：退出码 0，6 项测试全部通过（CPU/CUDA）。检查两处观测函数的索引重排、切片和输入保护，以及非轮噪声、轮列保持为零、Hydra 往返和列数错误。运行需启动 Isaac 应用，但不创建物理环境；不依赖历史配置基线。

核心脚本 SHA-256：`a0a2d5dfad5702c39f7b2731a855bf7d52a674fd38259e0fb5e42e67c1376f52`。

历史完整仿真验证与精简后核心回归是两次不同运行；核心回归通过不表示上述环境启动缺口已修复。

## 清理记录

删除最终汇总未引用的早期 `joint-position-observation-20260911-001.json`、`002.json`、`003.json`，以及本次临时控制台日志。其余历史策略评估和 Sim2Sim 记录保留。

提交前核对最终汇总及依赖引用的 79 个路径/校验和匹配；旧脚本使用上述历史快照核对。
