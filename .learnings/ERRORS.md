## [ERR-20260726-003] skill_creator_python_entrypoint

**Logged**: 2026-07-26T15:05:43+08:00
**Priority**: low
**Status**: resolved
**Area**: tests

### Summary
Skill 元数据生成首次使用裸 `python`，但系统 PATH 中没有该命令。

### Error
```text
/bin/bash: python: 未找到命令
```

### Context
- Command/operation attempted: 运行 `generate_openai_yaml.py`
- Input or parameters used: 更新训练监督调优 Skill 的 UI 元数据
- Environment details if relevant: robot_lab 要求使用 `isaacsim-5.1` conda 环境

### Suggested Fix
运行项目 Python 工具时使用
`conda run -n isaacsim-5.1 python ...`，不安装或假定系统 Python。

### Metadata
- Reproducible: yes
- Related Files: `.agents/skills/monitor-tune-isaaclab-training/agents/openai.yaml`

### Resolution
- **Resolved**: 2026-07-26T15:05:43+08:00
- **Commit/PR**: N/A
- **Notes**: 改用项目规定的 conda 环境后，元数据生成和 Skill 校验通过。

---

## [ERR-20260611-001] ROADeploymentWrapper

**Logged**: 2026-06-11T11:13:36+08:00
**Priority**: high
**Status**: resolved
**Area**: backend

### Summary
Play script failed due to ROADeploymentWrapper concatenating a tuple instead of unpacked tensors.

### Error
```python
TypeError: expected Tensor as element 1 in argument 0, but got tuple
```

### Context
- Command/operation attempted: Playing the amp_roa trained policy (`play.py`).
- Input or parameters used: `obs_history_flat` passed to `history_encoder`.
- Environment details if relevant: `ActorCriticROA` history encoder returns `(hist_latent, code_vel)`, but `ROADeploymentWrapper` assigned them to a single variable `hist_latent`.

### Suggested Fix
Unpack the returned tuple `hist_latent, code_vel = self.history_encoder(obs_history_flat)` and concatenate in the correct order `[current_obs, code_vel, hist_latent]` as expected by the actor network.

### Metadata
- Reproducible: yes
- Related Files: scripts/reinforcement_learning/rsl_rl/play.py

### Resolution
- **Resolved**: 2026-06-11T11:13:36+08:00
- **Commit/PR**: N/A
- **Notes**: Unpacked `history_encoder` return values into `hist_latent` and `code_vel`, and updated `torch.cat` order to `(current_obs, code_vel, hist_latent)`.

---

## [ERR-20260726-001] policy_storage_summary_pipeline

**Logged**: 2026-07-26T13:05:00+08:00
**Priority**: low
**Status**: resolved
**Area**: tests

### Summary
策略仓库只读检查的 JSON 汇总管道依赖未安装的 `jq`，并使上游
`conda run` 报告 BrokenPipe。

### Error
```text
/bin/bash: jq: 未找到命令
BrokenPipeError: [Errno 32] Broken pipe
```

### Context
- Command/operation attempted: 将 `inspect_policy_storage.py` 输出通过 `jq` 缩减
- Input or parameters used: 真实只读 `policy_storage --hash-artifacts` 检查
- Environment details if relevant: `isaacsim-5.1`，系统无 `jq`

### Suggested Fix
不增加依赖；使用环境内 Python 导入检查器并选择需要的 JSON 字段。

### Metadata
- Reproducible: yes
- Related Files: `.agents/skills/monitor-tune-isaaclab-training/scripts/inspect_policy_storage.py`

### Resolution
- **Resolved**: 2026-07-26T13:05:00+08:00
- **Commit/PR**: N/A
- **Notes**: 已改用 Python 汇总；真实策略仓库检查通过且保持干净。

---

## [ERR-20260717-001] OnPolicyRunnerAmpROA_no_log_mode

**Logged**: 2026-07-17T14:47:37+08:00
**Priority**: low
**Status**: resolved
**Area**: tests

### Summary
AMP_ROA runner crashes after its first iteration when `log_dir=None` unless logging is explicitly disabled.

### Error
```text
TypeError: expected str, bytes or os.PathLike object, not NoneType
```

### Context
- Command/operation attempted: Reduced in-memory AMP_ROA training smoke with `log_dir=None`.
- Input or parameters used: `OnPolicyRunnerAmpROA(..., log_dir=None)` followed by `learn()`.
- Environment details if relevant: At the first iteration, `store_code_state(self.log_dir, ...)` is called when `disable_logs` is false, even though `self.log_dir` is `None`.

### Suggested Fix
Guard the initial code-state snapshot with `self.log_dir is not None`, matching the other logging/save branches. Until fixed, set `runner.disable_logs = True` for in-memory smoke tests.

### Metadata
- Reproducible: yes
- Related Files: rsl_rl/rsl_rl/runners/on_policy_runner_amp.py, rsl_rl/rsl_rl/runners/on_policy_runner_amp_roa.py, rsl_rl/rsl_rl/runners/on_policy_runner_amp_dwaq.py

### Resolution
- **Resolved**: 2026-07-22T15:51:11+08:00
- **Commit/PR**: 8d6090c
- **Notes**: Code-state snapshot calls are now guarded by `log_dir is not None` in AMP, AMP_DWAQ, and AMP_ROA. One-iteration no-log runner tests passed for the ordinary AMP and AMP_DWAQ paths, and the AMP_ROA path was previously verified.

---

## [ERR-20260726-002] markdown_fence_shell_expansion

**Logged**: 2026-07-26T13:12:00+08:00
**Priority**: low
**Status**: resolved
**Area**: tests

### Summary
内联 Python 校验命令把 Markdown 三反引号直接放进双引号 shell 参数，
触发命令替换并破坏了 Python 源码。

### Error
```text
/bin/bash: json,1)[1].split(: 未找到命令
ValueError: empty separator
```

### Context
- Command/operation attempted: 从 `session-spec.md` 提取首个 JSON 示例
- Input or parameters used: `python -c` 中直接包含三反引号
- Environment details if relevant: Bash 会在双引号内执行反引号命令替换

### Suggested Fix
不要把反引号放入 shell 命令参数；在 Python 内用 `chr(96)` 构造 Markdown
围栏，或使用不经过 shell 展开的安全输入方式。

### Metadata
- Reproducible: yes
- Related Files: `.agents/skills/monitor-tune-isaaclab-training/references/session-spec.md`

### Resolution
- **Resolved**: 2026-07-26T13:12:00+08:00
- **Commit/PR**: N/A
- **Notes**: 改用 `chr(96)` 构造围栏后，版本 4 会话示例验证通过。

---
## [ERR-20260727-001] unittest_path_invocation

**Logged**: 2026-07-27T16:07:08+08:00
**Priority**: low
**Status**: resolved
**Area**: tests

### Summary
`python -m unittest` 把包含隐藏目录和连字符的文件路径当成模块名时返回
`ValueError: Empty module name`。

### Error
```text
ValueError: Empty module name
```

### Context
- Command/operation attempted: 直接把 `.agents/.../test_first_run_configuration.py` 传给 `python -m unittest`
- Input or parameters used: 文件路径而非可导入模块名
- Environment details if relevant: `isaacsim-5.1` Python 3.11

### Suggested Fix
对该测试目录使用 `python -m unittest discover -s ... -p 'test_first_run_configuration.py'`。

### Metadata
- Reproducible: yes
- Related Files: `.agents/skills/monitor-tune-isaaclab-training/tests/test_first_run_configuration.py`
- Pattern-Key: tests.unittest_hidden_path_invocation
- Recurrence-Count: 4
- First-Seen: 2026-07-27
- Last-Seen: 2026-07-31

### Resolution
- **Resolved**: 2026-07-27T16:07:08+08:00
- **Commit/PR**: N/A
- **Notes**: 改用 unittest discovery 或直接执行测试文件。2026-07-31 再次因把
  `.agents/...` 路径传给 `python -m unittest` 复现，随后改为直接执行并通过。

---

## [ERR-20261022-001] isaacsim_persistent_physx_thread_setting

**Logged**: 2026-10-22T17:28:04.302329+08:00
**Priority**: high
**Status**: resolved
**Area**: config

### Summary
此前临时 CPU 评估使用了可持久化的 PhysX 线程参数且未禁用偏好保存；server 后续读取 numThreads=1，在初始化阶段阻塞。

### Error
在 2026-10-22 重启后的受控 CPU 空场景测试中，实际设置为
`/persistent/physics/numThreads=1`、
`/physics/physxDispatcher=true`、
Carbonite threadCount=20 时，`SimulationContext.reset()` 40 秒未返回。
堆栈停在 `SimulationManager.initialize_physics -> _physx_interface.update_simulation`。
同一源程序仅将实际 numThreads 改为 0 或 8，reset 和 3 个物理步均通过。
0/8 空场景另有默认 app.close 阶段超时，不能误报为所有关闭问题均已修复。

### Context
- 我此前发起的 CPU 评估命令明确包含
  `--kit_args=--/plugins/carb.tasking.plugin/threadCount=2 --/persistent/physics/numThreads=1`，
  未传 `--/app/settings/persistent=false`。随后 server 用户配置中的值为 1，重启后仍保留。
- 操作错误是把可保存的用户偏好当成临时诊断参数，并缺少事前配置快照、实际值核验及事后字节检查。
  没有故障前快照，不能严格断言 1 首次由哪次启动写入，或声称 server 故障前一定为 8。
- 用户提供 younghit 机器该项为 8、其启动日志确认加载配置且启动成功的只读核验结果，
  并明确批准 server 仅将 1 改为 8。这是用户提供的跨机参考，未从 server 独立访问 younghit。
- 环境：conda `isaacsim-5.1`，IsaacLab 2.3.2，PhysX Core 107.3.26；
  wheel task、腿部延迟 0..3、参考 NPZ 和策略导出文件不属于本次修改范围。

### Suggested Fix
临时 Isaac Sim 诊断应在启动前禁用偏好保存，定位 Applied configs 指向的真实配置文件，
保存原始字节与 SHA-256，并在退出后确认不变。该开关仍会加载既有偏好，
因此必须打印实际 numThreads、dispatcher 等值，不能只依据 CLI 参数推断。
持久修复须获得用户授权、备份、只改目标字段，并用无线程覆盖的新进程复核。
初始化诊断不应运行会加载/导出策略的完整 play 流程。

### Metadata
- Reproducible: yes
- Related Files: /home/server/miniconda3/envs/isaacsim-5.1/lib/python3.11/site-packages/isaacsim/kit/data/Kit/Isaac-Sim/5.1/user.config.json; learnings/diagnostics/physx-initialization-20261022-1700/summary.json; learnings/diagnostics/physx-preference-repair-20261022-1724/repair.json; learnings/diagnostics/physx-preference-repair-20261022-1724/wheel-env-from-repaired-config.log; learnings/policy_tuning/RobotLab-Isaac-BeyondMimic-Flat-LW-wheel-v0/2026-10-22_11-01-24/evaluations/wheel-to-leg-live-cpu-review-20261022-001/raw/play-equivalent-a01/console.log
- See Also: LRN-20261022-001

### Resolution
- **Resolved**: 2026-10-22T17:28:04.302329+08:00
- **Commit/PR**: N/A
- **Notes**: 按用户明确授权，在启动日志确认的 server user.config.json 中仅将
  persistent.physics.numThreads 从 1 改为 8；只改变 1 个字节，保留其他字节、权限、
  属主和 inode。原始文件备份为 `learnings/diagnostics/physx-preference-repair-20261022-1724/user.config.before.json`。
  SHA-256 从 `993f07e8c0a9d1d19cb8c8a10719c0e12bf1950021bcff6b5f2e2a8252a301ea`
  变为 `1ae98b6b715be429ab10ffb13fcfbd10bcc1497675297b02b09067ff1d7409dc`。
  全新进程不传线程覆盖参数、也不调用线程 setter，直接从该配置读取实际值 8；
  1 个 CUDA wheel 环境构造用时 9.796 秒、
  reset 用时 0.205 秒，3 步观测和奖励均有限，
  无终止/超时，子进程在 15.344 秒内以 rc=0 退出。
  验证使用 app.close(wait_for_replicator=False)，未加载或导出策略；
  配置在验证前后字节和修改时间不变，受保护源文件及旧证据哈希不变。
  原生线程依赖为何在 1 下阻塞仍未定位，本记录不推断 SDK 历史默认值。

---
