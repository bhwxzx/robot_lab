# Session handoff: IsaacLab automated policy tuning skill

## Handoff metadata

- Created: `2026-07-31T15:37:50+08:00`
- Repository root: `/home/young/liufengrong/robot_lab`
- Branch and HEAD at live check: `main`, `83e6540`
- Local tracking state at live check: `main` was 3 commits ahead of the locally
  recorded `origin/main`; no network fetch was performed, so remote state may
  have changed.
- Trigger: the user stated that this long session had undergone too much context
  compaction and requested a durable new-session handoff.
- Compaction count: unavailable; no exact count is claimed.
- Evidence boundary: Git, process, GPU, local configuration-file existence, and
  worktree facts below were checked live. Older PC B and remote coordination
  facts are conversation-derived and explicitly marked unverified.

## Current objective

Maintain and continue improving
`.agents/skills/monitor-tune-isaaclab-training` as an algorithm-extensible,
approval-gated system for IsaacLab training supervision, recovery, tuning,
simulation/deployment evaluation, policy archiving, real-robot feedback, and
two-computer Git-mailbox coordination.

The immediate task in this session was to add a durable rule that creates a
verified handoff document and a new-session prompt after repeated context
compaction. That implementation is complete but uncommitted at handoff time.

## User decisions and protected constraints

- The user chooses between supervision/recovery-only mode and tuning-authorized
  mode, including which parameters may be tuned.
- The skill must support algorithms beyond AMP-ROA and safely accommodate new
  algorithm profiles.
- Use fixed seed `42` by default. Do not require a final multi-seed ensemble or
  duplicate identical seed/parameter runs across both computers unless the user
  later authorizes a host-effect experiment.
- Split dual-host work by trial/parameter combination. Host invariance is not a
  default claim.
- Do not add MuJoCo validation for now.
- Simulation curves alone are insufficient. Final evaluation must include Play
  and closed-loop Native/JIT/ONNX evidence, human video review, and ultimately
  supervised real-robot feedback before hardware-readiness claims.
- A deployable policy may be promoted to the shared `policy_storage` only after
  qualification, with ONNX and PT/JIT artifacts plus a policy description.
- Real-robot deployment feedback may change later tuning rounds.
- Do not overlap camera/evaluation work with active training on the same GPU
  unless separately approved.
- Use Conda environment `isaacsim-5.1`.
- Propose a modification plan and obtain explicit approval before code edits.
- Do not store Git credentials, usernames/passwords, or PATs in project
  configuration or handoff documents.
- Do not push, deploy, start real training, perform remote writes, or remove
  files merely because they are mentioned in this handoff.

## Completed work

- Built the `monitor-tune-isaaclab-training` skill through multiple
  approval-gated rounds: algorithm profiles, supervision/recovery, bounded
  tuning, fixed-seed campaigns, local W&B history priors, executable RSL-RL
  campaign control, two-host Git mailbox metadata, shared policy-storage leases,
  evaluation handoff, artifact parity, and hardware-feedback retuning.
- Configured Git Credential Manager on both computers according to the
  conversation. PC B applied its version-2 first-run configuration and performed
  a read-only verify, but reported `ready_for_training=false` because its
  `robot_lab` source worktree was dirty.
- Recorded that the real two-host Git-mailbox transaction still needs continued
  validation in commit `7f2b8b4`.
- Added the hash-bound human visual-review bundle in commit `6cb00a8`. It keeps
  canonical `motion.mp4` files unchanged and creates readable relative symlinks
  containing candidate, artifact, scenario, and seed under
  `<evaluation.output_dir>/review/required_videos/`.
- Ignored the VS Code C/C++ browse database in commit `83e6540`; the database
  remains a regenerable local IntelliSense cache.
- The visual-review implementation passed 107 complete skill tests, the skill
  structure validator, and validation of 11 algorithm profiles before commit.
- A previous AMP-ROA evaluation attempt was abandoned and its temporary work was
  cleaned after user approval. Camera-follow recording remains deferred for a
  no-training period.

## Live-verified state

Live checks were performed at `2026-07-31T15:37:50+08:00`.

- Repository HEAD: `83e6540 chore: ignore VS Code browse database`.
- Local branch view: `main...origin/main [ahead 3]`. This is based on the local
  remote-tracking ref, not a fresh network fetch.
- The only pre-existing dirty files before this handoff implementation were:
  - `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/LW/LW_Leg/rough_env_cfg.py`
  - `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/terrains/terrains_cfg.py`
- Their combined diff was 13 insertions and 13 deletions. Treat both as
  user-owned and do not stage, overwrite, revert, stash, or reinterpret them
  without a separate request.
- No `train.py`, `evaluate_policy.py`, or `play.py` workload was found. The
  process query matched only its own inspection shell.
- GPU 0 was an NVIDIA GeForce RTX 4080 SUPER at 27% utilization, 264 MiB of
  16376 MiB memory, and 41 C. This is only a point-in-time observation, not
  authorization to launch GPU work.
- PC A first-run files existed at:
  - `/home/young/.config/robot-lab/monitor-tune-isaaclab-training/configuration.json`
  - `/home/young/.config/robot-lab/monitor-tune-isaaclab-training/setup_receipt.json`
- No files were listed within two levels of
  `/home/young/liufengrong/robot_evaluation` during the live check.
- This handoff feature adds uncommitted changes to `AGENTS.md`,
  `.agents/skills/self-improvement/SKILL.md`,
  `.agents/skills/self-improvement/references/session-handoff.md`,
  `.learnings/FEATURE_REQUESTS.md`, and this document.

## Pending and unverified work

1. Validate and, only when requested, commit the five session-handoff files
   listed above without including the two user-owned training configuration
   files.
2. The actual bidirectional two-host Git-mailbox exercise remains pending.
   PC B's reported dirty-source blocker and all PC B state are
   conversation-derived as of July 29 and must be rechecked on that computer.
3. The shared `policy_storage` lease workflow is implemented, but a real
   two-host archive transaction and remote-write behavior have not been
   revalidated in this handoff.
4. Camera-follow evaluation must be tested only during an approved no-training
   period: first use a short Native video and manually verify that environment
   0's robot remains in frame before a full Native/JIT/ONNX matrix.
5. No new AMP-ROA policy evaluation was completed after the prior evaluation
   was abandoned. Any later evaluation requires fresh checkpoint, process, GPU,
   configuration, and artifact verification.
6. Real-robot qualification remains the decisive deployment gate and requires
   a separately supervised, bounded test.

## Risks and do-not-repeat actions

- Do not restart or duplicate a training run based only on a stale conversation
  summary. Use monotonic training evidence and exact process identity.
- Do not treat Isaac Sim reward curves or video filenames as proof of hardware
  readiness.
- Do not promote a policy merely because
  `visual_reviews.draft.json` exists or remains `pending`.
- Do not copy, rename, or replace canonical evaluation videos; readable review
  names are symlinks only.
- Do not introduce multi-seed or same-configuration cross-host duplication by
  default.
- Do not add MuJoCo, push Git branches, write remote mailboxes, archive policies,
  install packages, or delete files without the required approval.
- Do not include `rough_env_cfg.py` or `terrains_cfg.py` in a handoff-feature
  commit.
- Do not assume local `origin/main` metadata is current until a permitted fetch
  verifies it.

## Recommended next steps

1. In a new session, read this document, `AGENTS.md`, the `self-improvement`
   skill, and the monitor/tune skill only as needed.
2. Recheck HEAD, worktree status, relevant processes, and the five intended
   handoff-feature paths.
3. Run the self-improvement skill validator, Markdown/diff checks, required
   heading checks, and a secret-pattern scan.
4. Report the exact validation and staging scope. Commit only after the user
   explicitly asks.
5. After that atomic task, ask whether to resume the two-host mailbox live test
   or begin the next bounded monitor/tune skill optimization.

## New-session prompt

```text
请在 /home/young/liufengrong/robot_lab 继续当前任务。先完整阅读：
1. /home/young/liufengrong/robot_lab/AGENTS.md
2. /home/young/liufengrong/robot_lab/.learnings/session_handoffs/20260731-153750-monitor-tune-skill-long-session.md
3. /home/young/liufengrong/robot_lab/.agents/skills/self-improvement/SKILL.md

先只读核对 git status、HEAD、相关训练/评估进程和交接文档中可能变化的状态，不要把旧会话内容直接当成当前事实，也不要重复已完成工作。当前首要任务是检查“多次上下文压缩后生成会话交接”的未提交实现：仅涉及 AGENTS.md、self-improvement/SKILL.md、references/session-handoff.md、.learnings/FEATURE_REQUESTS.md 和本交接文档。运行 Skill 完整性、diff、必需章节及敏感信息检查并报告结果；提交前遵守 AGENTS.md，先给出范围并等待我的明确指令。

保护现有 rough_env_cfg.py 和 terrains_cfg.py 修改，不得暂存、覆盖、回退或清理。不要自动启动训练/评估、操作远端 Git 邮箱、推送、归档策略、增加 MuJoCo/多 seed、安装或删除任何内容。完成这一原子任务后，再询问我是继续双机 Git 邮箱实测，还是开始下一轮自动调优 Skill 优化。
```
