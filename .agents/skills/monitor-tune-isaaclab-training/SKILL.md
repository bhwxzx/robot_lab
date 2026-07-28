---
name: monitor-tune-isaaclab-training
description: Configure, supervise, diagnose, recover, distribute, execute, orchestrate, rank, closed-loop evaluate, safely archive, and improve IsaacLab policies from supervised physical-deployment feedback across RSL-RL and other backends through versioned algorithm profiles and explicit authorization. Use for first-run machine, Conda, GPU, path, policy-storage, or HTTPS Git-mailbox setup; training watchdogs; safe checkpoint resume; shadow or executable campaign control; bounded local-W&B history priors, adaptive fixed-seed rounds, or synchronous multi-fidelity training; bounded fixed-single-seed or multi-seed tuning on one or multiple Git-connected computers; effective-config gates; learning-quality anomaly detection; Play and Native/JIT/ONNX deployment-artifact tests; video-based motion review; bounded real-robot qualification; final-policy promotion; qualified policy storage; real-robot feedback retuning; new-algorithm discovery; or approval-gated profile upgrades.
---

# Monitor and Tune IsaacLab Training

Operate every training session under one explicit mode:

- `monitor`: supervise and recover only; never change training parameters.
- `tune`: supervise and recover, then run bounded experiments using only parameters authorized for this session.

Treat authorization as non-transferable. Never inherit mode, parameter choices, restart limits, algorithm identity, profile version, or cleanup permission from another run.

## Complete first-run configuration

Before the first use on each computer, read
[references/first-run-configuration.md](references/first-run-configuration.md).
Run `configure_skill.py locate` to find the versioned machine-local
`configuration.json` and `setup_receipt.json` outside the source worktree. If
either is absent, changed, or stale, run `configure_skill.py plan`, present the
exact operations, discovery path, and plan SHA-256, and pause for explicit
approval. Run `apply` only with that approved hash, then run `verify`.

Do not begin monitoring, training, evaluation, archival, physical feedback, or
Git-mailbox publication unless first-run verification reports the required
local source, `isaacsim-5.1` environment, GPU, output paths, and optional
mailbox clone. For multi-host use, require the same private HTTPS remote,
machine table, and unique branches on all computers. The setup tool may create
approved local directories and clone an existing initialized remote; it never
creates a provider repository, stores credentials, pushes, overwrites, resets,
stashes, deletes, installs packages, or grants session authority.

## Resolve the algorithm before approval

Inspect repository instructions, the exact training entry point, task registry, dumped effective configuration, runner, algorithm class, log format, checkpoint layout, and available monitoring tools.

Read [references/algorithm-profile-schema.md](references/algorithm-profile-schema.md). For a known algorithm, resolve the most specific entry from `references/algorithm-profiles.json`. For an unknown algorithm, create a draft session and run:

```bash
conda run -n isaacsim-5.1 python \
  .agents/skills/monitor-tune-isaaclab-training/scripts/scan_algorithm_coverage.py

conda run -n isaacsim-5.1 python \
  .agents/skills/monitor-tune-isaaclab-training/scripts/discover_algorithm_profile.py \
  DRAFT_SESSION.json --config EFFECTIVE_CONFIG --log TRAINING_LOG
```

Use a generic profile for monitor-only work. Never tune or promote a final
strategy through a generic profile. When discovery returns a candidate, inspect
its progress semantics, metrics, checkpoint state, resume behavior, protected
parameters, evaluation runner coverage, deployment tensor contract, and
smoke-test needs before proposing a persistent profile upgrade.

## Establish the session contract

Create a version 7 session JSON document for approved multi-host Git-mailbox
execution. Use version 6 for automated staged trial execution on one host.
Versions 6 and 7 support either robust multi-seed ranking or explicit
fixed-single-seed selection whose final authority is supervised hardware.
They also support physical feedback.
Version 5 remains valid for physical-feedback workflows without the new
executor, version 4 remains valid for evaluation and archival without hardware
feedback, and version 3 remains valid only for legacy sessions without policy
archival.
Use [references/session-spec.md](references/session-spec.md). Require the user
to approve the session before starting, attaching, recovering, tuning,
evaluating, archiving, or preparing a feedback-driven tuning draft. Require:

- exact backend, algorithm, runner, profile ID, profile version, and fingerprint;
- mode, commands, working directory, log, optional TensorBoard source, PID, and GPU;
- check interval, stale threshold, restart limit, and cleanup permission;
- in `tune` mode, every parameter path, domain, seed, trial budget, objective, and constraint.
- in version-6-or-7 `tune` mode, the explicit seed-strategy mode,
  screening/confirmation seeds, paired-baseline and Pareto rules, exact child
  argv, state directory, GPU exclusivity, retry budget, effective-config
  baseline, and learning-quality stop rules.
- for history-informed adaptive search, the exact local W&B roots and project,
  parameter/metric mappings, time/run/point limits, first-round influence cap,
  source/context compatibility policy, progress/stability gates, round count,
  trials per round, exploration fraction, and early-stop thresholds.
- for synchronous multi-fidelity training, exact increasing rung budgets,
  promotion targets, objective margin, conservative pruning delays, and
  adapter-reviewed budget/resume/checkpoint mappings.
- for campaign control, shadow or execute mode, single-host or distributed
  role, all worker mailbox paths, and a mandatory stop before evaluation.
- in version-7 `tune` mode, a dedicated HTTPS coordination repository,
  coordinator/worker branches, `by_seed` or fixed-seed `by_trial` assignment,
  clean source commit, per-worker paths/GPU, an explicit host-calibration
  choice, poll/grace intervals, metadata-only artifact exchange, and, when
  sharing policy storage, one coordinator-granted archive lease with exact
  per-worker clones and a common storage remote/branch.
- for final selection, exact Native/deployment artifacts, evaluation commands,
  scenarios, seeds, runtime overrides, gates, videos, and retuning authority.
- for archival, the exact storage root and collection, JIT and ONNX formats,
  recorded training source Git state, description notes, clean-worktree
  requirement, and no Git action.
- for physical feedback, proposal-only or pending-draft mode, exact output
  directory, archive-manifest and artifact-hash binding, safety-stop behavior,
  and mandatory approval of a new session before any trial. When physical
  evidence is final authority, also approve the repeated-test count, required
  scenarios, telemetry channels, and bounded qualification label.

Validate the profile registry and session:

```bash
conda run -n isaacsim-5.1 python \
  .agents/skills/monitor-tune-isaaclab-training/scripts/validate_algorithm_profiles.py

conda run -n isaacsim-5.1 python \
  .agents/skills/monitor-tune-isaaclab-training/scripts/validate_session_spec.py \
  SESSION.json
```

Stop on validation failure. Do not silently select a nearby profile or relax a rule.

## Distribute trials only when authorized

For version 7, read
[references/distributed-git-mailbox.md](references/distributed-git-mailbox.md).
Use a dedicated private coordination repository rather than the source
repository. Publish immutable jobs on the coordinator branch and let each
worker publish receipts, progress, and results only on its own branch.

Require a successful remote claim before local execution. In `by_seed` mode,
keep every seed's baseline and candidates on one worker. In
`fixed_single_seed` mode, give every worker the same seed and distribute
candidate trials deterministically with `by_trial`; publish each exact
seed-and-overrides combination once across the campaign. Default host-effect
calibration to disabled so reward-weight and parameter search does not repeat
identical work. Record that host effects are uncontrolled by design and make
no host-invariance claim. Only after explicit approval, enable a separate
unchanged calibration baseline on every host to diagnose machine effects; do
not include those jobs in candidate ranking. Refuse dirty or mismatched source
worktrees, invalid assignments, changed immutable events, embedded
credentials, large artifacts, duplicate run IDs, and result/hash mismatches.
Treat stale Git progress after a claim as `remote_state_unknown`, not proof of
a dead run; never reassign or stop it without independent local evidence.
Materialize the remotely published claim with `git_mailbox.py prepare-job`,
then pass it with the exact worker ID to `execute_trial_plan.py
--distributed-job`; never run the unfiltered full plan on a worker.

## Start or attach to training

Use `conda run -n isaacsim-5.1` for robot_lab IsaacLab commands. Start new training in the background with stdout and stderr redirected to the approved log. Record PID, process group, argv, run directory, checkpoint, and initial progress snapshot.

Prefer a recurring scheduler or task-monitoring tool. If no persistent scheduler exists, say that autonomous recurring monitoring cannot be guaranteed; do not emulate it with a long blocking sleep.

Do not assume wrapper scripts accept resume flags. Use only the exact approved backend-specific resume argv.

## Supervise and recover

Follow [references/monitoring-and-recovery.md](references/monitoring-and-recovery.md). On every check:

1. Run `collect_training_health.py` with the resolved profile.
2. Pass the previous log progress, TensorBoard step, and observation time.
3. Parse metrics with `summarize_training_log.py`.
4. Store the new progress snapshot for the next check.
5. Recover only after independent evidence confirms an incomplete stalled run.

Treat monotonic training progress or recent TensorBoard scalar time as primary evidence. Process existence, GPU utilization, checkpoint time, and `.wandb` transaction-file growth are auxiliary evidence only.

Never kill a healthy, observing, suspect, or merely slow run. Before recovery, resolve the exact PID/process group and readable checkpoint. Use the approved resume command unchanged, honor cooldown and restart limits, and preserve prior logs and checkpoints.

## Tune only when authorized

Skip this section in `monitor` mode. Reject `tune` mode when the selected profile is generic.

Read [references/tuning-policy.md](references/tuning-policy.md). For version-6-or-7
execution also read
[references/execution-and-robust-ranking.md](references/execution-and-robust-ranking.md).
Resolve each requested parameter to its current source, effective value, type,
override mechanism, and profile-specific risk. Let the user choose paths and
domains. Suggestions do not grant permission.

Generate a deterministic plan:

```bash
conda run -n isaacsim-5.1 python \
  .agents/skills/monitor-tune-isaaclab-training/scripts/build_trial_plan.py \
  SESSION.json --output TRIAL_PLAN.json
```

When the approved fixed-seed session enables `history_prior` and
`adaptive_search`, first read
[references/history-informed-adaptive-search.md](references/history-informed-adaptive-search.md).
Index only the bounded local W&B evidence, merge it without cloud access, and
pass the hash-bound prior with `--history-prior`. Never widen parameter domains,
reuse an exact historical combination, or let history supply more than the
approved half of first-round candidates. Add a later round only after every
existing trial has one valid completed result; the plan must be an append-only
deterministic expansion. Preserve the hash-bound stop decision when progress,
feasibility, improvement patience, budget, grid, or round limits say not to
publish more work.

When the approved fixed-seed session enables `multi_fidelity`, read
[references/multi-fidelity-training.md](references/multi-fidelity-training.md).
Wait for every active run at the same rung, eliminate hard-constraint failures
immediately, and performance-prune only after the approved minimum rungs and
consecutive underperformance. Preserve the baseline, exact parent checkpoint
hash/step, and same-worker affinity. Expand the plan only through
`build_multifidelity_rung.py`; terminal decisions create no new run.

When the session enables `campaign_controller`, read
[references/campaign-controller.md](references/campaign-controller.md).
Use `status` for a read-only next-action report. Use `advance` only with
approved `mode=execute`; it may perform one exact transition and must stop
after training ranking with `evaluation_required`.

When the session enables `policy_export` or `evaluation_handoff`, read
[references/policy-export.md](references/policy-export.md) and
[references/evaluation-handoff-controller.md](references/evaluation-handoff-controller.md).
Require separate execute permission, exact Native/JIT/ONNX parity, the
designated worker, and stop at `awaiting_visual_review`.

For version 6 or 7, initialize the hash-bound state and launch no more than one
authorized child at a time:

```bash
conda run -n isaacsim-5.1 python \
  .agents/skills/monitor-tune-isaaclab-training/scripts/execute_trial_plan.py \
  SESSION.json TRIAL_PLAN.json --action initialize
```

Use `--action launch-next` only when reconcile reports no active child, and use
`--action reconcile` on each scheduled check. Require exact session-plan
matching, unique per-stage run IDs, an idle approved GPU, the inherited GPU
lock, state-transition lock, disk and temperature preflight, total and
per-trial time limits, exact Linux process identity, attempt-isolated outputs,
finite structured results, and a passing effective-config diff. A quality or
timeout stop may signal only that executor's exact recorded process group,
using SIGTERM before the approved SIGKILL grace threshold. Recover a damaged
state only from the verified hash-chained execution journal.

For `rsl_rl`, use the included `rsl_rl_trial_adapter.py`. Bind every approved
tuning path to its reviewed Hydra path, and bind dumped seed/run-name fields as
runtime identity fields rather than tuning parameters. Require its terminal
receipt and checkpoint hash. If approved, only the first unchanged baseline
may bootstrap an absent effective-config baseline; never overwrite it.

Require the adapter to stream an atomic rolling summary after each complete
iteration and immediately on non-finite metrics. Use an approved
`minimum_progress` for finite quality rules that must ignore warm-up. Record
TensorBoard scalar progress as secondary evidence; never let GPU activity or a
fresh event file override stalled monotonic training progress.

Treat launch as a journaled two-phase transaction: reserve an isolated attempt,
write its reproducibility evidence, start the exact child, then persist a
hash-bound launch receipt. Consume failed starts without reusing their output
directories. Recover receipt-backed exact processes after scheduler failure,
but block and never signal an orphan with no valid receipt. Accept only a
truncated final journal record during explicit state recovery.

When reproducibility capture is enabled, bind each run to source Git state,
exact argv, profile, runtime/package versions, GPU identity, and explicitly
approved critical input hashes. Reject completion if that manifest changes.

Run an unchanged baseline in the same seed stage as every candidate. Isolate
trial and retry outputs. Use structured argv and verified configuration
overrides. If a parameter requires editing tracked code or data, pause and use
the repository modification workflow.

Require an exact double unlock for every parameter matched by the resolved
profile's protected patterns. Run every seed required by the applicable stage
and stop on non-finite state, crashes, hard constraints, or budget exhaustion.

In robust multi-seed mode, use the fixed screening seeds for all trials, then
run the remaining confirmation seeds only for the baseline and approved top-k
candidates. In `fixed_single_seed` mode, run that same exact seed for every
baseline and candidate; after screening, record the top-k selection without
creating confirmation-seed jobs. Do not describe this as robust multi-seed
evidence.

Rank results:

```bash
conda run -n isaacsim-5.1 python \
  .agents/skills/monitor-tune-isaaclab-training/scripts/rank_trials.py \
  SESSION.json RESULTS.json
```

Do not choose from total reward alone. Require each-seed constraints, identical
paired baseline seeds, minimum-improvement gates when declared, and Pareto
membership before weighted ordering. Require dispersion and confidence
evidence only in robust multi-seed mode. Label fixed-seed output
`single_seed_selected`, keep `generalization_claim=false`, and treat simulation
tuning as candidate selection, not real-robot readiness.

## Evaluate motion before final selection

Training curves can rank candidates but cannot produce a final policy. Read
[references/policy-evaluation.md](references/policy-evaluation.md). Require a
reviewed non-generic profile and inspect the current Play, export, observation,
normalization, history, state-reset, and deployment-runtime paths.

Build the approved candidate/artifact/scenario/seed matrix:

```bash
conda run -n isaacsim-5.1 python \
  .agents/skills/monitor-tune-isaaclab-training/scripts/build_evaluation_plan.py \
  SESSION.json CANDIDATES.json --output EVALUATION_PLAN.json
```

When the approved session contains `evaluation.execution`, initialize and
advance the matrix with:

```bash
conda run -n isaacsim-5.1 python \
  .agents/skills/monitor-tune-isaaclab-training/scripts/execute_evaluation_plan.py \
  SESSION.json EVALUATION_PLAN.json --action initialize
```

Use `--action launch-next` for one cell and `--action reconcile` on each
scheduled check. Require attempt isolation, a shared training/evaluation GPU
lock, exact process identity, timeout escalation, checkpoint/artifact
revalidation, finalized-video size, canonical result/video hashes, and
hash-chained state recovery. Never treat `awaiting_visual_review` as qualified.

Use the training task for stress evaluation. Do not rely on a Play-only
configuration that disables corruption, forces, pushes, or dynamics
randomization. For RSL-RL, use
`scripts/reinforcement_learning/rsl_rl/evaluate_policy.py` to run the exact
Native, JIT, or ONNX artifact in the closed simulation loop. Other backends
must supply a reviewed evaluator command through their specific profile.
Before launch, verify the authorized GPU is idle. Never overlap evaluation with
an active training process unless a separate approved contract explicitly
allows the resource interference.

Require:

1. one required nominal scenario and at least one required stress scenario;
2. Native plus at least one supported deployment artifact;
3. finite closed-loop metrics and every approved hard gate;
4. deployment-artifact action parity against Native inference;
5. approved closed-loop metric deltas against the matching Native
   scenario/seed;
6. recorded motion, peak-step review windows, and an actual visual review with
   notes.

Consolidate and validate:

```bash
conda run -n isaacsim-5.1 python \
  .agents/skills/monitor-tune-isaaclab-training/scripts/collect_evaluation_results.py \
  EVALUATION_PLAN.json \
  --execution-state EVALUATION_STATE.json \
  --visual-reviews VISUAL_REVIEWS.json \
  --output EVALUATION_RESULTS.json

conda run -n isaacsim-5.1 python \
  .agents/skills/monitor-tune-isaaclab-training/scripts/validate_policy_evaluation.py \
  SESSION.json EVALUATION_PLAN.json EVALUATION_RESULTS.json
```

Pass policy-evaluation results to `rank_trials.py`. Without a complete passing
evaluation, leave `final_selection` null. A passing simulation result is only
`simulation_qualified_hardware_candidate`; never label it hardware-ready.
Evaluation-triggered retuning requires tune mode, remaining budget,
`allow_retune_on_failure=true`, and an already-authorized parameter.

## Archive a qualified deployment candidate

Read [references/policy-archive.md](references/policy-archive.md). Archive only
when the approved version-4, version-5, version-6, or version-7 tune session enables
`archive`, both
JIT and ONNX are required evaluation artifacts, and final ranking reports
`simulation_qualified_hardware_candidate`.

Inspect the destination without changing it:

```bash
conda run -n isaacsim-5.1 python \
  .agents/skills/monitor-tune-isaaclab-training/scripts/inspect_policy_storage.py \
  /absolute/policy_storage --hash-artifacts
```

Require the exact storage Git worktree to be clean. Refuse symlinked artifacts,
changed hashes, duplicate artifact pairs, missing collections, collisions, or
incomplete promotion evidence. Then run:

```bash
conda run -n isaacsim-5.1 python \
  .agents/skills/monitor-tune-isaaclab-training/scripts/archive_policy_candidate.py \
  SESSION.json TRAINING_RESULTS.json EVALUATION_PLAN.json \
  EVALUATION_RESULTS.json --output /absolute/ARCHIVE_RECEIPT.json
```

For a version-7 shared policy-storage repository, first build and publish the
hash-bound archive request, let the coordinator grant the only active lease,
and materialize that grant. Pass the exact worker ID and grant to the archiver.
Read [references/policy-archive.md](references/policy-archive.md) for the
commands and state machine. Never infer lease ownership from time, a local lock,
or a stale remote observation. Only explicit coordinator release or approved
revoke closes a lease.

Create one timestamped directory atomically with `policy.pt`, `policy.onnx`,
`策略说明.txt`, and `archive_manifest.json`. State that the policy passed
simulation and is eligible only for supervised hardware testing. Never call it
hardware-ready. Do not commit or push the policy-storage repository unless the
user separately authorizes those Git actions. A shared-storage lease can be
completed only after that separately approved commit is the exact remote branch
head; release it only after the coordinator verifies the completion evidence.

## Adjust tuning from supervised physical feedback

Read
[references/hardware-feedback-retuning.md](references/hardware-feedback-retuning.md).
Accept only a version-1 feedback record under an approved version-5 or
version-6 or version-7
`hardware_feedback` contract. Bind it to the exact archive manifest, candidate,
JIT/ONNX hashes, deployment configuration, robot, firmware, control rate,
supervised test envelope, observation segments, safety outcomes, and available
video or telemetry.

Validate before interpretation:

```bash
conda run -n isaacsim-5.1 python \
  .agents/skills/monitor-tune-isaaclab-training/scripts/validate_hardware_feedback.py \
  SESSION.json HARDWARE_FEEDBACK.json \
  --output /absolute/approved/output/HARDWARE_FEEDBACK_VALIDATION.json
```

When the session authorizes supervised hardware as final authority, validate
each separately recorded test first, then aggregate the repeated-test matrix:

```bash
conda run -n isaacsim-5.1 python \
  .agents/skills/monitor-tune-isaaclab-training/scripts/validate_hardware_qualification.py \
  SESSION.json HARDWARE_QUALIFICATION_BUNDLE.json \
  --output /absolute/approved/output/HARDWARE_QUALIFICATION.json
```

Accept only `hardware_validated_for_test_envelope`. It binds one exact
artifact/deployment identity, unique test times and evidence files, required
scenario coverage, high-confidence video/telemetry, all-pass assessments, and
zero safety events. It deliberately keeps `hardware_ready=false` and
`generalization_claim=false`; validity ends at the recorded physical envelope.

On emergency stop, fall, limit violation, communication timeout, damage,
critical observation, or an unsafe user assessment, stop physical testing and
diagnose first. Also diagnose before tuning when the deployed configuration,
observation contract, history initialization, emergency stop, artifact hashes,
runtime timing, calibration, or mechanism is unverified.

For a possible training or Sim-to-Real gap, reproduce the reported segment in
closed-loop simulation and define measurable objectives and hard constraints.
Then build a non-executable proposal:

```bash
conda run -n isaacsim-5.1 python \
  .agents/skills/monitor-tune-isaaclab-training/scripts/build_feedback_retune_proposal.py \
  SESSION.json HARDWARE_FEEDBACK.json \
  --output /absolute/approved/output/RETUNE_PROPOSAL.json
```

Subjective-only feedback may guide evidence collection but cannot authorize
retuning. `proposal_only` never emits a parameter-choice draft.
`prepare_authorized_draft` may list only relevant paths and domains already in
the approved `tuning.allowed_parameters`; it leaves the final path selection
empty. Require the user to choose paths, ranges, objectives, constraints,
seeds, budget, and evaluation gates, then approve a new session. Never modify a
running plan or launch a feedback-driven trial from the proposal.

## Upgrade for a new algorithm

Keep runtime adaptation separate from persistent self-modification:

1. Detect that only a generic profile matches.
2. Generate a candidate profile without editing the registry.
3. Validate its identity and raw metric aliases.
4. Inspect algorithm-specific checkpoint, resume, progress, Play, deployment
   artifact, history/normalization, risk, and smoke-test contracts.
5. Present an exact registry modification plan.
6. Apply only after explicit approval.
7. Validate the registry, parsers, session contract, representative logs, and smoke behavior.
8. Forward-test the upgraded skill on a fresh task.

Never let a new algorithm automatically rewrite `SKILL.md`, scripts, or the registry without an approved diff. Automatic candidate generation is allowed; persistent upgrade is approval-gated.

## Preserve and report

Do not install packages or delete pre-existing files, logs, checkpoints, or user changes. Create temporary artifacts outside the repository and remove only those covered by cleanup permission.

Report run identity, profile, progress evidence, state, recovery count,
authorized parameters, budget, constraint failures, training ranking,
evaluation matrix coverage, parity, metric gates, video-review evidence,
simulation-promotion state, archive receipt, storage Git state, uncertainty,
artifact paths, feedback evidence confidence, root-cause classification,
safety stop, proposed metrics, and pending authorization decisions. Report
generic-profile limitations explicitly. Keep real-robot readiness separate and
require supervised hardware telemetry and safety tests.
