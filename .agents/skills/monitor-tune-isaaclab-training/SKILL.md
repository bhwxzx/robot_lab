---
name: monitor-tune-isaaclab-training
description: Supervise, diagnose, recover, execute bounded multi-seed tuning, robustly rank, closed-loop evaluate, safely archive, and improve IsaacLab policies from supervised physical-deployment feedback across RSL-RL and other backends through versioned algorithm profiles and an explicit per-run authorization contract. Use for training watchdogs, safe checkpoint resume, staged seed trials, effective-config gates, learning-quality anomaly detection, multi-metric tuning, Play and Native/JIT/ONNX deployment-artifact tests, video-based motion review, final-policy promotion gates, qualified JIT/ONNX policy storage, real-robot feedback diagnosis, feedback-driven retuning proposals, new-algorithm discovery, or approval-gated skill profile upgrades.
---

# Monitor and Tune IsaacLab Training

Operate every training session under one explicit mode:

- `monitor`: supervise and recover only; never change training parameters.
- `tune`: supervise and recover, then run bounded experiments using only parameters authorized for this session.

Treat authorization as non-transferable. Never inherit mode, parameter choices, restart limits, algorithm identity, profile version, or cleanup permission from another run.

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

Create a version 6 session JSON document for automated staged trial execution
and robust multi-seed ranking. Version 6 also supports physical feedback.
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
- in version-6 `tune` mode, screening/confirmation seeds, paired-baseline and
  Pareto rules, exact child argv, state directory, GPU exclusivity, retry
  budget, effective-config baseline, and learning-quality stop rules.
- for final selection, exact Native/deployment artifacts, evaluation commands,
  scenarios, seeds, runtime overrides, gates, videos, and retuning authority.
- for archival, the exact storage root and collection, JIT and ONNX formats,
  recorded training source Git state, description notes, clean-worktree
  requirement, and no Git action.
- for physical feedback, proposal-only or pending-draft mode, exact output
  directory, archive-manifest and artifact-hash binding, safety-stop behavior,
  and mandatory approval of a new session before any trial.

Validate the profile registry and session:

```bash
conda run -n isaacsim-5.1 python \
  .agents/skills/monitor-tune-isaaclab-training/scripts/validate_algorithm_profiles.py

conda run -n isaacsim-5.1 python \
  .agents/skills/monitor-tune-isaaclab-training/scripts/validate_session_spec.py \
  SESSION.json
```

Stop on validation failure. Do not silently select a nearby profile or relax a rule.

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

Read [references/tuning-policy.md](references/tuning-policy.md). For version-6
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

For version 6, initialize the hash-bound state and launch no more than one
authorized child at a time:

```bash
conda run -n isaacsim-5.1 python \
  .agents/skills/monitor-tune-isaaclab-training/scripts/execute_trial_plan.py \
  SESSION.json TRIAL_PLAN.json --action initialize
```

Use `--action launch-next` only when reconcile reports no active child, and use
`--action reconcile` on each scheduled check. Require exact session-plan
matching, unique per-stage run IDs, an idle approved GPU, the inherited GPU
lock, exact Linux process identity, attempt-isolated outputs, finite structured
results, and a passing effective-config diff. A quality-stop rule may signal
only that executor's exact recorded process group.

Run an unchanged baseline in the same seed stage as every candidate. Isolate
trial and retry outputs. Use structured argv and verified configuration
overrides. If a parameter requires editing tracked code or data, pause and use
the repository modification workflow.

Require an exact double unlock for every parameter matched by the resolved
profile's protected patterns. Run every seed required by the applicable stage
and stop on non-finite state, crashes, hard constraints, or budget exhaustion.

In version 6, use the fixed screening seeds for all trials, then run the
remaining confirmation seeds only for the baseline and approved top-k
candidates. Do not reuse a single RNG continuation as independent seed
evidence.

Rank results:

```bash
conda run -n isaacsim-5.1 python \
  .agents/skills/monitor-tune-isaaclab-training/scripts/rank_trials.py \
  SESSION.json RESULTS.json
```

Do not choose from total reward alone. Require each-seed constraints, identical
paired baseline seeds, dispersion and confidence evidence, minimum-improvement
gates when declared, and Pareto membership before weighted ordering. Treat
simulation tuning as candidate selection, not real-robot readiness.

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
5. recorded motion and an actual visual review with notes.

Consolidate and validate:

```bash
conda run -n isaacsim-5.1 python \
  .agents/skills/monitor-tune-isaaclab-training/scripts/collect_evaluation_results.py \
  EVALUATION_PLAN.json --visual-reviews VISUAL_REVIEWS.json \
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
when the approved version-4, version-5, or version-6 tune session enables
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

Create one timestamped directory atomically with `policy.pt`, `policy.onnx`,
`策略说明.txt`, and `archive_manifest.json`. State that the policy passed
simulation and is eligible only for supervised hardware testing. Never call it
hardware-ready. Do not commit or push the policy-storage repository unless the
user separately authorizes those Git actions.

## Adjust tuning from supervised physical feedback

Read
[references/hardware-feedback-retuning.md](references/hardware-feedback-retuning.md).
Accept only a version-1 feedback record under an approved version-5 or
version-6
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
