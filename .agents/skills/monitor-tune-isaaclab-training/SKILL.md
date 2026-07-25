---
name: monitor-tune-isaaclab-training
description: Supervise, diagnose, recover, tune, and closed-loop evaluate IsaacLab policies across RSL-RL and other backends through versioned algorithm profiles and an explicit per-run authorization contract. Use for training watchdogs, safe checkpoint resume, multi-metric tuning, Play and Native/JIT/ONNX deployment-artifact tests, video-based motion review, final-policy promotion gates, new-algorithm discovery, or approval-gated skill profile upgrades.
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

Create a version 3 session JSON document from [references/session-spec.md](references/session-spec.md). Require the user to approve it before starting, attaching, recovering, tuning, or evaluating. Require:

- exact backend, algorithm, runner, profile ID, profile version, and fingerprint;
- mode, commands, working directory, log, optional TensorBoard source, PID, and GPU;
- check interval, stale threshold, restart limit, and cleanup permission;
- in `tune` mode, every parameter path, domain, seed, trial budget, objective, and constraint.
- for final selection, exact Native/deployment artifacts, evaluation commands,
  scenarios, seeds, runtime overrides, gates, videos, and retuning authority.

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

Read [references/tuning-policy.md](references/tuning-policy.md). Resolve each requested parameter to its current source, effective value, type, override mechanism, and profile-specific risk. Let the user choose paths and domains. Suggestions do not grant permission.

Generate a deterministic plan:

```bash
conda run -n isaacsim-5.1 python \
  .agents/skills/monitor-tune-isaaclab-training/scripts/build_trial_plan.py \
  SESSION.json
```

Run an unchanged baseline first. Isolate trial outputs. Use structured argv or verified configuration overrides. If a parameter requires editing tracked code or data, pause and use the repository modification workflow.

Require an exact double unlock for every parameter matched by the resolved profile's protected patterns. Run all approved seeds and stop on non-finite state, crashes, hard constraints, or budget exhaustion.

Rank results:

```bash
conda run -n isaacsim-5.1 python \
  .agents/skills/monitor-tune-isaaclab-training/scripts/rank_trials.py \
  SESSION.json RESULTS.json
```

Do not choose from total reward alone. Require every seed and constraint. Treat simulation tuning as candidate selection, not real-robot readiness.

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
simulation-promotion state, uncertainty, and artifact paths. Report
generic-profile limitations explicitly. Keep real-robot readiness separate and
require supervised hardware telemetry and safety tests.
