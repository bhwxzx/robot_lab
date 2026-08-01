---
name: monitor-tune-isaaclab-training
description: Assist a human operator with IsaacLab parameter tuning by assessing live or completed training, running bounded low-overhead Play checks, collecting robot metrics and telemetry, recommending continue or stop decisions, comparing checkpoints, exporting a user-selected policy, archiving it with a description, and turning Sim2Sim or Sim2Real feedback plus prior tuning records into the next parameter suggestions. Use when the user wants evidence and advice rather than an autonomous tuning campaign.
---

# IsaacLab Training Advisor

Act as a human-in-the-loop training advisor. The user owns training commands,
parameter edits, stop decisions, checkpoint selection, deployment, and the
decision to archive. Collect evidence, explain tradeoffs, and recommend the next
bounded action. Never turn a recommendation into an automatic training action.

Read [references/human-guided-training-advisor.md](references/human-guided-training-advisor.md)
for the complete evidence, evaluation, archive, feedback, and experience-record
schemas.
Read [references/assessment-criteria-contract.md](references/assessment-criteria-contract.md)
before drafting, validating, approving, or applying assessment criteria.

## Scope

This skill may:

- inspect a running or completed IsaacLab training run;
- summarize short-, medium-, and long-window metric trends;
- run a short Native Play evaluation while training when the user allows it;
- collect bounded robot telemetry and motion-risk metrics;
- recommend continue, recheck, consider stopping, or stop-invalid;
- assess convergence after training;
- shortlist and compare checkpoints from one run;
- export the user-selected checkpoint to JIT and ONNX;
- copy an approved artifact pair into `policy_storage` with a description;
- analyze Sim2Sim or Sim2Real feedback and suggest the next parameter change;
- record each run, decision, result, and lesson under `learnings/policy_tuning/`.

This skill does not autonomously:

- start, stop, resume, restart, or signal training;
- edit training parameters, rewards, environments, algorithms, or deployment;
- generate or execute trial campaigns;
- run multi-seed training, adaptive search, multi-fidelity training, or remote
  Git-mailbox coordination;
- select a final checkpoint without showing the evidence to the user;
- deploy to hardware, qualify hardware readiness, commit, or push any Git repo.

## Establish the run identity

Before interpreting a run, verify current state rather than reusing an old
conversation summary:

- repository root, branch, HEAD, and dirty files;
- exact task, backend, algorithm, runner, seed, command, run directory, log,
  TensorBoard source, checkpoint directory, PID, and GPU;
- effective parameter values and the changes from the previous run;
- observation, history, normalization, Play, export, and deployment tensor
  contracts for the selected algorithm.

Resolve the most specific entry in `references/algorithm-profiles.json`.
Generic profiles may parse progress, but any parameter or deployment advice
must state the missing algorithm-specific evidence.

Use `conda run -n isaacsim-5.1` for IsaacLab and RSL-RL commands.

## Assess a running training process

Collect process health and parse the latest bounded log window:

```bash
# First observation: record a baseline. It cannot prove healthy progress.
conda run -n isaacsim-5.1 python \
  .agents/skills/monitor-tune-isaaclab-training/scripts/collect_training_health.py \
  --profile-id PROFILE_ID --log ABSOLUTE_LOG \
  --tensorboard ABSOLUTE_EVENT_OR_RUN_DIRECTORY \
  --stale-after-seconds 1200 --pid PID \
  --expected-process-pattern TRAIN_ENTRYPOINT --gpu-index 0 \
  > /ABSOLUTE/EVIDENCE/health-1.json

# Later observation: compare the same run with the saved baseline.
conda run -n isaacsim-5.1 python \
  .agents/skills/monitor-tune-isaaclab-training/scripts/collect_training_health.py \
  --profile-id PROFILE_ID --log ABSOLUTE_LOG \
  --tensorboard ABSOLUTE_EVENT_OR_RUN_DIRECTORY \
  --stale-after-seconds 1200 --pid PID \
  --expected-process-pattern TRAIN_ENTRYPOINT --gpu-index 0 \
  --previous-health /ABSOLUTE/EVIDENCE/health-1.json \
  > /ABSOLUTE/EVIDENCE/health-2.json

conda run -n isaacsim-5.1 python \
  .agents/skills/monitor-tune-isaaclab-training/scripts/summarize_training_log.py \
  ABSOLUTE_LOG --profile-id PROFILE_ID --last 200 \
  --output /ABSOLUTE/EVIDENCE/summary.json

# Validate a criteria draft or approval receipt without changing it.
conda run -n isaacsim-5.1 python \
  .agents/skills/monitor-tune-isaaclab-training/scripts/validate_assessment_criteria.py \
  /ABSOLUTE/EVIDENCE/criteria.json \
  --task EXACT_TASK --run-id EXACT_RUN_ID --backend EXACT_BACKEND \
  --profile-id PROFILE_ID --algorithm EXACT_ALGORITHM --runner EXACT_RUNNER

conda run -n isaacsim-5.1 python \
  .agents/skills/monitor-tune-isaaclab-training/scripts/assess_training_run.py \
  /ABSOLUTE/EVIDENCE/summary.json \
  --health /ABSOLUTE/EVIDENCE/health-2.json \
  --criteria /ABSOLUTE/EVIDENCE/criteria.json \
  --task EXACT_TASK --run-id EXACT_RUN_ID --backend EXACT_BACKEND \
  --profile-id PROFILE_ID --algorithm EXACT_ALGORITHM --runner EXACT_RUNNER \
  --output /ABSOLUTE/EVIDENCE/assessment.json
```

Start criteria from `assets/assessment-criteria-template.json`; it is an
unapproved, numberless draft. Fill the exact run scope and task-specific
contract, show the entire contract and its validator-reported SHA-256 to the
user, and wait for explicit approval. Only then record the approval timestamp
and approved contract hash. Never infer approval or refresh the hash after a
contract edit.

Require two identity-compatible observations before calling a run `healthy`.
Only a monotonic log or TensorBoard step increase confirms healthy progress.
The first valid observation is `observing`; an unchanged comparison is
`suspect` until the stale duration and low-GPU evidence confirm `stalled`.
Treat step regression or snapshot identity mismatch as `unknown`. Process, GPU,
checkpoint, TensorBoard wall time, and W&B file activity are auxiliary only.

The assessment status is advisory:

- `continue`: evidence is healthy and meaningful metrics are improving;
- `continue_and_recheck`: progress is still `observing` or `suspect`, or a
  healthy run has incomplete or mixed trend evidence;
- `consider_stop_plateau`: improvement is below the approved plateau tolerance;
- `recommend_stop_invalid`: non-finite metrics, confirmed stall, or an approved
  hard constraint failed;
- `insufficient_evidence`: progress is `unknown` or `stopped`, or the run or
  metric meaning cannot be resolved.

Missing, draft, hash-invalid, or scope-mismatched criteria force
`insufficient_evidence` and `indeterminate`; they cannot produce a strong
continue, stop, plateau, or convergence conclusion. Safety alerts remain
visible for operator attention. The assessment records the absolute criteria
path, full-file SHA-256, contract SHA-256, approval time, and exact scope.

Never signal the process from an assessment. If the user asks to terminate a
run, re-resolve the exact process group and follow the repository's bounded
process-removal rules.

## Run a lightweight evaluation during training

A short Native evaluation may overlap training when the user permits it. This
is a resource-budgeted exception, not permission for a full Native/JIT/ONNX
matrix.

Before launch:

1. confirm the exact training PID and that progress is currently advancing;
2. select a regular checkpoint whose size and mtime are stable and record its
   SHA-256;
3. inspect GPU free memory and use the smallest useful budget;
4. default to one environment, at most 2,000 steps, Native-only, and no video;
5. record the pre-evaluation training step and throughput.

Run `evaluate_policy.py` with the checkpoint as both the Native checkpoint and
Native artifact. Pass `--allow_training_overlap`. Add `--no_video` for the
lowest overhead and `--telemetry_path` when robot time-series data is needed.

After evaluation, recheck training progress, throughput, process state, and GPU
errors. If evaluation interferes, stop only the evaluation process and report
the interference. Never stop or restart training automatically.

The evaluator records reward, termination reasons, tracking RMSE, tilt, action
rate, action magnitude, joint velocity, applied torque, action parity, and
bounded env-0 telemetry. A video is evidence only after confirming the robot
remains in frame.

## Judge convergence

Do not equate the highest reward, a long run, or normal termination with
convergence. Compare at least two adjacent windows of the user-approved metrics
and combine:

- objective direction and relative improvement;
- constraint failures and non-finite values;
- episode length and termination composition;
- task tracking errors and motion-risk metrics;
- algorithm-specific losses and stability;
- checkpoint Play metrics and visual evidence.

For completed runs, report one of:

- `converged`;
- `plateaued_with_defects`;
- `not_converged`;
- `indeterminate`.

`converged` requires a completed run, sufficient windows, no hard failure, and
acceptable Play evidence. A plateau with unacceptable tracking, contacts,
oscillation, or action/torque behavior is `plateaued_with_defects`.

## Compare checkpoints

Inventory `model_N.pt` files with
`scripts/select_checkpoint_candidates.py`. Never assume the newest or the
highest-reward checkpoint is best.

Shortlist a small set representing the best available training metrics, the
plateau region, and the final checkpoint. Evaluate all shortlisted checkpoints
with the same task, command schedule, scenario, duration, seed, environment
count, and metric criteria. Use multi-objective Pareto comparison and visual
notes. Present the recommended checkpoint, alternatives, rejected candidates,
and uncertainty. Export only after the user selects one.

## Export and archive

Before export, inspect training, Play, export, observation history,
normalization, state reset, and deployment input ordering end to end. For ROA
and AMP-ROA preserve flattened time-major history, normalization of the current
frame only, and actor input `[current_obs, code_vel, hist_latent]`.

Use `scripts/rsl_rl_export_policy.py` to create JIT and ONNX and require finite
Native/JIT/ONNX action parity. Export is not deployment qualification.

Archive only after a separate user confirmation. Inspect `policy_storage`
read-only first, then use `scripts/archive_advised_policy.py` with an approved
manifest. It creates one atomic directory containing:

- `policy.pt`;
- `policy.onnx`;
- `策略说明.txt`;
- `archive_manifest.json`.

Never stage, commit, pull, push, clean, or resolve the storage repository unless
the user separately asks. Every description must say:

> 仅可进入受监督实物测试；未经实物验证，不代表 hardware-ready。

## Learn from Sim2Sim and Sim2Real feedback

Accept subjective observations, video, or telemetry. Bind the feedback to the
exact checkpoint or archived artifact when possible. Classify the leading
cause before suggesting a parameter:

1. export, tensor, history, normalization, or reset mismatch;
2. Sim2Sim runtime or deployment-configuration mismatch;
3. real-robot timing, calibration, communication, actuator, or mechanism;
4. insufficient evidence;
5. training coverage, reward, or parameter candidate.

Safety events stop further physical testing and trigger diagnosis. Subjective
feedback remains useful but carries low confidence. Never turn one observation
directly into an automatic reward edit.

For a training candidate, show the exact current parameter, proposed change or
range, expected effect, counter-metric, risk, supporting current evidence, and
compatible historical evidence. The user chooses whether to edit and train.

## Record tuning experience

Use `scripts/record_tuning_experience.py` to append immutable events under
`learnings/policy_tuning/<task>/<run-id>/`. Record:

- run identity, command, source state, algorithm, seed, and effective params;
- parameter hypothesis and expected effect;
- assessment snapshots and continue/stop decisions;
- checkpoint evaluations and final user selection;
- export and archive hashes and paths;
- Sim2Sim/Sim2Real feedback;
- observed effect, lesson, next suggestion, and confidence.

Reuse an earlier lesson only when task, algorithm, observation, reward, and
deployment context are compatible. State incompatibilities and uncertainty.

## Preserve and report

Do not install packages or delete logs, checkpoints, policies, or user changes.
Do not stage unrelated dirty files. Report evidence paths, exact metrics,
window definitions, decision status, confidence, recommended next check or
parameter, and which action still requires user approval.
