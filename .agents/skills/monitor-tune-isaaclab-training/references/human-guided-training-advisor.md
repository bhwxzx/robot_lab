# Human-guided IsaacLab training advisor

This workflow replaces autonomous campaign control with bounded evidence and
user decisions. Existing campaign and distributed utilities remain available
only as legacy code and are not part of the advisor path.

## Evidence order

Use evidence in this order:

1. monotonic training iteration, epoch, or TensorBoard scalar step;
2. finite short-, medium-, and long-window training metrics;
3. explicit completion or failure output;
4. Native closed-loop metrics and bounded telemetry from an exact checkpoint;
5. motion video that has passed a robot-in-frame check;
6. process, GPU, log mtime, checkpoint mtime, and W&B activity as auxiliary
   evidence only.

Never allow a lower-ranked source to overrule stale or non-finite monotonic
training evidence.

## Assessment criteria

`assess_training_run.py` consumes a bounded summary and this JSON shape:

```json
{
  "version": 1,
  "window_size": 20,
  "minimum_records": 40,
  "plateau_required_metrics": 2,
  "metrics": {
    "mean_reward": {
      "direction": "maximize",
      "plateau_relative_tolerance": 0.01,
      "required": true
    },
    "error_vel_xy": {
      "direction": "minimize",
      "plateau_relative_tolerance": 0.02,
      "hard_max": 0.8,
      "required": true
    },
    "illegal_contact": {
      "direction": "minimize",
      "plateau_relative_tolerance": 0.02,
      "hard_max": 0.1,
      "required": false
    }
  },
  "play_gates": {
    "termination_rate": {"op": "<=", "value": 0.05},
    "tracking_xy_rmse": {"op": "<=", "value": 0.6},
    "max_tilt": {"op": "<=", "value": 0.7}
  }
}
```

Thresholds are task-specific. Examples are schema illustrations, not approved
LW_Leg thresholds. The user must review them before they affect a stop or
convergence recommendation.

For each metric, compare the preceding window with the latest window. Relative
improvement is positive when movement follows the configured direction. A
metric is plateaued when the absolute relative change does not exceed its
tolerance. Hard limits and non-finite metrics dominate plateau logic.

## Decision meanings

- `continue`: process evidence is healthy and at least one required metric has
  meaningful positive improvement without a hard failure.
- `continue_and_recheck`: the run is healthy but required trend or Play
  evidence is incomplete or mixed.
- `consider_stop_plateau`: enough adjacent windows exist and the configured
  number of required metrics are plateaued without meaningful improvement.
- `recommend_stop_invalid`: confirmed stall, non-finite metric, or approved hard
  training/Play gate failure.
- `insufficient_evidence`: profile, metric direction, records, or process
  identity is unresolved.

The tool emits advice only. It never signals a process.

## Training-overlap Play budget

When the user allows a short Play check during training:

- use one environment by default and never more than four;
- use at most 2,000 steps;
- evaluate Native only;
- disable video unless motion evidence is needed;
- record env-0 telemetry only and use a configurable stride;
- check GPU memory and training step before and after;
- stop the evaluator if it causes OOM, a material throughput drop, or training
  progress interruption; do not signal training.

The exact Native command uses the checkpoint for both artifact paths and hashes:

```bash
conda run -n isaacsim-5.1 python \
  scripts/reinforcement_learning/rsl_rl/evaluate_policy.py \
  --task EXACT_TASK --headless --device cuda:0 \
  --artifact_kind native \
  --checkpoint /absolute/model_N.pt \
  --checkpoint_sha256 SHA256 \
  --artifact_path /absolute/model_N.pt \
  --artifact_sha256 SHA256 \
  --candidate_id model-N --scenario_id quick-native \
  --duration_steps 500 --num_envs 1 --seed 42 \
  --allow_training_overlap --no_video \
  --run_id RUN_ID --result_path /absolute/result.json \
  --telemetry_path /absolute/telemetry.json
```

Telemetry contains the selected environment's command, reward, done/timeout,
base linear and angular velocity, projected gravity, root position, joint
position and velocity, applied torque, and action. Missing optional robot
signals are recorded as unavailable rather than substituted with reward.

## Checkpoint comparison

First inventory regular `model_N.pt` files and reject files that are too recent
to be considered stable. Shortlisting is evidence selection, not final policy
selection. Include:

- the newest stable checkpoint;
- checkpoints nearest the best available configured training metrics;
- one or more checkpoints spanning the plateau region.

Run every shortlisted checkpoint under an identical evaluation contract. When
evaluation results are supplied, compare configured metrics by Pareto
dominance. A checkpoint dominates another only when it is no worse on every
available metric and strictly better on at least one. Missing required metrics
make a checkpoint ineligible.

## Archive manifest

`archive_advised_policy.py` consumes a version-1 JSON object:

```json
{
  "version": 1,
  "archive_authorized": true,
  "storage_root": "/absolute/policy_storage",
  "collection": "LW/leg_loco",
  "task": "exact task",
  "algorithm": "AMP-ROA",
  "runner": "OnPolicyRunnerAmpROA",
  "selected_checkpoint": {
    "path": "/absolute/model_N.pt",
    "sha256": "64 hex characters",
    "iteration": 50000
  },
  "artifacts": {
    "jit": {"path": "/absolute/policy.pt", "sha256": "64 hex characters"},
    "onnx": {"path": "/absolute/policy.onnx", "sha256": "64 hex characters"}
  },
  "source": {"commit": "full commit", "dirty": true},
  "parameters": {"reward.path": -0.15},
  "evaluation": {"result_paths": ["/absolute/result.json"]},
  "description_notes": "User-approved notes"
}
```

The archiver verifies a clean storage Git worktree, existing non-symlinked
collection, source hashes, extensions, duplicate artifact pairs, and
destination collision. It creates one atomic directory and performs no Git
action.

## Feedback and experience records

Record feedback as an immutable `feedback` event with `source` set to
`sim2sim` or `sim2real`. Include the exact policy identity when known,
deployment configuration, scenario, observations, safety events, evidence
paths, user assessment, root-cause classification, and next suggestion.

Each experience event uses:

```json
{
  "version": 1,
  "event_id": "unique-ascii-id",
  "event_type": "run_snapshot",
  "recorded_at": "2026-07-31T18:00:00+08:00",
  "task": "exact-task",
  "run_id": "training-run-id",
  "algorithm": "AMP-ROA",
  "context": {
    "observation_fingerprint": "sha256 or explicit unknown",
    "reward_fingerprint": "sha256 or explicit unknown",
    "deployment_fingerprint": "sha256 or explicit unknown"
  },
  "parameters": {},
  "evidence": {},
  "analysis": {"summary": "", "confidence": "low"},
  "next_suggestion": ""
}
```

Allowed event types are `run_snapshot`, `assessment`, `decision`,
`checkpoint_evaluation`, `checkpoint_selection`, `export`, `archive`,
`feedback`, and `recommendation`. Records are append-only. Reuse prior advice
only when task, algorithm, and all three context fingerprints match; otherwise
state the mismatch.
