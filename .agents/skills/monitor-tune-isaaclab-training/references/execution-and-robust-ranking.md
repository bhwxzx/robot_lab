# Staged execution and robust ranking

Use this workflow only with an approved version-6 tune session. Versions 3–5
retain their prior static-plan behavior and do not gain execution authority.

## Contents

- Seed stages
- Execution contract
- Child command outputs
- Effective-config gate
- State transitions and recovery
- Quality anomaly rules
- Robust ranking
- Commands

## Seed stages

Use the same fixed screening seed set for the unchanged baseline and every
candidate. After screening, retain the baseline and exactly
`confirmation_top_k` eligible candidates. Run only the remaining confirmation
seeds for those trials.

The screening set must be a proper subset of the confirmation set.
Confirmation must contain at least two seeds, and
`minimum_final_training_seeds` controls whether the final ranking has enough
independent training evidence. The grid must contain at least
`confirmation_top_k` non-baseline trials. Choose every seed before observing
outcomes.

Do not count repeated resume from the same RNG state as an independent training
seed. If every run starts from one common pretrained checkpoint, report that
the seed evidence covers fine-tuning variability rather than full-training
variability.

## Execution contract

Add root-level `execution` to a version-6 tune session:

```json
{
  "enabled": true,
  "state_dir": "/absolute/path/to/tuning_execution",
  "run_command": [
    "conda", "run", "-n", "isaacsim-5.1", "python",
    "/absolute/path/to/approved_training_adapter.py",
    "--seed={seed}",
    "--trial-id={trial_id}",
    "--stage={stage}",
    "--run-id={run_id}",
    "--run-dir={run_dir}",
    "--overrides-json={overrides_json}",
    "--result={result_path}",
    "--summary={summary_path}",
    "--effective-config={effective_config_path}",
    "--device=cuda:{gpu_index}"
  ],
  "gpu_index": 0,
  "require_idle_gpu": true,
  "max_retries_per_run": 0,
  "effective_config": {
    "enabled": true,
    "baseline_path": "/absolute/path/to/baseline_effective_config.json",
    "require_exact_override_match": true
  },
  "quality_rules": [
    {
      "id": "throughput-collapse",
      "metric": "steps_per_second",
      "op": "<",
      "value": 10000,
      "consecutive_windows": 3,
      "action": "mark_suspect"
    }
  ],
  "nonfinite_action": "stop_trial"
}
```

Treat the values as illustrative. The user must approve the exact argv,
placeholders, GPU, state directory, retry count, baseline config, metric rules,
thresholds, consecutive windows, and stop actions.

The executor never invokes a shell. Every required placeholder must appear in
the argv template. `{run_id}` is unique across stage, trial, and seed, which
prevents accidental W&B ID reuse when the adapter passes it through unchanged.

## Child command outputs

The approved adapter writes `result_path` only after a finite completed run:

```json
{
  "trial_id": "trial-001",
  "seed": 42,
  "status": "completed",
  "metrics": {
    "mean_reward": 148.2,
    "illegal_contact": 0.0
  }
}
```

It writes the complete final effective configuration as a JSON object to
`effective_config_path`. It may refresh `summary_path` atomically while
training using the output schema from `summarize_training_log.py`.

Do not write `completed` before checkpoints, logs, effective config, and
metrics are finalized. A missing or invalid result cannot advance the state.

## Effective-config gate

`validate_effective_config.py` flattens nested JSON configuration objects to
dotted paths. A baseline run must match the approved baseline exactly. A
candidate must apply every declared override with the exact approved type and
value and differ at no undeclared path. An override may equal its baseline
value when another parameter makes that grid combination a distinct trial.

The gate rejects:

- an override that did not become effective;
- a value or type mismatch;
- an extra changed path;
- an unauthorized override;
- missing, non-JSON, linked, relative, or non-finite config input.

Store the baseline/candidate hashes and exact diff in the run state.

## State transitions and recovery

`execute_trial_plan.py` persists one atomic `execution_state.json`:

```text
pending -> running -> completed
                    -> pending (approved retry remains)
                    -> failed -> blocked
running -> stopping_quality_rule -> failed -> blocked
screening -> confirmation -> completed
```

The state binds the exact session and plan SHA-256. The plan must also exactly
equal the deterministic plan rebuilt from that session. Reusing the state
directory with another contract or plan is rejected. The executor launches at
most one run, refuses a duplicate live run-ID token, holds an inherited per-GPU
file lock, requires the GPU to be idle, records exact argv, PID, process group
and Linux process start ticks, and returns without waiting. Each retry gets a
new attempt directory so stale artifacts cannot satisfy a later attempt.

On reconcile:

1. leave the exact recorded live process untouched;
2. analyze a valid structured summary when available;
3. signal only the exact argv/start-time/PID-matched recorded process group
   after an approved stop rule;
4. accept completion only with a matching finite result and valid config diff;
5. consume an approved retry only after the child is gone;
6. select confirmation candidates only after every screening run completes.

Never infer ownership from a command substring alone. Never stop an unknown,
PID-reused, or different process group.

If screening leaves fewer constraint-satisfying candidates than the approved
top-k, persist a blocked state and the selection failure instead of silently
lowering top-k or launching an unauthorized substitute.

## Quality anomaly rules

A quality rule states that its comparison is anomalous. For example,
`steps_per_second < 10000` for three consecutive windows triggers the configured
action only when all three recent finite values match. One isolated value does
not satisfy the rule.

Use `mark_suspect` when the metric needs diagnosis. Use `stop_trial` only for a
threshold whose meaning, direction, units, warm-up behavior, and safe response
were reviewed for the exact algorithm. Any recorded non-finite training metric
uses the mandatory `stop_trial` action.

The anomaly detector creates a decision; it does not signal a process. Only the
executor may act, and only on its exact child group.

## Robust ranking

Version-6 ranking requires:

- every confirmation seed for the baseline and selected candidates;
- identical seed sets for paired comparison;
- constraints checked on each seed;
- mean, sample standard deviation, range and 95% t interval;
- oriented paired improvement relative to the same-seed baseline;
- optional per-objective `minimum_improvement`;
- Pareto-front membership before weighted-score ordering.

When an objective specifies `minimum_improvement`, its candidate mean
improvement must meet the value and no seed may regress. Omit the field when a
trade-off is allowed; the Pareto front will expose the trade-off rather than
silently converting it into a hard constraint.

The weighted score ranks surviving candidates. It does not override a seed
constraint, paired-improvement gate, missing result, or Pareto evidence.

## Commands

Build the staged plan:

```bash
conda run -n isaacsim-5.1 python \
  .agents/skills/monitor-tune-isaaclab-training/scripts/build_trial_plan.py \
  SESSION.json --output TRIAL_PLAN.json
```

Initialize, launch one run, and reconcile later:

```bash
conda run -n isaacsim-5.1 python \
  .agents/skills/monitor-tune-isaaclab-training/scripts/execute_trial_plan.py \
  SESSION.json TRIAL_PLAN.json --action initialize

conda run -n isaacsim-5.1 python \
  .agents/skills/monitor-tune-isaaclab-training/scripts/execute_trial_plan.py \
  SESSION.json TRIAL_PLAN.json --action launch-next

conda run -n isaacsim-5.1 python \
  .agents/skills/monitor-tune-isaaclab-training/scripts/execute_trial_plan.py \
  SESSION.json TRIAL_PLAN.json --action reconcile
```

Run `launch-next` only after reconcile reports no active run and at least one
pending run. A persistent scheduler may call reconcile at the approved
interval; without one, do not claim continuous autonomous supervision.
