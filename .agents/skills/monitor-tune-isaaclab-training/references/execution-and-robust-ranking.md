# Staged execution and evidence-aware ranking

Use this workflow only with an approved version-6 or version-7 tune session. Versions 3–5
retain their prior static-plan behavior and do not gain execution authority.

## Contents

- Seed modes
- Execution contract
- Child command outputs
- Effective-config gate
- Synchronous multi-fidelity execution
- State transitions and recovery
- Quality anomaly rules
- Evidence-aware ranking
- Commands

## Seed modes

Choose the seed mode before observing results:

- `robust_multi_seed`: use the same screening seed set for the unchanged
  baseline and every candidate. After screening, retain the baseline and exact
  top-k candidates, then run only their remaining confirmation seeds.
- `fixed_single_seed`: set `tuning.seeds`, `screening_seeds`, and
  `confirmation_seeds` to the same one-element array. Rank top-k immediately
  after screening and create no extra confirmation jobs. Require
  `final_authority=supervised_hardware`.

Robust mode requires screening to be a proper subset of at least two
confirmation seeds. Fixed mode requires
`minimum_final_training_seeds=1`. Both modes require at least
`confirmation_top_k` non-baseline trials and choose the seed before outcomes.

Do not count repeated resume from the same RNG state as an independent training
seed. Fixed mode reports `single_seed_selected` and makes no generalization or
cross-seed robustness claim. Its final acceptance depends on Play/deployment
artifact checks followed by the approved supervised physical-test matrix.

## Execution contract

Add root-level `execution` to a version-6 or version-7 tune session:

```json
{
  "enabled": true,
  "state_dir": "/absolute/path/to/tuning_execution",
  "run_command": [
    "conda", "run", "-n", "isaacsim-5.1", "python",
    "/absolute/path/to/rsl_rl_trial_adapter.py",
    "--contract={adapter_contract_path}",
    "--executor-run-id={run_id}",
    "--overrides-json={overrides_json}",
    "--result={result_path}",
    "--summary={summary_path}",
    "--effective-config={effective_config_path}",
    "--terminal={terminal_path}",
    "--log-path={log_path}"
  ],
  "gpu_index": 0,
  "require_idle_gpu": true,
  "max_retries_per_run": 0,
  "effective_config": {
    "enabled": true,
    "baseline_path": "/absolute/path/to/baseline_effective_config.json",
    "require_exact_override_match": true,
    "allow_baseline_bootstrap": true
  },
  "adapter": {
    "id": "rsl-rl",
    "parameter_cli_map": {
      "agent.algorithm.learning_rate": "agent.algorithm.learning_rate"
    },
    "runtime_config_paths": {
      "agent.seed": "seed",
      "agent.run_name": "run_id",
      "env.seed": "seed"
    },
    "summary_last": 100,
    "require_checkpoint": true
  },
  "resource_limits": {
    "campaign_timeout_minutes": 1440,
    "min_free_disk_gb": 50,
    "max_gpu_temperature_c": 85,
    "stop_grace_seconds": 30
  },
  "reproducibility": {
    "enabled": true,
    "capture_git_diff": true,
    "capture_gpu": true,
    "package_names": ["torch", "rsl-rl-lib", "PyYAML"],
    "input_paths": [
      "/absolute/path/to/critical_training_config.py"
    ]
  },
  "quality_rules": [
    {
      "id": "throughput-collapse",
      "metric": "steps_per_second",
      "op": "<",
      "value": 10000,
      "consecutive_windows": 3,
      "minimum_progress": 10,
      "action": "mark_suspect"
    }
  ],
  "nonfinite_action": "stop_trial"
}
```

Treat the values as illustrative. The user must approve the exact argv,
placeholders, GPU, state directory, retry count, baseline config, metric rules,
thresholds, consecutive windows, and stop actions.

The executor never invokes a shell. The built-in RSL-RL adapter receives an
attempt-specific contract and adds the exact seed, unique run name, and only
the approved Hydra mappings to the reviewed base training argv. It resolves
the actual RSL-RL run directory from stdout, loads the dumped `env.yaml` and
`agent.yaml`, summarizes the same log, discovers the newest `model_N.pt`, and
writes a terminal receipt. Pass `{run_id}` as the adapter's exact executor
identity token as well as the child run name. This makes duplicate and orphan
checks exact rather than substring-based.

`runtime_config_paths` is not tuning authority. Each path must equal the
adapter-managed `seed` or `run_id` for that run. The effective-config gate
checks those exact identity values, normalizes them only for the baseline diff,
and still rejects every other undeclared difference.

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
`effective_config_path`. While training, it atomically replaces `summary_path`
after each complete progress record and immediately after any non-finite
metric. The live summary includes update time, line count, and resolved RSL-RL
run directory. Reconcile records TensorBoard scalar progress from that run
directory as independent secondary evidence.

The RSL-RL adapter also writes `terminal_path` with the exact child argv, exit
code, resolved run directory, checkpoint path and SHA-256, status, timestamps,
and failure reason. Do not write `completed` before checkpoints, logs,
effective config, and metrics are finalized. A missing, failed, identity-
mismatched, or hash-invalid receipt cannot advance the state.

## Effective-config gate

`validate_effective_config.py` flattens nested JSON configuration objects to
dotted paths. A baseline run must match the approved baseline exactly. A
candidate must apply every declared override with the exact approved type and
value and differ at no undeclared path. An override may equal its baseline
value when another parameter makes that grid combination a distinct trial.
When explicitly approved, the first unchanged baseline run may create a
previously absent baseline file exactly once. Later attempts cannot overwrite
it.

The gate rejects:

- an override that did not become effective;
- a value or type mismatch;
- an extra changed path;
- an unauthorized override;
- missing, non-JSON, linked, relative, or non-finite config input.

Store the baseline/candidate hashes and exact diff in the run state.

## Synchronous multi-fidelity execution

When the approved fixed-single-seed session contains `multi_fidelity`, follow
`multi-fidelity-training.md`. The initial version-6 plan contains only rung 1.
After every current-rung result is valid, `build_multifidelity_rung.py`
reconstructs the immutable plan, records one conservative promotion decision,
and either appends the next rung or records a terminal zero-job decision.

For single-host execution, pass the expanded plan to
`execute_trial_plan.py --action adopt-plan`. The executor accepts only
unchanged trials, prior runs and decisions plus one valid appended rung. After
final completion, `finalize_multifidelity_results.py` extracts the final
hash-bound result snapshot for `rank_trials.py`.

When `campaign_controller` is approved, follow `campaign-controller.md`.
`status` reports the next transition without creating controller state;
`advance` reuses these executor functions and performs at most one bounded
transition. The controller adds its own session/plan-bound hash-chain journal
but does not replace the executor journal or process-identity checks.

## State transitions and recovery

`execute_trial_plan.py` persists one atomic `execution_state.json`:

```text
pending -> running -> completed
                    -> pending (approved retry remains)
                    -> failed -> blocked
running -> stopping_quality_rule -> failed -> blocked
running -> stopping_trial_timeout -> failed -> blocked
running -> stopping_campaign_timeout -> failed -> blocked
stopping_* -> stopping_forced -> failed -> blocked
screening -> confirmation -> completed
```

The state binds the exact session and plan SHA-256. Every mutating CLI action is
serialized by an execution-state lock and appends a full hash-chained snapshot
to `execution_events.jsonl` before atomically replacing the state. If the state
file is damaged, `--action recover-state` validates the complete journal chain
and restores its latest session/plan-bound snapshot. It may discard and rewrite
only one incomplete, non-newline-terminated final record; corruption in any
earlier record remains fatal. The plan must also exactly
equal the deterministic plan rebuilt from that session. Reusing the state
directory with another contract or plan is rejected. The executor launches at
most one run, refuses a duplicate live run-ID token, holds an inherited per-GPU
file lock, requires the GPU to be idle, records exact argv, PID, process group
and Linux process start ticks, and returns without waiting. Each retry gets a
new attempt directory so stale artifacts cannot satisfy a later attempt.

Launch is a two-phase transaction. First reserve and journal the attempt with
status `launching`; then create its artifacts and process; finally write a
hash-bound `launch_receipt.json` before journaling `running`. A failed spawn or
identity capture consumes the attempt and records `launch-failed`. If the
scheduler exits between receipt and state persistence, reconcile restores the
exact process identity from the receipt. A live run-ID without a valid receipt
is blocked as an orphan and is never signaled.

On reconcile:

1. leave the exact recorded live process untouched unless an approved quality,
   per-trial, or campaign limit has fired;
2. analyze a valid structured summary when available;
3. signal only the exact argv/start-time/PID-matched recorded process group
   with SIGTERM, then use SIGKILL only after the approved grace period;
4. accept completion only with a matching finite result, adapter receipt,
   checkpoint hash, and valid config diff;
5. consume an approved retry only after the child is gone;
6. select confirmation candidates only after every screening run completes.

Never infer ownership from a command substring alone. Never stop an unknown,
PID-reused, or different process group.

Before each launch, the executor checks total campaign time, free disk, GPU
temperature, GPU idleness, and the inherited per-GPU lock. A failed
`nvidia-smi` health query is a launch blocker, including driver/library
mismatch conditions.

If screening leaves fewer constraint-satisfying candidates than the approved
top-k, persist a blocked state and the selection failure instead of silently
lowering top-k or launching an unauthorized substitute.

## Quality anomaly rules

A quality rule states that its comparison is anomalous. For example,
`steps_per_second < 10000` for three consecutive windows triggers the configured
action only when all three recent finite values match. One isolated value does
not satisfy the rule. Set `minimum_progress` to keep finite metric rules out of
the approved warm-up interval. Non-finite metrics still stop immediately.

## Reproducibility manifest

When enabled, every reserved attempt writes `reproducibility.json` before
launch. It binds the session and plan hashes, algorithm profile, seed, exact
training and executor argv, adapter contract hash, Git root/HEAD/dirty-status
hash, optional tracked-diff hash, Python/platform/package versions, optional
GPU name/UUID/driver, and SHA-256 for every explicitly listed critical input.
Completion rejects a missing or changed manifest. List dataset/config files
explicitly; no directory is recursively hashed by implication.

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

conda run -n isaacsim-5.1 python \
  .agents/skills/monitor-tune-isaaclab-training/scripts/execute_trial_plan.py \
  SESSION.json TRIAL_PLAN.json --action recover-state
```

Run `launch-next` only after reconcile reports no active run and at least one
pending run. A persistent scheduler may call reconcile at the approved
interval; without one, do not claim continuous autonomous supervision.
