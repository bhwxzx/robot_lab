# Session authorization specification

Use JSON version `3`. Resolve the algorithm profile before approval, then validate with `scripts/validate_session_spec.py`.

## Contents

- Monitor example
- Tune fields
- Final policy evaluation
- Runtime progress snapshot
- Results format

## Monitor example

```json
{
  "version": 3,
  "mode": "monitor",
  "algorithm": {
    "backend": "rsl_rl",
    "name": "PPO",
    "runner_class": "OnPolicyRunner",
    "profile_id": "rsl-rl-ppo",
    "profile_version": 2,
    "profile_fingerprint": "e6e31a40f6934323",
    "unknown_algorithm_policy": "reject"
  },
  "training": {
    "command": [
      "conda", "run", "-n", "isaacsim-5.1", "python",
      "scripts/reinforcement_learning/rsl_rl/train.py",
      "--task=EXACT_TASK",
      "--headless"
    ],
    "resume_command": [
      "conda", "run", "-n", "isaacsim-5.1", "python",
      "scripts/reinforcement_learning/rsl_rl/train.py",
      "--task=EXACT_TASK",
      "--resume",
      "--load_run=RUN_DIRECTORY",
      "--checkpoint=CHECKPOINT_FILE",
      "--headless"
    ],
    "cwd": "/absolute/path/to/robot_lab",
    "log_path": "/absolute/path/to/train_watchdog.log",
    "run_id": "user-chosen-run-id",
    "checkpoint_path": "/absolute/path/to/model_N.pt"
  },
  "monitoring": {
    "check_interval_seconds": 600,
    "stale_after_seconds": 1200,
    "pid": null,
    "gpu_index": 0,
    "tensorboard_path": "/absolute/path/to/run-or-event-file",
    "expected_process_pattern": "rsl_rl/train.py",
    "low_gpu_utilization_percent": 5
  },
  "recovery": {
    "enabled": true,
    "max_restarts": 2,
    "cooldown_seconds": 600
  },
  "tuning": null,
  "evaluation": null,
  "cleanup": {
    "remove_created_temp_files": true
  }
}
```

Keep commands as argv arrays. Resolve task, run, checkpoint, backend, algorithm, and runner to exact values. The profile version and fingerprint bind authorization to the reviewed profile semantics.

Use `unknown_algorithm_policy` as follows:

- `reject`: require a known specific profile.
- `runtime_generic`: allow monitor-only operation through a generic profile.
- `propose_persistent`: allow monitor-only operation and generate a candidate profile upgrade.

Generic profiles are invalid in tune mode.

Set `pid` after starting or attaching. The validator checks profile-required resume tokens; it cannot prove that the selected checkpoint is healthy.

## Tune fields

Change `mode` to `tune` and replace `tuning`:

```json
{
  "allowed_parameters": [
    {
      "path": "agent.algorithm.learning_rate",
      "values": [0.0003, 0.0005],
      "baseline": 0.0005
    },
    {
      "path": "env.rewards.action_rate_l2.weight",
      "range": {
        "min": -0.2,
        "max": -0.1,
        "step": 0.05
      },
      "baseline": -0.15
    }
  ],
  "protected_parameters_unlocked": [],
  "max_trials": 6,
  "seeds": [42, 43, 44],
  "trial_timeout_minutes": 120,
  "max_concurrent_trials": 1,
  "mutation_scope": "overrides_only",
  "objectives": [
    {
      "metric": "mean_reward",
      "goal": "maximize",
      "weight": 1
    },
    {
      "metric": "error_vel_xy",
      "goal": "minimize",
      "weight": 2
    }
  ],
  "constraints": [
    {
      "metric": "illegal_contact",
      "op": "<=",
      "value": 0.06
    }
  ]
}
```

Treat all example parameter paths as illustrative. Inspect the current effective configuration first.

`max_trials` includes the unchanged baseline. Every configuration runs every listed seed. The plan builder selects a deterministic bounded subset when the authorized grid is larger than the budget.

Any allowed path matched by the selected profile's protected patterns also requires the exact same path in `protected_parameters_unlocked`.

## Final policy evaluation

Set `evaluation` before calling any checkpoint a final policy. Read
`policy-evaluation.md` for the complete result and visual-review workflow.

```json
{
  "enabled": true,
  "require_for_final_selection": true,
  "artifacts": [
    {
      "kind": "native",
      "required": true,
      "command": [
        "conda", "run", "-n", "isaacsim-5.1", "python",
        "scripts/reinforcement_learning/rsl_rl/evaluate_policy.py",
        "--task=EXACT_TRAINING_TASK",
        "--headless",
        "--device=cuda:{gpu_index}",
        "{require_idle_gpu_flag}",
        "--checkpoint={checkpoint_path}",
        "--checkpoint_sha256={checkpoint_sha256}",
        "--artifact_kind={artifact_kind}",
        "--artifact_path={artifact_path}",
        "--artifact_sha256={artifact_sha256}",
        "--candidate_id={candidate_id}",
        "--scenario_id={scenario_id}",
        "--scenario_overrides_json={scenario_overrides_json}",
        "--command_schedule_json={command_schedule_json}",
        "--duration_steps={duration_steps}",
        "--seed={seed}",
        "--run_id={run_id}",
        "--result_path={result_path}",
        "--video_path={video_path}"
      ]
    },
    {
      "kind": "onnx",
      "required": true,
      "command": [
        "conda", "run", "-n", "isaacsim-5.1", "python",
        "scripts/reinforcement_learning/rsl_rl/evaluate_policy.py",
        "--task=EXACT_TRAINING_TASK",
        "--headless",
        "--device=cuda:{gpu_index}",
        "{require_idle_gpu_flag}",
        "--checkpoint={checkpoint_path}",
        "--checkpoint_sha256={checkpoint_sha256}",
        "--artifact_kind={artifact_kind}",
        "--artifact_path={artifact_path}",
        "--artifact_sha256={artifact_sha256}",
        "--candidate_id={candidate_id}",
        "--scenario_id={scenario_id}",
        "--scenario_overrides_json={scenario_overrides_json}",
        "--command_schedule_json={command_schedule_json}",
        "--duration_steps={duration_steps}",
        "--seed={seed}",
        "--run_id={run_id}",
        "--result_path={result_path}",
        "--video_path={video_path}"
      ]
    }
  ],
  "scenarios": [
    {
      "id": "nominal-command-sweep",
      "category": "nominal",
      "required": true,
      "seeds": [42, 43],
      "duration_steps": 2000,
      "overrides": {},
      "command_schedule": [
        {"start_step": 0, "end_step": 199, "command": [0.0, 0.0, 0.0]},
        {"start_step": 200, "end_step": 699, "command": [0.5, 0.0, 0.0]},
        {"start_step": 700, "end_step": 899, "command": [0.0, 0.0, 0.0]},
        {"start_step": 900, "end_step": 1399, "command": [-0.4, 0.0, 0.0]},
        {"start_step": 1400, "end_step": 1599, "command": [0.0, 0.0, 0.0]},
        {"start_step": 1600, "end_step": 1999, "command": [0.0, 0.0, 0.4]}
      ],
      "video": true
    },
    {
      "id": "mass-friction-stress",
      "category": "dynamics",
      "required": true,
      "seeds": [42, 43],
      "duration_steps": 2000,
      "overrides": {
        "events.randomize_rigid_body_mass_base.params.mass_distribution_params": [-1.0, 3.0]
      },
      "command_schedule": [
        {"start_step": 0, "end_step": 1999, "command": [0.4, 0.0, 0.0]}
      ],
      "video": true
    }
  ],
  "gates": [
    {
      "metric": "termination_rate",
      "op": "<=",
      "value": 0.001,
      "aggregation": "max",
      "artifacts": ["*"],
      "scenarios": ["*"]
    },
    {
      "metric": "tracking_xy_rmse",
      "op": "<=",
      "value": 0.6,
      "aggregation": "mean",
      "artifacts": ["*"],
      "scenarios": ["*"]
    }
  ],
  "parity": {
    "required": true,
    "reference_artifact": "native",
    "max_abs_action_error": 0.00001
  },
  "visual_review": {
    "required": true,
    "minimum_reviewed_videos": 2,
    "require_notes": true
  },
  "output_dir": "/absolute/path/to/policy_evaluation",
  "gpu_index": 0,
  "require_idle_gpu": true,
  "max_concurrent_runs": 1,
  "run_timeout_minutes": 30,
  "allow_reject_candidate": true,
  "allow_retune_on_failure": false
}
```

The example thresholds and overrides are illustrative. Inspect the exact task,
units, event configuration, command ranges, and hardware limits before approval.
Use the training task rather than a Play-only task that disables disturbances.

If evaluation is absent, disabled, incomplete, or failed, ranking may report a
training candidate but must leave `final_selection` null. Enabling
`allow_retune_on_failure` does not add parameters; it only permits another
bounded trial using the existing authorized tuning paths.

## Runtime progress snapshot

After every check, persist:

- observation timestamp;
- profile ID and fingerprint;
- latest log progress value;
- latest TensorBoard scalar step and wall time;
- process identity and GPU evidence;
- restart count.

Pass the prior progress values to the next health check. A live process, active GPU, fresh checkpoint, or growing `.wandb` file does not replace monotonic progress evidence.

## Results format

Provide one training row per trial and seed:

```json
{
  "runs": [
    {
      "trial_id": "baseline",
      "seed": 42,
      "status": "completed",
      "metrics": {
        "mean_reward": 148.2,
        "error_vel_xy": 0.52,
        "illegal_contact": 0.05
      }
    }
  ]
}
```

Use `completed` only for a finite, fully evaluated run. The ranker rejects missing seeds, metrics, non-finite values, failed constraints, and an incomplete baseline.
