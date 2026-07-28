# Final policy evaluation and promotion

Training curves select candidates; they do not prove motion quality or
deployment readiness. A final selection requires closed-loop simulation with
the exact checkpoint and deployment artifact, hard metric gates, and an actual
review of recorded robot motion.

## Contents

- Promotion states
- Authorization boundary
- Scenario design
- Artifact evaluation
- Metrics and gates
- Visual review
- Execution workflow
- Result schemas
- Qualified policy archive
- Failure and retuning rules
- Real-robot boundary

## Promotion states

Use these exact meanings:

- `training_ranking_only`: ranked from training metrics; not Play-qualified.
- `awaiting_policy_evaluation`: selected for evaluation but the matrix is
  missing or incomplete.
- `simulation_rejected`: at least one required run, metric, parity check, or
  visual review failed.
- `simulation_qualified_hardware_candidate`: passed approved Native and
  deployment-artifact tests; eligible only for supervised hardware testing.

Never emit `hardware_ready=true`. Simulation cannot prove wiring, timing,
actuator calibration, communication, estimator, mechanical, or operator-safety
behavior on the physical robot.

## Authorization boundary

The version-4, version-5, version-6, or version-7 session owns evaluation and optional
archival authority. Version 3 remains valid only for legacy evaluation without
archival. Require the user
to approve:

- candidate count and exact checkpoint/artifact paths;
- artifact formats;
- scenarios, seeds, duration, and runtime overrides;
- metric thresholds and aggregations;
- video-review count;
- output directory, concurrency, and time limit;
- executor state directory, retry budget, termination grace period, disk,
  temperature, and minimum finalized-video size;
- exact GPU and the requirement that it be idle before launch;
- whether evaluation may reject a candidate;
- whether failure may start another already-authorized tuning trial.

Evaluation does not grant tuning authority. `allow_retune_on_failure=true`
requires tune mode and still permits only paths and domains already listed in
`tuning.allowed_parameters`.

## Scenario design

Every final evaluation requires a nominal scenario and at least one
non-nominal stress scenario. Build the matrix from the task's real command and
event capabilities rather than copying ranges from another robot.

Useful categories:

- `nominal`: standing, start, stop, forward/backward motion, ordinary turns;
- `command`: steps, ramps, reversals, deadband transitions, command boundaries;
- `terrain`: roughness, slopes, stairs, edges, gaps, and friction changes that
  the task and robot are designed to handle;
- `dynamics`: mass, center of mass, actuator gains, motor strength, payload;
- `disturbance`: impulses, sustained forces, slip, and recovery;
- `latency`: observation delay, action delay, dropped updates, timing jitter.

Use multiple fixed seeds. Include boundary values only when they are physically
meaningful and within the approved safety envelope. Do not add deployment-only
filters, delays, or observation transformations unless the same tensor contract
is intentionally under evaluation.

For deterministic start/stop, reversal, and turn tests, define a contiguous
`command_schedule` covering every scenario step. Each segment contains
`start_step`, `end_step`, and `[vx, vy, yaw_rate]`. The RSL-RL evaluator freezes
random command resampling and refreshes observations without updating history a
second time. Check every command against the exact task range before approval.

The RSL-RL evaluator applies scenario overrides to the training environment
configuration. Use a training task that retains its randomization and
disturbance events. Do not use a Play-only task whose configuration disables
them.

## Artifact evaluation

Always evaluate `native`. When a reviewed profile supports JIT or ONNX, require
at least one exported artifact for final selection.

For each artifact:

1. Load the same native checkpoint as a reference.
2. Feed the exact observation layout defined by the profile.
3. Run the artifact in the closed simulation loop.
4. Compare its action against native inference on the same observations.
5. Reset native recurrent state at episode boundaries.
6. Reject non-finite inputs, outputs, or metrics.

`flat_time_major_history` means flattening `[batch, time, observation]` to
`[batch, time * observation]` without permuting time and feature axes. Verify
normalization and history initialization from current code before approving the
profile.

## Metrics and gates

The bundled RSL-RL evaluator can emit:

- `mean_reward`;
- `termination_rate` and `timeout_rate`;
- `termination_term_<name>_rate` for every manager-based termination term,
  plus the `illegal_contact_rate` alias when that term exists;
- `tracking_xy_rmse` and `tracking_yaw_rmse`;
- `tilt_rms` and `max_tilt`;
- `action_rate_rms` and `max_abs_action`;
- `max_abs_joint_velocity` and `max_abs_applied_torque`;
- `max_abs_action_error`;
- `real_time_factor`.

These are a portable baseline, not a complete robot-specific safety model.
Manager-based tasks expose their termination reasons automatically, but foot
slip, stumble severity, impact, joint/torque margins, power, recovery time,
gait symmetry, or wheel behavior still require task-specific adapters backed by
trustworthy signals.

Every gate defines a metric, comparison, threshold, aggregation, artifact
selector, and scenario selector. Missing metrics fail the gate. Do not replace
a missing safety metric with reward.

Thresholds must come from the exact task, actuator and hardware envelope. An
example threshold in `session-spec.md` is not authorization to reuse it.

## Visual review

Visual review is mandatory because scalar summaries may miss:

- torso roll/pitch oscillation;
- jitter, chatter, or limit cycling;
- foot dragging, scuffing, crossing, or asymmetric placement;
- unnatural expert-motion imitation;
- abrupt start/stop behavior;
- unstable turns;
- unrealistic recovery or contact behavior;
- visually obvious sim-to-real hazards.

Record required matrix cells. The reviewer must inspect the motion and submit
one review per candidate with exact video paths, a `pass` or `fail` status,
reviewer identity, and notes. The validator checks coverage and declaration;
it does not pretend to infer that a file was meaningfully watched.

When automated video inspection is available, use it as an additional reviewer,
not as the sole physical-safety authority. Preserve videos for failed segments
and cite timestamps in notes.

## Execution workflow

1. Resolve and validate the non-generic algorithm profile.
2. Confirm the current Play/export/deployment tensor path.
3. Approve the version-4, version-5, version-6, or version-7 session including
   `evaluation` and, when requested,
   the separate `archive` contract.
4. Export required artifacts and verify that paths are isolated by candidate.
   When `policy_export` is approved, require its transactional Native/JIT/ONNX
   parity manifest before building the evaluation plan.
5. Create a candidate manifest:

```json
{
  "candidates": [
    {
      "candidate_id": "trial-001",
      "checkpoint_path": "/absolute/run/model_50000.pt",
      "checkpoint_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
      "artifacts": {
        "jit": "/absolute/run/exported/policy.pt",
        "onnx": "/absolute/run/exported/policy.onnx"
      },
      "artifact_sha256": {
        "jit": "0000000000000000000000000000000000000000000000000000000000000000",
        "onnx": "0000000000000000000000000000000000000000000000000000000000000000"
      }
    }
  ]
}
```

6. Build the exact matrix:

```bash
conda run -n isaacsim-5.1 python \
  .agents/skills/monitor-tune-isaaclab-training/scripts/build_evaluation_plan.py \
  SESSION.json CANDIDATES.json --output EVALUATION_PLAN.json
```

The builder reads every file and rejects a mismatched SHA-256 before creating
the plan. The RSL-RL evaluator verifies both hashes again before allocating a
GPU, preventing a checkpoint or deployment artifact from being replaced after
approval.

For an approved Campaign Controller result, the separate
`evaluation-handoff-controller.md` workflow may create this manifest and plan
from the immutable training ranking and checkpoint inventory, then drive the
same executor one cell at a time. It only locates already exported artifacts
and still stops before visual review.

7. Review every argv and runtime override before execution.
8. Confirm the approved GPU is idle. Do not overlap evaluation with training or
   another GPU-heavy task unless the user creates a separate contract that
   explicitly allows and budgets that interference.
9. Initialize the hash-bound evaluation state:

```bash
conda run -n isaacsim-5.1 python \
  .agents/skills/monitor-tune-isaaclab-training/scripts/execute_evaluation_plan.py \
  SESSION.json EVALUATION_PLAN.json --action initialize
```

10. Use `--action launch-next` for one matrix cell and `--action reconcile` on
    scheduled checks. The executor reserves a unique attempt before launch,
    records exact PID, process group, start ticks, argv, and receipt, enforces
    timeout and resource gates, and promotes only hash-stable complete results
    and finalized videos. Recover a damaged state only with
    `--action recover-state`.
11. Inspect the required videos and their `motion_evidence.review_windows`,
    then create visual reviews.
12. Consolidate per-run results and bind them to the executor state:

```bash
conda run -n isaacsim-5.1 python \
  .agents/skills/monitor-tune-isaaclab-training/scripts/collect_evaluation_results.py \
  EVALUATION_PLAN.json \
  --execution-state /absolute/policy_evaluation/.executor/evaluation_state.json \
  --visual-reviews VISUAL_REVIEWS.json \
  --output EVALUATION_RESULTS.json
```

13. Validate promotion:

```bash
conda run -n isaacsim-5.1 python \
  .agents/skills/monitor-tune-isaaclab-training/scripts/validate_policy_evaluation.py \
  SESSION.json EVALUATION_PLAN.json EVALUATION_RESULTS.json
```

14. Pass the plan and results to `rank_trials.py`. Only a training-eligible
    candidate that also passes policy evaluation may populate `final_selection`.

## Qualified policy archive

Read `policy-archive.md` before writing outside the training repository. Policy
storage requires the final candidate's JIT and ONNX artifacts to be required,
evaluated, hash-stable, and simulation-qualified. Archival is not evidence of
hardware readiness and performs no Git commit or push.

## Result schemas

Each matrix cell writes:

```json
{
  "version": 1,
  "run_id": "trial-001__onnx__nominal-command-sweep__seed-42",
  "candidate_id": "trial-001",
  "artifact": "onnx",
  "scenario_id": "nominal-command-sweep",
  "seed": 42,
  "status": "completed",
  "video_path": "/absolute/evaluation/trial-001/onnx/nominal-command-sweep/seed-42/motion.mp4",
  "metrics": {
    "termination_rate": 0.0,
    "tracking_xy_rmse": 0.42,
    "max_abs_action_error": 0.000002
  },
  "motion_evidence": {
    "step_dt_seconds": 0.02,
    "peak_steps": {
      "max_tilt": 731,
      "max_abs_action": 730
    },
    "termination_first_steps": {},
    "review_windows": [
      {
        "start_step": 701,
        "end_step": 761,
        "start_seconds": 14.02,
        "end_seconds": 15.24,
        "evidence": ["max_abs_action", "max_tilt"]
      }
    ]
  }
}
```

A visual-review file contains:

```json
{
  "visual_reviews": [
    {
      "candidate_id": "trial-001",
      "status": "pass",
      "reviewer": "user",
      "reviewed_video_paths": [
        "/absolute/evaluation/trial-001/native/nominal-command-sweep/seed-42/motion.mp4",
        "/absolute/evaluation/trial-001/onnx/mass-friction-stress/seed-42/motion.mp4"
      ],
      "notes": "No sustained roll oscillation, foot drag, chatter, or unstable recovery observed."
    }
  ]
}
```

Use `completed` only after the requested number of simulation steps finishes
with finite outputs and the video is finalized. Missing, crashed, timed-out, or
partial runs remain ineligible.

The evaluator's action-parity metric compares Native and deployment actions on
the exact same observations. `parity.closed_loop_metrics` separately compares
Native and deployment outcomes for matching scenario/seed cells. Both layers
must pass. Peak-step review windows are evidence-selection hints, not a
replacement for watching the recorded motion.

## Failure and retuning rules

Evaluation failure rejects the candidate when `allow_reject_candidate=true`.
Classify the evidence before suggesting a response:

- export parity failure: inspect tensor ordering, normalization, state, and
  deployment runtime; do not tune rewards;
- native and artifact both fail the same scenario: inspect policy/task
  robustness and training coverage;
- visual-only failure: identify timestamps and a measurable signature before
  proposing tuning;
- latency/dynamics-only failure: verify that the stress range matches expected
  hardware before changing training.

Retuning requires `mode=tune`, `allow_retune_on_failure=true`, remaining budget,
and an already-authorized parameter. Otherwise report the rejection and stop.

## Real-robot boundary

After simulation qualification, require a separate supervised hardware plan:

- support frame or tether where appropriate;
- conservative command, speed, torque, and workspace limits;
- tested emergency stop and communication timeout;
- verified observation order, units, history initialization, and control rate;
- telemetry for IMU, commands, actions, joint state, torque/current, contacts,
  estimator health, and timing;
- staged standing, low-speed motion, turning, disturbance, and terrain tests.

Only supervised physical evidence can advance the candidate beyond
`simulation_qualified_hardware_candidate`.

Record that evidence through the separate version-5, version-6, or version-7 contract in
`hardware-feedback-retuning.md`. Physical feedback may propose another bounded
simulation cycle, but it does not extend tuning paths, start trials, or excuse
deployment-runtime and hardware diagnosis.
