# Hardware-feedback retuning

Physical deployment feedback can inform another bounded simulation-training
cycle, but it does not grant tuning authority and it must not bypass diagnosis.
Bind every report to one archived policy, exact deployment configuration, test
envelope, and time segment.

## Decision order

Process feedback in this order:

1. Stop after an emergency stop, fall, limit violation, communication timeout,
   damage, critical observation, or `unsafe` user assessment. Preserve evidence
   and diagnose the physical system before another test.
2. Verify JIT and ONNX hashes against `archive_manifest.json`.
3. Verify observation ordering, units, normalization, history initialization,
   reset behavior, control rate, deployment configuration, emergency stop, and
   communication timeout.
4. Separate deployment-runtime and hardware/calibration failures from
   training or simulation-coverage gaps.
5. Reproduce a suspected training gap in closed-loop simulation with a
   measurable signature.
6. Offer only parameter paths and domains already authorized by the approved
   tune session. Require another approved session before running a trial.

Do not translate one subjective observation directly into a reward edit.
Subjective-only feedback remains useful, but has low evidence confidence and
can produce only evidence-collection and reproduction suggestions.

## Session authorization

Use session version `5` and add:

```json
{
  "enabled": true,
  "output_mode": "prepare_authorized_draft",
  "output_dir": "/absolute/path/to/hardware_feedback_results",
  "require_policy_manifest": true,
  "verify_artifact_hashes": true,
  "stop_on_safety_event": true,
  "require_new_session_approval": true
}
```

Set the root-level `hardware_feedback` field to this object.
`proposal_only` is valid in monitor or tune mode. It produces analysis but no
parameter-choice draft. `prepare_authorized_draft` requires tune mode and can
list only the existing `tuning.allowed_parameters`; its output remains
non-executable and pending user selection and approval.

## Feedback record

Create a version-1 JSON object:

```json
{
  "version": 1,
  "feedback_id": "lw-leg-standing-2026-07-26-001",
  "policy": {
    "archive_manifest_path": "/absolute/policy/archive_manifest.json",
    "archive_manifest_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
    "candidate_id": "trial-001",
    "artifacts": {
      "jit": "0000000000000000000000000000000000000000000000000000000000000000",
      "onnx": "0000000000000000000000000000000000000000000000000000000000000000"
    }
  },
  "deployment": {
    "runtime": "exact runtime and version",
    "artifact_kind": "jit",
    "robot_id": "exact robot identifier",
    "firmware": "exact firmware identifier",
    "control_frequency_hz": 50,
    "config_files": [
      {
        "path": "/absolute/deployment/config.yaml",
        "sha256": "0000000000000000000000000000000000000000000000000000000000000000"
      }
    ],
    "observation_contract_verified": true,
    "history_initialized": true,
    "emergency_stop_verified": true,
    "notes": ""
  },
  "test": {
    "started_at": "2026-07-26T14:00:00+08:00",
    "operator": "user",
    "supervision": "supervised",
    "scenario": "standing",
    "surface": "level non-slip floor",
    "payload_kg": 0,
    "duration_seconds": 30,
    "command_envelope": {
      "max_linear_speed_mps": 0,
      "max_yaw_rate_rps": 0
    }
  },
  "observations": [
    {
      "symptom": "standing_roll_oscillation",
      "severity": "moderate",
      "start_seconds": 5,
      "end_seconds": 18,
      "notes": "sustained left-right torso motion"
    }
  ],
  "safety": {
    "emergency_stop": false,
    "fall": false,
    "joint_limit_violation": false,
    "torque_limit_violation": false,
    "communication_timeout": false,
    "mechanical_damage": false,
    "operator_intervention": false,
    "notes": ""
  },
  "evidence": {
    "video_files": [
      {
        "path": "/absolute/evidence/standing.mp4",
        "sha256": "0000000000000000000000000000000000000000000000000000000000000000"
      }
    ],
    "telemetry_files": [
      {
        "path": "/absolute/evidence/telemetry.csv",
        "sha256": "0000000000000000000000000000000000000000000000000000000000000000"
      }
    ],
    "telemetry_channels": [
      "command",
      "imu_roll",
      "imu_pitch",
      "base_angular_velocity",
      "action",
      "joint_position",
      "joint_velocity",
      "applied_torque",
      "control_timestamp"
    ],
    "sample_rate_hz": 100,
    "clock_synchronized": true
  },
  "user_assessment": {
    "overall": "fail",
    "notes": "standing motion is not acceptable for the next deployment stage"
  }
}
```

Supported symptom tags are defined by
`scripts/validate_hardware_feedback.py`. Use `other` with a precise note when a
new algorithm or robot exhibits a behavior not yet represented. The generic
contract does not authorize a persistent schema or algorithm-profile change.

## Evidence expectations

The validator verifies every supplied file and hash. It also checks that:

- the archive manifest and candidate match the session algorithm profile;
- the archived `policy.pt` and `policy.onnx` still match the report;
- the physical test was supervised and its observation segments fit within
  the test duration;
- telemetry declares channels, sample rate, and clock synchronization;
- the result never claims `hardware_ready`.

For standing oscillation, prefer time-aligned command, raw IMU roll and pitch,
angular velocity, projected gravity or equivalent estimator state, action,
actual joint state, torque/current, contact, and control timestamps. Compare
the same signals in simulation before choosing a training change.

## Commands

Validate the report:

```bash
conda run -n isaacsim-5.1 python \
  .agents/skills/monitor-tune-isaaclab-training/scripts/validate_hardware_feedback.py \
  SESSION.json HARDWARE_FEEDBACK.json \
  --output /absolute/approved/output/HARDWARE_FEEDBACK_VALIDATION.json
```

Build a deterministic, non-executable proposal:

```bash
conda run -n isaacsim-5.1 python \
  .agents/skills/monitor-tune-isaaclab-training/scripts/build_feedback_retune_proposal.py \
  SESSION.json HARDWARE_FEEDBACK.json \
  --output /absolute/approved/output/RETUNE_PROPOSAL.json
```

The proposal classifies the leading cause as safety, incomplete deployment
contract, deployment runtime/tensor path, hardware/calibration, insufficient
evidence, no retune, or a training/Sim-to-Real candidate. Training candidates
receive proposed simulation scenarios, measurable metrics, parameter
categories, and matching existing authorized paths.

The optional `authorization_draft` deliberately leaves
`selected_parameter_paths` empty. The user must choose the paths, exact
domains, objectives, constraints, seeds, trial budget, evaluation scenarios,
and gates, then approve a new version-5 or version-6 session. Never feed the
proposal itself
to `build_trial_plan.py`.

When `--output` is used, it must be a new absolute `.json` path beneath the
approved `hardware_feedback.output_dir`. The tools refuse an existing file or
an output path that escapes that directory.
