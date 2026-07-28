# Automatic policy export and parity gate

Use this workflow to turn approved training checkpoints into evaluation-ready
JIT and ONNX artifacts. Export is not policy qualification: it is a
deterministic gate before closed-loop Play testing and visual review.

## Contents

- [Permission boundary](#permission-boundary)
- [What the RSL-RL adapter exports](#what-the-rsl-rl-adapter-exports)
- [Build and inspect a plan](#build-and-inspect-a-plan)
- [Execute one transition](#execute-one-transition)
- [Parity evidence](#parity-evidence)
- [Failure and recovery](#failure-and-recovery)
- [Limitations](#limitations)

## Permission boundary

Add `policy_export` only to an approved version-6-or-7 tune campaign with:

- an executable Campaign Controller;
- executable policy evaluation selecting Native, JIT, and ONNX;
- an enabled Evaluation Handoff Controller;
- one exact checkpoint seed and Pareto Top-K;
- the same export and evaluation worker;
- an absolute export directory inside `evaluation.output_dir`.

`policy_export.mode=shadow` permits validation and plan inspection only.
`mode=execute` separately authorizes checkpoint loading and JIT/ONNX creation.
It does not authorize tuning, visual acceptance, policy-storage promotion, Git
operations, or physical deployment.

Version 6 uses `worker_id=null`. Version 7 names the exact approved distributed
worker. Git mailbox transport remains metadata-only, so selected checkpoints
must already exist on that worker with the inventory-recorded SHA-256.

## What the RSL-RL adapter exports

The bundled `rsl_rl_export_policy.py` is export-only. It starts Isaac Sim to
construct the exact task and runner, loads the approved checkpoint, captures
one observation batch, exports JIT and ONNX, reloads both artifacts, and
compares their actions with Native inference. It does not step the simulation
or replace the later Play matrix.

The reviewed tensor contracts are:

- ordinary RSL-RL: profile history contract plus
  `backend_export_helper`;
- DWAQ: flattened time-major history and `combined_actor_input`;
- ROA and AMP-ROA: flattened time-major history,
  `[current_obs, code_vel, hist_latent]`, and `current_frame_only`
  normalization.

Do not guess these values for a new runner. Add or upgrade its algorithm
profile only after reviewing training, Play, export, and deployment paths.

The command template must pass every validated placeholder. A typical RSL-RL
command is:

```bash
conda run -n isaacsim-5.1 python \
  .agents/skills/monitor-tune-isaaclab-training/scripts/rsl_rl_export_policy.py \
  --task=EXACT_TRAIN_TASK --headless --device=cuda:{gpu_index} \
  {require_idle_gpu_flag} \
  --checkpoint={checkpoint_path} \
  --checkpoint_sha256={checkpoint_sha256} \
  --candidate_id={candidate_id} --trial_id={trial_id} \
  --export_run_id {export_run_id} \
  --jit_path={jit_path} --onnx_path={onnx_path} \
  --result_path={result_path} --seed={seed} \
  --history_contract={history_contract} \
  --normalization_contract={normalization_contract} \
  --minimum_parity_samples={minimum_parity_samples} \
  --max_abs_action_error={max_abs_action_error}
```

Use the training task, not a Play-only configuration that changes observation
or randomization behavior.

## Build and inspect a plan

Build a deterministic plan from the immutable Campaign Controller outputs:

```bash
conda run -n isaacsim-5.1 python \
  .agents/skills/monitor-tune-isaaclab-training/scripts/build_policy_export_plan.py \
  SESSION.json TRAINING_RANKING.json CHECKPOINT_INVENTORY.json \
  --output POLICY_EXPORT_PLAN.json
```

For version 7, add `--worker-id EXACT_APPROVED_WORKER`. The builder selects
only Pareto candidates, the approved checkpoint seed, and the newest
unambiguous checkpoint on the designated worker. It verifies every checkpoint
hash before emitting work.

The plan binds the session, ranking, inventory, algorithm, adapter, worker,
GPU, parity contract, command template, artifact filenames, and source
checkpoint hashes. Treat it as immutable.

## Execute one transition

The Evaluation Handoff Controller normally owns the sequence. Status is
read-only:

```bash
conda run -n isaacsim-5.1 python \
  .agents/skills/monitor-tune-isaaclab-training/scripts/evaluation_handoff_controller.py \
  SESSION.json TRAINING_RANKING.json CHECKPOINT_INVENTORY.json \
  --action status
```

After both execute permissions are approved, `--action advance` performs only
one transition per invocation. It prepares the plan, initializes export state,
reconciles one active process, or launches one pending export. It cannot skip
the export manifest and enter evaluation.

The standalone executor is available for diagnosis:

```bash
conda run -n isaacsim-5.1 python \
  .agents/skills/monitor-tune-isaaclab-training/scripts/execute_policy_export_plan.py \
  SESSION.json POLICY_EXPORT_PLAN.json --action status
```

Its other actions are `initialize`, `launch-next`, `reconcile`, and
`recover-state`. Use the handoff controller for normal operation so export and
evaluation provenance remain in one hash-bound journal.

## Parity evidence

Each successful result must contain:

- exact export run, candidate, checkpoint path, and checkpoint SHA-256;
- JIT and ONNX paths, hashes, sizes, input/output shapes, and finite flags;
- maximum absolute action error for each artifact;
- sample count;
- observation-batch and Native-output digests;
- exact history and normalization contracts.

The executor independently rereads every artifact and checkpoint, verifies
hashes and minimum sizes, enforces the approved error limit and sample count,
and rejects missing or additional result fields. Only when every candidate
passes does it atomically publish `export_manifest.json`.

The manifest remains training evidence, not motion evidence. Closed-loop
Native/JIT/ONNX scenarios, videos, scalar gates, and human visual review still
follow.

## Failure and recovery

Every attempt uses a fresh directory, launch receipt, exact Linux process
identity, idle-GPU check, shared GPU lock, disk/temperature preflight, timeout,
and bounded retry count. A parity or artifact validation failure consumes an
attempt. Exhausted retries set the export stage to `blocked`; no candidate
manifest or evaluation plan is published.

The state journal is hash-chained. A torn final journal write is discarded
before the next append. If the state file is missing but the journal is intact,
use `recover-state`; bindings to the exact session and plan are rechecked.

Do not manually replace a failed artifact inside an attempt. Diagnose the
checkpoint, tensor ordering, normalization, recurrent/history state, exporter,
or runtime dependency, then approve a new session or retry budget as required.

## Limitations

Open-loop action parity proves that one captured batch produces sufficiently
close actions. It does not prove recurrent behavior across resets, robustness,
contact dynamics, motion quality, or real-robot readiness. Those claims require
the subsequent closed-loop Play matrix, video review, and supervised hardware
qualification.

The bundled adapter currently automates reviewed RSL-RL runner families. A new
backend or runner must supply a versioned exporter adapter and reviewed profile;
generic fallback profiles cannot authorize automatic export.
