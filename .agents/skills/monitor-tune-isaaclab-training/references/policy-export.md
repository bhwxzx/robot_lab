# Transactional policy selection, export, and archive

Selection/export/archive validators accept both legacy v1 identity/config paths and
new content-addressed v2 provenance through `evidence_provenance.require_source_path`.
Closed-loop result v3/gzip bundles are fully revalidated alongside legacy v2 bundles.
Selection and export receipt formats and lifecycle directories stay unchanged; this
is compatibility with additional validated inputs, not a relaxed parity/reset gate.


## Contents

- [Prepare evidence paths](#prepare-evidence-paths)
- [Record the user selection](#record-the-user-selection)
- [BeyondMimic export and archive](#beyondmimic-export-and-archive)
- [GPU headroom and training overlap](#gpu-headroom-and-training-overlap)
- [Export transaction](#export-transaction)
- [ONNX export contract](#onnx-export-contract)
- [Parity contract](#parity-contract)
- [Export receipt and validation](#export-receipt-and-validation)
- [Archive gate](#archive-gate)

## Prepare evidence paths

Allocate a fresh selection ID and export ID with
`prepare_evidence_layout.py`. The final paths are:

- `CHECKPOINT_SELECTION_PATH`;
- `EXPORT_JIT_PATH`;
- `EXPORT_ONNX_PATH`;
- `EXPORT_RECEIPT_PATH`.

Never reuse either ID. Existing final files, a claim, or a partial export
directory are evidence of an earlier attempt, not permission to overwrite it.

## Record the user selection

Run `policy_export_evidence.py record-selection` only after the user explicitly
chooses one stable checkpoint. Supply the exact checkpoint-selection report and
SHA-256, checkpoint path/hash/filename stem, source identity, effective config,
every supporting version-2 evaluation result, and the reviewed tensor contract.

The checkpoint must be one stable inventory entry named
`model_<iteration>.pt`; `checkpoint_id` must equal its filename stem. The
receipt revalidates all evaluation bundles and requires their task, run,
runner, checkpoint path, and checkpoint hash to match. AMP-ROA also requires
complete evaluation telemetry.

The runner-specific tensor contracts are deterministic:

- ROA/AMP-ROA: flattened time-major history, current-frame-only normalization,
  environment history reset, actor input `[current_obs, code_vel, hist_latent]`;
- DWAQ/AMP-DWAQ: flattened time-major history, combined actor-input
  normalization, environment history reset;
- supported stateless runners: current observation, backend export-helper
  normalization, stateless environment reset.

The selection receipt embeds the complete run identity and binds the effective
config, report, checkpoint, evaluation results, and tensor contract by hash.

```bash
python3 .agents/skills/monitor-tune-isaaclab-training/scripts/policy_export_evidence.py \
  record-selection --selection-id "$SELECTION_ID" --approved-at "$APPROVED_AT" \
  --checkpoint-id "$CHECKPOINT_ID" --checkpoint "$CHECKPOINT" \
  --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --selection-report "$SELECTION_REPORT" \
  --selection-report-sha256 "$SELECTION_REPORT_SHA256" \
  --run-identity "$SOURCE_IDENTITY_PATH" \
  --run-identity-file-sha256 "$SOURCE_IDENTITY_FILE_SHA256" \
  --effective-config "$EFFECTIVE_CONFIG_PATH" \
  --effective-config-sha256 "$EFFECTIVE_CONFIG_SHA256" \
  --evaluation-result "$PLAY_RESULT_PATH" \
  --tensor-contract-json "$TENSOR_CONTRACT_JSON" \
  --output "$CHECKPOINT_SELECTION_PATH"
```

## BeyondMimic export and archive

Use the existing selection/export/archive validators for BeyondMimic. Resolve
the actual algorithm and runner from the saved `params/agent.yaml`; the verified
LW Leg history10 run uses `PPO` / `OnPolicyRunner` and the `rsl-rl-ppo` profile.
BeyondMimic names a task family, not an ROA history encoder or a new runner.

Read `params/env.yaml`, `params/agent.yaml` and the run's provenance before
exporting. Check observation terms and order, history length/layout, action and
joint order, normalization, motion reference and control/physics timing against
the export runtime. The CLI resolves the registered task from the checkout;
binding saved configuration evidence does not automatically restore that
configuration. Resolve any interface mismatch before launching, preserving live
training and unrelated edits.

For a supported stateless `OnPolicyRunner`, retain the existing contract strings:
`history_contract=current_observation`,
`normalization_contract=backend_export_helper`, and
`reset_contract=stateless_environment_reset`. Here `current_observation` describes
the rank-2 tensor presented to the exporter. It does not imply a single physical
frame. For the verified LW Leg history10 example, input `[1, 590]` contains ten
time-major frames of 59 values and output `[1, 10]` contains ten joint actions.
The environment or deployment consumer owns the history buffer and must reset it
at episode boundaries, even though the network itself is stateless. Read the
selected run's normalization flag; the example has actor observation normalization
disabled. Do not generalize these dimensions or flags to every BeyondMimic run.

Reuse completed, hash-valid evaluations matching the chosen checkpoint; archiving
does not require rerunning them. Record a fresh selection and export ID, perform
the bounded Native/CPU/JIT/ONNX parity check including reset, validate its receipt,
then use the archive gate below. Existing GUI exports without a valid receipt are
not a validated export bundle.

Include the saved env/agent configuration references and hashes, motion NPZ path
and hash, frame count/frequency, control/physics timestep, and policy input/output,
history, normalization and reset details in archive `parameters` and the policy
description. Keep the selected run's source HEAD and dirty state, rather than
substituting the current checkout's HEAD. For LW `leg_to_wheel`, use the authorized
`LW/leg_to_wheel` collection; run `2026-10-06_11-58-04` maps to directory
`2026-10-06-11-58-04` via explicit `--timestamp`. The standard bundle contains
`policy.pt`, `policy.onnx`, `策略说明.txt` and `archive_manifest.json`.

## GPU headroom and training overlap

An authorized ROA/AMP-ROA or BeyondMimic export may run while training occupies
the selected GPU. Complete GPU idleness is not an export/archive prerequisite.
The JIT/ONNX conversion and artifact parity use CPU, but this export entrypoint
also initializes Isaac Sim and samples real observations for Native parity and
reset coverage. Budget for that simulation overhead, not just the actor weights.
Archiving an already validated pair launches no simulator and needs no GPU wait.
This allowance is for bounded export parity; full evaluation batches and video
work keep their existing resource and authorization rules.

Before the attempt, record the selected GPU, its processes, utilization, used/free
memory, and the active training run's identity, log/TensorBoard progress and
throughput. Match the physical GPU index/UUID to `EXPORT_DEVICE`, accounting for
any `CUDA_VISIBLE_DEVICES` mapping. Use fresh measurements, not a historical GPU
snapshot:

```bash
nvidia-smi -i "$GPU_INDEX" \
  --query-gpu=index,name,uuid,utilization.gpu,memory.used,memory.free,memory.total \
  --format=csv,noheader,nounits
nvidia-smi -i "$GPU_INDEX" \
  --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader,nounits
```

Proceed when current free memory covers the estimated additional exporter peak
plus an explicit reserve for training fluctuations and unsampled startup peaks,
and training is progressing normally. Record the estimate, reserve and their
basis. Prefer comparable export telemetry from the same GPU, runner, robot/scene,
environment count, history and headless mode. GPU utilization is a contention
signal, not an idle/busy veto or proof of adequate resources. If headroom cannot
be established, defer the GPU attempt or use the idle mode; do not assume an
unmeasured ROA export fits merely because a BeyondMimic example did.

The completed 2026-10-08 export of LW Leg run `2026-10-06_11-58-04` used one
environment and eight temporal samples. Its exporter process reached a sampled
2,494 MiB, with about three seconds between GPU samples. The evidence is under
that run's `evidence/export/storage-overlap-20261008-002/`, in
`overlap-monitor.json` and `overlap-impact.json`. This is a past observation, not a
fixed budget or a continuous peak. Check the original telemetry before reusing
it; another runner, scene or configuration needs its own justified estimate.

For overlap, reuse an already approved finite parity budget. A typical small
attempt is headless, one environment, eight steps, reset at step four, at least
eight samples, and maximum absolute action error `1e-5`; record a finite timeout
(for example, 600 seconds) and the attempt limit. Sample GPU/process usage during
the attempt and compare training progress and throughput before, during and after
using stated windows. Report observed slowdown and sampled memory limits without
claiming zero impact or a causal performance estimate from one observation.

Omit `--require_idle_gpu` for this headroom-checked route. The flag remains the
explicit idle-only mode; omitting it does not make the exporter perform the
resource check automatically. Stop only the owned export process or its verified
process group on OOM, timeout, resource reserve exhaustion, or clear training
disruption. Never stop, restart or signal training to make room. Preserve logs
and partial-attempt evidence, publish no success receipt after a failed check,
and use fresh IDs for any later authorized retry. Existing session authorization
for the export and its overlap scope is sufficient; do not request it again just
because the GPU has an active training PID.

## Export transaction

Pass the new selection receipt and whole-file SHA-256 to
`rsl_rl_export_policy.py`. Also pass the same task, run, checkpoint ID/path/hash,
tensor-contract strings, a fresh export ID, canonical output paths, and the
bounded parity contract. For ROA/AMP-ROA or BeyondMimic, the example below uses the
GPU headroom route above. Add `--require_idle_gpu` when idle-only execution is
required; other profiles retain their existing resource policy.

Before Isaac Sim starts, the exporter validates all source evidence and takes
an exclusive `.publish-claim`. JIT, ONNX, and receipt work files are generated
inside its owned `.attempt/`. It loads both artifacts, completes every parity
gate, and revalidates all input evidence before publishing. JIT and ONNX are
linked without overwrite; `receipt.json` is linked last as the only completion
marker. Normal failure removes only owned attempt objects and final links whose
inode still belongs to that publisher.

```bash
conda run -n isaacsim-5.1 python \
  .agents/skills/monitor-tune-isaaclab-training/scripts/rsl_rl_export_policy.py \
  --task "$TASK" --run_id "$RUN_ID" --export_run_id "$EXPORT_ID" \
  --checkpoint_id "$CHECKPOINT_ID" --checkpoint "$CHECKPOINT" \
  --checkpoint_sha256 "$CHECKPOINT_SHA256" \
  --selection_receipt_path "$CHECKPOINT_SELECTION_PATH" \
  --selection_receipt_sha256 "$CHECKPOINT_SELECTION_SHA256" \
  --jit_path "$EXPORT_JIT_PATH" --onnx_path "$EXPORT_ONNX_PATH" \
  --result_path "$EXPORT_RECEIPT_PATH" \
  --history_contract "$HISTORY_CONTRACT" \
  --normalization_contract "$NORMALIZATION_CONTRACT" \
  --reset_contract "$RESET_CONTRACT" \
  --onnx_export_profile static_batch_1_simplified \
  --parity_steps 8 --reset_step 4 \
  --minimum_parity_samples 8 --max_abs_action_error 1e-5 \
  --num_envs 1 --seed "$SEED" --device "$EXPORT_DEVICE" --headless
```

## ONNX export contract

Choose the profile explicitly; there is no implicit default. Use
`static_batch_1_simplified` for a single-robot deployment. It fixes input and
output batch dimensions at 1, uses stable `obs` and `actions` names, exports
with opset 17, requires `onnxsim` to pass its model check, then reloads and
validates the final ONNX graph. Use `dynamic_batch` only when a consumer truly
needs batches larger than 1; that profile retains dynamic batch axes and opset
18 without simplification.

The static profile evaluates the multi-sample parity corpus one row at a time
and concatenates the actions, so it preserves the full temporal and reset
coverage. A static-batch artifact rejects direct batch sizes greater than 1 by
design.

## Parity contract

Use 2 through 64 temporal steps and 1 through 64 environments. Choose an
explicit reset step strictly inside the window. The exporter captures every
environment at every step and records at least these labeled boundaries:
`initial`, `pre_reset`, `post_reset`, and `final`.

Native device, Native CPU, JIT, and ONNX actions must have matching shapes and
finite values. JIT and ONNX maximum absolute action errors must not exceed the
approved limit. The receipt records observation/action digests, shape evidence,
boundary steps, sample count, reset contract, and Native device-to-CPU error.
This is bounded open-loop parity around real environment observations; it is
not deployment or hardware qualification.

## Export receipt and validation

A newly completed version-4 receipt binds:

- task, run, runner, checkpoint ID, and export ID;
- complete run identity and effective-config reference/fingerprints;
- approved checkpoint-selection receipt;
- tensor, ONNX export, and parity contracts;
- final ONNX input/output names, dtypes, shapes, opset/profile, simplifier
  result, and pre/post-simplification node counts;
- JIT/ONNX canonical paths, sizes, and SHA-256 values;
- multi-time/reset-boundary parity evidence.

Use `policy_export_evidence.py validate-export RECEIPT` before any downstream
use. Missing receipt, source drift, hash mismatch, incomplete boundaries,
parity failure, or a changed artifact invalidates the entire export.
Completed version-3 receipts created before this contract upgrade remain
validation-compatible, but every new export must publish version 4.

## Archive gate

Validate a completed export receipt before reuse or archive. This CPU/file-only
operation does not wait for an idle GPU, even while ROA or BeyondMimic training
continues. It still requires the authorization and storage checks below.

The archive directory MUST use the selected checkpoint's training run start
time in `YYYY-MM-DD-HH-MM-SS` format. Obtain this time from the verified run
identity and training run directory; preserve its recorded local time. For
example, run `2026-09-11_17-04-10` must be archived under
`policy_storage/LW/wheel_loco/2026-09-11-17-04-10`.
Always pass that value explicitly as `--timestamp 2026-09-11-17-04-10` to
`archive_advised_policy.py`; omitting the option currently defaults to the
archive time and violates this workflow. Do not substitute export time,
archive time, checkpoint mtime, or the time of evaluation. If the training
start time cannot be verified, resolve it before archiving rather than inventing
a timestamp. Use the same destination in collision checks, `archive_path`,
the policy description, and current navigation. A collision must follow the
replacement approval rules below; never avoid it by choosing a newer timestamp.

Archive only after a separate user authorization. An explicit request covering
both export and archive supplies both decisions; reuse it without asking again.
A version-2 archive manifest must reference the export receipt by path and
SHA-256 and list evaluation
results as `{path, sha256}` objects. After authorization and immediately before
the archive write, require a clean storage worktree and index, including no
untracked paths, then run `git -C "$STORAGE_ROOT" pull --ff-only`. Recheck the
updated HEAD and Git state, destination collision, and duplicate JIT/ONNX pair.
Stop without archiving if the upstream is missing, the pull fails or cannot
fast-forward, the repository is dirty, or a target or duplicate appears. Never
stash, merge, rebase, reset, checkout, clean, or resolve storage state
automatically.

The archiver revalidates the complete export and evaluation bundles, then
cross-checks checkpoint, task, runner, algorithm, source HEAD/dirty state, and
both artifacts before creating its atomic destination. It performs no Git
action; the required pull belongs to the outer advisor workflow. Legacy
version-1, path-only manifests are ineligible.

An existing destination remains an error by default. Replace one only after a
separate destructive-operation approval based on a read-only inventory. Add
this exact-shape object to the otherwise complete version-2 manifest:

```json
{
  "replace_existing": {
    "authorized": true,
    "path": "/absolute/policy_storage/collection/existing-directory",
    "files": {
      "policy.pt": "old SHA-256",
      "policy.onnx": "old SHA-256",
      "策略说明.txt": "old SHA-256",
      "archive_manifest.json": "old SHA-256"
    }
  }
}
```

Immediately after the required pull, recheck the exact four-file set and every
old hash. The archiver builds and verifies the new bundle in a private sibling,
atomically exchanges the directories, and deletes only the displaced bundle.
Any missing/extra file, symlink, changed hash, unavailable atomic exchange, or
cleanup failure aborts the operation. The original tracked bundle remains
recoverable from its prior Git commit, but never restore it automatically.
