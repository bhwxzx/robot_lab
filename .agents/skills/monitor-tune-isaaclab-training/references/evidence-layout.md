# Policy-tuning evidence layout

Use one deterministic repository-local tree for raw observations and keep it
separate from immutable tuning-experience events:

```text
learnings/policy_tuning/<task>/<run-id>/
├── evidence/
│   ├── criteria/criteria-<snapshot-id>.json
│   ├── health/health-<snapshot-id>.json
│   ├── source/identity-<snapshot-id>.json
│   ├── source/effective-config-<snapshot-id>.json
│   ├── source/source-<snapshot-id>.patch
│   ├── training/summary-<snapshot-id>.json
│   ├── training/assessment-<snapshot-id>.json
│   └── play/<evaluation-id>/
│       ├── result.json
│       ├── telemetry.json
│       └── video.mp4
└── <timestamp>__<event-id>.json
```

## Prepare paths

Run the standard-library-only helper from the repository root before writing
evidence. Use a new snapshot ID for every observation and a new evaluation ID
for every Play attempt:

```bash
eval "$(
  python3 \
    .agents/skills/monitor-tune-isaaclab-training/scripts/prepare_evidence_layout.py \
    --task "$TASK" --run-id "$RUN_ID" \
    --snapshot-id "$SNAPSHOT_ID" --evaluation-id "$EVALUATION_ID" \
    --format shell
)"
```

The assignments provide `RUN_ROOT`, `EVIDENCE_ROOT`, `CRITERIA_PATH`,
`HEALTH_PATH`, `SOURCE_IDENTITY_PATH`, `EFFECTIVE_CONFIG_PATH`,
`SOURCE_PATCH_PATH`, `SUMMARY_PATH`, `ASSESSMENT_PATH`, `PLAY_RESULT_PATH`,
`TELEMETRY_PATH`, and `VIDEO_PATH`.
Omit `--evaluation-id` when no Play
artifacts are needed; shell output then explicitly unsets the three Play
variables so values from an earlier evaluation cannot leak into the current
snapshot. The default JSON format returns `null` for those paths and also
reports the created directories.

The helper accepts only bounded ASCII identifiers, rejects traversal and every
existing symlink component, creates directories but not evidence files, and
fails if any returned target already exists. It performs no Git operation.
Do not bypass a failure by deleting or overwriting evidence; choose a new
snapshot or evaluation ID.

## Write evidence once

Pass the returned paths directly to producers. `collect_training_health.py`
keeps stdout behavior when `--output` is omitted, but file output must be a new
absolute path. `summarize_training_log.py` always requires a new absolute
`--output` path. Both reject existing targets and symlinked parent components.

Create or copy a criteria draft into `CRITERIA_PATH` as a new file. After the
user approves its exact contract and hash, do not edit that file. Use the
matching `HEALTH_PATH`, `SUMMARY_PATH`, and `ASSESSMENT_PATH` for the same
snapshot. A later observation gets a new snapshot ID and a fresh set of paths.

After writing `SOURCE_IDENTITY_PATH`, pass it and the exact absolute RSL-RL run
directory to `capture_effective_training_config.py`; write only to the matching
new `EFFECTIVE_CONFIG_PATH`. The artifact embeds both effective YAML dumps and
their fingerprints. Do not substitute console, TensorBoard, or W&B metadata.

The evaluator requires `PLAY_RESULT_PATH` and accepts `TELEMETRY_PATH` and
`VIDEO_PATH` only when they exactly match its `--task`, `--run_id`, and
`--evaluation_id`. Also pass the matching immutable source identity through
`--run_identity_path` and `--run_identity_file_sha256`. Use `--no_video` and
omit `VIDEO_PATH` when video was not authorized or needed.

Before Isaac Sim starts, the evaluator verifies the checkpoint, deployment
artifact, run-identity file, complete scenario contract, all supplied hashes,
and every canonical output path. It rejects traversal, symlinked components,
existing final targets, and a scenario that conflicts with the identity. A
`.publish-claim` and private `.attempt/` directory serialize writers. Work
files stay in that attempt directory; final video and telemetry are published
with non-overwriting hard links, and `result.json` is linked last as the sole
completion marker. Normal failure removes only claim and attempt objects owned
by that process. A publication collision rolls back only final links created
by that process and never overwrites the competing file.

Version-2 results bind the exact checkpoint, artifact, source identity,
scenario fingerprint, and resource mode. Their `outputs` object binds every
published telemetry/video file by canonical absolute path and SHA-256.
Telemetry version 3 repeats the same `evaluation` and `inputs` objects. Downstream
consumers must revalidate the complete bundle rather than trusting path strings
or the presence of a partial work file.

## Keep events immutable and separate

Raw criteria, health, summary, assessment, result, telemetry, and video files
stay below `evidence/`; source identity, effective configuration, and optional
controlled patch evidence stay below `evidence/source/`.
`record_tuning_experience.py` writes timestamped, append-only event JSON
directly below `RUN_ROOT`; never place those event files inside `evidence/`.

An event may reference raw evidence only by its absolute path plus a SHA-256.
Once referenced, the evidence file is immutable. Record later observations
under new snapshot or evaluation IDs and append a new event instead of editing
the earlier evidence or event.
