# Monitoring and recovery

## Evidence hierarchy

Use evidence in this order:

1. Monotonic progress from the resolved profile, such as iteration, epoch, or global step.
2. Latest TensorBoard scalar step and wall time.
3. Explicit completion or failure output.
4. Console-log modification time.
5. Exact process identity and GPU utilization.
6. Checkpoint time and W&B transaction activity.

Items 5 and 6 are auxiliary. A process can remain alive and `.wandb` can grow after training progress has stopped.

Run the first check to establish a baseline:

```bash
conda run -n isaacsim-5.1 python \
  .agents/skills/monitor-tune-isaaclab-training/scripts/collect_training_health.py \
  --profile-id PROFILE_ID \
  --log ABSOLUTE_LOG \
  --tensorboard ABSOLUTE_EVENT_OR_RUN_DIRECTORY \
  --stale-after-seconds 1200 \
  --pid PID \
  --expected-process-pattern TRAIN_ENTRYPOINT \
  --gpu-index 0
```

On later checks pass `--previous-log-progress`, `--previous-tensorboard-step`, and `--previous-observed-at`.

Interpret states:

- `observing`: initial progress baseline recorded; do not recover.
- `healthy`: progress advanced or a TensorBoard scalar is recent.
- `completed`: profile progress reached its target.
- `stopped`: process disappeared before progress became stale; recheck.
- `suspect`: progress did not advance but stall evidence is incomplete.
- `stalled`: incomplete progress is stale and either the process disappeared or GPU utilization is low.
- `unknown`: progress sources or process identity are unresolved.

Only `stalled` is an automatic recovery candidate.

## Recovery sequence

1. Re-run the health check to exclude a transient pause.
2. Confirm the same exact profile, PID, process group, and command.
3. Confirm stale monotonic progress and incomplete target.
4. Confirm the restart count and cooldown.
5. Resolve the exact intended checkpoint and verify it is a readable regular file.
6. Record all evidence and the backend-specific resume argv.
7. Stop only the resolved stalled process through the available task mechanism.
8. Execute the approved resume argv without adding or changing parameters.
9. Establish a new progress baseline and confirm subsequent advancement.

Never delete or overwrite the old run. Never tune during recovery. A recovery resumes the same trial.

Stop and report when process identity is ambiguous, progress semantics are unknown, checkpoint state is incomplete for the selected algorithm, the resume command violates its profile, the restart limit is reached, or the same failure repeats.
