# Read-only experience query

Use `scripts/query_tuning_experience.py` to find historical tuning events that
may be relevant to the current run. The query is an inventory and
compatibility check. It never selects a parameter, generates an experiment,
or changes training state.

## Query contract

Pass an existing absolute history root and the complete current context:

```bash
conda run -n isaacsim-5.1 python \
  .agents/skills/monitor-tune-isaaclab-training/scripts/query_tuning_experience.py \
  --root "$ABSOLUTE_POLICY_TUNING_ROOT" \
  --task "$TASK" \
  --algorithm "$ALGORITHM" \
  --host-id "$HOST_ID" \
  --observation-fingerprint "$OBSERVATION_FINGERPRINT" \
  --reward-fingerprint "$REWARD_FINGERPRINT" \
  --deployment-fingerprint "$DEPLOYMENT_FINGERPRINT"
```

The tool scans only direct event files matching
`<root>/<task>/<run-id>/*.json`. It does not recurse into `evidence/`, invoke
Git, contact another host, or write an output file. JSON is emitted only to
stdout. `--max-events` and `--max-event-bytes` bound the scan; exceeding the
event-count limit is an error, while oversized event files are reported as
invalid history.

Each event must:

- pass `record_tuning_experience.py` validation;
- be a regular non-symlinked file under non-symlinked task and run paths;
- match the task and run ID encoded by its storage directory;
- use the immutable filename derived from `recorded_at` and `event_id`.

The query hashes the exact event bytes. A file that changes during reading is
rejected rather than classified.

## Classification

The result contains `compatible_events`, `conflicting_events`,
`unknown_events`, and `invalid_events`:

- `compatible`: algorithm, host ID, and all three context fingerprints are
  known and exactly match the query;
- `conflicting`: at least one known field differs; `classification_reasons`
  lists every known mismatch and any additional unknown field;
- `unknown`: no known mismatch exists, but the query or event contains an
  explicit `unknown` or lacks host identity;
- `invalid`: JSON, schema, path binding, size, or immutability validation
  failed.

Version-1 events remain readable but have no complete host-local identity, so
they cannot be `compatible`. Never treat `unknown == unknown` as a match.

## Output and evidence limits

Each valid result preserves the event's own confidence, parameter snapshot,
analysis summary, next suggestion, absolute event path, and event SHA-256.
`evidence_refs` extracts absolute paths stored in structured `path` or
`*_path` fields and includes a valid adjacent SHA-256 when present.

An evidence reference is not proof that the artifact still exists or is
available on the current host. Verify the referenced file and hash separately
before using it.

Interpret `historical_support.status` as follows:

- `compatible_history_available`: compatible events exist and the scan is
  valid;
- `no_compatible_history`: no compatible event exists;
- `query_context_incomplete`: at least one current query field is unknown;
- `history_invalid`: at least one scanned event could not be validated.

`direct_parameter_change_supported` is always `false`. Compatible history is
candidate evidence only. Combine it with current run evidence, state conflicts
and uncertainty, and present any proposed change to the user for approval.
