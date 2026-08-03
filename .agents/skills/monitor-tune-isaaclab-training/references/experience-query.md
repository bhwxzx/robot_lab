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
  --run-identity "$SOURCE_IDENTITY_PATH" \
  --effective-config "$EFFECTIVE_CONFIG_PATH" \
  --effective-config-sha256 "$EFFECTIVE_CONFIG_SHA256" \
  --observation-fingerprint "$OBSERVATION_FINGERPRINT" \
  --deployment-fingerprint "$DEPLOYMENT_FINGERPRINT"
```

The query revalidates the current identity and effective-config artifact, then
derives task, algorithm, host ID, run ID, and reward fingerprint. It rejects a
current artifact outside that run's `evidence/source/` directory, a mismatched
whole-file SHA-256, or any internally inconsistent embedded YAML evidence.

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

Every version-3 event must also reference one effective-config artifact under
its own `evidence/source/` directory. For events whose algorithm, host,
observation, and deployment context match the current query, the tool verifies
that artifact's path, whole-file SHA-256, run identity, reward fingerprint, and
internal semantic fingerprints before classifying it as usable history.

The query hashes the exact event bytes. A file that changes during reading is
rejected rather than classified.

## Classification

The result contains `compatible_events`, `conflicting_events`,
`unknown_events`, and `invalid_events`:

- `compatible`: a version-3 event has verified effective-config evidence and
  its algorithm, host ID, and all three context fingerprints exactly match;
- `conflicting`: at least one known field differs; `classification_reasons`
  lists every known mismatch and any additional unknown field;
- `unknown`: no known mismatch exists, but the query or event contains an
  explicit `unknown` or lacks host identity;
- `invalid`: JSON, schema, path binding, size, or immutability validation
  failed.

Version-1 and version-2 events remain readable but lack the required verified
effective-config binding, so they are always `unknown`. Never treat
`unknown == unknown` as a match.

A version-3 event with matching algorithm, host, observation, and deployment
context remains comparison-eligible when only the reward fingerprint differs.
It stays `conflicting`, but its `parameter_diff` shows the verified historical
configuration as baseline and the current configuration as current. Events
from another host or incompatible observation/deployment context are not
dereferenced on the current host.

## Output and evidence limits

Each valid result preserves the event's own confidence, parameter snapshot,
analysis summary, next suggestion, absolute event path, and event SHA-256.
`evidence_refs` extracts absolute paths stored in structured `path` or
`*_path` fields and includes a valid adjacent SHA-256 when present.

An evidence reference is not proof that the artifact still exists or is
available on the current host. Verify the referenced file and hash separately
before using it.

`effective_config_verification` reports whether a version-3 artifact was
verified, skipped because its context belongs elsewhere, or unavailable on a
legacy event. A verified `parameter_diff` is complete and deterministic. It
contains semantic JSON-Pointer changes plus separate reward-weight and selected
training-parameter changes. `--max-diff-entries` bounds complete comparison;
exceeding it makes that candidate invalid instead of silently truncating it.

Interpret `historical_support.status` as follows:

- `compatible_history_available`: compatible events exist and the scan is
  valid;
- `no_compatible_history`: no compatible event exists;
- `query_context_incomplete`: at least one current query field is unknown;
- `history_invalid`: at least one scanned event could not be validated.

`direct_parameter_change_supported` is always `false`. Compatible history is
candidate evidence only. Combine it with current run evidence, state conflicts
and uncertainty, and present any proposed change to the user for approval.
