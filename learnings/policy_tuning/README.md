# Policy tuning experience records

This directory stores append-only evidence and lessons produced by the
human-guided IsaacLab training advisor.

Records are organized as:

```text
<task>/<run-id>/<timestamp>__<event-id>.json
```

Each event binds one run, algorithm, parameter snapshot, evidence, analysis,
next suggestion, and observation/reward/deployment context fingerprints.
Feedback events additionally identify `sim2sim` or `sim2real` as their source.

Do not copy advice between runs merely because the symptom name matches. Treat
history as compatible only when task, algorithm, and all three context
fingerprints match. Unknown or mismatched context lowers confidence and must be
reported.
