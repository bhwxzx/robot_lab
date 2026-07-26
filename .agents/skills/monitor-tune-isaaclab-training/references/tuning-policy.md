# Bounded tuning policy

## Authorization

Require a reviewed non-generic algorithm profile. Discover candidates from current source and dumped effective configuration. For every candidate show:

- exact path, source, effective value, and type;
- proposed values or bounded range;
- verified override mechanism;
- expected effect and algorithm-specific risk;
- matching protected-profile rules.

Let the user select paths and domains. A suggestion, previous session, monitor permission, or generic request to optimize does not authorize a parameter.

Keep `mutation_scope` at `overrides_only`. If a parameter requires editing tracked configuration, rewards, data, or algorithm code, stop and use the repository modification workflow.

## Experimental design

Choose screening and confirmation seeds before observing outcomes. Run an
unchanged baseline with the same environment count, seed stage, evaluation
window, hardware, code state, algorithm profile, and checkpoint semantics as
candidates. Screen every trial on the same small seed set, then confirm only
the baseline and top-k eligible candidates on the remaining seeds. Use at least
two total confirmation seeds; prefer three or more for a final recommendation
when the budget permits.

Do not treat a resume from the same RNG state as an independent seed. If trials
share a pretrained checkpoint, label the result as fine-tuning variability
rather than full-training variability.

Store each trial and retry separately with effective configuration and diff,
exact argv, unique run ID, Git state, profile ID/fingerprint, stage, seed, log,
checkpoints, parsed metrics, anomaly decision, and stop reason.

Prefer one GPU-heavy trial at a time. Bind state to the exact approved session
and deterministic plan, require an idle GPU and exclusive lock, and never reuse
a prior attempt's outputs. Run a short algorithm-appropriate smoke stage before
full trials. Reject crashes, non-finite state, sustained approved anomaly
conditions, incomplete checkpoint state, unauthorized effective-config
differences, or hard constraints.

## Metrics

Choose objectives and constraints before candidate outcomes. Start from metrics exposed by the resolved profile, then add task metrics from the environment.

Common locomotion metrics include reward, episode length, velocity errors, timeout, illegal contact, terrain bounds, stability rewards, iteration time, and throughput.

Add only relevant algorithm metrics:

- PPO: value, surrogate, entropy, KL, and learning rate.
- Distillation: student/teacher action loss and validation error.
- DWAQ: autoencoder, velocity, KL, and latent losses.
- AMP variants: discriminator loss, gradient penalty, policy/expert predictions, and style/task balance.
- ROA variants: history latent, velocity, privileged regularization, and DAgger-route losses.
- New algorithms: metrics whose meaning and direction were verified during profile review.

Do not compare algorithms using metrics they do not share. Use profile-specific
constraints before weighted objectives. Check hard constraints on every seed,
not only on their mean.

For final training ranking, report each objective's mean, sample standard
deviation, range, and 95% t interval. Pair every candidate seed with the same
baseline seed and report mean and worst paired improvement. Enforce any
predeclared minimum improvement before scoring. Identify the multi-objective
Pareto front, then use the approved normalized weighted score only to order
surviving Pareto candidates.

## Physical-feedback input

Use `hardware-feedback-retuning.md` when supervised deployment feedback
motivates another experiment. Treat the user report as evidence about a
specific archived policy and test, not as permission to edit a reward or
algorithm.

First exclude artifact, observation/history, normalization, reset, timing,
configuration, communication, calibration, actuator, and mechanism failures.
Reproduce the reported segment in simulation and define a measurable signature.
Only then offer relevant paths already present in
`tuning.allowed_parameters`. Do not automatically select a path or narrow,
expand, or replace its user-approved domain. A newly approved session is
required even when every proposed path existed in the earlier session.

## Stop and promotion

Stop when budget is exhausted, all candidates violate constraints, infrastructure prevents fair comparison, seed variation dominates, or the next useful experiment needs unauthorized changes.

Report the screening selection, training-ranked candidate, paired baseline
deltas, seed uncertainty,
exclusions, profile limitations, and next authorization. Do not apply
candidates to tracked files automatically. A training-ranked candidate is not a
final strategy. Require the closed-loop Native/deployment-artifact and visual
workflow in `policy-evaluation.md`; only a passing candidate may become
`simulation_qualified_hardware_candidate`. Do not infer hardware readiness from
simulation.
