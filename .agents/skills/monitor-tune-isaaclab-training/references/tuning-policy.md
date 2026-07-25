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

Run an unchanged baseline with the same environment count, seed set, evaluation window, hardware, code state, algorithm profile, and checkpoint semantics as candidates.

Store each trial separately with effective configuration, argv, git state, profile ID/fingerprint, seed, log, checkpoints, parsed metrics, and stop reason.

Prefer one GPU-heavy trial at a time. Run a short algorithm-appropriate smoke stage before full trials. Reject crashes, non-finite state, throughput collapse, incomplete checkpoint state, or hard constraints.

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

Do not compare algorithms using metrics they do not share. Use profile-specific constraints before weighted objectives. Require all approved seeds.

## Stop and promotion

Stop when budget is exhausted, all candidates violate constraints, infrastructure prevents fair comparison, seed variation dominates, or the next useful experiment needs unauthorized changes.

Report the training-ranked candidate, baseline deltas, seed uncertainty,
exclusions, profile limitations, and next authorization. Do not apply
candidates to tracked files automatically. A training-ranked candidate is not a
final strategy. Require the closed-loop Native/deployment-artifact and visual
workflow in `policy-evaluation.md`; only a passing candidate may become
`simulation_qualified_hardware_candidate`. Do not infer hardware readiness from
simulation.
