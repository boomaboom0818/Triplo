# Sampling diagnostics compatibility

Legacy visit suffixes `enumerated`/`sampled` identify owner/other in the original
external sampler. They are not reliable branch labels for trajectory/prefix
algorithms. Historical counters and checkpoints are retained without relabeling.

New trajectory/prefix visits use `sampling-v2:<traversal>:street=<s>:active=<n>:
actor=<owner|other>:<sampled|enumerated>`. Trajectory regret/averaging actions are
sampled; prefix regret enumerates only its first configured owner decisions.
Trainer metadata records `sampling_diagnostics_schema=actual-actions-v2` and the
first completed-iteration boundary. Legacy resume starts this boundary at its
saved iteration; existing actual-sampling boundaries survive resume.

Tests compare labels enabled/legacy across identical seeds: regrets,sums,RNG,
nodes,iterations,weight moments and per-fork statistics remain identical. A legacy
checkpoint retains old counters and unchanged learning while gaining an explicit
new namespace/boundary. This is an observational correction, not a policy change.
Direct epoch weight moments remain separate; their concentration proxies use
correlated node weights and are not inferential independent effective samples.
