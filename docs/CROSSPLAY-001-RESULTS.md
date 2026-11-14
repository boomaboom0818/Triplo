# Independent-learner crossplay screening

All six reports (candidate, own parent, uniform control for both learners) were
verified against checkpoint/opponent hashes, evaluation mode, seed 5,600,000 and
1,000 retained deal blocks. Both candidates were frozen at 1,500 iterations; each
faced five copies of the other seed's independently trained 1,000-iteration policy.
Neither opponent was used in candidate training. This is an additional local
screen, not final testing, live OpenPoker evidence or an established strong
benchmark. The original validation criteria were unchanged.

| Candidate | Paired difference vs own 1,000 parent, BB/100 [95% interval] | Difference vs uniform |
|---|---|---|
| A vs frozen B | -72.33 [-167.58, 32.24] | +123.96 [-52.48, 295.03] |
| B vs frozen A | -16.41 [-124.48, 88.40] | +85.52 [-86.63, 261.46] |

All four nominal intervals include zero. This screen establishes neither
incremental improvement nor a statistically clear regression or equivalence.
Absolute A/B payoffs cannot rank the candidates because their opponents differ.
It does not replicate the gains observed against the weak 250-iteration validator.
Robustness therefore remains an unresolved readiness requirement.

The already authorized 1,750/2,000 diagnostic extension remains exploratory and
preserves both 1,500 checkpoints. It measures averaging-weight concentration
without changing the estimator. Crossplay uncertainty does not justify clipping
weights, resetting learning, raising resource caps or claiming promotion.
Results and exact hashes remain under `runs/bot/crossplay-001`. Final-test seed
6,100,000 and live OpenPoker configuration remain untouched.
