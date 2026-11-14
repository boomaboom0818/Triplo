# Bounded 1,250/1,500 extension results

Both independent learners reached 1,500 iterations with target_reached. All four
stage evaluations and paired reports were validated against the immutable parent
and opponent hashes, stage recipe, seed 5,400,000, 1,000 retained block payoffs,
report means and independently recomputed paired statistics. No worker failure,
memory stop, table/node exhaustion or recovery occurred.

At 1,500 iterations, A has 190,529 union information sets and 8,103.6 cumulative
training seconds; B has 185,079 sets and 7,864.4 seconds. Original parents,
intermediate checkpoints and evaluations remain preserved.

| Learner | Learned-opponent difference vs 1,000 parent, BB/100 [95% paired interval] | Difference vs uniform |
|---|---|---|
| A | +45.24 [-54.17, 145.43] | +24.13 [-150.42, 199.33] |
| B | +42.67 [-65.14, 147.82] | +138.01 [-45.37, 312.93] |

Incremental payoff progress on this development seed is inconclusive. Tight-script
paired differences vs parent are also inconclusive (A +0.175 [-0.375,0.725];
B -0.208 [-0.858,0.417]), and the gap vs uniform persists. These results do not
change the failed earlier two-seed promotion gate or establish competitive play.
Additional crossplay against the other independently trained 1,000-iteration
policy is separate screening and was still running at this review.

## Next explicitly exploratory extension

Under the user's overnight authorization, continue the identical recipe from
1,500 into new 1,750 and 2,000 checkpoints. This bounded extension will also collect
previously unavailable averaging-weight moments: update counts, total/squared
weight, and maximum weight grouped by street and active-player count. Only
completed iterations contribute; resume restores the moments and records the
iteration where collection began (1,500 for these legacy parents).

This instrumentation tests a possible explanation for noisy learning: concentration
of importance-weighted averaging updates. That explanation is a hypothesis, not
a diagnosed defect. Branch visits are correlated; a total-weight-squared over
squared-weight proxy is not a count of independent samples. It must not be used
as an uncertainty interval or evidence of strength. Instrumentation neither changes
the sampler nor clips/reweights learning. Tests compare exact regrets, averages,
RNG and nodes with instrumentation disabled, and verify exact resume/rollback.

Use fresh development seed 5,700,000 for paired comparisons against the immutable
1,500 parent and uniform control at each stage. Preserve 3 GiB worker limits,
existing table/node caps, 50-iteration checkpoints and one-hour invocation
budgets. Keep the 17:31 UTC deadline and evaluate every completed stage.
Promotion remains subject to actual payoff/robustness evidence; more iterations
and completion alone are not success criteria. Final-test seed 6,100,000 and all
live OpenPoker settings remain untouched.
