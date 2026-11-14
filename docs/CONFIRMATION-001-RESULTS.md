# Independent development check: 1,000 iterations

Both learners completed 1,000 iterations. Six independent-development reports
(candidate, original 250-iteration parent, uniform control for each learner)
were verified against checkpoint hashes, opponent hash, evaluation mode, seed
5,500,000, and 2,000 retained deal blocks. Final-test seed 6,100,000 was untouched.

| Learner | Learned-opponent difference vs parent, BB/100 [95% paired interval] | Difference vs uniform |
|---|---|---|
| A | +110.88 [8.50, 217.19] | +147.10 [31.98, 261.68] |
| B | +90.42 [-3.75, 186.95] | +147.24 [37.71, 261.73] |

Both pass the predeclared postflop coverage screen on calling/mixed/learned
populations. A passes the paired payoff screen; B's parent comparison remains
inconclusive. **The predeclared two-seed gate did not pass.** The interval is not
rounded up or reinterpreted as proof of equivalence/superiority.

Both are worse than their parents on the tight script (A: -2.41 [-3.11,-1.73];
B: -1.49 [-2.13,-0.83]). Tight postflop coverage remains zero. This is a robustness
limitation, although that script is weak and uniform play exploits it easily.
Other scripted results vary by population and must not be pooled into a strength
claim. Nominal intervals are not a multiplicity-adjusted certification.

## Authorized next bounded experiment

The overnight user instruction explicitly authorizes continued local exploration
and autonomous judgment. Continue the unchanged recipe into 1,250 and 1,500 total
iterations, using the verified 1,000 checkpoints as immutable parents. This is
**exploratory**, not evidence-qualified promotion or indefinite sustained training.
The evidence suggests learning progress and broad coverage gains while leaving
uncertainty and robustness gaps; a bounded extension can measure whether those
improve without increasing caps or changing multiple mechanisms at once.

Use new development seed 5,400,000, one-copy evaluations with 1,000 independent
deal blocks at each stage, paired against the 1,000 parent and uniform policy.
Preserve all stages and compare the tight-script gap as well as the primary
learned-opponent and coverage results. Review before further extension.

The runner records purpose=exploratory, exact initial/per-stage parent hashes and
configurations. It enforces 3 GiB per worker, the existing 1,250,000-information-set
and 500,000-node iteration budgets, one-hour invocations, validated atomic
reports, writer locks and the 17:31 UTC overnight deadline. No reset or relabeling
of existing learning, no live deployment, and no cloud spending are involved.
