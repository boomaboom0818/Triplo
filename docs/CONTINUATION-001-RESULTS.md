# Continuation 001 evidence review

Both v1 controls completed 5,250 iterations. Both independent v2 pilots completed
250 and 1,000 iterations. The original 5,000-iteration checkpoints are preserved.
All sixteen one/three-copy evaluation reports were checked against their candidate
and frozen validation SHA-256 hashes, seed 5,100,000 and 1,000 deal blocks.
No training or evaluation worker remains. No sustained campaign was launched.

## Primary one-copy results

Postflop coverage counts decisions with positive average-policy weights. The
samples below reuse deals across seat rotations; decisions are not independent
samples. Calling, mixed and learned populations are separate tests.

| Policy | Calling coverage | Mixed coverage | Learned coverage | Learned BB/100 (95% block interval) |
|---|---:|---:|---:|---|
| A v1 5,000 | 0.02% | 0.12% | 1.29% | 136.6 [-97.9, 385.8] |
| A v1 5,250 | 0.03% | 0.12% | 1.72% | 157.1 [-75.9, 398.5] |
| A v2 1,000 | 0.17% | 12.32% | 46.03% | 33.4 [-199.6, 285.5] |
| B v1 5,000 | 0.03% | 0.00% | 1.42% | 121.7 [-110.7, 353.7] |
| B v1 5,250 | 0.03% | 0.00% | 1.43% | 135.7 [-97.8, 369.5] |
| B v2 1,000 | 0.20% | 21.27% | 46.61% | -15.8 [-239.1, 224.0] |

V2's learned-opponent postflop coverage is 110/239 decisions for A and 110/236
for B. Against calling opponents it is only 16/9,169 and 18/9,030. Thus the
coverage improvement is population-dependent and broad postflop coverage remains
inadequate. In particular, the proposed 5% diagnostic gate is not met across the
relevant populations. The 250-iteration pilots and three-copy results remain in
`runs/bot/continuation-001`; three-copy results are secondary table compositions.

The intervals above estimate each policy's payoff against the weak independent
250-iteration validation agent. They are not paired policy-difference intervals.
All include zero. Lower v2 point estimates do not establish a regression, and
coverage improvement does not establish better poker play. The uniform-policy
control and retained per-deal samples required for a stronger promotion decision
have not yet been implemented. Consequently credible payoff evidence remains
an unmet requirement, regardless of the coverage result.

## What traversal diagnostics add

At 1,000 iterations, v2 A had 80,882 union information sets and B had 80,235.
Their total recorded training seconds were 2,447 and 1,908, respectively.
These are separate runs, not a controlled throughput comparison.

Postflop nonterminal visits in the averaging traversal:

| Active players | Candidate A | Candidate B |
|---|---:|---:|
| 2 | 131,225 | 123,774 |
| 3 | 41,157 | 40,606 |
| 4 | 10,252 | 11,262 |
| 5 | 2,049 | 2,018 |
| 6 | 253 | 215 |

Regret traversal also had very few six-player postflop visits: 200 and 149.
These counters include sampled-opponent and enumerated-owner visits; they are
neither independent hands nor unique information sets. They confirm that these
training traversals spend very little effort in six-player postflop situations.
This supports investigating sampling coverage, but does not by itself prove the
sampler is biased or that changing it will improve payoffs.

## Decision and next experiment

**The evidence gate is not met. Keep sustained training stopped.**

The v1 continuation provides little additional postflop coverage. V2 demonstrates
that state fragmentation can be reduced, but leaves multiway calling populations
largely uncovered. Further blind iteration increases are not justified by this
pilot.

Next, instrument evaluation by active-player count and retain per-deal payoff
samples. Add a uniform legal-policy control and paired block comparisons. Then
specify a bounded sampling experiment that increases support for multiway
postflop situations while retaining correct sampling probabilities and importance
weights. Validate it against exact small-game expectations and deterministic
resume/rollback tests before any Hold'em pilot. Do not simply force calls or
start from arbitrary postflop states under the existing estimator.

Keep the v1/v2 checkpoints as controls, use a new named recipe for any sampler
change, retain 3 GiB worker ceilings and existing node/table budgets, and leave
final test seed 6,100,000 untouched. A stronger independently validated opponent
is still needed before any competitive-strength claim.
