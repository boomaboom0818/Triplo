# Averaging-weight investigation

Candidate A completed 1,750 iterations. Collection began at the preserved
1,500-iteration checkpoint. All recorded weight moments were finite; the largest
single averaging update weight was approximately 8.12e22. Several street/active
aggregates are dominated by a few updates, but these aggregates combine different
information sets. Their concentration cannot by itself diagnose unstable action
probabilities, numerical failure or independent effective sample size.

A read-only diagnostic compared the 1,500 and 1,750 average policies on newly
sampled development deals (seed 5,850,000, 100 deal blocks per population, six
seat rotations, one candidate against calling, mixed, and frozen validation
opponents). It retained checkpoint hashes and used a 3 GiB address-space ceiling.
The executable protocol and full results are under
`runs/bot/weight-investigation-001/diagnose.py` and `report.json`.

| Population | Postflop observed decisions | Previously covered | Covered decisions with action-distribution TV change >0.1 | Covered decisions with TV >0.5 |
|---|---:|---:|---:|---:|
| Frozen validation | 18 | 14 | 1 | 0 |
| Calling | 893 | 286 | 0 | 0 |
| Mixed | 102 | 65 | 1 | 0 |

TV is half the summed absolute difference between normalized legal-action
probabilities. Counts above refer to decisions, often repeated states and reused
deals, not independent samples or unique information sets. Previously uncovered
rows are excluded from probability-drift counts. Their prevalence remains a
coverage limitation; absence of measured drift is not evidence of adequate play.

Across all streets/populations, only one previously covered observed decision had
TV change above 0.5. None of the twenty largest-mass rows in the 1,750 checkpoint
appeared in this limited diagnostic sample. A larger sample or different population
may encounter those rows; this observation cannot prove their irrelevance.

## Decision

Large global/group weights are real, but this check does not establish that they
are the cause of noisy payoff results. The observed common covered policies were
mostly stable. Do not clip weights, reset tables or modify the estimator based
solely on the aggregate values. Preserve the unchanged recipe through the already
authorized 2,000-iteration experiment, compare the next stage evidence, and record
any numerical/normalization failure explicitly if one occurs.

This diagnostic does not repair sparse coverage, establish convergence, replicate
crossplay gains or certify competitive strength. No live OpenPoker setting or
final-test seed was touched.
