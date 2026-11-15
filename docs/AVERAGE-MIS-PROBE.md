# Prospective read-only balanced averaging proposal001

All prior protocols remain closed with original failures intact. The guarded
curriculum failed raw-average coverage and learning; candidate regret backoff
failed its fixed improvement-over-average gate. Do not extend either unchanged,
promote regret fallback, or count regrets as average rows. Earlier external-prefix
enumeration FAILED its original variance/work metric and remains rejected.

This is a DISTINCT engineering hypothesis: choose ONE whole-path external
proposal A or B with probability1/2. A is unchanged call-mixture-v2; B is the
public-only guarded-v2 profile, without epsilon or retuning. Sample owner actions
on CURRENT strategy as before. Track the likelihood of the observed external
prefix under BOTH proposals QA,QB. At each owner information set add
`scale * sigma / (.5 QA + .5 QB)`; owner action sampling supplies its own reach.
The denominator is the MARGINAL prefix proposal, not the selected component's
likelihood. A has full support even when B assigns a free fold zero probability.
For each prefix, this weight is <=2times the original A weight. Log likelihoods
avoid underflow; nonfinite squared weights still STOP, never clip/reset learning.

First require exact finite-tree expectation against exhaustive averaging with
multiple external players, repeated owner decisions, zero owner support and zero
B support; prove/test the pointwise2xbound and nonfinite stop. No production
algorithm/config/checkpoint changes. Only a read-only prototype is implemented.

Then ONE prospectively fixed conditional variance/work probe on immutable
curriculum A4906/B5150, fresh deal fixtures9900000/9900001,128replicates, baseline
original A versus balanced mixture. Same per-replicate seeds, independent copied
RNG, frozen regret strategies; new sum deltas ONLY in memory, never committed to
checkpoint. Record complete variance vector trace over key/action increments,
variance*nodes, variance*seconds, owner postflop active2/3 visits per node,
weight concentration/maxima, max pointwise ratio and hashed vector sums.

Eligible for investigating a NEW versioned training pilot ONLY if >=3/4fixtures
have variance*nodes ratio<.75 AND variance*seconds ratio<1, >=3/4have postflop
active2/3 visit/node gain>=2, all maxtraces<=10000 and all weight ratios<=2+1e-12.
Freeze these criteria, code, parents, deals and128replicates before execution.
No posthoc seed/depth/mix changes or metric substitutions. Failure rejects this
proposal; success is not permission for sustained training or a learning PASS.

Initial finite invocation deadline2026-11-15T02:15UTC, hardAS3GiB, hostreserve2GiB,
diskreserve10GiB, per-trace node500000. No concurrent trainer/evaluator writers.
Atomic output/progress, output lock, immutable source hashes, lifetime peak metrics.
No automatic resource/nonfinite retry. No cloud/liveOpenPoker/reserved6100000.

These four fixed-deal fixtures are conditional engineering observations, not
held-out payoff evidence, independent ESS, complete rare-tail coverage or strength.
The mixture cannot give positive target updates to prefixes with ZERO current
owner-policy reach. It does not solve historical/current-strategy support mismatch
or guarantee improved mean-policy coverage. Any production adoption needs explicit
version/lineage, rollback/resume tests, unchanged learning preservation and a new
prospective bounded two-seed learning/coverage confirmation with original gates.
