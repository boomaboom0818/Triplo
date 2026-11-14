# Sampling pilot 001

Fresh v2 learners, seeds 11/29, 250 iterations each. The recipe
`call-mixture-v1` changes only external-action sampling in average-strategy
traversal: q(a)=0.5/|A| + 0.5 I[a=call]. Every legal action has support and
inverse sampling reach is multiplied by 1/q(a). Regret traversal is unchanged.
An exhaustive small-tree test verifies expected averaging weights; exact resume
and legacy config compatibility are covered by tests. This does not establish
Hold'em convergence or control estimator variance. Increased call sampling can
make trees more expensive; budget stops require investigation.

Outputs: `runs/bot/sampling-001`. Each child has a 3 GiB address-space cap,
3,600-second training invocation allowance, 500,000-node iteration cap and the
existing 1,250,000-set cap. No retries or reset-on-failure. The new recipe is stored
in checkpoint configuration; resumes from legacy recipes are rejected. All
previous checkpoints remain untouched.

Evaluate one-copy primary results on seed 5,200,000, 1,000 independent deal blocks.
Compare the pilot with an explicitly labeled uniform candidate and the original
v2 learner at 250 iterations. All share deals, seat rotations and frozen validation.
Retain per-deal block payoffs and use paired whole-block bootstrap differences.
Also report postflop coverage by active-player count. Final seed 6,100,000 stays
untouched. The old validation agent remains weak and independently trained.

Predeclared screening gate: both seeds must achieve >=5% positive-weight postflop
coverage and >=200 postflop decisions in each of calling, mixed, and learned
populations. Against the frozen learned opponent, paired payoff differences
versus both the uniform candidate and the original v2 control must have a lower
95% bootstrap bound above zero. Other populations remain reported, including
aggressive/tight where postflop counts may be too small for a coverage gate.
Intervals are screening evidence, not a multiplicity-adjusted final certification.
Passing requires review before a bounded 1,000-iteration confirmation stage;
sustained training follows only if that confirmation retains coverage, payoff
and resource evidence. Failed/inconclusive results do not authorize automatic
scaling or relaxed resource limits.
