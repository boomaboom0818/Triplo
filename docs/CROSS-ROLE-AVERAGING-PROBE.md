# Prospective cross-role averaging v9 engineering

V8 remains closed NOTREADY: operations and all four primary fresh parent/uniform
payoff tests pass, but A's tight coverage improvement is1.71556pp, below2pp.
Both natural-trajectory diagnostics and their full independent replay show most
remaining tight mean gaps have positive CURRENT own reach; many have regret rows.
No earlier positive means were erased. Calling gaps also expose missing regret
experience. This does not make every gap an averaging error or prove causality.

Select ONE unregistered estimator change: reuse every actor's observations from
the existing six owners' regret trees AND averaging paths. Do not add traversals,
change sampling distributions/regret updates/RNG, mix in heldout opponents, reset
learning, or modify the future-linear target. This is not a revival of failed
guarded-proposal reallocation or external enumeration. No production registration
or training yet; mathematical tests and both frozen engineering runs must pass.

## Literal-prefix expectation

For actor j, r_j is CURRENT own-action reach. Source A_k is the existing owner-k
average path: own actions sampled from current strategy, others original call
proposal. Source B_k is the existing owner-k regret tree: first two own decisions
enumerated (inclusion1), later own actions existing half-uniform proposal, others
the selected frozen population. Track all12 prefix inclusion probabilities and
all6 own reaches and owner-decision depths, irrespective of the current source.

At every visited actor-j node use
`w_j = (future_multiplier/8) * r_j / sum_k(P_Ak + P_Bk)`.
There are eight samples/trees of EACH source, so their summed expected contribution
is exactly `future_multiplier * r_j * strategy_j` at that literal prefix. Chance
sampling is unchanged. Aliased information keys sum literal prefixes. Zero own
reach contributes zero; A_j supplies support for a positive target. Pointwise
weight divided by future_multiplier is bounded by the original A_j estimator,
since its inclusion appears in the denominator. This bound alone does not imply
lower variance. Nonfinite/squared-overflow/positive-underflow values stop safely.

Keep old future origin and all inherited numerators. This explicitly changes only
the estimator, not v8's inherited-uniform/future-linear target. No clock restart.
The prototype is deliberately absent from production config/loaders/fork choices.

## Tests before observations

Exhaustively enumerate all source draws on finite trees, all owners, repeat-owner
prefixes, aliasing, both enumeration depths, zero support, guarded/current frozen
opponents and future multipliers1/50. Check summed expectation against a separately
enumerated exact target. Real six-role Holdem paired full steps must have identical
regret dictionaries, RNG states and node counts. Preserve inherited means, failed
step transaction state and split-step clock equivalence. No production promotion.

## Fixed read-only engineering matrix

After tests, freeze one run `cross-role-probe-001`, BOTH immutable v8 A8906/B9150.
Four fresh11800000..11800003 deals x two population fixtures (historical archive
selection, whole guarded-v2) x32replicates. Each replicate is the FULL equal-count
six-role source set: per role eight regret trees plus eight original average paths.
The guarded fixtures are engineering strata, not a new training population mix.
RNG11900000+candidate*100000+population_index*10000+deal_index*100+replicate.
Use the next preserved v8 clock2001 for BOTH paired methods. No parameter sweep.

For every pair require exact regret/RNG/node checksums. Record each role-batch's
actual runtime/nodes, all role/street/active positive updates/new mean keys and raw
weights, plus stable sparse Welford vector variance over the FULL source set.
A single owner's subset is NOT an unbiased cross-role estimator.

Both learners must meet ALL gates: aggregate variance*nodes ratio<=1.5,
variance*time ratio<=2, total batch runtime ratio<=2, active2/3postflop positive
updates/node>=3x, >=20 unique newly positive mean keys for EACH of all6owners and
EACH postflop street (active2/3 combined), normalized defensive ratio<=1,
finite nonzero actual peakweights, each role8+8batch<=2seconds/10000nodes.
These gates are prospective engineering checks, not payoff evidence or ESS.
All failures are preserved; no repeated unchanged failure or selection of a
favorable subset. Parent bytes, source/config/protocol/test hashes frozen first.

Two hard/soft3GiB read-only workers, reserve2GiB RAM/10GiB disk. Finite invocation
deadline2026-11-15T13:00UTC. No learning deltas committed to parents. Independently
review both complete reports/matrix/ratios/checksums/resources before registration.
Any cost/resource/nonfinite stop requires exact investigation, not blind retry.

## Further work is conditional

Only if BOTH engineering passes: strict separate v8->v9 fork preserving all tables,
old future origin/clock, RNG/archive/lineage, new diagnostic epoch only; exact disk
resume/failed-fork/evaluation/full relevant tests. No real pilot without prospective
fresh payoff and coverage criteria. Explicitly resolve the near100% coverage ceiling
and qualification-reference provenance BEFORE choosing a new training protocol.
No gate or reference changes are authorized by this engineering document; v8's
failed gate remains immutable. No earlier-stage promotion, blind extension,
confirmation redraw, cloud/liveOpenPoker changes, reserved6100000 or strength claim.
User renewed concrete independent work until actual sustained readiness.
