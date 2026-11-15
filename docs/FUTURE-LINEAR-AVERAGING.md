# Prospective future-linear averaging v8

The completed v7 pilot remains closed: operational/coverage PASS, parent payoff
FAIL-inconclusive. No unchanged extension or repeated confirmation is allowed.
The independently reviewed all-role inventory and secondary regret-drift audit
found both sparse revisitation and mean inertia. At many revisited coordinates
current regret strategies changed substantially while their averages barely moved.
This experiment isolates average inertia; it does not claim to fix revisitation.

Select ONE deterministic intervention, with no multiplier/seed sweep: from the
preserved latest v7 A6906/B7150, keep all old regrets, means, RNG, archives, counts,
units and fork history. At the kth NEW completed-iteration attempt since the
explicit fork, multiply ONLY that iteration's genuine v7 average contributions
by k (first new iteration1, second2, etc). Regret updates, policies used within a
step, traversal sampling and RNG draws remain v7. No old row is reset, rescaled,
discounted or clipped. Archives may subsequently change because means change.

This is an EXPLICIT new averaging target, not a relabeling of uniform averaging:

`S_new(I,a) = S_parent(I,a) + sum(k * estimated_original_target_k(I,a))`.

The inherited numerator keeps its original values. At each literal prefix the
original target is ownreach times strategy, conditional on the selected
opponent population; v7's two inclusion-corrected sources have expectation that
target. Multiplication by deterministic k gives the new weighted target exactly,
including repeated owners, imperfect-recall aliases and zero-own-reach cases.
The clock derives from absolute completed iterations minus a persisted fork
boundary. Rollback/resume must never restart it or commit a partial iteration.

Reweighting output strategies is studied in Brown/Sandholm2019,
[Solving Imperfect-Information Games via Discounted Regret Minimization](https://arxiv.org/abs/1809.04040).
This hybrid inherited-uniform/future-linear variant is NOT their full Linear CFR
or Discounted CFR (regrets remain unchanged), and their results do not establish
convergence or strength for this six-player imperfect-recall setup. It is a
cause-supported experimental hypothesis requiring fresh evidence.

Prototype names: algorithm future-linear-average-mccfr-v8, traversal
future-linear-average-v8, sampler future-linear-joint-average-v8. Do NOT register
production loader/trainer/fork until engineering checks BOTH immutable learners.

## Exact correctness and engineering before registration

Finite-tree tests: scalar target on both sources, repeated owner/alias/zero support,
all-six-owner step exact regret/RNG/node equality, legacy means retained, invalid
clock rejected, failed-step full rollback, full/split in-memory clock equivalence.

Freeze a read-only matrix BEFORE measurements: four11300000..11300003 deals x
all6owners x clock1/50/2000 x four paired replicates per learner. Each is one FULL
8regret+8average batch on unchanged frozen policies/population. New RNG formula
11400000+candidate*1000000+owner*10000+deal_index*100+replicate; reuse the same
paired draws for the three clock values. Require every regret/RNG/node checksum
match v7; every candidate mean coordinate equals k*the v7 update within1e-12
relative tolerance; source owner/street/active counts match; no zero-reach inflation.
Report actual raw weights/moments, worst batches and input/source hashes.

Weights and raw variance grow with k and k squared respectively; this is NOT a
variance-reduction claim. The pointwise defensive bound is k times originalA,
or equivalently weight/k <= originalA, not the old unweighted absolute bound.
All squared weights must remain finite. No clipping or erased failures. Require
both learners each paired batch<=2seconds/10000nodes and total candidate seconds
<=2x baseline, no extra nodes/RNG/traversals. Fixed clocks do not select a winner;
all must pass. Hard3GiB read-only workers, reserve2GiB/disk10GiB, finite deadline
2026-11-15T09:30UTC. These engineering checks cannot qualify sustained training.

## Only after engineering passes

Register strict v7->v8 fork, clock boundary/schema, loader/evaluation support and
exact failed-fork/resume/archive/RNG/old-byte tests. Freeze a NEW bounded two4GiB
learner pilot, added50/200/500/1000/2000; ONLYfinal qualifies. Fresh200block
screens11500000, final2000block11600000 versus unchanged own v7parents/uniform
on frozen750/OTHERseed2000 held out from training. Existing all4parentlower98.75%>0,
all4uniformlower95%>0, no materialpopulationregression, requiredaveragepostflop
5%/200 and BOTHtight5%/200/+2pp over ownparents remain. Final gate failure closes
this protocol; no redraw/earlier-stage promotion/blind extension. Every new job
own finite deadline, trainers4GiB/evaluators3GiB, oldnode/table/archive/cp protections.
No cloud/liveOpenPoker/reserved6100000/strength claims. User renewed work untilready.
