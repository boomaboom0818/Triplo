# Local poker training readiness handoff — 2026-11-15

The bounded local training setup is operationally validated. The current models
are **not qualified for sustained training**: required learning gains and raw
average-policy postflop coverage remain unproven/failed. No competitive strength
is established. All completed stages and the final planned evidence were reviewed;
there is no unmonitored training job to resume or extend automatically.

## Preserved state and evidence

Latest complete training checkpoints are:

|Learner|Iterations|Checkpoint|SHA256|
|---|---:|---|---|
|seed11|4906|`runs/bot/coverage-curriculum-001/candidate-0-plus-300.json`|`39a85c00d35e70d1e41057e7bcdd3be1e6d15d097034ddbabfab1c60ae0e5675`|
|seed29|5150|`runs/bot/coverage-curriculum-001/candidate-1-plus-300.json`|`73969044a0db8397b7a6c054855f23e619465f596377d4ab0b1a2cc9817a930b`|

They retain `prefix-outcome-mccfr-v3`, texture abstractionv2, eight mean-scaled
regret/average paths, prefix depth2 and the explicit guarded-tables-v2 population
fork. Original regrets/sums/RNG/archive/iteration and earlier fork lineage remain
preserved. These are useful research parents, not a promoted policy. Immutable
v3parents4606/4850 and earlier checkpoints/failures remain available.

|Evidence root|Operational result|Learning/component result|
|---|---|---|
|`scaling-validation-001`|FAIL: original iteration node cap|Incomplete; closed, never retry unchanged|
|`traversal-pilot-001`|PASS|FAIL required learning gains|
|`prefix-pilot-001`|PASS|FAIL required learning gains|
|`coverage-curriculum-001`|PASS|FAIL all4parent lower98.75% bounds; extra tight average coverage fails both seeds|
|`policy-backoff-001`|PASS|FAIL all4required average-inference gain bounds; all4uniform bounds positive|
|`average-mis-probe-001`|PASS|Engineering FAIL; rejected, no production adoption|

Each root contains its original manifest, reports, metrics and journal. Latest
`qualification.json`/`QUALIFICATION.md` independently reconstruct all12curriculum
confirmation reports and all24fallback screen/confirmation reports. All completed
training stages were screened. All populations, regressions and street/active-count
coverage remain available; positive regret rows never count as positive averages.

## What the investigations establish

The guarded curriculum added positive regret rows in sampled tight postflop
states while average rows stayed missing. Standard CFRPolicy uses only averages
and falls back to uniform. This identified a concrete inference/averaging gap,
not proof that those regrets improve play. The candidate-only positive-regret
fallback passed exact control replay/held-out identity tests but failed its fixed
fresh2000block improvement-over-average gate. Do not promote or extend it unchanged.

The distinct whole-path balanced averaging proposal passed exact expectation and
pointwise2xweight-bound tests but failed its engineering eligibility criteria.
Its first-actor conditional fixtures often only update the root (fold/all-in
current support), leaving baseline active2/3postflop visit rates undefined. That
is a limitation of the evidence, not a positive coverage result or proof of a
poker bug. No owner/deal/mix/metric retuning was used to rescue it. Earlier failed
external-prefix averaging enumeration remains rejected separately.

A diagnostic numerical bug was fixed for future use: subtracting large raw
vector moments produced tiny false variance on a constant vector and can hide
small variation beside a huge constant coordinate. The replacement uses sparse
per-coordinate Welford moments, including missing coordinates as zero. Nineteen
targeted tests cover the prototype and repair, exact baseline replay, immutable
checkpoint bytes, zero-support expectations, bounds, constant/large/sparse vectors
and nonfinite/shape rejection. No production trainer or checkpoint changed.
Original probe reports/failure were not rerun or rewritten; frozen source46790b8
is archived in its run root and remains in git. Qualification hashes were checked
before the diagnostic-only repair; future code declares its new diagnostic schema.

## Required next research design

No further unchanged long run is justified by this evidence. A future sampling
study must prospectively represent all six updater/owner roles and relevant
postflop situations, disclose zero-current-owner-reach cases, and keep a fixed
work/variance/coverage objective. Do not select replacement fixtures for a favorable
result. The rejected proposals remain rejected. A different abstraction or
averaging estimator needs its own explicit version, exact target/full-support
tests and source lineage, preservation and interrupted-resume tests before use.

Before another sustained campaign require BOTH operational and learning passes:
all4independent parent-change lower98.75% bounds>0, all4learned-versus-uniform
lower95% bounds>0, no declared material population regression, required
calling/mixed/learned positive-average postflop coverage>=5% on>=200decisions,
and BOTH seeds' tight positive-average coverage>=5%,>=200decisions and>=2pp gain
over their unchanged parents. Any replacement criterion must be prospectively
declared for a new question; it cannot erase these failed gates. Weak checkpoint,
scripted and empty-table blueprint agents do not establish competitive strength.
OpenPoker/live evidence remains separate and was not changed.

## Safe execution and recovery

Use at most two trainers with4GiB hard address-space limits each, at most two
evaluation children with3GiB each, and reserve at least2GiB host RAM/10GiB disk.
Preserve per-iteration500000node/table1250000 protections, checkpoint50, archive2
and a finite invocation deadline for EVERY new job. The old global morning cutoff
is not a stop condition; each individual job still has its own deadline.

Inspect actual processes/limits, checkpoint/log mtimes, iteration and stop reason,
health and lifetime attempt metrics. Never infer a stop solely from stale status,
and never start a duplicate writer. Verify manifest/config/input/source hashes and
the latest COMPLETE checkpoint/RNG/lineage before any resume. Preserve `.previous`,
partial logs and failed attempts; pending reports are not completed evidence.

An OOM/node/table/nonfinite failure needs an investigated, tested change and new
prospective protocol where applicable. Never loop an unchanged retry, silently
reset/relabel learning or clip learning to force progress. The original scaling
seed11iteration2607 node failure stays closed. Completed old deadlines/commands
are historical; copying them is not a new safe invocation. No cloud spending,
live OpenPoker changes or reserved final-test6100000 use.

All bounded jobs are complete and reviewed. Routine monitoring can be disabled
because the tested setup/evidence review and this actionable handoff are complete;
this is not a claim that the model passed, nor an irrecoverable-block diagnosis.
Any later authorized research should start from these preserved facts with its
own bounded design, not an automatic extension seeking favorable statistics.
