# Prospective live-owner stratified averaging v5

V4engineering remains FAILED: seed11variance*nodes1.606>fixed1.5, despite all-role
coverage gains; seed29 passed. No limit/mix/seed change rescues that proposal.
Source b3c34c7 and original reports are preserved. No real v4training occurred.

New evidence-supported computational fix: averaging still simulates other players
after its owner folds or has zero chips. That owner has no further decisions,
so this tail contributes EXACTLYzero to every average coordinate. Stop that tail
using public fold/stack status, with correct player-to-seat mapping. Do not prune
regret traversals, live owners or any positive average contribution. Per-path
updates must match unpruned v4 exactly under the same starting RNG. Subsequent
RNG consumption changes because unused tails disappear; disclose/version it and
require exact interrupted resume for the NEW algorithm. No learning reset/clipping.

Explicit algorithm live-owner-stratified-average-mccfr-v5, traversal
stratified-live-v5, sampler live-owner-stratified-average-v5. Keep all v4math:
fixed6A/2B, original current owner target,1/8scale,1/(.75QA+.25QB),pointwise4/3bound,
v3regretprefix2/8paths andguarded25%regretpopulation. Old rejected proposals remain
failed. No allocation tuning, favorable-seed replacement or criterion relaxation.

## Fixed engineering check

REUSE the SAME prospective10500000-series8deals,6owners,32replicates/8pathbatches,
same10400000-series RNG and baseline as v4. Change ONLY the unused-tail optimization.
Same eligibility limits, BOTH learners: aggregatevariance*nodes ratio<=1.5,
variance*seconds<=2, active2/3postflop updates/node gain>=3, >=20distinct new positive
parent-missingkeys withALL6owners/EACHflop-turn-river, everybatch<=10000nodes,
pointwise<=4/3+1e-12, no nonfinite/resource failures. Undefined ratios fail. All
fixtures/strata/zero data retained. Source/criteria/parent hashes freeze before
execution. Initial finite deadline2026-11-15T03:30UTC, two3GiBread-only workers,
hostreserve2GiB/disk10GiB,node500000. No checkpoint writes or automatic retries.

Tests must establish exact frozen-policy per-path sum/weight equality to unpruned
v4 across ALL6roles/bothcomponents and real fold/all-in states, smaller/equal
node counts, unchanged inputs, plus v4exactexpectation/4/3bound/rollback. Do not
promote on engineering alone. Independent review and unchanged estimator hashes
are required before trainer/loader registration.

## Prospective learning pilot only if engineering eligible

Explicitly fork preserved curriculum A4906/B5150, preserving all regrets/sums/RNG/
archive/iterations/previousfork chain. Strict v5config checks and rollback/exact
resume tests; only declared new diagnostic epochs start at fork. No v4checkpoint
relabel. Stages+50/+200/+500, unchanged ownparent controls. Freshscreen10200000/200
afterEVERYstage onvalidator750; any95%populationupper<-10stops/investigates.
Finalfresh10300000/2000 candidate/parent/uniform onvalidator750andOTHERseed2000,
bothheldoutfromtraining/archives. ALL4parentlower98.75%>0,ALL4uniformlower95%>0,
no98.75%materialpopulationregression, requiredcalling/mixed/learned avgpostflop
>=5%/200decisions, BOTHtight>=5%/200/+2pp over unchangedparents. No regret-count
substitution, silent gates or unsupportedstrength. Scripted overlap is disclosed.

Two4GiBtrainers/evaluators3GiB,hostreserve2GiB/disk10GiB,oldnode500000/table1250000/
archive2/cp50 limits. EVERYnewjob itsown finite deadline. No cloud/liveOpenPoker/
reserved6100000. Continue user-renewed evidence-guided work untilactualreadiness,
notblindextensions orfailed-handoff shutdown. Test/commit/pushtodevelop authorized.
