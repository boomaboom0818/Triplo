# Prospective joint-traversal averaging v7

All prior protocols remain preserved with failed outcomes. V6adaptive enumeration
passed finite-tree correctness tests but its uniform-policy cost smoke hit the
strict2second guard; it is NOT qualified and no real-parent v6retry is authorized.
The passing interface test uses a deterministic six-role tree separately from
feasibility, not an erased performance failure. No v4/v5/v6realtraining occurred.

## Exact same-target reuse, with no extra traversal tree

Existing v3per owner/iteration hasNregret-prefix trees B andNaveraging paths A.
They already sample useful states differently. Reuse BOTH for averaging, while
leaving every regret action, update and RNG draw unchanged. Condition on the
iteration's selected historical/guarded opponent policies. For literal prefixh:
- ownreach r is the product of CURRENT owner action probabilities;
- PA=r*QA, whereQA is original call-mixture external prefix likelihood;
- PB is regret-tree inclusion probability: external selected-policy probabilities,
  inclusion1for the first2enumerated owner decisions, then sampled ownerq thereafter.
At each visited owner node from EITHERsource add
`(1/N) * r/(PA+PB) * sigma`. Eachsource hasNtraversals, so expectation of the pair
is `(PA+PB)*r/(PA+PB)*sigma`, the ORIGINALtarget per iteration. Regret-tree
branching is handled by prefix inclusion probability, not a falsely normalized
whole-tree density. Zero ownreach gives ZEROupdate. A is full-support for every
positive-own-target prefix, even if B's opponents have zero support. The new
weight on any prefix is <=the original Aweight `(1/N)/QA`; no clipping/reset.

Explicit algorithm joint-traversal-average-mccfr-v7, traversaljoint-average-v7,
samplerregret-path-mis-average-v7. Same8regret+8average traversals, prefix2,
guarded25%regret objective. This is neither failed fixed6A/2Ballocation nor
conditional unweighted copying of regrets into averages. These are genuine
reach/inclusion-corrected strategy averages, preserving all prior table units.
No extra sampled trees, averaging target change, sampling/action/RNG change or
extra exploration parameter. Future archives may change because averages improve.

Finite-tree tests must match the exact original target with repeated owners,
multiple external players, first-owner enumeration and later sampling, zero-own/
zero-opponent support, historical/current/guarded opponents and imperfect-recall
key aliases. Test full-step EXACTregret/RNG/node equality against v3 with same
depth/config, no-average-inflation bound and transaction rollback.

## Fixed all-role engineering BEFORE production registration

Two preserved A4906/B5150,8fresh10800000..10800007deals x6owners x32replicates.
Eachreplicate is one FULLowner batch:8regret-prefix trees PLUS8averaging paths,
for BOTHv3baseline andcandidate. Same10400000+candidate*1000000+owner*10000+
deal_index*100+replicate RNG. Opponent selection uses a separate copy of this
initial RNG under the unchanged25%archive/25%guarded recipe, identical for the
pair. Fixedconditional root deal, both consume the original factory-seed draws.
Require exact regret-delta/finalRNG/node checksum equality for EVERYpairedbatch.
Never commit frozen local regret deltas or temporary mean sums to checkpoints.

Same engineering risk/coverage limits, BOTHlearners: aggregatevariance*nodes
ratio<=1.5, variance*seconds<=2, active2/3postflop averageupdates/node>=3x,
>=20newpositiveparent-missingkeys acrossALL6owners/EACHstreet, allbatchnodes<=10000,
and newprefixweight/originalAweight<=1+1e-12. Undefinedratiosfail. Node/work here
include regret trees alreadyrequiredbytraining, rather than incorrectly pricing
reused observations as new simulations. Source counts distinguish actual A/B
average contributions; regret presence alone is not coverage. Stable variance,
allfixtures/zero cells retained. No independentESS/payoff/strength claims.

Strict per-full-owner-batch2second/10000node cost guard, actual3GiBread-onlyworkers,
hostreserve2GiB/disk10GiB, initial finite deadline2026-11-15T04:30UTC. Exact failed
batch retained if stopped; no unchanged resource retry. Freeze allsources/parents/
criteria before execution. Verify all paired checksums/gates/lifetime peaks.

Onlyifeligible register strict v7loader/trainer/config/fork with same original
learning/RNG/archive/iterations/chain preserved, only declared new diagnosticepoch.
Exact interrupted resume/failed-fork/parent-byte tests before real training.
Prospectively fix BOTHparents +50/+200/+500/+1000/+2000 BEFORE any candidate
payoff measurement. The final exposure is larger than earlier short pilots so
the new estimator can accumulate learning; this is not a posthoc extension of a
failed confirmation. Intermediate stages NEVER qualify/promote. Fresh10200000/
200screens every stage (any95%upper<-10 stops/investigates), final+2000 only:
10300000/2000candidate/unchangedparent/uniform on
validator750andOTHERseed2000heldoutfromtraining. ALL4parentlower98.75%>0,
ALL4uniformlower95%>0,no materialpopulationregression, requiredcalling/mixed/
learnedpositive-averagepostflop5%/200 andBOTHtight5%/200/+2pp remain unchanged.
Two4GiBtrainers/eval3GiB,oldnode500000/table1250000/archive2/cp50 protections.
Everynewjobownfinite deadline. No cloud/liveOpenPoker/reserved6100000.
Userrenewedworkuntilactualreadiness;30minute cadence, no failed-handoffshutdown,
no statisticalshopping orblindunchangedextensions. Commit/push testedchanges.
