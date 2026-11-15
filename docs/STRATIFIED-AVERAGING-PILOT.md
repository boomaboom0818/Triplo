# Prospective stratified averaging v4 engineering and learning pilot

The six-role isolation001 independently completed all cells/roles. On tight
mean-policy postflop observations A78average misses include68positive-regret rows,
27zero-current-own-reach; B84misses include79positive-regret,29zero-own-reach.
Thus reachable misses dominate, while zero target reach remains a separate issue.
Unchanged root averaging visited active2/3postflop only49/59times over1536paths per
learner, with very large weights. No payoffs from this diagnosis qualify learning.

ONE evidence-guided change: deterministic stratification of averaging paths,
6original call-mixtureA and2public guardedB per owner per iteration, retaining8
mean-scaled regret/average paths and v3prefix-depth2regret unchanged. Owner
strategy stays current, never a scripted profile. For each sampled external prefix
track BOTHQA,QB and use weight `(1/8)/(.75QA+.25QB)` for its average update. The
fixed component counts supply the marginal mixture probabilities. A has full
support, even if B assigns freefold0. Weight<=4/3times original A on the same
prefix. No reset/clipping, target substitution or change to the regret population.
This cannot create positive target mass on zero-current-own-reach prefixes.

Explicit algorithm `stratified-average-mccfr-v4`, traversal `stratified-v4`, sampler
`stratified-call-guarded-average-v4`. This is a new deterministic allocation, not
requalification of the rejected RANDOM50/50whole-path MIS proposal. Its failure
and original eligibility metric remain unchanged. No mix/seed/fixture search.
The fixed25%B allocation matches the existing guarded-regret curriculum share;
no payoff-based ratio selection. All earlier failed protocols remain preserved.

## Prospective engineering check BEFORE production use

Exact finite-tree expectation of the fixed6A/2B average versus exhaustive original
target, repeated owners/multiple external players, zero-owner and zero-B support,
4/3pointwise bound/nonfinite stop and transaction rollback must pass. Trainer
registration/real learning wait until the engineering check is reviewed.

Both immutable A4906/B5150;8fresh10500000..10500007deals x ALL6owners x32replicates,
one complete8path averaging batch per replicate for baseline AND candidate.
RNG10400000+candidate*1000000+owner*10000+deal_index*100+replicate. Frozen current
regrets, temporary sums only. Report per role/street/active counts, stable full
vector variance, nodes/wall and all raw fixtures, including zero variance/visitation.

This proposal trades bounded variance/work inflation for missing-state coverage;
it is NOT evaluated under the rejected proposal's original variance-reduction
criterion and does not erase that failure. Before any candidate measurements,
fixed eligibility requires BOTH learners:
- aggregate(sum over48fixtures variance_trace*mean_nodes) candidate/baseline<=1.5;
- aggregate variance*seconds ratio<=2, measured on the same host;
- owner active2/3postflop updates per node>=3times baseline;
- >=20distinct positive target information keys missing/zero in parent averages,
  in active2/3postflop, including at least one visit for EVERYowner and EACHstreet;
- all batch traces<=10000nodes, ratio<=4/3+1e-12, no nonfinite/resource failures.
Zero denominator stays undefined and fails, never treated as automatic success.
These are engineering risk/coverage gates only, not independent ESS or strength.
Initial finite diagnostic deadline2026-11-15T03:30UTC, two3GiBread-only workers,
hostreserve2GiB/disk10GiB, node500000. Freeze all inputs/code/criteria before run.

## Only after engineering eligibility

Register the new version with strict config/parent-fork validation. Preserve all
existing regrets/sums/RNG/archive/iterations and fork chain. Only declared new
diagnostic epoch starts at the fork. Test wrong-config rejection, byte-identical
parent preservation, rollback and exact interrupted/resumed result. Both parents
fork to new outputs; no old checkpoint relabel or unchanged failed-run resume.

Stages +50/+200/+500 (A4956/5106/5406,B5200/5350/5650), screen10200000/200 after
each completed stage against unchanged ownparent on frozenvalidator750. Material
nominal95%upper<-10 for ANYpopulation stops/investigates. Final fresh2000block
10300000 candidate/parent/uniform on validator750 AND OTHERseed2000. Both heldout
opponents stay out of training/archives. Require all4parent lower98.75%>0,
all4uniform lower95%>0,no98.75%material population regression, calling/mixed/learned
average postflop>=5%/200decisions and BOTHtight>=5%/200/+2pp over unchangedparents.
No silent gate changes; regrets do not count as averages. All regressions/coverage
and sampling counts retained. Approximate paired intervals on weak agents do not
establish competitive strength; scripted training-style overlap is disclosed.

Two trainers4GiB/evaluators3GiB; hostreserve2GiB/disk10GiB, archive2/checkpoint50,
node500000/table1250000 protections. Every training/evaluation invocation has its
own finite deadline, chosen before launch. No cloud/live/reserved6100000. Fixed
evidence-guided work continues until actually qualified; never blind extensions
or parameter/statistical shopping. Test/commit/push authorized changes todevelop.
