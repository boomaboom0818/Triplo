# Prospective adaptive averaging v6

V4/V5remain failed: improving coverage or removing unused tails did not reliably
clear the fixed variance/work gate. V5seed11workratio28.55 was driven by rare
verylargeimportanceweights (~1.86e13), not a license to change RNG/seeds/limits.
All original reports/failures remain unchanged. No v4/v5realtraining occurred.

New cause-supported method: adaptive EXTERNALbranch inclusion, with original
call-mixtureA and current-owner-policy sampling. No guarded mixture. At prefix
weightw, fixed per-pathcap4096*scale, partition actions H={a:w/q(a)>cap} andL=rest.
Enumerate EVERYHaction withunchangedw (inclusion1), sample ONE Laction under
conditionalq(a)/sumL andweightw*sumL/q(a). Every external action remains represented
without bias. Inductively w<=cap at every node, with no clipped/dropped learning.
Owner actions remain on-policy, supplying own reach. Dead-owner fold/all-in tails
have zero average contribution and are pruned; regret remains v3prefix2/8paths.

Explicit version adaptive-average-mccfr-v6, traversaladaptive-average-v6, sampler
adaptive-external-enumeration-average-v6. One fixedcap4096, no cap/seed/mixsearch.
This is distinct from rejected fixed EXTERNAL-prefix enumeration: branching is
selected by the prospective importance bound at any depth, rather than a fixed
early prefix. Tests must sum the entire variable-length sampling tree and match
the exact original averaging target across multiple players/repeated owner/zero
support, and prove no correction exceeds the cap. No counterfactual target change.

## Bounded engineering before registration/learning

Use the SAME8fixed10500000-seriesdeals x6owners x32replicates/8pathbatches and
10400000-seriesRNG as v4/v5, baseline originalA, bothpreserved4906/5150parents.
Every batch has a STRICT2second/10000node safety budget, inside global500000node/
3GiBmemory protections. Failure rejects this proposal at this budget; no retry
unchanged or tuning cap to escape failure. Initial finite deadline2026-11-15T04:00UTC.
Hostreserve2GiB/disk10GiB, two3GiBread-onlyworkers. Freeze allsources/inputs/criteria.
No checkpoints orlearningwrites. Record exact failedowner/deal/replicate onfailure.

Eligibility BOTHlearners: sameaggregatevariance*nodes<=1.5baseline,variance*time<=2,
active2/3postflopupdates/node>=3x,>=20newpositiveparent-missingkeys acrossALL6owners/
EACHstreet, allbatchnodes<=10000, EVERYaverageweight<=4096/8 for these8paths,
andno nonfinite/cost/resource failure. Undefined ratios fail. No oldfailure erased,
no claim independentESS, payoffgain, generalization or strength. Allrawfixtures
retained. A costfailure must be investigated; do not keep enumerating unchanged.

Onlyifeligible: strict newtrainer/config/version/fork/preservation/resume tests,
thennewboundedboth-parent+50/+200/+500learningpilot. Freshscreen10200000/200each
stage vsownparent onvalidator750, final10300000/2000candidate/parent/uniform on
validator750+OTHERseed2000 heldoutfromtraining. ALL4parentlower98.75%>0,
ALL4uniformlower95%>0,no materialpopulationregression, requiredcalling/mixed/
learnedavgpostflop5%/200 andBOTHtight5%/200/+2ppvsparents remain unchanged.
No reset/relabel/clipping/gateweakening. Trainers4GiB/eval3GiB,alloldnode/table/
archive/cp protections. Everynewjobownfinite deadline. No cloud/live/reserved6100000.
Userrenewedworkuntilactualreadiness; keep30mincadence andevidence-guided progress.
