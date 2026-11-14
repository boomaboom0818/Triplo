# Prospective trajectory pilot 001

This is a new algorithm experiment following scaling-validation-001's node-limit
failure. It does not change or pass that closed protocol. User authorized continued
local investigation, implementation and bounded runs without routine approval.

## Estimator and immutable lineage

Parents are the preserved latest complete seed11@2606 and seed29@2850 states:
`runs/bot/scaling-validation-001/candidate-{0,1}-2850.json`. The first filename
contains a partial stage, not2850 completed iterations. New algorithm identity is
`outcome-sampling-mccfr-v2`, traversal `trajectory-v2`, average sampler
`trajectory-call-mixture-v2`. Fork requires a new output, explicit flag, compatible
recipe and parent hash. All regrets, sums, RNG, archive and completed iterations
carry forward. No old checkpoint is rewritten, relabeled or discarded.

Each iteration averages8 independent regret paths and8 averaging paths per player,
with frozen tables during the iteration and transactional commit/rollback. Regret
updater sampling is q(a)=0.5*sigma(a)+0.5/|A|. An action-value estimate divides by
the sampled action's q; descendant regret updates also divide by the product of
earlier updater sampling probabilities. Other players use the original current
or selected frozen population policy, so their probabilities cancel in expectation.
For averaging, owner actions are sampled on-policy: probability of visiting an
owner history supplies its own reach. Other actions use the full-support call
mixture and inverse external sampling reach. Each path's update is scaled1/8.
These corrections retain the existing regret/average update targets in expectation;
variance and practical learning remain empirical questions. Importance weights are
not clipped. Nonfinite estimates stop and roll back the iteration.

The general sampling/importance-weighting basis is described by
[Lanctot et al.,2009](https://proceedings.neurips.cc/paper_files/paper/2009/file/00411460f7c92d2124a67ea0f4cb5f85-Paper.pdf).
The particular implementation is checked by exact enumeration on a multiplayer
toy with repeated owner decisions, zero-probability target actions, independent
resume/rollback tests, known two-player Kuhn value and evaluator coverage tests.
Neither that paper's two-player guarantees nor Kuhn tests establish strength or
equilibrium for this six-player imperfect-recall abstraction.

## Bounded design and resources

Before pilot launch, reproduce/profile the blocked original iteration once on a
read-only fork, with500000 nodes,20minutes and4GiB. Retain phase/owner/street/active
counts, including aborted traversal work. Do not retry the original learner.

Continue each parent by50/250/750/1500 additional complete iterations. Absolute
targets differ by parent; report both added iterations and sampled path work.
Preserve abstractionv2, historical population probability0.25, archive2, inherited
node500000 and table1250000 caps. Two independent training workers,4GiB hard
address-space limits each,2GiB host reserve,10GiB disk reserve, checkpoint50,
one-hour child allowances. Evaluators: at most2,3GiB each. Every invocation has
an explicit finite UTC deadline no more than12hours from launch; planned initial
deadline2026-11-15T04:30UTC. No cloud/liveOpenPoker/final-test6100000 activity.

Freeze algorithm, orchestration, criteria and all parent/opponent hashes before
training. Resume only verified complete saved state with matching lineage and
manifest after inspecting failures. Never unchanged-retry node/table/OOM/nonfinite
stops. A protocol change requires explicit versioning, not relaxing frozen gates.

## Evidence and readiness gates

Every completed stage gets200 fixed screening blocks on development seed8300000
against its unchanged parent on frozen independent validator750. Report all six
populations, paired payoffs, street/active coverage and negative point estimates.
A screen with any nominal95% paired upper bound below-10bb/100 stops the pilot
for material-regression investigation. Screening never promotes a model.

After1500 additional iterations, fresh2000-block confirmation on seed8400000
compares candidate/unchanged parent/uniform separately on validator750 and the
OTHER seed's frozen2000 policy from sustained-002. All4primary learned-opponent
paired changes require positive lower98.75% bootstrap bounds; uniform comparisons
require positive lower nominal95% bounds. Require >=5%positive-weight postflop
coverage with >=200postflop decisions in calling,mixed and the learned population.
Report every population and mark any98.75% upper bound below-10bb/100 as a material
regression preventing readiness. Bootstrap intervals are approximate. Do not use
screens, repeated seeds or extensions to shop for favorable confidence intervals.

Operational pass separately requires both1500-additional targets, observed two
training workers, all measured attempts successful, lifetimeRSS/VMpeak<80%of4GiB,
sampled host available>=2GiB, final table<80%of inherited cap, and each stage's
total training-attempt wall time divided by newly requested iterations<=20seconds.
Retain maximum per-iteration nodes and importance weights plus street/active
regret/averaging diagnostics. Operational success alone cannot pass learning.

Readiness requires operational and learning pass plus evidence review. On a failed
or inconclusive gate, investigate a concrete issue rather than start blind long
extensions. Preserve this run and any failed attempt. Any subsequent experiment
needs its own bounded design and evidence rationale. No competitive-strength or
live promotion claim follows from these weak independent opponents.
