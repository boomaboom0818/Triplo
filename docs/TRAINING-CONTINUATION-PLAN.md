# Campaign 001 investigation and continuation plan

Investigated locally on 2026-11-13. No new training has been launched.
The original 5,000-iteration learners remain intact.

## Findings

The checkpoint provenance hashes for `cfr.py`, `game.py`, and `abstraction.py`
match the current files for both learners. Both training and evaluation use
`information_key` and the same default button. No version/key-encoding mismatch
was found. This is not proof that every training implementation is correct.

A separate read-only diagnostic used seed 4,100,000, 100 deals per population,
six candidate seat rotations, and one candidate against calling agents and the
frozen validation agent. Both checkpoint loads and all diagnostics ran with a
3 GiB address-space ceiling. The diagnostic does not update weights.
Raw counts and source-match checks are in
`runs/bot/campaign-001-investigation/candidate-{a,b}.json`; the executable local
protocol is `runs/bot/campaign-001-investigation/diagnose.py`.

Postflop exact keys matched only 1 of 917 candidate-A decisions and 2 of 905
candidate-B decisions. These are observed decisions with repeated deals and
rotations, not 917/905 independent samples. Removing all stack-ratio fields did
not add any matches. Removing card features raised matches only to 76/917 and
81/905. Individually, card features matched 800/917 and 767/905 decisions, and
legal-menu, stack-ratio, call-ratio, position and raise-history features each
matched every sampled postflop decision. Their joint combinations are sparse.

The exact active-seat mask alone was unseen in 530/917 A decisions and 234/905
B decisions. The average-policy tables overwhelmingly describe two-player
postflop states: A has 247,617 two-player keys but only 93 six-player keys; B has
244,462 and 105 respectively. These count distinct keys, not training visits,
and do not by themselves measure the training-state frequency distribution.
They support investigating a training/evaluation distribution mismatch.

The averaging traversal enumerates the owner's actions and samples other
players uniformly over legal slots, with an importance correction. Regret
traversal instead samples current/historical policies. Uniform legal-slot
sampling gives each raise size its own probability; it is not uniform over
strategic categories such as fold/call/raise. An inference to test is that these
sampled paths and the detailed joint abstraction make multiway postflop states
hard to revisit. Do not remove the importance correction or simply force more
postflop visits: changing a sampling distribution requires corresponding
unbiased weighting and tests.

The evidence supports sparse state reuse and a distribution mismatch, not an
absent postflop table, a proven solver defect, or proof that no amount of further
training could help. Uniform fallback can itself change which states evaluation
reaches, reinforcing this mismatch.

## Resource headroom

At 5,000 iterations, A has 1,044,525 union information sets and B 1,029,665.
Over iterations 4,000–5,000, growth was about 203 and 198 sets per iteration,
and elapsed checkpoint time about 2.20 and 2.17 seconds per iteration.
At that local linear rate, the existing 1,250,000-set cap would be reached near
6,010 and 6,110 iterations. These are planning estimates, not guarantees;
per-iteration overshoot and memory exhaustion can occur first.

A 250-iteration extension is roughly nine minutes per learner at the historical
rate and roughly 50,000 additional sets. It will not establish postflop quality
merely by completing. JSON file size is not live process memory. Measure RSS,
virtual address size and archive/checkpoint-save peaks before any extension.

## Recommended sequence

1. **Instrument and specify, before changing learning.** Add training counters
   by street and active-player count for regret versus averaging traversals,
   including sampled versus enumerated visits. Retain aggregate per-deal payoff
   samples and coverage counts in evaluation so candidate differences can be
   paired correctly. Add an evaluation-only uniform-policy control using the
   same legal menus. Test counters, unchanged seeded outputs and whole-deal
   paired resampling. Keep validation separate from training.

2. **Measure a small continuation control.** Resume each existing learner to
   5,250 total iterations into a new output path. Keep its seed, abstraction,
   population probability .25, archive interval 100 and archive size 2 exactly
   unchanged; retain the 1,250,000-set cap, 500,000-node iteration budget,
   50-iteration checkpoints and 3 GiB worker ceiling. Use an 1,800-second
   invocation budget and run learners sequentially initially to measure peaks.
   Record baseline hashes and the resume parent before launch. Stop on any
   resource failure and investigate it; never retry an unchanged limit failure.
   Preserve 5,000 and 5,250 snapshots. This control tells us whether a little
   more of the same training creates useful postflop reuse; it is not the
   proposed long-term solution.

3. **Run an abstraction-design pilot separately.** Use the measured state
   counts to design `texture-buckets-v2`, reducing joint fragmentation without
   merging incompatible legal action vectors. Start with relative opponent
   position/count features rather than all exact seat masks; compare a small
   predeclared set of postflop card/betting groupings. Do not blindly delete
   stack/card features: the projection results show neither deletion solves
   the problem by itself. Test observation privacy, suit invariance, legal-menu
   compatibility and train/evaluation key identity. Keep the MCCFR estimator
   unchanged for this experiment. A v2 policy is a new experiment: old v1
   regrets/averages/archives must not be relabeled or merged into it. Existing
   v1 learning remains available as the baseline and continuation control.

4. **Pilot at 250, then 1,000 iterations using seeds 11 and 29.** Compare each
   v2 stage with the retained v1 stage at equal iterations and report measured
   CPU time, nodes, table growth and memory as well. Use the same resource
   ceilings and stop explicitly when any ceiling is hit. Freeze the current
   validation checkpoint; do not train against it. Run diagnostic evaluations
   on predeclared development deals across calling/mixed and learned opponents,
   separately by street and active-player count. Keep one-copy primary and
   three-copy secondary. A v2 implementation may increase traversal cost, so
   iteration count alone is not a compute-matched comparison.

5. **Gate the next scale-up on evidence.** An initial engineering gate is
   positive-weight postflop coverage of at least 5% overall on development deals,
   with at least 200 postflop decisions in each reported population; always
   show street/active-player breakdowns and uncertainty. This is a proposed
   diagnostic threshold, not a strength standard. Require progress on both
   seeds, measured memory headroom and no clearly demonstrated payoff regression
   against the uniform control or v1. Inconclusive payoff intervals mean more
   evaluation is needed, not that the candidate passed. If coverage still fails,
   investigate sampling with correct importance weights as a separate change,
   rather than increasing the table cap or changing multiple mechanisms at once.

6. **Then schedule a bounded 1,000→5,000 campaign for the selected recipe.**
   Recheck resource projections at every stage. Evaluate shortlisted policies
   with at least 1,000 independent deal blocks on fresh development deals and
   a paired difference analysis. Reserve another seed range, e.g. 6,100,000,
   untouched for final testing; do not inspect it during recipe selection.
   Predeclare any stopping/selection rule before running. Add independently
   established stronger opponents when a compatible source is available.
   A higher-iteration independent learner can broaden screening, but cannot be
   called strong without separate evidence. No cloud resources are required.

## Decision

Recommend the instrumented 250-iteration continuation control and a separately
versioned abstraction pilot, rather than a blind 5,000→10,000 extension.
Do not schedule sustained training until the diagnostic instrumentation and
new experiment manifest are reviewable. Keep all original checkpoints and
memory protections. The monitor schedule still requires its actual task ID to
be disabled or repurposed; none was supplied in the monitoring run.

## Implemented bounded experiment

`scripts/continuation_control.py` runs the controls and independent v2 pilots
sequentially under 3 GiB per-child address-space limits. It refuses an existing
experiment manifest and uses a supervisor lock. It stops on a failed child or
incomplete training target; it does not retry failures. Outputs are isolated in
`runs/bot/continuation-001` and parent checkpoint hashes are recorded.

Diagnostics count nonterminal traversal visits by street, active-player count,
and whether the owner enumerates actions or an opponent action is sampled.
Only completed iterations contribute. Legacy resumes explicitly record the
iteration at which diagnostic collection starts; earlier visits are unknown.

The experimental v2 groups active seats by count and number before the acting
seat, replaces the stack vector with an effective stack/pot bucket, and coarsens
postflop card features. Own position and ordered legal-menu ratios remain.
It is intentionally lossy; coverage improvement cannot establish quality.
Training and loading explicitly recognize v2, and v1/v2 resumes are rejected.

Each stage and the original 5,000-iteration baseline receive one/three-copy
1,000-block evaluations on seed 5,100,000. Final status is
`completed_pending_evidence_review`; no sustained campaign is launched.
Promotion requires the coverage and payoff review above. The current aggregate
reports do not provide paired per-deal differences, so close/inconclusive results
must trigger further evaluation instrumentation rather than automatic promotion.
