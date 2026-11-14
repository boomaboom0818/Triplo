# Campaign 001 evaluation audit

## Scope and evidence

Audited the completed local 250/1,000/5,000-iteration campaign on 2026-11-13.
Both final checkpoints parsed successfully with `target_reached`; all twelve
original evaluation reports matched their candidate checkpoint SHA-256 and the
same frozen 250-iteration validation checkpoint. Evaluation stderr was empty.
Original campaign files remain unchanged under `runs/bot/campaign-001/`.

## What the measurements mean

`CFRPolicy.act` normalizes positive average-policy weights. Missing rows and
all-zero rows both choose uniformly among legal action slots. Uniform here is
not a conservative poker default: numeric raise sizes each have their own slot.
A positive-weight row may also describe a uniform distribution; positive weight
indicates stored averaging mass, not a useful or strong strategy.

The original `policy_coverage` reports row presence, not positive averaging mass.
The evaluator now additionally reports missing, zero-weight and positive-weight
row counts overall and by street, plus `uniform_fallback_fraction`. The old
metric remains available for comparison. Non-CFR checkpoints have no CFR
coverage measurements (the fraction remains null).

The candidate's payoff is measured in big blinds per 100 candidate hands, with
stacks reset each hand. Each independent deal is reused over six candidate seat
rotations. Resampling whole deal blocks correctly retains dependence between
rotations and, for three copies, between candidate seats. Three copies do not
create three times as many independent deals. Their payoff measures a different
table composition and is not interchangeable with the one-copy result.

Reported latency includes all table decisions, including opponents. It is now
explicitly labeled `latency_scope: all_table_decisions`; do not treat it as
candidate-only performance. Bootstrap intervals are per population, not a joint
confidence statement or a paired candidate-versus-candidate significance test.
The same deals are reused across candidates and stages; do not treat those
reports as independent experimental replications.

## Evaluation limitations

Calling and tight agents ignore card strength. Aggressive uses a fixed raise
probability. Blueprint with an empty baseline table is uniform over legal
slots; mixed combines the simple scripted styles. None is a strong benchmark.
The frozen learned opponent is independent of candidate training but trained
for only 250 iterations with the same abstraction and unestablished strength.
Hash/population overlap checks protect against direct checkpoint reuse, not
all forms of shared lineage or benchmark selection bias.

The original four final learned-opponent intervals all include zero. The
results establish neither superiority over that opponent nor a ranking of the
two candidates. Higher row coverage alone is not evidence of stronger play.
The abstraction deliberately loses history and card detail, so more iterations
cannot resolve every information loss. No multiplayer equilibrium or competitive
strength guarantee follows from this experiment.

## Recommended next experiment

Keep these checkpoints as the baseline. Before extending training, integrate an
independently validated stronger six-player opponent through the existing
checkpoint contract, or add a process-isolated adapter with an explicit action
mapping and licensing review. The repository contains no such validated strong
agent; this audit does not invent one or silently promote the pilot opponent.

Predeclare fresh deal seeds and the primary one-copy comparison. Keep three-copy
results separate. Increase deal blocks after setting a meaningful precision goal;
200 blocks is a diagnostic smoke sample, not a competitive certification. A
rough fourfold increase in blocks only halves sampling uncertainty when variance
is similar. Use paired per-deal candidate differences before claiming a winner;
the current aggregate reports do not retain the samples needed for that test.

Only then choose a bounded training extension based on measured positive-weight
coverage, held-out payoff and table growth. Retain the existing 3 GiB worker
address-space protection and checkpoint lineage; do not reset learning or change
the abstraction in place. No additional training was launched during this audit.

## Fresh-seed diagnostic protocol

Re-evaluated both 5,000-iteration candidates with seed 3,100,000, 200 independent
deal blocks, one and three copies, against all five existing scripted populations
and the unchanged frozen validation agent. Each run is local with a 3 GiB
address-space ceiling. Raw JSON is retained separately in
`runs/bot/campaign-001-audit/`. These new deals diagnose coverage/generalization;
they do not strengthen the opponent or constitute a final benchmark.

Both final average-policy tables contain postflop rows (candidate A: 70,058 flop,
89,722 turn, 137,495 river; candidate B: 70,377 flop, 89,061 turn, 135,057 river).
Neither table has all-zero average rows. Low held-out postflop coverage therefore
reflects unseen abstract states rather than an absent postflop table. This is a
state-distribution/coverage finding, not proof of a specific training defect.

## Fresh-seed results

Against the frozen learned opponent (200 independent deal blocks per row):

| Candidate | Copies | Uniform fallback | Postflop positive / decisions | BB/100 | 95% interval |
|---|---:|---:|---:|---:|---|
| candidate-a | 1 | 53.3% | 0 / 66 | 187.2 | [-361.1, 771.2] |
| candidate-a | 3 | 52.0% | 1 / 178 | 167.6 | [-75.0, 412.4] |
| candidate-b | 1 | 52.1% | 0 / 63 | 345.6 | [-216.5, 933.8] |
| candidate-b | 3 | 51.3% | 3 / 198 | 229.0 | [-1.1, 468.4] |

Fresh-seed results remain diagnostic only. Postflop samples against the learned opponent are small; broader scripted populations also show sparse postflop coverage. No ranking or competitive-strength claim is justified.
