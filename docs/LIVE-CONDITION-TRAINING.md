# Live-condition training and checkpoint promotion

`live-conditions-v1` is a separate continuation distribution for the compact
history model. It retains the old abstraction and trained action menu; the new
`history-live-mixture-es-v1` algorithm tag and curriculum metadata distinguish it
from equal-stack six-max runs. Existing optimizer directories and the live
checkpoint are never modified by initialization.

## Training distribution

- 3, 4, 5 and 6 players, sampled with weights 1:2:3:4.
- 20, 50, 75, 100, 200 and 500 BB depths with weights 1:4:2:4:2:1.
- Half the deals use unequal stacks. One quarter of those introduce a 1,000-BB
  stack. These are deliberate coverage weights, not an empirical fitted population.
- Short tables use the live adapter's six-seat embedding with forced early folds.
  Heads-up starting tables remain unsupported; hands can fold down to heads-up.
- Population deals mix independent training-only loose callers and aggressive
  profiles. The other deals remain self-play. All six traversal roles are retained.
- Fixed opponents can use full menus, including all-ins. Aggressive profiles avoid
  tiny reraises in large pots, which otherwise cause very deep recursive sequences.

A bounded continuation example (source must be a resumable optimizer directory):

```sh
.venv/bin/python -m scripts.history_train \
  --root runs/bot/live-new/candidate-581 --minutes 3 --seed 581 \
  --stack-bb 100 --workers 6 --population .75 --abstraction history-compact-v1 \
  --curriculum live-conditions-v1 \
  --initialize-from runs/bot/history-sustained-001/state --max-rows 11000000
```

`stack_bb` remains nominal checkpoint metadata for compatibility; `curriculum`
determines actual deal depths. Existing regret, average-policy and visit arrays are
retained. This is continued training, not evidence that old visits were collected
under the new distribution. Resume the new run with `--resume`, the same curriculum,
seed and population settings, and without `--initialize-from`.

## Evaluation and promotion

`scripts/live_gate.py evaluate` always uses the deployed 64-visit guard and
`equity-v1` fallback. Nine frozen conditions cover equal 50/100-BB tables,
3/4/5-player unequal stacks, tight/aggressive/calling-station/mixed opponents, and
five separate instances of the incumbent. Learned opponents retain their trained
menu; rule opponents retain full menus. Opponents are reproducible checks, not
certified strong players.

Each deal is replayed with the hero in every occupied seat. The independent sample
is the complete deal block, not its seat rotations. Policy RNGs are independent
per seat and reset per deal, allowing paired candidate/control comparisons. The
aggregate gives equal weight to each condition. Reports retain all block outcomes,
checkpoint/source hashes, learned/postflop coverage and decision latency.

Promotion requires all of:

1. Positive aggregate lower bounds for profit, improvement over the incumbent,
   and improvement over the guarded equity-only fallback.
2. Every condition's lower bound for improvement over **each** control at least
   -10 bb/100 (a predeclared noninferiority margin).
3. At least 5% learned decisions overall and 1% learned postflop decisions.
4. Per-condition p99 decision latency at most 250 ms.

Bounds are one-sided normal approximations with Bonferroni adjustment for a total
5% family error budget over the 21 payoff comparisons. These finite-suite bounds
are not multiplayer exploitability estimates, and they do not establish universal
or live profitability. Increasing visits or choosing the newest checkpoint cannot
bypass the gate. A fallback-equivalent candidate cannot pass strict improvement.

`scripts/live_campaign.py --root ROOT` consumes a frozen `manifest.json`, trains
its candidate seeds, screens them on the development seed, selects one using the
smaller aggregate advantage over the two controls, and evaluates that single
selection on an untouched confirmation seed. It writes `selection.json` and
`promotion.json` and never changes the running client. A completed holdout cannot
be reused by rerunning the campaign. New campaigns need new confirmation seeds;
repeated campaigns also require a policy for the total statistical error budget.

Only a `promote: true` confirmation report authorizes replacing the champion, and
the deployment must match both the candidate hash and the incumbent hash in that
report. Failed or inconclusive gates leave the incumbent running.

## First bounded campaign

`runs/bot/live-curriculum-001` preserves a failed first batch: deep-stack tiny
reraise sequences exceeded Python's recursive traversal limit before any complete
training round. No candidate was promoted from it. The corrected campaign is
`runs/bot/live-curriculum-002`; it preserves two three-minute continuations, frozen
training/evaluation seeds, source copies, the original manifest, and gate reports.
The training deadline is checked at round boundaries; initialization, an in-flight
round, export and evaluation add wall time beyond the traversal budget.


## Completed campaign result (2026-11-16)

Both bounded continuations completed. Seed 581 reached 900,520 total iterations
(+24,000), with 9,043,526 rows; seed 582 reached 897,520 (+21,000), with 8,921,318
rows. Seed 581 was selected using only the 100-deal development screens. The
confirmation evaluated 500 independent deal blocks per condition across nine
conditions: 24,000 actual hero hands per policy, 72,000 across candidate, incumbent
and fallback controls. Training and gate source hashes matched their frozen plans.

The candidate's equally weighted aggregate return was +61.665 bb/100, compared
with +59.361 for the incumbent and +39.728 for the equity-only fallback. Its paired
advantage over the incumbent was +2.304 bb/100 with adjusted one-sided lower bound
-2.573; over fallback it was +21.937 with lower bound -8.935. It failed both strict
improvement claims and the per-condition noninferiority requirement. Positive
aggregate profit, learned coverage and latency requirements passed. Learned
coverage was 72.7% overall and 49.2% postflop.

**Promotion rejected.** The deployed 876,520-iteration checkpoint continues using
the equity fallback. This is an inconclusive small continuation, not proof that
mixed-condition training is ineffective or that the incumbent is strongest.
Confirmation seeds are now consumed and must not be reused for selecting/tuning.
The next experiment should measure longer training stages on development seeds
and address the aggressive 100-BB weakness before spending another fresh holdout.

Artifacts: `runs/bot/live-curriculum-002/selection.json`, `promotion.json`,
`confirmation-*.json`, `gate-plan.json`, `source-manifest.json`, and both candidate
optimizer states/checkpoints. Full regression: **401 tests passed**; focused tests:
40 passed; `git diff --check` passed. No live checkpoint replacement occurred.

## Longer aggressive curriculum (`aggressive-stages-001`)

`live-aggressive-v2` adds a targeted distribution while preserving v1 behavior:
60% of roots are six-player 100-BB tables; the remainder uses the original varied
stack/table distribution. On population deals, 80% of fixed opponents are varied
training-only aggressors and 20% loose callers. Population probability is 0.8,
leaving 20% self-play deals. These opponents are separate from held-out rule styles.

`scripts/aggressive_campaign.py` runs 5 and 10 cumulative additional training
minutes from the best previous development candidate, seed 581's optimizer state,
with a new training seed (681). The 10-minute snapshot resumes the 5-minute state.
Each immutable snapshot gets the same 300-deal development suite. Sources and
parent/incumbent hashes are retained in `runs/bot/aggressive-stages-001/manifest.json`.
Initialization, evaluation and checkpoint exports add wall time to training budgets.

A stage needs all original development gate checks plus strictly positive adjusted
lower bounds for aggregate advantage on the three aggressive-opponent conditions
versus **both** controls. If any stage is ready, selection uses its smaller mean
aggressive advantage; the selected stage must repeat readiness on a new 500-deal
development validation seed. Only then is the untouched 1,000-deal confirmation
seed consumed. Otherwise `result.json` records a rejection and the confirmation
seed remains untouched. The final confirmation still uses all nine conditions.

This campaign uses a 2.5% family error budget for the original payoff gate and
Bonferroni bounds across the two additional aggressive development comparisons.
The aggressive checks are development readiness filters. The final promotion claim
uses only the untouched confirmation suite. Previous campaigns used their own
budgets, so there is no claim of a cumulative 5% error guarantee across campaigns.

Existing regrets, averages and visits are inherited. Older average-policy mass
may slow adaptation; these stages measure the effect rather than assuming a longer
run automatically improves strength. No automatic reset or threshold relaxation
occurs after results are seen. The running bot is replaced only if confirmation
passes and candidate/incumbent hashes still match.


### Completed longer-stage results (2026-11-16)

Both planned stages completed: 930,520 iterations after the first five minutes
(+30,000 from parent) and 969,520 after ten cumulative minutes (+69,000). Final
training inventory reached 9,805,095 rows. Each immutable stage and both controls
were evaluated on 300 whole-deal blocks in each of nine conditions, for 14,400
hero hands per policy (57,600 across the two stages and two controls). The same
blocks were reused across policies for pairing; seat rotations are not independent
samples. All source and incumbent hashes matched the frozen campaign manifest.

| Development metric, bb/100 | 5-minute stage | 10-minute stage |
|---|---:|---:|
| Equal-condition aggregate payoff | +26.84 | +26.70 |
| Aggregate advantage over incumbent | -1.72 | -1.86 |
| Aggregate advantage over fallback | -21.24 | -21.37 |
| Aggressive-condition advantage over incumbent | -7.31 | -3.31 |
| Aggressive-condition advantage over fallback | -5.63 | -1.63 |

The aggressive-condition one-sided adjusted lower bounds versus incumbent were
-22.90 and -21.54 respectively; versus fallback they were -53.59 and -51.05.
These development estimates do not establish either improvement or deterioration
in the underlying win rate. Neither stage passed payoff/improvement/noninferiority
requirements; coverage and latency passed. Overall learned coverage was 72.8% and
73.1%; postflop learned coverage was 48.8% and 49.2%.

The remaining six-player 100-BB aggressive case returned +3.11 and +4.03 bb/100
for the two stages, versus +13.86 incumbent and +38.78 fallback on these blocks.
The 10-minute stage improved the unequal-stack four-player aggressive point
estimate to +64.42 versus +57.62 incumbent, but that local gain does not establish
an overall improvement.

**No promotion and no fresh confirmation test.** Neither the second development
validation seed nor the confirmation seed was consumed; both remain reserved.
The current 876,520-iteration live champion with equity fallback remains running.
All 404 regression tests passed; focused curriculum/gate tests passed and
`git diff --check` passed. New regression tests cover v2 distribution emphasis,
continued diversity, aggressive-readiness rejection and tighter error bounds.

Artifacts: `runs/bot/aggressive-stages-001/result.json`, `manifest.json`,
`stage-5.json`, `stage-10.json`, `development-*.json`, `progress-*.json`, copied
sources, final resumable `training/state`, and `regression.log`.

Further compute should follow a development-only diagnosis of the weak 100-BB
aggressive decisions, including inherited average-policy mass and abstraction
collisions/range information. These runs alone do not isolate which mechanism is
responsible; training longer has not yet demonstrated added strength.
