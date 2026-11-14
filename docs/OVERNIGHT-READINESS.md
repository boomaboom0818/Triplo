# Overnight training handoff

Completed 2026-11-14. The user authorized work beyond the morning checkpoint;
all planned jobs finished normally before any finite job deadline was reached.

**The local training setup is operational and recoverable. The evidence threshold
for a broad sustained campaign has not passed. No competitive-strength claim or
live promotion is made.**

## Preserved checkpoints

All historical checkpoints remain intact, including the original v1 campaign,
both v2 seeds through2000, matched historical controls and explicit diversity
forks through2100, and separate frozen validation250/750 snapshots.

| Learner | Recipe | Most recent reviewed v2 snapshot | SHA256 prefix |
|---|---|---|---|
| A | control | [2100 checkpoint](/home/boomaboom0818/Triplo-Texas-Holdem-Poker-/runs/bot/population-pilot-001/candidate-0-control.json) | a08a05444e16 |
| A | diverse | [2100 checkpoint](/home/boomaboom0818/Triplo-Texas-Holdem-Poker-/runs/bot/population-pilot-001/candidate-0-diverse.json) | f375ce4d1120 |
| B | control | [2100 checkpoint](/home/boomaboom0818/Triplo-Texas-Holdem-Poker-/runs/bot/population-pilot-001/candidate-1-control.json) | f6296bb388ce |
| B | diverse | [2100 checkpoint](/home/boomaboom0818/Triplo-Texas-Holdem-Poker-/runs/bot/population-pilot-001/candidate-1-diverse.json) | df7403064db1 |

“Most recent” is not a statistical ranking. The diversity forks are experimental;
retain historical controls for continuation comparisons. Exact hashes, parent
configurations and paired results are in
[runs/bot/population-pilot-001/verified-review.json](../runs/bot/population-pilot-001/verified-review.json).
No source checkpoint was overwritten or learning silently reset.

## What improved and what remains unresolved

- Postflop coverage improved after the separately versioned abstraction and
  corrected call-mixture sampling pilots. The earlier near-zero coverage problem
  was measured rather than treated as proof of a version mismatch.
- The independent1000-iteration development check passed A's screen, but B's
  parent comparison remained inconclusive. The original two-seed gate failed.
- Later bounded extensions through2000 and independent-learner crossplay did not
  establish consistent incremental payoff gains. Scripted tight-agent gaps persist.
- Additional frozen750-validator screening was nominally favorable for B and
  inconclusive for A. It does not retroactively pass the original gate or certify
  the opponent as strong.
- The final matched population pilot preserved both2000 parents and compared100
  additional historical-control iterations with an explicit diversity fork.
  Diverse-minus-control learned-opponent payoff was A +30.53 BB/100
  [-18.64,80.14] and B -23.92 [-74.06,27.17]. Both paired95% intervals include0;
  the effect is inconsistent across seeds. The diversity recipe is NOT promoted.
- Large importance-weighted averaging updates were investigated. All moments
  were finite; sampled covered policies were mostly stable. This did not justify
  clipping weights, resetting learning or modifying the estimator.
- Scripted styles are training-related diagnostics for the population pilot;
  validation agents were never used as training opponents. There is no established
  strong benchmark. Nominal development intervals and repeated screening are not
  competitive certification, final-test evidence or equilibrium guarantees.

## Operational readiness

The runner preserves immutable parent hashes, exact recipe/RNG/archive state,
3 GiB per-child address-space caps, node/table limits, checkpoints and previous
snapshots, output locks, validated atomic reports, paired deal-block statistics,
explicit recipe forks and finite per-job deadlines. Known interrupted/time-stopped
runs can resume; unresolved OOM/node/table/corruption or no-progress failures stop
for investigation. No unchanged resource-failure retry loop was introduced.

Software checks passed: full83-test suite, the additional explicit-fork preservation
check, and two pilot-recovery tests. The fork/resume checks cover preserved parent
bytes, regrets/averages/archive/RNG, rollback, implicit-fork refusal, lineage checks,
completed-work skipping and avoidance of applying the fork twice. No new code has
been added since those relevant checks except documentation.

Verify completed historical continuation evidence:

```sh
.venv/bin/python -m scripts.review_sustained --output runs/bot/sustained-002
```

Exact pilot recovery command, only after an actual interruption and verifying
there is no active writer:

```sh
.venv/bin/python -m scripts.population_pilot --resume
```

No recovery or restart is needed now: all current runs completed and their
reports were verified. For a new planned research campaign, use the configurable
runner and finite-deadline procedure in SUSTAINED-TRAINING.md; do not call a failed
promotion gate passed or launch endless unchanged extensions.

## OpenPoker and resources

The adapter is installed/tested locally and logs checkpoint hash plus coverage/
action type without cards/tokens. It has a different unseen-state check/fold/call
fallback, evaluated separately. No local live session or live telemetry was
observed and authenticated behavior was not verified. No live checkpoint, rebuy
setting or connection was changed by this work. No live-play returns are pooled
with local controlled results.

No cloud resources were used. RAM/disk and actual worker processes were checked
throughout; no duplicate training writers were launched. All completed results
and source are retained. The reserved final-test range6100000 remains untouched.

## Next research decision

The setup can run further bounded, evaluated experiments. The model is not yet
qualified for a broad sustained campaign by the agreed evidence gate. A useful
next research proposal needs a specific remaining hypothesis, an independently
validated stronger opponent, both-seed controls and measured payoff/robustness
criteria; simply adding more iterations or choosing a favorable development
result is not enough. No statistically established best checkpoint was selected.

## Evidence index

- CONFIRMATION-001-RESULTS.md: independent development check.
- SUSTAINED-001-RESULTS.md and SUSTAINED-002-RESULTS.md: retained learning curves.
- CROSSPLAY-001-RESULTS.md and CROSSPLAY-002-RESULTS.md: independent learned opponents.
- AVERAGING-WEIGHT-INVESTIGATION.md: weight concentration and inference drift.
- SUPPLEMENTAL-SCREENING.md: separate frozen validator continuation.
- POPULATION-PILOT.md: controlled curriculum protocol and recovery.
- runs/bot/overnight-2026-11-14/journal.md: timestamped monitoring/recovery record.

Schedule state confirmed at 2026-11-14T17:28:13.291626+00:00: recurring checks and the morning checkpoint
are disabled, and no trainer/evaluator is active. All current authorized local
runs and the handoff are complete. Playing-strength limitations remain explicit;
no unbounded blind training or statistical promotion was triggered.
