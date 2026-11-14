# Overnight readiness — interim handoff

As of 2026-11-14 15:20 UTC; additional frozen-validator screening is still running.
The 17:31 UTC finalizer must update this report with its outcome and shutdown state.

**Operational software is ready for bounded, recoverable local training. Playing
strength is not established, and the two-seed promotion gate remains unmet.**
No candidate was deployed to OpenPoker or relabeled as a validated winner.

## Preserved candidates

| Learner | Most advanced checkpoint | Iterations | Union information sets |
|---|---|---:|---:|
| A | `runs/bot/sustained-002/candidate-0-2000.json` | 2000 | 228,969 |
| B | `runs/bot/sustained-002/candidate-1-2000.json` | 2000 | 223,760 |

These are the most advanced checkpoints of the unchanged v2/call-mixture recipe,
not a statistically established ranking. Initial/per-stage parent paths and
hashes, recipe settings and paired reports remain in the run manifests and
verified-review.json. Original v1 learners and all earlier v2 snapshots are intact.

## Evidence and limitations

- Both learners completed500/750/1,000/1,250/1,500/1,750/2,000 and evaluations.
- Independent development screening at1,000 found A's improvement, while B's
  parent comparison remained inconclusive: the two-seed gate failed.
- Later incremental payoff results and independent-learner crossplay did not
  establish robust additional gains. Tight-script and unseen-state coverage gaps
  persist. No established strong benchmark opponent is integrated.
- Large averaging weights were measured and investigated; all moments remained
  finite and sampled covered policies were mostly stable. No clipping/reset or
  estimator change was justified. The diagnostic does not prove harmlessness.
- The original250 validator remains frozen. A separate750 continuation completed
  within its original250K table cap; additional2000-vs1500/uniform screening is
  pending. More iterations do not certify that validator's strength.
- Live OpenPoker uses a different check/fold/call fallback on unseen states.
  Separate local evaluation modes/logging expose this difference. No local live
  client or telemetry was observed; no live-play results are included.
- Final-test seed6,100,000 remains untouched. Development/nominal intervals are
  not competitive certification or a multiplayer equilibrium guarantee.

## Software verification

The final software check passed all76 tests on2026-11-14 before this interim
handoff. It covers exact resume/rollback, resource stops/locks, sampler expectation,
weight instrumentation invariance, report provenance and deadline cleanup.

## Operating and recovery

The runner preserves recipe/seed/archive settings, immutable checkpoint lineage,
3 GiB per-child address-space limits, node/table caps, checkpoint intervals,
validated atomic evaluation reports and output locks. A hard deadline stops new
launches and requests graceful checkpointing. Known interruption/time stops can
resume; unresolved OOM/node/table/corruption failures halt for investigation.

Verify current learner evidence:

```sh
.venv/bin/python -m scripts.review_sustained --output runs/bot/sustained-002
```

For an interrupted current extension, first verify there is no active writer, then
use the exact original command plus --resume (deadline remains the authorized cutoff):

```sh
.venv/bin/python -m scripts.sustained_train \
  --output runs/bot/sustained-002 \
  --parents runs/bot/sustained-001/candidate-0-1500.json runs/bot/sustained-001/candidate-1-1500.json \
  --stages 1750 2000 \
  --validation runs/bot/campaign-001/validation-250.json \
  --seed 5700000 --purpose exploratory \
  --deadline 2026-11-14T17:31:00+00:00 --resume
```

The current learner extension is complete and does not need restarting. Future
training needs a new output directory and declared stages/deadline, following
SUSTAINED-TRAINING.md. Do not extend merely to claim the failed evidence gate passed.
No cloud resources were used, and no OpenPoker live session/rebuy was started.

## Source results

See CONFIRMATION-001-RESULTS.md, SUSTAINED-001-RESULTS.md,
SUSTAINED-002-RESULTS.md, CROSSPLAY-001-RESULTS.md and
AVERAGING-WEIGHT-INVESTIGATION.md. Full local protocol and checkpoint artifacts
remain under runs/bot; the overnight journal records each check and recovery.

## Checkpoint hashes

- A: `6971791302ac3c0d84507603972265fff7a61853cd1327217b2bf91b9e663628`
- B: `70b18f614d15a3ea7b9b1d9d13eada7c013fe747082ed1bec2029b04c51368fd`
