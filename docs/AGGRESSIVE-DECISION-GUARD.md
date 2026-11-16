# Measured response to weak aggressive decisions

The inherited-average diagnosis identified slow policy movement, but slow movement
is not necessarily the cause of poor play. Two bounded interventions were tested
without another training run. Both use the already consumed development suite:
seed 61161000, 300 deal blocks per condition, all occupied hero positions, nine
conditions and 14,400 hands per policy. These are development results, not fresh
confirmation or a claim about live win rate.

## Recent-average ablation

`scripts.recent_average` subtracts the 900,520-iteration parent's accumulated
strategy sums from the 969,520-iteration snapshot. A row replaces its completed
policy only if it has at least 64 **new** averaging visits. Other rows retain the
completed policy and original support. Replaced rows carry only their new visit
count. Visits are a support heuristic, not independent observations.

Exactly 10,302 of 5,767,752 rows qualified; none was newly created. On the target
six-player 100-BB aggressive condition, recent-only averages returned -30.83
bb/100, compared with +4.03 for the unchanged completed checkpoint. The average
advantage across the three aggressive conditions was -16.30 bb/100 versus the
incumbent and -14.62 versus the equity fallback. This variant failed development
readiness. Removing inherited averages is therefore **not** the selected fix.
These results do not prove that all averaging resets or mixtures are inferior.

The extraction tool preserves both input checkpoints and optimizer state. It
checks metadata, matching action-row lengths, finite nonnegative masses,
monotonic sums and counts, and rejects transformed inputs. It records source
checkpoint hashes and publishes outputs without overwriting existing files.
Monotonicity cannot prove ancestry: use raw exports from the same known optimizer
lineage. Exported probabilities must never be treated as raw training totals.

## Pressure decision guard

`equity-pressure-v1` is an explicit, experimental checkpoint option. It applies
only to actions selected from a supported learned row:

- Postflop, when facing a bet, veto a learned call or raise if the existing
  deterministic equity fallback chooses to fold.
- Preflop, after at least two public raises, veto a learned continuation if the
  tight preflop fallback chooses to fold.
- Preserve free actions, existing folds, and continuations supported by the
  fallback. Existing AA and river-nuts safeguards run first.

This directly rejects the audited AQ river raise and K8-suited continuation
under repeated preflop pressure. It is not a solved EV decision: heuristic ranges
can be wrong, and a fold veto can remove profitable bluffs or overfold. It needs
measured validation before deployment. Guard overrides are reported separately
and excluded from learned-policy coverage.

The guard candidate uses the **original completed checkpoint**, not the rejected
recent-only policy. The default policy behavior remains unchanged. Unknown guard
versions fail explicitly. The export records the original checkpoint hash and
does not modify training state or the live checkpoint.

## Development outcome and deployment decision

All four policies used identical current evaluator source hashes. The unguarded
original, incumbent, and fallback reproduced every payoff block from their prior
development reports exactly.

| Policy | Six-player 100-BB aggressive, bb/100 | Equal-condition suite mean, bb/100 |
| --- | ---: | ---: |
| Completed checkpoint, unchanged | 4.03 | 26.70 |
| Live incumbent with equity fallback | 13.86 | 28.56 |
| Equity fallback only | 38.78 | 48.08 |
| Completed checkpoint with pressure guard | 78.44 | 55.42 |

The guard made 540 overrides across the suite, including 132 in the target
condition. Learned coverage remained 71.30% overall and 46.26% postflop; the
largest per-condition p99 decision latency was 8.48 ms.

**Do not promote yet.** At family alpha .025, development readiness passed
aggregate profit, coverage and latency, but failed superiority over both controls
and per-condition noninferiority. The aggregate paired advantage was +26.86
bb/100 over the incumbent (adjusted lower bound -2.05) and +7.34 over fallback
(lower bound -27.78). Across aggressive conditions the lower bounds were also
negative. The target-condition gain over the unchanged checkpoint was +74.42
bb/100, but its paired standard error was 43.02; the large point estimate alone
is not reliable evidence of improved strength.

Other weaknesses remain: the guard's six-player 50-BB aggressive point estimate
was 20.17 versus the incumbent's 30.50; six-player 100-BB tight was 1.19 versus
fallback's 19.06. These noisy comparisons are not proven regressions. Before
increasing training, separate the preflop and postflop guard effects in
development and inspect those decision traces. Keep the candidate fixed for any
subsequent independent validation. Do not relax the promotion gate or tune on
the confirmation set.

The live checkpoint and optimizer were unchanged, no training iterations were
added, and reserved validation/confirmation seeds remain unused. This is a tested
candidate solution to the observed high-commitment decisions, not a completed
strength promotion.

Local artifacts under ignored `runs/`:

- `runs/bot/recent-average-001/{development,assessment}.json`
- `runs/bot/pressure-guard-001/{manifest,assessment,result}.json`
- `runs/bot/pressure-guard-001/development-{candidate,original,incumbent,fallback}.json`
- Guard candidate SHA-256:
  `f3ed3b73624edfa43568e1248abe80b61deeb320f217e6d9a123d32879f9b6ca`
- Source completed checkpoint SHA-256:
  `2bafe03ca0ea8ceb2ea37ce967a01fbd2b66d074b1e29530c6fc17460c35df54`

Validation: the full suite passed (413 tests at collection), followed by the
additional audited AQ river/nuts regression. Tests cover new-average support,
input preservation, malformed/changed rows, exclusive export, guard opt-in,
unknown versions, repeated pressure, free actions and learned coverage accounting.
The AQ regression uses the real equity estimate; it does not establish that the
heuristic fold is optimal against every range.

## Reproduce

Use new output paths for each run; exports and evaluation reports refuse to
overwrite existing artifacts.

```sh
.venv/bin/python -m scripts.recent_average \
  --completed runs/bot/aggressive-stages-001/stage-10.json \
  --parent runs/bot/live-curriculum-002/candidate-581/checkpoint.json \
  --out runs/bot/recent-average-new.json

.venv/bin/python -m scripts.export_pressure_guard \
  --checkpoint runs/bot/aggressive-stages-001/stage-10.json \
  --out runs/bot/pressure-guard-new.json

.venv/bin/python -m scripts.live_gate evaluate \
  --checkpoint runs/bot/pressure-guard-new.json \
  --incumbent runs/bot/history-sustained-001/checkpoint.json \
  --out runs/bot/pressure-guard-development-new.json \
  --seed 61161000 --deals 300
```

Run incumbent and `--fallback-only` controls with the same evaluator source,
seed, and deal count. Use `scripts.aggressive_campaign.readiness` with family
alpha .025 before consuming any new validation or confirmation seed. No guard
flag is added to the live client: loading a deliberately exported guard checkpoint
is the opt-in mechanism, subject to the existing promotion workflow.
