# History-aware poker implementation and first measured test

The new version fixes a representation defect and adds a basic playable floor.
It does not yet establish learned six-player strength. Existing lean checkpoints
and runs are preserved. No live session was started.

## What changed

`poker/sampled_cfr.py` is a shared external-sampling kernel used by both the known
game and Hold'em adapters. It uses linear regret/average weights and averages the
next player on each traverser's pass, following the practical simple-estimator
convention in [OpenSpiel](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/algorithms/external_sampling_mccfr.py).
This is original code, not imported pretrained weights. Multiplayer averaging and
imperfect-recall limitations remain; a two-player proof does not transfer to six.
It also retrieves regret coordinates after recursion so a descendant's update to
a merged coordinate cannot be overwritten on return.

`poker/kuhn.py` supplies a tiny independent poker game and an exhaustive evaluator:
every pure response respecting information sets is enumerated, rather than letting
a best responder see hidden cards. An analytic equilibrium independently verifies
the evaluator's zero NashConv and first-player value -1/18. The same kernel then
learns Kuhn across three RNG seeds. After 30,000 deals per seed, NashConv is
0.00500, 0.01474 and 0.01269, versus 0.91667 for uniform play. First-player values
are approximately -0.05555, -0.05553 and -0.05522. This validates the kernel on
Kuhn; it is not a Hold'em exploitability measurement.

`poker/history_cfr.py` introduces algorithm `history-linear-es-v1` and abstraction
`history-public-v1`. Keys preserve every public actor/street/action in order, exact
stacks, price, pot, active-player identities, and the actual legal action amounts.
Training records history on every branch. Previously colliding legal histories
now get different keys, and training/observation keys match. Other players' hole
cards never enter keys. The card abstraction remains lean-v3 to isolate the public
history change; future-card potential/range-aware card abstraction is outstanding.

`HistoryPolicy` applies learned rows only after 64 sampled average updates by
default; these are visits, not independent samples or a confidence guarantee.
Otherwise it uses the existing tight rule policy. It also never folds pocket aces
preflop, an unbeatable river hand, or a free check. River nuts are verified against
all opponent holdings consistent with the hero's cards and board. A board that
plays for everyone is called/checked instead of pointlessly raised. These guards
are hand-coded basic knowledge, not learned achievements. The policy remains weak
and the fallback can be exploited. No general tactical correctness claim is made.

## First experiment

`runs/bot/history-001` contains a complete from-scratch matched experiment:

- Training seeds91/92; 1,000 deals per arm; six players; 20BB stacks.
- History and historyless arms use the SAME new learning kernel, menus, card
  features and iteration budgets. This isolates public-state keys, rather than
  comparing two differently implemented solvers. Historyless controls are marked
  nonproduction and cannot be loaded as history checkpoints.
- Both raw policies and guarded policies are evaluated on 200 new independent
  deals beginning19310000, rotating the hero through all six seats, against four
  unchanged rule styles. Opponents keep the full raise menu.
- Guarded rule-only control uses the same tactical guards. Raw unseen states use
  uniform random play; the raw-vs-guarded comparison therefore combines support
  handling and safeguards. It is not an isolated measure of either mechanism.
- Per-arm deadline180seconds; 4GiB address-space cap; host/disk reserves2/10GiB.
  Actual training completed in13–21seconds per arm. Source hashes matched at
  completion and exact sources are archived in the run's `source/` folder.
- Reports retain independent deal-block samples. Paired normal95% intervals are
  pointwise screening estimates without multiple-comparison correction.

Result: raw history policies still lose and show no consistent advantage over the
historyless controls across two seeds. At the preselected 64-visit threshold,
learned-decision coverage was ZERO for both arms on this evaluation. Every guarded
policy therefore exactly matches the guarded rule-only baseline. This is a
successful support guard, not learned improvement. Raw history lookup coverage
ranged approximately22–45%, illustrating how much support exact history needs.

Guarded baseline winnings in bb/100 were +4.4 tight, -2.9 aggressive, +52.8 station,
and +5.7 shover; this short screen does not establish positive expected winnings
against every style. Stacks are20BB: do not compare these numbers directly with
lean-002's100BB evaluations or promote this checkpoint to live100BB play.

The original pilot reports contain a `preflop_allins_per_100_hands: 0` field
inherited from the common summary helper; shove frequency was NOT instrumented
and that field must be ignored. After the run, the current runner was updated to
omit it and expose stack/seed arguments for the standalone evaluator. The archived
source reproduces the original run; no checkpoint or original report was changed.

## Use locally

Fresh training, bounded to a finite time and table size:

```sh
.venv/bin/python -m scripts.history_train \
  --root runs/bot/history-next --minutes 2 --seed 93 --stack-bb 20
```

Resume with the same seed and stack setting and `--resume`. One writer is locked
per root. Checkpoints save atomically every30seconds and at normal shutdown.
SIGTERM/Ctrl-C finish the current iteration before saving; unexpected mid-iteration
failures retain the previous complete checkpoint. Limits are checked between
iterations, so a limit may overshoot by one traversal. The time limit excludes
final serialization. Default row cap500,000 applies separately to regret/average
rows. Start with a bounded run rather than spending hours on sparse tables.

Evaluate against the four fixed styles and the rule-only baseline on the same
fresh deals. The evaluator uses the checkpoint's stack size:

```sh
.venv/bin/python -m scripts.history_gate \
  --checkpoint runs/bot/history-001/history-91.json \
  --out runs/bot/history-001/new-gate.json --deals 500 --seed 20310000
```

For in-process local play:

```python
from poker.history_cfr import HistoryPolicy, load
policy = HistoryPolicy(load('runs/bot/history-001/history-91.json'))
action = policy.act(observation)  # uses learned rows only when supported
```

The new checkpoint deliberately does not load as a lean-v3 checkpoint. At the time
of this pilot, the OpenPoker CLI rejected it because snapshots omit betting history.
The subsequent [river/history implementation](RIVER-003.md) adds sequenced public
event reconstruction and a full-game history checkpoint loader. Incomplete replay
uses the rule fallback. The bounded river policy remains a separate local model.

## Validation and next step

Tests verify analytic Kuhn equilibrium, learned Kuhn performance against all pure
responses, recursive regret accumulation, legal replay of old key collisions,
training/inference parity, private-card exclusion, exact uninterrupted resume,
metadata rejection, sparse-row support and boundary behavior, nuts/aces/free-check
guards, complete legal hands with conserved chips, overwrite/config protection,
and finishing an interrupted iteration before checkpointing. The trainer CLI was
also run fresh and resumed, and the standalone evaluator ran a two-deal smoke.

This delivers a basic rule-backed player and a tested history-aware training path.
The priority is now to build useful support in a SMALLER poker task (for example,
a fixed-board river subgame with explicit ranges), demonstrate learned advantage
there, and then scale the validated representation. Merely training these huge
exact-history tables longer is not yet justified. Avoid lowering the visit gate
just to report learned coverage. Potential-aware cards and range-aware solving
remain necessary development work before a strong full-game bot can be claimed.

Final focused validation: **55 tests passed** across the shared kernel, history
model, lean compatibility, mixture, state audit, live adapter, fast engine and
rule gate suites. `git diff --check` also passed. No long training job remains
running after this experiment.
