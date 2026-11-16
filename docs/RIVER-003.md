# Bounded river learning and live public-history adapter

## Result

The bounded Hold'em river experiment passed its predeclared aggregate gate on
all three training seeds. This is measured learning in a small actual-card river
game, not full-game or six-player strength. The river policy is not deployed as a
live full-game player.

The game is heads-up with two known weighted ranges (up to three holdings each),
a fixed five-card board, and either a half-pot or pot bet. A player may check/bet;
the opponent may fold/call. A check allows the other player to bet. Raises and
additional bet sizes within one game are excluded. Showdowns use the real Hold'em
evaluator, including ties. Incompatible joint holdings are removed and joint
chance weights renormalized. Payoffs include each player's share of the initial
pot so they sum to zero.

Policies condition only on their own holding and public information. Exhaustive
best responses enumerate pure strategies per OWN holding and observed action
history. They cannot choose separate actions based on the opponent's hidden cards.
Expectations integrate all compatible private deals and action probabilities;
there is no Monte Carlo evaluation error to bootstrap in these finite games.

## Design and measured outcomes

Eight training games were fixed using generator seed311, with equal and unequal
range weights and both bet sizes. Eight untouched boards/ranges were then generated
that match the training games' public payoff/chance structures. Their cards differ,
but the abstract games are strategically identical. This is an exact structural
transfer test, NOT generalization to arbitrary unfamiliar ranges or betting trees.
Held-out generation uses no learned policy or payoff-vs-baseline filter.

Public keys retain the joint deal probabilities, showdown outcome matrix, bet/pot
ratio, acting player, own public-range index and action history. Binary chance
probabilities are encoded losslessly. A saved `RiverPolicy` refuses an unsupported
public structure rather than silently inventing a strategy. Candidate ranges and
their showdown relationships are public; the actual opponent holding is not.

Each of seeds121/122/123 trained40,000 sampled deals across the training cases with
the shared, Kuhn-validated external-sampling kernel. Policy snapshots were frozen
before evaluation. The fixed gate was mean held-out exploitability below0.03BB,
and positive mean exact seat-balanced advantage over `tight-v1` for EACH seed.

| Training seed | Mean exploitability, BB | Mean gain over tight baseline, BB/hand | Gate |
| --- | ---: | ---: | --- |
| 121 | 0.00836 | +0.01013 | pass |
| 122 | 0.01240 | +0.01638 | pass |
| 123 | 0.01074 | +0.01860 | pass |

For reference, uniform play's mean exploitability is0.47507BB and the tight rule
baseline's is0.51292BB in these same bounded cases. Exploitability is half of
NashConv (sum of each player's best-response improvement). It applies here because
this game is two-player zero-sum. These values must not be called six-player
exploitability or compared directly with full-game win rates.

The mean advantage does not mean a win in every case: some individual cases favor
the rule baseline. Seed122's worst-case exploitability is0.03522BB, so the
mean0.03 gate does not establish a0.03 bound for every board. The three seeds also
share the same cases. No confidence claim across arbitrary boards follows.

Dedicated tactical tests on a Kuhn-equivalent actual-card river verify learned
value bets with the highest holding, mixed bluffs with the lowest, folding a
holding that cannot win, and mixed bluff-catching with the middle holding. An
analytic Kuhn equilibrium independently verifies the exhaustive evaluator.

Artifacts: `runs/bot/river-003/manifest.json`, `summary.json`, `policy-121.json`
through `policy-123.json`, `execution.log` and archived exact sources. Earlier
`river-001/002` runs are preserved development checks;003 is the final-source run.

## Reproduce and inspect

```sh
.venv/bin/python -m scripts.river_experiment --root runs/bot/river-new
```

A new root is mandatory; reports/checkpoints are never overwritten. Seeds, game
ranges/weights/boards and gate are recorded before training. Source hashes must
match at completion. Saved policies explicitly identify `bounded-river-es-v1` and
are separate from all full-game checkpoints. This is a standalone local policy:

```python
from poker.river import RiverPolicy, RiverGame
from poker.river_solver import solve
# game supplies the public board and both public weighted ranges as integer cards
policy = RiverPolicy.load('runs/bot/river-003/policy-121.json')
probabilities = policy.probabilities(game, player=0, hole=own_cards, history='')
# Or solve a previously unseen public game on demand:
unseen_policy, checkpoint = solve(game, iterations=20000, seed=131)
# Empty history: [check, bet]; facing a bet: [fold, call].
```

### On-demand solving on unseen public structures (`river-unseen-001`)

To evaluate previously unseen boards, range weights (`[1,2,4]` vs `[4,2,1]`),
and bet/pot ratio (`1.5 / 2.0 = 0.75`), `poker/river_solver.py` and
`scripts/river_unseen.py` solve each of eight new public games from scratch
with 20,000 iterations per game across three seeds (131, 132, 133):

```sh
.venv/bin/python -m scripts.river_unseen --root runs/bot/river-unseen-new
```

| Seed | Mean exploitability, BB | Mean gain over tight baseline, BB/hand | Gate |
| --- | ---: | ---: | --- |
| 131 | 0.00509 | +0.08159 | pass |
| 132 | 0.00277 | +0.08142 | pass |
| 133 | 0.00420 | +0.07810 | pass |

Artifacts are recorded in `runs/bot/river-unseen-001/manifest.json`,
`summary.json`, `execution.log`, `policy-131-case-0.json` through
`policy-133-case-7.json`, and archived sources.

## OpenPoker history reconstruction

The full-game `history-public-v1` checkpoint can now be loaded by the existing
OpenPoker client. The bounded river model cannot; it lacks the other streets,
full-size ranges, raises, side pots and multiplayer play.

`poker/openpoker_history.py` retains sequenced public hand starts and player
actions. Reconnect replay is sorted and deduplicated before reconstructing the
current hand. Forward sequence jumps are accepted, because private messages can
consume unseen sequence numbers. A full hand start is required. Blind units,
actor order, streets, action contributions, stacks, bets, folds, final pot and
acting seat are validated against the authoritative snapshot. Initial stacks are
inferred from final snapshot stacks plus reconstructed contributions; they are
not independently certified starting-stack records.

Contributions are derived from public pot changes, avoiding ambiguity between
incremental event amounts and raise-to action amounts. Normalization must fall on
the trained half-BB grid; short blind posts and unsupported/inconsistent history
use the fallback. Partial replay, missing starts, conflicting duplicates, actor
mismatches and table/hand changes never enable learned inference. Short-handed
hands use the adapter's existing six-seat embedding with explicit early synthetic
folds, and tests check the corresponding history positions.

History reconstruction does not modify snapshot pot or stacks a second time.
The client logs `history_verified`, `history_reason` and actual learned coverage.
Duplicate turn tokens retain exactly the same action payload and action ID.
Cold reconnect asks for retained replay with `last_table_seq:0`; warm reconnect
uses the retained watermark. Learned full-game rows still require their configured
minimum visit count, so sparse checkpoints may mostly use the rule fallback.
These mechanics follow the maintained [OpenPoker message catalog](https://docs.openpoker.ai/api-reference/message-types/)
and [reducer rules](https://docs.openpoker.ai/building-bots/state-consistency/).

A short diagnostic, after securely configuring the API key:

```sh
.venv/bin/python -m scripts.openpoker_client \
  --checkpoint runs/bot/history-001/history-91.json \
  --min-visits 64 --max-minutes 2 \
  --log runs/bot/openpoker-history-diagnostic.jsonl
```

That checkpoint was trained at20BB while ordinary live tables may be100BB; exact
public keys will often be unsupported and use fallback. This session tests the
adapter and logs missing situations, not the bounded river model's playing strength.
Auto-rebuy remains disabled unless explicitly requested. The client stops at the
first completed hand after its time budget; it is not a hard socket timeout.

In the executed diagnostic (`runs/bot/openpoker-history-001`), the bot joined a
six-player table (`table_id=a675d2dc-c37c-444e-b9be-322c437dd3eb`), observed one
completed hand before acting, and played one full hand across all four streets
(4 decisions). Live opponent stacks and bet sizes (e.g., 526, 612, 572, 1168
chips on a 20-chip BB table) fell off the exact half-BB grid (`chip amount
outside trained half-BB grid`), so `PublicHistory` explicitly flagged
`history_verified: false` and the guarded client cleanly used its rule fallback
for all four decisions before exiting with `auto_rebuy=False`.

## Sustained multi-core training (`history-compact-v1`)

To resolve both the live off-grid chip rejection and the exact-history state
explosion discovered above:

1. `poker/openpoker_history.py` now validates the public action ledger in exact
   integer chips (`small_blind` / `big_blind` units) before normalizing output
   history entries to half-BB units, so off-grid live stacks and bet sizes
   (e.g., `4784` stacks or `112`/`612` chip raises) verify cleanly while still
   rejecting any missing, out-of-order, or inconsistent pot/stack events.
2. `poker/history_cfr.py` adds `history-compact-v1` (`compact_observation_key`)
   and `ParallelHistory` (shared-memory multi-core arrays + loose-caller
   population mixing). Keys retain `lean-v3` card/position/pot/price/SPR buckets
   plus per-street raise counts, relative last-aggressor identity, last raise
   pot-ratio token (`s`/`p`/`o`), current-street call count, and legal raise menu
   ratios. All `lean_state_audit` history collisions remain separated while
   generalizing across stack depths and off-tree bet sizes.
3. `scripts/history_train.py` supports multi-core sustained training and atomic
   checkpoint/state resume:

```sh
# Start or resume sustained 100BB multi-core training:
.venv/bin/python -m scripts.history_train \
  --root runs/bot/history-sustained-001 \
  --minutes 60 --workers 10 --population 0.5 \
  --abstraction history-compact-v1 --stack-bb 100 \
  --max-rows 15000000 --save-minutes 5 --resume
```

In the sustained training progression (`runs/bot/history-sustained-001`, 10
workers at 100BB with `population=0.5` up to `876,520` iterations, `1,663,968,933`
nodes, and `7,934,624` unique state rows), learned decision coverage at the
64-visit gate (`seed=21310000`) and win rates (`bb/100`) progressed as follows:

| Checkpoint | Style | Learned coverage | Candidate (`bb/100`) | Guarded fallback (`bb/100`) | Paired diff (`bb/100`) |
| --- | --- | ---: | ---: | ---: | ---: |
| `11,030` (`gate-1m.json`) | `tight-v1` | `3.3%` | `-49.17` | `-46.17` | `-3.00` |
| `11,030` (`gate-1m.json`) | `aggressive-v1` | `3.5%` | `-6.58` | `-2.42` | `-4.17` |
| `11,030` (`gate-1m.json`) | `station-v1` | `2.3%` | `+260.42` | `+283.08` | `-22.67` |
| `11,030` (`gate-1m.json`) | `shover-v1` | `4.7%` | `+20.42` | `+21.17` | `-0.75` |
| `51,520` (`gate-51k.json`) | `tight-v1` | `33.3%` | `-86.33` | `-30.29` | `-56.04` |
| `51,520` (`gate-51k.json`) | `aggressive-v1` | `39.4%` | `+7.54` | `+24.54` | `-17.00` |
| `51,520` (`gate-51k.json`) | `station-v1` | `25.8%` | `+222.88` | `+187.29` | `+35.58` |
| `51,520` (`gate-51k.json`) | `shover-v1` | `40.7%` | `+18.58` | `+12.00` | `+6.58` |
| `81,520` (`gate-81k.json`) | `tight-v1` | `41.7%` | `-53.92` | `-30.29` | `-23.62` |
| `81,520` (`gate-81k.json`) | `aggressive-v1` | `50.9%` | `-22.88` | `+24.54` | `-47.42` |
| `81,520` (`gate-81k.json`) | `station-v1` | `30.5%` | `+198.12` | `+187.29` | `+10.83` |
| `81,520` (`gate-81k.json`) | `shover-v1` | `48.9%` | `+53.04` | `+12.00` | `+41.04` |
| `151,520` (`gate-151k.json`) | `tight-v1` | `52.5%` | `-65.42` | `-30.29` | `-35.13` |
| `151,520` (`gate-151k.json`) | `aggressive-v1` | `60.8%` | `-44.29` | `+24.54` | `-68.83` |
| `151,520` (`gate-151k.json`) | `station-v1` | `41.3%` | `+209.33` | `+187.29` | `+22.04` |
| `151,520` (`gate-151k.json`) | `shover-v1` | `55.5%` | `+8.75` | `+12.00` | `-3.25` |
| `716,520` (`gate-716k.json`) | `tight-v1` | `72.5%` | `-46.25` | `-30.29` | `-15.96` |
| `716,520` (`gate-716k.json`) | `aggressive-v1` | `82.5%` | `-34.04` | `+24.54` | `-58.58` |
| `716,520` (`gate-716k.json`) | `station-v1` | `75.8%` | `+378.29` | `+187.29` | `+191.00` |
| `716,520` (`gate-716k.json`) | `shover-v1` | `61.7%` | `+42.08` | `+12.00` | `+30.08` |
| `856,520` (`gate-856k.json`) | `tight-v1` | `74.4%` | `+18.08` | `-30.29` | `+48.38` |
| `856,520` (`gate-856k.json`) | `aggressive-v1` | `83.6%` | `-18.83` | `+24.54` | `-43.38` |
| `856,520` (`gate-856k.json`) | `station-v1` | `78.5%` | `+351.62` | `+187.29` | `+164.33` |
| `856,520` (`gate-856k.json`) | `shover-v1` | `61.9%` | `+51.21` | `+12.00` | `+39.21` |
| `876,520` (`gate-876k.json`) | `tight-v1` | `76.2%` | `-18.04` | `-30.29` | `+12.25` |
| `876,520` (`gate-876k.json`) | `aggressive-v1` | `84.0%` | `-121.54` | `+24.54` | `-146.08` |
| `876,520` (`gate-876k.json`) | `station-v1` | `77.1%` | `+252.17` | `+187.29` | `+64.88` |
| `876,520` (`gate-876k.json`) | `shover-v1` | `61.7%` | `+40.21` | `+12.00` | `+28.21` |

## Verification

Tests cover exact payoff arithmetic, blocked deals, weighted chance, split pots,
private-card isolation, learned river behavior, board-transfer support, on-demand
solving of unseen structures (including zero-reach dominated branches), policy
snapshot immutability, compact-history collision separation, multi-core
`ParallelHistory` state resume/export, and the live adapter's sequenced replay
(including off-grid live chip amounts) and fallback paths.
An asynchronous fake WebSocket checks cold resync, replay, duplicate-token payload
reuse and session logging without any network access or credentials. Live traffic
may still reveal protocol differences not represented by these documented fixtures.

Final local verification: **72 focused tests passed**, covering both the new work
and existing lean/history/client/engine compatibility. `git diff --check` passed.
The live diagnostic is tracked separately under `runs/bot/openpoker-history-001`;
its result is not part of the bounded river gate.


