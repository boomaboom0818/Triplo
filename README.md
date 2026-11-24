# Triplo multiplayer poker

Bot-first six-max no-limit Hold'em research and training pipeline. Development is
on `develop`; `main` retains the repository initialization. The bot and future
human model share one codebase. No competitive-strength claim has been established.

## Install and test

Python 3.11+ on Linux (writer locks use `fcntl`):

```sh
uv venv .venv
uv pip install --python .venv/bin/python -e '.[test]'
.venv/bin/python -m pytest -q
```

The suite covers game integration, settlement, action order, privacy, learning
updates, abstraction invariance, exact resume, resource stops and output locks.
CI runs the suite and baseline/CFR smoke workflows.

## Train the bot

For the newer lean trainer's multiway-coverage diagnosis and controlled
loose-caller experiment, see [the mixture pilot](docs/LEAN-MIXTURE-PILOT.md).
Its candidates remain experimental; the measured improvement does not yet
establish an advantage over the tight rule baseline.

See the complete [training runbook](docs/TRAINING.md) for resource controls,
recovery, separate validation training, staged experiments and limitations.

```sh
.venv/bin/python -m poker.train_cfr --config configs/cfr-bot.json --output runs/bot/candidate.json
.venv/bin/python -m poker.train_cfr --config configs/cfr-bot-validation.json --output runs/bot/validation.json
.venv/bin/python -m poker.evaluate --checkpoint runs/bot/candidate.json --opponent-checkpoints runs/bot/validation.json --blocks 1000
.venv/bin/python -m poker.evaluate --checkpoint runs/bot/candidate.json --opponent-checkpoints runs/bot/validation.json --blocks 1000 --copies 3
```

Training uses external-sampling MCCFR with a versioned coarse abstraction and a
historical opponent population. Resume restores regrets, average policies, archive
and RNG state. Every checkpoint identifies its model, algorithm, abstraction,
configuration and compute usage. New checkpoints also record source file hashes.
The default configuration bounds each invocation to one hour; it may stop before
its total target. Review stop reasons and resume with suitable limits.

A staged run retains snapshots and evaluates one and three candidate copies:

```sh
.venv/bin/python -m poker.experiment --output runs/bot/learning-curve --stages 100 1000 10000 --blocks 1000 --opponent-checkpoints runs/bot/validation.json
```

A small preflight is available with `python -m poker.preflight --output runs/bot/preflight`.
It measures coverage and compute; it does not certify competitive readiness.

## Model targets

- `bot-opponent-v1`: active. Diverse-agent robustness and evidence-based exploitation
  are the objectives. Historical-population training is implemented; validated
  opponent exploitation remains unfinished.
- `human-opponent-v1`: reserved and deferred. It will add persistent per-player,
  position/street/sizing-aware adaptation with the same winnings objective.
  Human training/evaluation is explicitly blocked until implemented.

Model IDs prevent accidental cross-target checkpoint use. They identify training
targets, not separate neural architectures. No reduced-difficulty behavior exists.

## Architecture and API

- `poker/game.py`: trusted PokerKit adapter. `play(policies, seed, button)` accepts
  3–6 independent objects implementing `act(observation)`. Returns payoffs in BB,
  decision traces and timings. Policies return `call`, `fold`, or a listed integer
  raise-to amount. Chips are half-BB units; stacks reset to 100 BB each hand.
- `poker/abstraction.py`: own-card/public-state buckets. Action slots retain ordering;
  card and history abstraction is lossy and imperfect recall. See the runbook.
- `poker/cfr.py`: reference regret traversal, importance-weighted averaging,
  branching adapter and average-policy inference.
- `poker/train_cfr.py`: population training, checkpointing, resource budgets,
  graceful interruption and recovery. Single-process CPU implementation.
- `poker/evaluate.py`: per-population seat-balanced duplicate-deal evaluation,
  whole-deal-block bootstrap intervals, sample counts, coverage, latency and time.
- `poker/model.py`: preliminary opponent statistics, not wired into betting.
- `poker/policy.py` and `poker/train.py`: earlier tabular REINFORCE baseline, retained
  separately for comparison. Use `configs/train-bot.json` for that baseline.

Observations contain own private cards and public state/history only. Independent
policy instances have separate RNG/model state. Simulator internals are never
passed to policies, but in-process Python is not a security sandbox. Untrusted
external agents need process isolation and a serialized protocol before use.

The game engine supports multiple all-ins, side pots, folds down to heads-up and
configurable stacks. Rake and tournaments are unsupported. Current training and
evaluation use six players and reset 100-BB stacks. Tests verify several exact
payout and reopening cases; broader cumulative-short-raise/rules certification and
public showdown-history integration remain outstanding.

## Measured artifacts and limitations

`artifacts/bot/abstract-pilot/` contains an actual 100-iteration candidate, a separate
20-iteration validation checkpoint, and evaluation reports for one and three copies.
The snapshots are included so report hashes can be reproduced. This short pilot
is not a learning curve or evidence of competitive strength. Historical exact-state
and REINFORCE smoke artifacts are retained in other artifact directories.

Practical abstraction and population training are now implemented, but their
quality needs multi-seed learning curves and stronger held-out opponents. Coverage
remains sparse in the pilot. No validated range-aware limited-depth search, live
adaptation, native accelerated traversal backend or established strong benchmark
agent is included. Checkpoint opponents are supported; independently trained
pilots must not be described as strong merely because they are learned agents.
Multiplayer CFR and imperfect-recall abstractions do not inherit two-player
zero-sum equilibrium guarantees. No heads-up metric establishes multiplayer strength.

## Deployment setups

**Frontier setup: Triplo v7-fast** (released as "v7"). This is what runs live on OpenPoker (requires the `openpoker` and `river-solver` extras):

```sh
python -m scripts.openpoker_client --checkpoint runs/bot/pressure-fresh-001/base-guard.json \
  --min-visits 64 --buy-in 5000 --auto-rebuy --fallback equity-guarded-v5 \
  --stack-press --deep-value --no-learned-postflop --river-solver \
  --defend-3bet --flop-peel --turn-defend --opponent-stats runs/bot/opponent-stats.json \
  --avoid-strong leave --strong-bb100 220 --min-weak 2 --max-strong 1 --idle-timeout 300
```

**Triplo v7-fast** adds `--turn-defend` to v6-fast: heads-up, it calls a first turn bet from a player who
did not bet the flop with pairs and draws instead of folding them. Live v6 folded 79% to these turn
stabs. Dropping `--turn-defend` gives **v6-fast**. `--turn-probe` (turn bets after a checked flop) is
in the code but off. See [Triplo v7](docs/TRIPLO-V7.md).

**Conservative setup: Triplo v7-tight.** v6-tight plus the same `--turn-defend`: 50 BB buy-in, the learned
policy still plays the postflop spots it covers, and no Stack Press or Deep Value sizing:

```sh
python -m scripts.openpoker_client --checkpoint runs/bot/pressure-fresh-001/base-guard.json \
  --min-visits 64 --buy-in 1000 --auto-rebuy --fallback equity-guarded-v5 --river-solver \
  --defend-3bet --flop-peel --turn-defend --opponent-stats runs/bot/opponent-stats.json \
  --avoid-strong leave --strong-bb100 220 --min-weak 2 --max-strong 1 --idle-timeout 300
```

Dropping `--turn-defend` from v7-tight gives **v6-tight**.

**Triplo v6-fast** builds on **v5-fast** by adding `--defend-3bet --flop-peel` to defend opens against 3-bets and peel small/medium flop bets with draws and pairs. Dropping those two flags gives **v5-fast** verbatim.

**Conservative setup: Triplo v6-tight.** The same two fixes on top of **v5** instead of v5-fast: 50 BB buy-in, the learned policy still plays the postflop spots it covers, and no Stack Press or Deep Value sizing:

```sh
python -m scripts.openpoker_client --checkpoint runs/bot/pressure-fresh-001/base-guard.json \
  --min-visits 64 --buy-in 1000 --auto-rebuy --fallback equity-guarded-v5 --river-solver \
  --defend-3bet --flop-peel --opponent-stats runs/bot/opponent-stats.json \
  --avoid-strong leave --strong-bb100 220 --min-weak 2 --max-strong 1 --idle-timeout 300
```

Dropping `--defend-3bet --flop-peel` from v6-tight gives baseline **v5**. See [Triplo v6](docs/TRIPLO-V6.md) and [OpenPoker client setup](docs/openpoker.md) for full details and benchmarks.

**`base-guard.json` is now the 2,001,520-iteration model** (2026-11-22), which replaced the
969,520-iteration model after longer training improved it by +6.7 bb/100 [+1.1, +12.3] on a
fresh confirmation seed (+10.1 on the development seed). The old model is kept as
`base-guard-969520.json`. Weights are not committed; see
[Scale-001 results](docs/SCALE-001-RESULTS.md) to rebuild the model and for the results.

**Late-Position Steals is built into `equity-guarded-v3` through `equity-guarded-v5` for history checkpoints.**
The client automatically reads `runs/bot/late-position-folds.json` at startup.
It can turn a fold into a 2.5 BB open from the cutoff, button, or small blind
only with qualifying fold evidence for the exact remaining lineup. If the file
is absent or no lineup qualifies, the original action stands. The initial data
check found no qualifying lineups and did not establish live profitability. See
the [OpenPoker client setup](docs/openpoker.md),
[Triplo v4](docs/TRIPLO-V4.md), and the
[Late-Position Steals fitting guide](docs/LATE-POSITION-STEALS.md).

A deployment combines the **model** (`--checkpoint`), an optional
**PressureGuard** baked into that checkpoint, the **fallback** (`--fallback`) that
plays states the model does not cover, and opt-in switches (`--buy-in`, `--stack-press`,
`--deep-value`, `--no-learned-postflop`, `--river-solver`, `--defend-3bet`, `--flop-peel`,
and `--avoid-strong`).

| Setup | Checkpoint | Fallback & key flags | Good for | Watch out for |
|---|---|---|---|---|
| **Triplo v7-fast** (frontier, current live default; released as "v7") | `runs/bot/pressure-fresh-001/base-guard.json` | v6-fast + `--turn-defend` | Same fields as v6-fast, plus regulars that bet into Triplo's checks (live v6 folded 79% to heads-up turn stabs). 4,096-deal paired screen over v6-fast: +8.2 [+2.9, +13.6] BB/100 vs a Doppio-style turn-stab field, +1.9 [−1.8, +5.6] in `pressure`, 0.0 in every other field (`docs/TRIPLO-V7.md`). | Changes only heads-up turn calls facing a probe, so the gain is small outside stab-heavy tables. Not yet confirmed live. Drop `--turn-defend` to roll back to v6-fast. |
| **Triplo v6-fast** (previous frontier; was "v6") | `runs/bot/pressure-fresh-001/base-guard.json` | `equity-guarded-v5` + `--buy-in 5000 --stack-press --deep-value --no-learned-postflop --river-solver --defend-3bet --flop-peel` + `--opponent-stats` | Mixed OpenPoker tables with both calling-station presets and HUD-driven regulars. Closes v5-fast's fold-to-3bet (~89% → ~55–65%) and flop fold-to-cbet (49% → 34%) leaks while keeping 250 BB preset extraction. In 4,096-deal paired screens over v5-fast: +29.3 [+5.3, +53.2] BB/100 in `real_mix`, +40.0 [+13.3, +66.7] in `pressure`, +7.9 [+1.9, +13.9] head-to-head vs v5-fast, confirmed at +3.8 [+1.9, +5.7] on 32,768 fresh deals (`docs/TRIPLO-V6.md`). | Moves some flop folds to the turn (a double-barreling shark can still test turn folds); carries v5-fast's unequal-stack drag vs v4 (−8.5 [−15.3, −1.8] in crossplay). Drop `--defend-3bet --flop-peel` to roll back to v5-fast. |
| **Triplo v7-tight** (conservative v7) | `runs/bot/pressure-fresh-001/base-guard.json` | v6-tight + `--turn-defend` (`equity-guarded-v5` + `--buy-in 1000 --river-solver --defend-3bet --flop-peel --turn-defend` + `--opponent-stats`) | Tougher or shorter-stacked tables, with v6-tight's lower variance and v7's turn-stab defense. Head-to-head and field results vs v6-tight in `docs/TRIPLO-V7.md`. | Not yet tested live. Drop `--turn-defend` to roll back to v6-tight. |
| **Triplo v6-tight** (conservative v6) | `runs/bot/pressure-fresh-001/base-guard.json` | `equity-guarded-v5` + `--buy-in 1000 --river-solver --defend-3bet --flop-peel` + `--opponent-stats` | Tougher or shorter-stacked tables where you want v6's 3-bet defense and flop peel with v5's lower variance: 50 BB buy-in, learned postflop play, no Stack Press or Deep Value. Duplicate head-to-head (8,192 deals per cell): +5.0 [+1.3, +8.8] BB/100 vs v5 with measured profiles at unequal stacks, +8.1 [+4.4, +11.9] vs v6-fast cold at unequal stacks, and not significantly behind v5 or v6-fast in any cell (`docs/TRIPLO-V6.md`). | Not yet tested live, and the head-to-head ran at 250 BB, not its 50 BB buy-in. Like v5, the 50 BB buy-in caps chip rate on caller-heavy tables. Drop `--defend-3bet --flop-peel` to roll back to v5. |
| **Triplo v5-fast** (previous frontier) | `runs/bot/pressure-fresh-001/base-guard.json` | `equity-guarded-v5` + `--buy-in 5000 --stack-press --deep-value --no-learned-postflop --river-solver` + `--opponent-stats` (+ `--avoid-strong leave --strong-bb100 220 --min-weak 2 --max-strong 1`) | Maximizing chip rate on soft/mixed fields. Keeps the 2M-iteration learned policy preflop, routes all postflop play to the guarded fallback + river solver (cutting postflop air/weak-pair bluff spew), and unlocks `0.75×/1.5×/2×/half-stack/cover` value sizing at all depths plus sticky top-pair value raises. In 10,240 paired deals vs v5 @ 50 BB: +3,917 vs +1,082 on `sticky`, +1,928 vs +486 on `mixed`, +458 vs +157 on `real_mix`, and +27.7 vs −4.1 vs 5× Turbo. | Higher 250 BB swing variance than 50 BB v5; without v6-fast's `--defend-3bet --flop-peel`, HUD sharks can exploit its ~89% fold-to-3bet and ~57% flop fold-to-cbet. |
| **Triplo v5** | `runs/bot/pressure-fresh-001/base-guard.json` | `equity-guarded-v5` (defaults to `--buy-in 1000` / 50 BB with `--river-solver` ON) + `--opponent-stats` | Lower-variance short-stack play and tougher heads-up rivers. Adds the calibrated heads-up river subgame solver (`poker/calibrated_river_solver.py`) on verified fallback river spots; at 50 BB the solver added +4.66 [+0.87, +8.44] BB/100 vs 5× Turbo (`runs/bot/buyin-river-factorial-001`). Requires `.[openpoker,river-solver]`. | The 50 BB default buy-in caps chip rate on caller-heavy tables (~2–3× slower than 250 BB). Pass `--buy-in 2500` or `--buy-in 5000` with `--river-solver` for deeper games. |
| **Guarded + `equity-guarded-v4`** (previous frontier) | `runs/bot/pressure-fresh-001/base-guard.json` (2,001,520-iteration model + PressureGuard v1 since 2026-11-22) | `equity-guarded-v4` + `--opponent-stats runs/bot/opponent-stats.json` | Every field, without the optional `numpy`/`river-solver` dependency. Keeps every preflop and heads-up `v3` decision verbatim, and scales multiway postflop betting thresholds by fair-share equity `1 / (opponents + 1)` (using half-pot when newly unlocked below 0.40 usable equity). Untouched 16-scenario holdout (~132,000 hands): +6.21 ± 2.79 BB/100 aggregate, 3 significant wins, 0/16 significant regressions (`docs/TRIPLO-V4.md`). | Multiway simulation gains are strongest against loose/passive fields; without `--stack-press` / `--deep-value`, postflop raises stay capped at 1× pot. |
| **Guarded + `equity-guarded-v3`** | `runs/bot/pressure-fresh-001/base-guard.json` | `equity-guarded-v3` + `--opponent-stats runs/bot/opponent-stats.json` | Every field. All of v2's rules, with opponent ranges calibrated on 5,447 real OpenPoker showdown hands (`poker/range_calibration.py`): one multiplier per hand class for each postflop line (bet/raise, call, check). Predicts real holdings better than v2's hand-set weights on held-out hands (log-loss 1.127 vs 1.233). | Uses a flat 0.40 multiway postflop threshold that under-bets top pair and strong favorites in 3–6 player pots. |
| **Guarded + `equity-guarded-v2`** | `runs/bot/pressure-fresh-001/base-guard.json` | `equity-guarded-v2` + `--opponent-stats runs/bot/opponent-stats.json` | Every field. Keeps all of Guarded, and adds call pricing that splits main and side pots by who can win each, plus bet sizing from each opponent's measured postflop calling stats. Sizing adapts to a player only after they have faced 20 postflop bets; until then it keeps v1's size. | Against five copies of Guarded in duplicate play it was even (about +0.2 ± 0.5 bb/100, no table type worse, `runs/bot/h2h-002`); its edge is expected against sticky callers, which simulation cannot supply, so confirm it with live results. Needs `--opponent-stats`, or bet sizing stays at v1's. |
| **Guarded** | `runs/bot/pressure-fresh-001/base-guard-969520.json` (stage-10 model + PressureGuard v1; was `base-guard.json` until 2026-11-22) | `equity-guarded-v1` | Mixed, tight and aggressive fields, and short or medium stacks. Best simulated average (58.2 bb/100, +11 over Classic) and +38 to +45 bb/100 against tight/aggressive 100 BB tables. | Fell from +32 to about -19 bb/100 against copies of the Classic model. That gap is not statistically significant but appeared in every guarded setup. |
| **Guarded with PressureGuard v2** | `runs/bot/pressure-fresh-001/stage-10-pressure-guard-v2.json` or `history-sustained-001-pressure-guard-v2.json` | `equity-guarded-v1` | Same niche as Guarded; v2 adds the JJ/QQ 100 BB cap, river-overbet fold and reraise limit inside the guard. Statistically tied with Guarded. | Not in the full nine-table screen for the stage-10 model. |
| **Classic** (the original baseline) | `runs/bot/history-sustained-001/checkpoint.json` | `equity-v1` | Fields of strong trained bots: it was the only setup that beat copies of itself (+32 bb/100 in simulation). | Loses to tight and aggressive rule-style players at 100 BB (about -10 and -17 bb/100). Can call off huge pots with one pair or JJ/QQ. |
| **Classic + guarded fallback** | same | `equity-guarded-v1` | Same as Classic with the one-pair and JJ/QQ disasters removed. | Statistically no different from Classic in simulation (0.0 bb/100); the model plays most hands. |
| **Any guarded setup + Stack Press** | any | `equity-guarded-v1..v5` + `--stack-press` | Weak or loose fields (calling stations, over-callers) and deep stacks (> 100 BB). In duplicate self-play it added +47 bb/100 at 100 BB and +117 bb/100 at 200 BB against rule bots. | Strictly more risk when you are beaten. No gain against an equal-strength field (0.0 bb/100). Turn it off in tight, competitive games. |

All lean-menu setups now offer the *effective* all-in: a raise to the deepest live
opponent's total, so a deep stack can still put a short stack all in (it was offered
only from our own stack before; worth about +9 bb/100 against callers when we are deep).
Build or refresh the stats file from client traces with
`python -m poker.opponent_stats runs/bot/*/trace.jsonl --out runs/bot/opponent-stats.json`;
the client keeps it updated after every hand. Refit the v3 range table from new
showdowns with `scripts.range_sim_lines` and `scripts.range_fit`; `scripts.range_report`
compares real holdings by line with the model. `scripts.h2h_live` replays the same deals
with a candidate in one seat against the deployed setup.

Choosing quickly: by default run **Triplo v7-fast** (the frontier command above, with
`--opponent-stats` and `--avoid-strong leave --strong-bb100 220 --min-weak 2 --max-strong 1`);
drop `--turn-defend` for **v6-fast**; for lower variance run **Triplo v7-tight** (its command above), or drop `--turn-defend` from it for **v6-tight**; from v6-fast, to roll back its 3-bet defense and flop peel, drop `--defend-3bet --flop-peel` (**v5-fast**);
for lower variance on tougher tables, run **Triplo v6-tight** (drop `--stack-press --deep-value
--no-learned-postflop` and buy in at `--buy-in 1000`), or drop `--defend-3bet --flop-peel` from it
for plain **v5**; without the `river-solver` extra, use
`--fallback equity-guarded-v4` (**v4**) or earlier `v3`/`v2`/`v1` fallbacks; on a pure
trained-bot table where Guarded underperforms, fall back to **Classic**. Startup logs confirm
each active switch (`Stack Press`, `Deep Value`, `River Solver`, `Defend 3-Bet`, `Flop Peel`,
and `learned_postflop`).

Evidence and limits: these figures come from simulated tables (`runs/bot/deploy-combo-001`
for model/guard/fallback combinations, `runs/bot/selfplay-ab-001` for Stack Press,
reproducible with `scripts.live_gate evaluate --hero-fallback ...` and
`scripts.selfplay_ab`). Opponents are fixed rule bots or copies of this project's own
policies, not real opponents. The simulator offers fewer bet sizes than the
table client, so it cannot measure Stack Press inside the learned model. Treat
all of this as a development screen; compare results per 100 hands over a long
sample of real games before trusting any setup.

## Research and licensing

[PokerKit](https://pokerkit.readthedocs.io/en/stable/simulation.html) 0.7.7 supplies
the game engine under its [MIT license](https://github.com/uoftcprg/pokerkit/blob/main/LICENSE).
Installed distributions retain upstream notices. Algorithms here are original
project implementations informed by [Lanctot et al.'s MCCFR paper](https://www.mlanctot.info/files/papers/nips09mccfr_techreport.pdf)
and [Brown and Sandholm's Pluribus paper](https://noambrown.com/papers/19-Science-Superhuman.pdf).
This is not a Pluribus reproduction. Python was chosen for an inspectable reference;
profile throughput, table growth and learning before budgeting larger compute.

## History-aware experimental model

[History-001](docs/HISTORY-001.md) adds public betting history, a learning kernel
validated on Kuhn poker, resumable bounded training, local evaluation, and basic
play guards with a tight-rule fallback. The first short Hold'em pilot did not
establish learned improvement; guarded play used the fallback throughout. New
history checkpoints use a separate format. The [river/history work](docs/RIVER-003.md)
adds verified OpenPoker history reconstruction and demonstrates learning in bounded
heads-up river games. See those reports for commands and measured limitations.
