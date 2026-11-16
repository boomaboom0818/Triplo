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
