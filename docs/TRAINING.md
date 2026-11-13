# Bot training runbook

The current training candidate uses `texture-buckets-v1` and historical-population
MCCFR. It is ready to run bounded learning experiments. Competitive strength and
the benefit of large-scale training are unproven. Human-model work remains deferred.

## Install and verify

```sh
uv venv .venv
uv pip install --python .venv/bin/python -e '.[test]'
.venv/bin/python -m pytest -q
```

## Run training and an independent validation population

Run each command in a separate terminal if sufficient CPU/RAM is available. These
are independent runs, not distributed updates to one table. Do not share output
paths. The validation seed and current-policy-only recipe differ from training.

```sh
.venv/bin/python -m poker.train_cfr --config configs/cfr-bot.json --output runs/bot/candidate.json
.venv/bin/python -m poker.train_cfr --config configs/cfr-bot-validation.json --output runs/bot/validation.json
```

Config targets are total iterations. Each invocation has a one-hour wall budget,
100,000 nodes per iteration and a soft 500,000-information-set limit. Checkpoint
messages report why a run stopped. `target_reached` means the requested iterations
completed, not that a strength target was met. Adjust limits after profiling; do
not interpret budget exhaustion as convergence. JSON config overrides CLI training
settings; output and resume remain CLI arguments.

```sh
.venv/bin/python -m poker.train_cfr --config configs/cfr-bot.json --resume runs/bot/candidate.json --output runs/bot/candidate.json
```

One process owns an output lock. SIGINT/SIGTERM rolls back an incomplete traversal
and saves completed iterations. Resume includes current regrets, average strategy,
historical policies, RNG state and algorithm configuration. Retained archive size
multiplies memory and checkpoint costs. Recovery uses `.previous` into a new output.
The process is CPU-only; checkpointing and Python engine copies dominate overhead.
No cloud spending or GPU provisioning is performed.

## Evaluate before increasing the budget

```sh
.venv/bin/python -m poker.evaluate --checkpoint runs/bot/candidate.json --opponent-checkpoints runs/bot/validation.json --blocks 1000 --seed 2000000 > runs/bot/eval-single.json
.venv/bin/python -m poker.evaluate --checkpoint runs/bot/candidate.json --opponent-checkpoints runs/bot/validation.json --blocks 1000 --seed 2000000 --copies 3 > runs/bot/eval-copies.json
```

Reports separate scripted populations, mixed styles and checkpoint opponents,
with duplicate-deal cluster bootstrap intervals, BB/100, decision counts, table
coverage, latency and wall time. Checkpoint content hashes identify competitors.
Identical candidate policies and retained training archive members are rejected
as held-out opponents. This does not detect all historical ancestry; maintain
separate seed/config/lineage records and never train on validation checkpoints.
Validation agents are independent learned opponents, **not established strong
opponents**. Add stronger independently trained/legal licensed checkpoint sources
as available. Keep a final untouched test population once selection begins.

Save separate checkpoint snapshots for learning curves. Compare multiple training
seeds and population-mixture ablations on identical evaluation deal blocks. Do not
select a model using a pooled scripted win rate or treat confidence intervals from
very few blocks as reliable. Coverage is not strategy quality. No automated strength
promotion or promised iteration count exists.

## Abstraction and algorithm limitations

Preflop buckets retain rank pair and suitedness. Postflop buckets retain made-hand
category, coarse private ranks, board multiplicities/high-card texture, flush suit
counts and straight connectivity. They omit exact kickers, blockers and detailed
ranges. Public buckets retain position, active seats, coarse stack/pot/call ratios,
ordered legal-menu pot ratios and street-level raise counts. Past action order is
lost, making this an **imperfect-recall abstraction**. Distinct actions retain their
menu slots even when ratio buckets collide. Training/deployment must use the same
version; old exact-state and new bucket checkpoints are not interchangeable.

The chance sampler deals real cards through PokerKit; each policy lookup only sees
its own cards and public observations. Neither bucket construction nor historical
policies can access actual opponent cards. Population sampling freezes per-seat
historical average policies for regret traversal; averaging traversal still measures
the current policy. This is a nonstationary population training heuristic, not a
multiplayer equilibrium proof. The existing exact-observation reference remains an
ablation and algorithm debugging option. Live range-aware search and validated
opponent exploitation are not implemented. Do not call the resulting bot GTO,
superhuman or strong without appropriate evidence.

## Staged learning-curve runner

After training the independent validation checkpoint, run:

```sh
.venv/bin/python -m poker.experiment --output runs/bot/learning-curve-seed1 --stages 100 1000 10000 --blocks 1000 --opponent-checkpoints runs/bot/validation.json
```

This freezes a checkpoint at every stage and evaluates both one and three candidate
copies, using common validation deals across stages. It records configuration and
stage metadata, retains all reports, and stops explicitly if a training budget is
hit. It does not silently promote an incomplete stage or overwrite an experiment.
Resource budgets apply to each stage invocation. To recover a budget-stopped
checkpoint use `train_cfr --resume` and evaluate it explicitly. For another training
seed, copy the config and change `seed`; keep the final test set untouched.
