# Triplo multiplayer poker research baseline

Working local simulation, training, and evaluation prototype. **Not a competitive poker bot yet.** No strength claim is supported by the included smoke runs. Development branch: `develop`; `main` is the initialization point.

## Install and run

Python 3.11+; CPU only. With uv:

```sh
uv venv .venv
uv pip install --python .venv/bin/python -e '.[test]'
.venv/bin/python -m pytest -q
.venv/bin/python -m poker.train --hands 1000 --seed 1 --output artifacts/baseline.json
.venv/bin/python -m poker.evaluate --blocks 1000 --seed 100000
.venv/bin/python -m poker.evaluate --blocks 1000 --seed 100000 --copies 3
```

The committed checkpoint is only a 200-hand smoke training run. Evaluation JSON files contain actual measurements, sample sizes, wall time, decision latency, and 95% percentile bootstrap intervals. Ten independent deal blocks are far too few for reliable strength conclusions. Wall seconds are measured compute cost; no cloud dollar cost is inferred.

## Architecture and API

- `poker/game.py`: trusted PokerKit adapter, integer half-big-blind chips, stacks reset each hand, rotating positional mapping, legal discrete raise menu. Call `play(policies, seed, button)` with 3–6 separately instantiated policy objects. A policy implements `act(observation)` and returns `call`, `fold`, or a legal integer raise-to amount. Payoffs are BB.
- `poker/policy.py`: stochastic tabular blueprint and benchmark fixtures.
- `poker/model.py`: independent public-action counts indexed by player, position, street, including sizing. This is not yet connected to action selection.
- `poker/train.py`: terminal-return REINFORCE over coarse information buckets, mixing historical checkpoints and scripted opponents. No CFR guarantees claimed.
- `poker/evaluate.py`: held-out deal seeds, seat rotations, candidate copies, population-separated results. Bootstrap resamples whole duplicate-deal blocks, preserving within-block seat and deal dependencies.
- `configs/`: bot and human configuration declarations. Both use exactly the same policy. Evaluation loads them with `--config configs/bot.json` or `--config configs/human.json`. Neither has validated adaptation.

Observations are frozen detached records with own hole cards, board, public stacks, active players and public action history. No engine or future deck is passed to policies. This protects ordinary API consumers, not hostile Python code running in the same process. Untrusted external agents require process isolation and serialized observations before deployment. Independent policies have their own RNG and opponent model; do not share model state between seated agents.

The current action menu is call/check, fold when legal, minimum raise, half-pot, pot and all-in (deduplicated and engine-validated). It does not expose all possible legal no-limit sizes. The state abstraction loses exact bet history and stack sizes, uses coarse private ranks, and retains board ranks without suits. It is severely lossy, imperfect-recall, and unsuitable for a strength claim. Unvisited buckets choose uniformly; policy-gradient learning is high variance and sparse. No LLM selects bets.

## Engine and research decisions

[PokerKit documentation](https://pokerkit.readthedocs.io/en/stable/simulation.html) supplies a maintained no-limit engine, including side-pot and betting mechanics. Version 0.7.7 is pinned. [Its license is MIT](https://github.com/uoftcprg/pokerkit/blob/main/LICENSE); installed distributions retain upstream notices. Engine behavior is covered here by initial integration tests, **not a complete independent rules certification**. The 19 tests include short-all-in reopening, folded blinds, uncalled returns, button rotations and fold-down to heads-up. Exact multi-side-pot, cumulative-short-raise, split-pot and odd-chip fixtures remain necessary before trusting large experiments.

[Brown and Sandholm, 2019](https://noambrown.com/papers/19-Science-Superhuman.pdf) and [supplement](https://noambrown.com/papers/19-Science-Superhuman_Supp.pdf) motivate an offline blueprint plus imperfect-information limited-depth search. This implementation is a preliminary learning baseline, not a reproduction. Multiplayer CFR does not inherit two-player zero-sum convergence guarantees.

Python and tabular learning were chosen for an inspectable smoke baseline, not because Python happened to be installed. Competitive training needs a profiled native traversal backend, appropriate information abstraction and far larger training. No GPU stack is justified for this table-based baseline; neural methods would require a separate measured design decision.

## Staged work remaining

1. Add exact rule regression cases covering every settlement/reopening edge case, odd-chip seat order, and explicit public showdown observations. Add serialized bot protocol with process isolation.
2. Implement and validate external-sampling multiplayer MCCFR on a small tractable game, then transfer to hold'em with versioned abstraction, checkpoint resume and population manifests. Current checkpoint schema 1 records engine, abstraction, seed, hand count, table and measured runtime; it cannot resume an identical training trajectory.
3. Add range-conditioned multiway belief sampling and limited-depth search with multiple continuation policies. Never determinize using actual opposing cards or future cards. Validate search against blueprint with independent seeds and paired comparisons.
4. Wire per-player model updates and shrinkage into conservative adaptation; evaluate both modes with identical strength objectives. Currently neither mode adapts.
5. Integrate licensed strong held-out agents and genuinely disjoint training/evaluation populations. Current evaluation scripts overlap training styles; held-out seeds alone do not establish population generalization. Test heterogeneous tables and multiple candidate copies, persistence across sessions, longer latency distributions and training cost scaling.
6. Run preregistered large benchmarks with enough independent deal blocks to resolve a chosen BB/100 effect. Exact competitive compute requirements cannot be known before profiling the implemented CFR/search backend and learning curves. No amount of training the present smoke implementation is guaranteed to produce competitive play.

Rake and tournaments are unsupported. Unequal stacks are supported by the game adapter but not represented adequately by the learner. Multiway activity is bucketed, but strategic interactions and opponent ranges are not yet modeled. There is no heads-up exploitability claim or human-strength evidence.
