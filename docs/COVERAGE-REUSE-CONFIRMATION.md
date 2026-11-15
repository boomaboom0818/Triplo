# Coverage-only reuse v10: amended engineering confirmation

Status: prospective protocol, approved by the user on 2026-11-15 before any
observation under it. Source proposal: `runs/bot/coverage-reuse-probe-001/PROTOCOL-PROPOSAL.md`.

The first v10 probe (`runs/bot/coverage-reuse-probe-001`) remains FAILED under its
frozen 3x raw-update gate (2.3147x / 2.4309x). Nothing here reinterprets it.

## What changed

Only the coverage selection metric:

- `new_unique_key_rate`: unique positive-average keys that are absent or zero in
  the immutable v8 parent, active 2/3 postflop, per traversal node, must be at
  least 3x the v8 baseline.
- `total_update_rate_not_regressed`: active 2/3 postflop positive updates per
  node must be at least 1x the baseline.

Unchanged: exact regret/RNG/nodes, exact covered updates, variance*nodes <= 1.5,
variance*time <= 2, runtime <= 2, each of the 18 role/street cells >= 20 new keys,
normalized bound <= 1, finite actual weights, 2 s / 10000 nodes per role batch.

## Matrix

Immutable parents A8906 / B9150 from `future-linear-training-001`. Fresh deals
12600000..12600007, populations historical and guarded, 32 replicates, all 6
roles, 8 regret + 8 average each. RNG seed
`12700000 + candidate*100000 + population_index*10000 + deal_index*100 + replicate`.

Two read-only 3 GiB workers, one finite deadline of at most 2 h, host reserve
2 GiB, disk reserve 10 GiB. Both learners must pass the whole matrix. No
alternate-seed retry; a failure is preserved as a failure.

    .venv/bin/python -m scripts.coverage_reuse_confirm --candidate N --deadline ISO

This is engineering evidence only: no training, payoff or strength claim.
A pass permits strict v8->v10 integration work, not registration by itself.

## Later readiness pilot (also approved, registered separately)

Tight-coverage improvement is referenced to immutable v7 A6906 / B7150 (>= 2
percentage points, >= 5% positive coverage on >= 200 decisions) with no
tight-coverage decline against the immediate v8 parents. Payoff gates stay on
the immediate v8 parents A8906 / B9150: all 4 learned-opponent lower 98.75% > 0,
all 4 uniform lower 95% > 0, no material population regression.
