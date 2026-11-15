# Coverage-only reuse v10: bounded readiness pilot

Frozen before launch. Protocol approved by the user on 2026-11-15
(`runs/bot/coverage-reuse-probe-001/PROTOCOL-PROPOSAL.md`). No gate changes after
results. A failed gate is preserved as a failure.

## Preconditions

Both learners passed the amended engineering confirmation and its independent
review (`runs/bot/coverage-reuse-confirm-001`); `scripts.coverage_reuse_training`
refuses to start otherwise or if the qualified math sources or parents changed.

## Training

Explicit strict fork of immutable v8 parents A8906 / B9150
(`future-linear-training-001/candidate-{0,1}-plus-2000.json`) to
`coverage-only-cross-role-average-mccfr-v10`. All regrets, means, RNG, archives,
fork chain, the v8 future-averaging origin and the absolute weighting clock are
preserved; only a new diagnostic epoch starts. Same regret objective and same
future-linear averaging target; 8 regret + 8 average samples, no extra trees.

Stages +50 / +200 / +500 / +1000 / +2000. Only the final stage can qualify.
Two 4 GiB trainers, 3 GiB evaluators, host reserve 2 GiB, disk reserve 10 GiB,
child training budget 7200 s, existing node/table/archive/checkpoint
protections, one finite controller deadline chosen before launch.

## Evidence

Screen after every stage: seed 13100000, 200 blocks, stops only for clear harm.

Final: seed 13200000, 2000 blocks, on the frozen validator-750 and the
other-seed-2000 held-out opponent, for candidate, its own unchanged v8 parent,
the uniform control, and the fixed v7 coverage reference A6906 / B7150
(`joint-average-training-001/candidate-{0,1}-plus-2000.json`) on the same deals.

Gates, all required for both learners on both populations:

- Payoff vs immediate v8 parent: paired lower 98.75% bound > 0 (all 4).
- Payoff vs uniform: paired lower 95% bound > 0 (all 4).
- No material population regression (upper 98.75% bound < -10 on any population).
- Positive-average postflop coverage >= 5% on >= 200 decisions for calling,
  mixed and the learned opponent.
- Tight coverage >= 5% on >= 200 decisions, at least +2 percentage points over
  the fixed v7 reference, and no decline against the immediate v8 parent.
- Operational: resource, throughput and table-budget checks of the driver.

The scripted tight population overlaps the training style, and the fixed-reference
coverage result is a repeated scripted measurement, not independent opponent
evidence. These validators are weak; passing does not establish competitive
strength. No cloud, no live OpenPoker, reserved seed 6100000 untouched.
