# High-power payoff confirmation of v10

Frozen before any observation. Approved by the user on 2026-11-15.

The v10 pilot (`runs/bot/coverage-reuse-training-001`) passed operations, coverage,
uniform and regression gates but failed payoff against its v8 parents at 2000
blocks: point estimates +43 to +96 bb/100, all four lower 98.75% bounds below
zero. A read-only power analysis of those vectors (`power-analysis.json`) showed
2000 blocks resolve only about 130 bb/100 at 80% power. That failure stands.

This is ONE new evaluation of the same immutable checkpoints, with no training:

- Candidates A10906 / B11150 (v10), parents A8906 / B9150 (v8), uniform control.
- Opponents: frozen validator-750 and the other-seed-2000 held-out checkpoint.
- Fresh seed 15000000, 20000 blocks, one copy. Detectable effect about 44 bb/100.
- Two 3 GiB evaluators at a time, host reserve 2 GiB, disk reserve 10 GiB, one
  finite controller deadline. Interrupted runs may resume completed outputs only.

Gates, all required for both learners on both opponents:

- Payoff vs immediate v8 parent: paired lower 98.75% bound > 0 (all 4).
- Payoff vs uniform: paired lower 95% bound > 0 (all 4).
- No material regression: no population's upper 98.75% bound below -10.

No retry on another seed, no larger sample afterwards, no gate change. A pass
means v10 improves on v8 against these weak local opponents; together with the
pilot's operational and coverage results it makes v10 the base for further
training. It is not evidence of competitive strength. A fail closes v10's
payoff question and v8 stays the base.
