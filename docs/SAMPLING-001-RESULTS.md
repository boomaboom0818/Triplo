# Sampling 001 results

Both fresh call-mixture-v1 learners reached 250 iterations with target_reached.
Checkpoint provenance, all six evaluation reports and all four paired-comparison
outputs were verified. Evaluations used 1,000 shared deal blocks, seed 5,200,000,
one candidate copy and the unchanged independent 250-iteration validation agent.
Original v1/v2 checkpoints remain intact. No sustained training was launched.

## Predeclared decision

Both learners passed the coverage screen. Neither passed the payoff screen.
The original combined gate therefore fails; do not call this a promotion or a
confirmed improvement in poker strength.

| Learner | Calling postflop coverage | Mixed | Learned |
|---|---:|---:|---:|
| A | 1,030/9,336 (11.03%) | 383/870 (44.02%) | 69/247 (27.94%) |
| B | 855/9,379 (9.12%) | 357/864 (41.32%) | 69/250 (27.60%) |

All three screened populations meet 5% positive-weight coverage and at least 200
postflop decisions. Counts reuse deals across rotations and are not independent
samples. Tight-agent coverage remains zero for both learners; aggressive-agent
postflop samples are only eight and seven decisions. Those results should not be
hidden by pooling populations.

Paired learned-opponent payoff differences, in BB/100:

| Learner | Versus uniform, estimate [95% interval] | Versus original v2 at 250, estimate [95% interval] |
|---|---|---|
| A | +28.38 [-60.80, 116.61] | +41.38 [-65.12, 155.32] |
| B | +5.65 [-86.89, 95.67] | -52.29 [-168.19, 55.52] |

Every primary interval includes zero. The pilot does not establish superiority
or equivalence. Tight-script differences versus uniform are -3.83 [-4.55, -3.15]
and -3.13 [-3.85, -2.40]. These are nominal per-population intervals, not a
multiple-comparison-adjusted competitive claim. Diagnose this scripted weakness
before judging robustness from learned-opponent coverage alone.

## Resource evidence and next decision

A took 1,542.8 recorded training seconds and used 53,758 union information sets;
B took 1,393.8 seconds and used 51,656 sets. At the observed average rate,
an additional 750 iterations would take roughly 70–77 minutes per learner;
that is only an estimate, and exceeds a one-hour invocation budget. A future
extension needs intermediate checkpoints/targets, rather than retrying an
unchanged timeout. Memory protections must stay in place.

The coverage bottleneck now has a promising experimental remedy. The unresolved
question is whether that coverage becomes useful strategy with further learning.
More evaluation alone cannot turn the current point estimates into evidence of
improvement. A bounded 250→500→1,000 exploratory learning curve is a reasonable
next experiment, with the same paired controls, fresh predeclared development
deals, and focused tight-agent diagnostics. It would explicitly revise the
screening sequence: the 250-iteration payoff gate did not pass, so this must not
be described as the previously qualified confirmation stage. Sustained training
still requires credible payoff and resource evidence after that experiment.

Preserve all pilot outputs and use separate stage paths for any continuation.
Final test seed 6,100,000 remains untouched. No established strong opponent has
been integrated; no competitive-strength claim is warranted.
