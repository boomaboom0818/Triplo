# Additional screening after 2,000 iterations

The independently trained validator completed750 iterations within its original
250,000-set cap, at170,052 union information sets. The original250 checkpoint/hash
and all previous gates remain unchanged. All six candidate/parent/uniform reports
were checked for hashes, modes, seed5,900,000,1,000 samples and sample means.

| Learner | Paired difference vs1,500 parent, BB/100 [95% interval] | Difference vs uniform |
|---|---|---|
| A | +47.78 [-38.06,136.75] | +132.71 [-52.05,316.73] |
| B | +91.30 [0.825,181.89] | +230.66 [51.45,419.30] |

B has positive nominal intervals on this additional population; A remains
inconclusive. The tight-script gap persists. This does not retroactively pass
the original failed gate, establish consistent robustness or certify strength.
The new validator is independent of candidates but not established strong.
Repeated development screening and nominal intervals must remain explicit.

## Last development crossplay check

The frozen2,000 learners will face the other seed's frozen1,500 policy, with their
own1,500 parent and uniform control on the same deals. This uses seed6,000,000,
2,000 independent deal blocks, one candidate versus five opponent copies. It
is separate from the untouched final-test range6,100,000. The opponent and parent
lineages/configurations/hashes are checked; no learner training or live action
occurs. Absolute A/B returns cannot rank them because their opposing policies
differ. The primary diagnostic is paired within-opponent change versus parent
and uniform with95% block intervals; report every population, not just favorable
results. Even positive results remain development screening, not an automatic
promotion or established strength.

Exact bounded evaluation command:

```sh
.venv/bin/python -m scripts.crossplay_check \
  --output runs/bot/crossplay-002 --curve runs/bot/sustained-002 \
  --stage 2000 --seed 6000000 --blocks 2000 \
  --deadline 2026-11-14T17:31:00+00:00
```

The shared evaluator enforces3 GiB child limits, report locks, atomic validation
and the overnight deadline. The command refuses an existing manifest and does
not retry failures blindly. Completion must be reviewed and included in the final
morning handoff; original benchmarks and reserved final-test deals remain intact.
