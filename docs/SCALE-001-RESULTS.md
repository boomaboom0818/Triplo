# Scale-001: longer training for base-guard

**Result: continuing the base-guard model's training from 969,520 to 2,001,520
iterations improved it by about +6.7 to +10 bb/100 on two separate evaluation seeds.
The 2,001,520-iteration model replaced `runs/bot/pressure-fresh-001/base-guard.json`
on 2026-11-22 and runs live with `equity-guarded-v4`.**

Earlier short continuations (+15,000 to +39,000 iterations) were flat or slightly
negative (`docs/LIVE-CONDITION-TRAINING.md`, `pressure-fresh-001`). This run added
about 1.9M iterations, roughly 3x the original training.

## Rebuilding the weights

Model weights are not committed: everything under `runs/` is gitignored, and the
model alone is 843 MB. Rebuild it from the code with the steps below. The parent
state `runs/bot/aggressive-stages-001/training/state` comes from the earlier
training chain described in `docs/LIVE-CONDITION-TRAINING.md`.

## Training

Same recipe as the parent, forked from `runs/bot/aggressive-stages-001/training/state`
(969,520 iterations; the source state is never written):

```
python -m scripts.history_train --root runs/bot/scale-001/trained --minutes 300 --seed 2027 \
  --stack-bb 100 --workers 6 --population .8 --curriculum live-aggressive-v2 \
  --abstraction history-compact-v1 --max-rows 18000000 --save-minutes 15 \
  --snapshots 1100000 1250000 1500000 1750000 2000000 \
  --initialize-from runs/bot/aggressive-stages-001/training/state
```

About 106 iterations/second on 6 workers. The run was killed once by an unrelated
session teardown right after a complete save at 1,176,520 iterations and resumed
with `--resume`; the resumed run reproduced the killed run's progress exactly.
It stopped at its deadline at 2,886,520 iterations and 16.6M table rows.

Each snapshot was turned into a guarded checkpoint exactly as the previous
base-guard was; this produces the deployed model (sha256 `d553036c…`):

```
python -m scripts.export_pressure_guard --checkpoint runs/bot/scale-001/trained/snapshot-2001520.json \
  --out runs/bot/pressure-fresh-001/base-guard.json --version equity-pressure-v1
```

## Evaluation

`scripts.live_gate evaluate`, 2,400 deals in each of the nine standard setups,
hero fallback `equity-guarded-v4`, incumbent opponent
`runs/bot/history-sustained-001/checkpoint.json`. Every arm plays identical deals,
so differences are paired per block and averaged with equal weight over the nine
setups. Intervals are 95%.

```
python -m scripts.live_gate evaluate --checkpoint <guard.json> \
  --incumbent runs/bot/history-sustained-001/checkpoint.json \
  --out <eval.json> --seed 87419000 --deals 2400 --hero-fallback equity-guarded-v4
python -m scripts.paired_compare --baseline <eval-old-guard.json> <eval-new-guard.json>
```

### Development seed 93,531,000

| Iterations | Diff vs base-guard, bb/100 | 95% CI | Postflop decisions from the model |
|---:|---:|---|---:|
| 969,520 (old base-guard) | — | — | 47.0% |
| 1,101,520 | +2.36 | [-0.45, +5.17] | 48.4% |
| 1,251,520 | +2.84 | [-0.74, +6.42] | 49.8% |
| 1,500,520 | +5.41 | [+0.77, +10.05] | 53.0% |
| 1,752,520 | +6.87 | [+1.67, +12.07] | 55.4% |
| 2,001,520 | +10.06 | [+4.45, +15.66] | 57.1% |
| 2,886,520 | +10.30 | [+3.76, +16.85] | 61.7% |

### Confirmation seed 87,419,000 (pass rule written before results)

Pass rule: 95% lower bound above zero; with two candidates, also report the
Bonferroni 97.5% lower bound (z = 2.24).

| Iterations | Diff vs base-guard, bb/100 | 95% CI | 97.5% lower bound | 6-max 100 BB aggressive | Verdict |
|---:|---:|---|---:|---:|---|
| 2,001,520 | +6.67 | [+1.08, +12.25] | +0.29 | +11.83 | **Pass** |
| 2,886,520 | +6.53 | [-0.15, +13.20] | -1.11 | +24.83 | Narrow miss |

The 2,001,520 model passed on both seeds and was promoted. The confirmation was
rerun with the committed `equity-guarded-v4` code (release `6a20179`) and every
block matched the original run exactly. Training past about
2M iterations added no measurable value. No setup showed a significant loss.
p99 decision latency stayed about 7 ms.

## Limits

These are simulated opponents (rule bots and an older checkpoint), not real
OpenPoker players; the live gain is probably smaller. The training trend had not
flattened in decision coverage, but payoff did not improve between 2.0M and 2.9M.

## Deployment and rollback

The new model has sha256 `d553036c8e2b5a4d7e590d15195486c6b2387fdfedb6ead19a8469261a9d7674`.
The previous model (sha256 `f3ed3b73…`) is kept at
`runs/bot/pressure-fresh-001/base-guard-969520.json`; to roll back, copy it over
`base-guard.json` and restart the client.
