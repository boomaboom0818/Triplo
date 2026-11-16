# Loose-caller continuation pilot

This tests whether adding multiway experience helps the lean-002 policy. It is
separate from the stopped run and leaves its original state and snapshots intact.

## Design fixed before execution

- Parent: `runs/bot/lean-002/state`, the last durable training state, rather than
  the later policy-only snapshot. Hashes are in the experiment manifest.
- Two continuation RNG seeds, 71 and 72; both share the same trained parent.
- For each seed: 10,000 additional iterations of original self-play as control,
  and 10,000 iterations with population probability 0.5 as treatment.
- Population deals: for each of the six traversers, choose three other seats as
  fixed loose callers; retain two other learned seats. Profiles independently
  vary preflop price tolerance and postflop strength/price thresholds. Other deals
  remain original self-play. All training uses the lean action menu.
- Four workers, batches of 100, at most 180 seconds per arm checked between rounds,
  4 GiB address-space limit per process, 2 GiB host and 10 GiB disk reserves.
- Evaluate the parent and all four arms on the same fresh 500 deal seeds starting
  at 18,310,000, each rotated through six seats against all four unchanged rule
  styles. Opponents retain the full action menu. Compare both original inference
  and the preselected support cutoff 1e-4; also evaluate tight fallback alone.
- Paired intervals use whole deal blocks. They are pointwise normal 95% intervals,
  without multiple-testing correction; this is a small screen, not qualification.

Training opponents are implemented independently in `poker/loose_opponents.py`;
they do not import gate policies. Their fixed behavior is used directly for
sampled opponent actions and hence for traverser utility estimates. Their actions
are never accumulated into learned regrets or policy averages. On population
passes only the remaining learner opponents contribute average rows. This leaves
the existing sampled-node averaging estimator unchanged, while changing its
visitation distribution. Average mass is not independent sample size.

This is a heuristic multiplayer mixture objective with imperfect recall, not a
claim of unbiased full CFR averaging or equilibrium convergence. OpenSpiel also
[distinguishes simple sampled-node from full averaging](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/algorithms/external_sampling_mccfr.h)
for multiplayer use. This experiment tests practical payoff and coverage effects;
it does not resolve the existing solver's general averaging limitations.

Mixture checkpoints use algorithm ID `lean-loose-mixture-es-mccfr-v1`, carry the
population probability/version/count, and resume through `ParallelLean.load_state`.
The single-process legacy resume path rejects them. Zero-population workers retain
identical RNG consumption and updates to the original implementation.

## Reproduction

```sh
.venv/bin/python -m scripts.lean_mixture_pilot \
  --parent runs/bot/lean-002/state --root runs/bot/lean-mixture-001
```

Choose a new root for another run; existing experiment output is never replaced.
Manifest source hashes are checked again after evaluation, and parent file hashes
must remain identical. Each arm retains a resumable state, exported policy, and
training metrics. Reports retain per-deal payoff samples and paired differences
against the matched control, parent and fallback-only policy.

A positive result should improve multiway support and outperform matched controls
without clear losses elsewhere. A short, inconclusive result does not justify
promoting a checkpoint. OpenPoker can expose missing live situations later; an
uncontrolled live session is not a substitute for these matched evaluations.

## Completed screen: lean-mixture-001

All four arms completed, adding 40,000 training iterations total. Controls took
81.3/75.3 seconds and mixture arms 97.4/98.6 seconds, within the per-arm bound.
The mixture arms used 5,072/4,980 population deals. Both ended at 1,952,060 total
iterations. Parent hashes were unchanged; all nine source hashes matched at
completion. The exact run sources are archived under the run's `source/` folder.
After completion, the runner received input-validation and automatic source-copy
improvements only; use the archived runner to reproduce its exact original hash.

Mean added average mass per linear iteration weight in four-plus-player states
increased about 9x on the flop, 55x on the turn and 425x on the river relative to
controls. This verifies increased training exposure, not independent sample counts
or policy strength. The counts of distinct table rows rose to 781,820/781,646 for
the mixture arms, versus 773,231/773,268 for controls.

Station results (bb/100), with paired differences against the matching control:

| Seed / inference | Self-play control | Mixture | Difference | Pointwise paired 95% interval |
| --- | ---: | ---: | ---: | --- |
| 71 / raw | -55.5 | -4.7 | +50.8 | +9.8 to +91.7 |
| 72 / raw | -59.3 | -0.2 | +59.1 | +0.4 to +117.9 |
| 71 / supported | +7.5 | +52.5 | +45.0 | -31.5 to +121.5 |
| 72 / supported | +15.1 | -29.1 | -44.2 | -105.4 to +17.0 |

The raw policy shows an encouraging direction on both continuation seeds, but
absolute station win-rate intervals still include zero. Both continuations share
a parent and evaluation deals, so this is not two independent replications. The
seed-72 paired lower bound is barely positive and is not corrected for the many
comparisons. Other-style differences against the controls include zero. This does
not prove absence of regressions.

The support-cutoff policy is inconclusive and inconsistent across seeds. Tight
fallback alone scores +187.3 bb/100 against station and exceeds every mixture
variant with a paired interval excluding zero on this screen. No checkpoint is
qualified or promoted; no OpenPoker connection was made. The result supports a
longer predeclared mixture-vs-self-play experiment, evaluated on new deals, while
keeping fallback-only as a required control. It does not support simply deploying
the best-looking point estimate or retuning the cutoff on these evaluation deals.

Full reports: `runs/bot/lean-mixture-001/summary.json`, the per-policy evaluation
JSON files, `manifest.json`, and `execution.log`. Every evaluation includes all
four styles and all deal-block samples. Tests cover fixed-policy utility and
non-contamination, all-seat updates, deterministic resume, metadata compatibility,
and the complete pilot pipeline with immutable parent verification.
