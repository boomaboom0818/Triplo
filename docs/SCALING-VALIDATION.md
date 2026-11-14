# Prospective local scaling validation 001

This experiment separately qualifies (1) operational capacity and (2) learning
value. Passing operations does not certify the recipe's strength. Results can
be pass, fail or incomplete; no guaranteed improvement or automatic promotion.

## Frozen design

- Parents: both historical CONTROL2100 checkpoints from population-pilot-001.
  Diversity was not promoted and is excluded. Preserve immutable paths/hashes,
  seeds11/29, abstractionv2, call-mixture sampler, archive2 and table/node caps.
- Two independent training processes concurrently, separate outputs/writer locks.
  User explicitly authorized more cores/RAM:4 GiB address-space cap per trainer,
  at least2 GiB host reserve; no more than2 trainers. Start only with10 GiB
  available for both. Evaluation keeps its existing3 GiB per-child protection.
- Stages2350,2600,2850,3150,3400,3650,3950,4200. Final4200 is twice parent2100.
  Every stage resumes completed learning, saves50-iteration checkpoints, retains
  prior snapshots and has a one-hour trainer allowance. No unchanged resource
  retries, silent learning reset or algorithm change.
- Overall finite deadline2026-11-15T00:00UTC. On timeout/interrupt, signal trainers
  for graceful checkpointing, then bound shutdown waits. No global indefinite job.
- Record lifetime VmPeak/peakRSS/CPU/wall time for each training invocation and
 10-second host/process samples. Preserve attempt records and checkpoint lineage.

## Operational qualification

Both seeds must complete4200 with validated config/parent lineage and no unresolved
worker failures. Both workers must be observed concurrently, with evidence of
at least1.5 CPU-cores worth of combined utilization on a two-worker batch. Every
recorded invocation must be successful, peakRSS and VmPeak below80%of its4 GiB cap,
minimum sampled host available memory >=2 GiB, and final table occupancy below
80%of its inherited budget. Missing measurements mean incomplete qualification.
Peak measurements are per worker; sampled host statistics are not continuous
host-memory maxima. Report stage throughput, memory/table/checkpoint growth and
remaining uncertainty. Parent snapshots and their original caps are never changed.

## Learning qualification

Every stage receives200 fixed screening blocks on development seed8,100,000,
paired against parent2100 and frozen independent validator750. Screening can
identify failures/regressions but is not final confirmation or a source of model
promotion. Every completed stage is evaluated before the next stage starts.

After4200, use fresh development seed8,200,000,2,000 blocks, one copy. Compare
candidate versus unchanged2100 parent against (a) frozen validator750 and (b)
OTHER seed's frozen2000 policy. Neither was used as a training opponent. Freeze
both opponent hashes in the manifest. Also evaluate uniform controls separately.

Primary gain requirement: all four seed/opponent paired changes have a positive
lower98.75% bootstrap interval (nominal Bonferroni adjustment across4primary
comparisons). Bootstrap coverage is approximate, not an exact confidence guarantee.
Require positive nominal95% candidate-versus-uniform learned-opponent intervals
and >=5%positive-weight postflop coverage with >=200postflop decisions on those
learned populations. Report every scripted population, including tight-regression
intervals; uncertainty is not proof of non-regression. Qualification remains
subject to review of material regressions, compute use and assumptions.

This is a NEW prospective scaling protocol. It does not retroactively pass or
rewrite any earlier failed gate. Independent learned opponents are not established
strong. No competitive-strength claim, live promotion or final-test certification
follows; reserved final-test range6,100,000 and OpenPoker/live settings stay untouched.

## Run and recover

```sh
.venv/bin/python -m scripts.scaling_validation
```

After diagnosing a stop and checking no supervisor/writer remains:

```sh
.venv/bin/python -m scripts.scaling_validation --resume
```

Manifest and per-stage parent hashes must match. Complete stages/reports are
validated/skipped, known interruption/time stops resume complete saved state;
node/table/unknown failures require investigation. Measurements are retained
across attempts, and any failed invocation is visible in qualification. No altered
policy/config can be substituted under the same protocol. Two-worker launch or
runtime failure stops the pair and preserves latest complete learning.

Output: runs/bot/scaling-validation-001. Final status reports operational_pass and
learning_pass separately; completed_pending_review still requires evidence review.
Only operational scaling inside this measured envelope can be concluded. Larger
worker counts or model sizes require another measured resource check.

The manifest freezes the criteria document and algorithm source hashes as well
as model/opponent inputs. Each stage rechecks them. An algorithm or criteria
change requires explicit new protocol handling, not a silent resumed qualification.
