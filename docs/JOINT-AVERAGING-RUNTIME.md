# Bounded v7 training orchestration

This runtime appendix preserves the already-frozen estimator, parent set,
50/200/500/1000/2000 added-iteration stages,10200000screens/10300000confirmation
and ALL learning/coverage gates in JOINT-AVERAGING-PILOT.md. It does not qualify
the model or replace the two-learner engineering prerequisite.

The stage child runtime is explicitly7200seconds. A1000iteration stage at
4+seconds per iteration can exceed the older3600second setting; declaring a
two-hour cap before launch avoids an artificial mid-stage stop. Every controller
invocation separately requires a finite timezone-aware future deadline<=12hours.
The outer deadline, reserve monitor and stop/cleanup remain authoritative.

Two training children retain4GiB hard address-space limits each; evaluations3GiB.
Host reserve2GiB,disk10GiB,iterationnode500000/table1250000,archive2/checkpoint50
remain unchanged. Resource failures are investigated, never retried unchanged.
A known wall-budget/interruption can resume only from verified complete state,
matching config/source/input/lineage and no duplicate writer, under the same
outer finite invocation deadline. No learning reset, clipping or failed-gate erasure.

The launch wrapper refuses unless BOTH independently reviewed engineering gates
pass, the qualified mathematical source hashes match and BOTH parent bytes match.
New registration must pass strict metadata/fork/preservation/rollback/resume and
evaluation tests before any real training launch. Freeze this appendix and the
chosen outer deadline in the training manifest. No automatic extension seeking
favorable statistics, cloud spending, live OpenPoker or reserved6100000 use.
