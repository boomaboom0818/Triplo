# Natural-trajectory coverage diagnosis after v8

The frozen v8 pilot is closed and NOT READY. Independent reconstruction of all
12 final reports passes operations, all four primary parent payoff gains, all
four primary uniform gains, required absolute postflop coverage, and the material
population-regression checks. A's tight coverage improves only 1.71556 percentage
points against its v7 parent, below the frozen 2-point requirement. B improves
2.45617 points and passes. This failure is preserved, not waived or rerun.

A's 59 uncovered postflop decisions are missing average rows, not existing rows
with zero weight: flop15, turn20, river24. All are heads-up. B has52 missing rows.
The tight scripted population is also part of the broader training-style family;
coverage there is not heldout generalization. B still loses modestly to uniform
in tight-player payoff. No competitive-strength claim is justified.

Before another training intervention, distinguish absent experience from
zero CURRENT own reach, tiny positive reach, existing regret-only rows, and
changes in the mean policy's visited situations. A positive parent average row
cannot be lost by the preserved-history update; differing trajectory denominators
must not be interpreted as erasure of learning.

Freeze one read-only diagnostic, `coverage-gap-audit-001`, BEFORE its observations:

- Immutable paired v7 A6906/B7150 and v8 A8906/B9150; no heldout750/OTHER2000
  inputs. No model writes, table deltas, policy fallback, or learning.
- Both natural average-policy trajectories, all six seat rotations, fixed fresh
  deal seed11700000. 2000 tight blocks and128 calling/mixed blocks each. Report
  every owner/street/active2..6 cell, including zero-visit cells. No forced contexts.
- Query both checkpoints' mean/regret rows on EACH same observation. Classify
  missing/zero/positive means, regret presence, and prior-action CURRENT own reach.
  Keep exact zero support separate from tiny positive reach using logarithms.
- Check instrumented/uninstrumented action traces and final RNG state for the
  first two blocks of every population, variant and role. Do not export payoff
  estimates or private cards. This diagnoses coverage, not fresh efficacy.
- Record counts and hashed gap-key multiplicities, immutable input/source/protocol
  hashes, progress and actual lifetime memory peaks. No cherry-picked deal search.
- Two read-only workers, each hard/soft3GiB; reserve2GiB RAM/10GiB disk. Invocation
  deadline2026-11-15T12:15UTC; no overwrite or unchanged resource-failure retry.

Independently review all role/cell totals, missing-key multiplicities, replay
counts, zero/tiny reach partitions and input preservation before selecting ONE
cause-supported change. If gaps mainly have zero current own reach, distinguish
the defined averaging target from sampling failure; filling them arbitrarily is
not a valid estimator repair. If gaps lack regret rows too, more averaging weight
alone cannot create missing experience. A correction needs an explicit target,
preservation/resume tests and prospective fresh engineering/payoff/coverage gates.

Before any future training protocol, audit coverage-ceiling feasibility and
explicitly identify training parents versus qualification references. The current
v8 result's roughly98% tight coverage makes another automatic additive2-point
requirement against each new parent near or above100%; this is NOT permission
to silently replace references, weaken gates, or shop fresh baseline samples.
No next training recipe, gate amendment or extension is selected by this audit.
All prior failed protocols and the current unsuccessful overall qualification
remain immutable. User renewed autonomous concrete work until actual readiness.
