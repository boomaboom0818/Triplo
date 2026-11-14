# Controlled population-diversity pilot

The user explicitly authorized autonomous fixes/experiments and continuation past
the morning cutoff if needed. The frozen2,000 learners showed incomplete robust
incremental payoff evidence. Their historical/self-play population is narrow.
This is a hypothesis-driven exploratory pilot, not proof that curriculum diversity
will fix the model and not an evidence-qualified sustained campaign.

For both independent seeds, preserve the2,000 checkpoint and run matched continuations
to2,100: unchanged historical control and explicitly forked diverse-v1 population.
The fork retains regrets, averages, archives, RNG and counters; source files are
never overwritten. A new output and --fork-population are required, and only the
historical-v1→diverse-v1 population change is allowed. Any seed/abstraction/sampler
change is refused. Fork metadata records parent hash, config and iteration.

Diverse-v1 overrides25% of per-seat historical selections, equally choosing:

- passive-v1: call weight.75, fold.15, total ordered raise slots.10;
- pressure-v1: call.35, fold.10, total ordered raise slots.55.

Available actions are renormalized; profiles use only the legal menu and no hidden
cards/state. Other seats retain the existing historical/current-policy recipe.
Regret traversal samples those target behaviors directly. Average traversal and
its corrected sampler remain unchanged. This retains the existing nonstationary
population heuristic, not a multiplayer equilibrium guarantee. Stochastic profiles
are related to scripted evaluation styles, so those diagnostics must not be
misrepresented as independent held-out evidence. Validation checkpoints are never
loaded as training opponents.

Each arm receives100 additional iterations with preserved table/node/archive and
3 GiB limits and a one-hour invocation allowance. Report actual nodes/seconds too:
equal iterations may require different compute. Stop on budget/unknown failures
and investigate; do not retry unchanged resource failures or silently reset.
Pilot deadline20:31 UTC (12:31Pacific) is an explicit finite job allowance under
the extended authorization, not a new mandatory global shutdown.

Evaluate control, diversity fork, original2,000 parent and uniform on shared
new development seed7,000,000 with1,000 independent deal blocks against the frozen
750 validator plus all scripted populations. Primary diagnostic is paired
diverse-minus-control learned-opponent payoff; retain population-specific coverage,
paired regression intervals, training style-visit counts and compute/memory costs.
No automatic promotion; both-seed consistency and robustness must be reviewed.
The original gate and final-test range6,100,000 remain unchanged. The validator
has no established competitive strength.

Run: `.venv/bin/python -m scripts.population_pilot`.
Outputs: `runs/bot/population-pilot-001`; originals stay in sustained-002.
The pilot refuses an existing manifest unless --resume is explicit. After checking
actual processes and diagnosing the failure, run:

```sh
.venv/bin/python -m scripts.population_pilot --resume
```

Resume verifies the immutable parent/validator/design manifest, completed or
partial checkpoint recipe, iteration bounds, RNG and explicit fork parent hash.
It skips completed arms/reports and resumes partial arms without applying the
population fork again. Node/table/unknown failures and runtime stops with no
iteration progress require investigation. Global/per-report writer locks and
validated caches prevent duplicate output writers. The finite job deadline stays
20:31UTC; no unchanged failure-retry loop or learning reset is introduced.
All live OpenPoker configuration remains untouched.
