# Instrumented extension through 2,000 iterations

Both learners reached 2,000 iterations with target_reached. The manifest-based
review verified both 1,750 and 2,000 checkpoints, immutable parent/configuration
lineage, all stage reports, block counts/means and recomputed paired statistics.
All averaging-weight moments were finite. No memory/node/table stop, recovery,
checkpoint reset or algorithm change occurred in this extension.

| Learner | Union information sets | Cumulative training seconds | Paired learned-opponent difference vs1,500 parent, BB/100 [95% interval] |
|---|---:|---:|---|
| A | 228,969 | 10,493.4 | +53.33 [-38.56,148.33] |
| B | 223,760 | 10,172.4 | +12.93 [-71.87,103.68] |

Development seed5,700,000,1,000 independent deal blocks, one candidate copy. Both
intervals include zero, so this increment did not establish further improvement.
Both have positive nominal paired differences vs uniform play on this development
seed, which is a weak baseline. Tight-script differences vsparent are inconclusive
and the gap vsuniform persists. Different development seeds and opponent policies
must not be pooled into a competitive-strength claim.

The largest recorded update weights since iteration1,500 are approximately8.12e22
for A and1.30e17 for B. These aggregate observations are not independent sample
sizes. The separate weight investigation found no basis for clipping/resetting
learning; see AVERAGING-WEIGHT-INVESTIGATION.md. No estimator modification was made.

## Preserved candidates and operational state

- A: `runs/bot/sustained-002/candidate-0-2000.json`.
- B: `runs/bot/sustained-002/candidate-1-2000.json`.
- Exact hashes/lineage: manifest, per-stage `*-parent.json` and verified-review.json
  in that directory. No candidate has been promoted to live OpenPoker.
- Both are the most advanced checkpoints of this recipe, not established winners.
  Independent-development evidence and separate crossplay remain mixed.

The bounded learner extension is complete. A separate independent validator
continuation is running under its original protections before additional frozen
screening of these candidates. This does not replace the original validation gate.
The two-seed promotion/readiness requirement remains unmet; no further blind
learner extension is scheduled solely because previous jobs completed.

Final-test seed6,100,000 remains unused. Live OpenPoker results are not included;
no local live client or telemetry was observed. Software supports recoverable,
evaluated, deadline-bounded training; robust playing strength still needs evidence.
