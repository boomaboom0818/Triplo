# Prospective candidate inference component001

The guarded curriculum completed operations but failed learning/average-coverage
qualification. All previous failed gates and original learning stay preserved.
On identical candidate-observed states its audit found many newly positive regret
rows and no newly positive averages in tight postflop play. The deployed local
evaluation policy reads averages only, then uses uniform at missing/zero rows.
This motivates ONE fixed inference experiment, not more unchanged training.

Candidate variant `average-positive-regret-backoff-v1`: use the unchanged average
strategy wherever its row has positive weight; otherwise use normalized positive
regrets if available; otherwise uniform. No blending, tuning, table writes,
bootstrap, clipping or relabeling. This is NOT pure average CFR. Both unchanged
curriculum checkpoints A4906/B5150 are used, with all learning intact. The
held-out validator750/OTHERseed2000 always use their original average policies.
Candidate controls are the SAME checkpoint with original average inference and
uniform. Controls have exact payoff replay tests against the original evaluator.
Thus this measures an inference-component effect, NOT additional learning gains.

Two candidate seeds x two frozen opponents x three variants. Fresh development
screen9700000,200whole-deal blocks, six seat rotations, all six populations. Any
nominal95% backoff-minus-average interval upper<-10bb/100 stops this proposal for
investigation, preserving every report. Otherwise run fresh2000block9800000
confirmation for ALL combinations. No seed/variant search or blind extension.
All4learned-opponent paired lower98.75% backoff-minus-average bounds must exceed0;
all4backoff-minus-uniform lower95% bounds must exceed0. Any population98.75%
upper<-10 prevents component qualification. These approximate intervals and weak
agents do not establish competitive strength. Training-style tight overlap is
not held-out generalization; the empty-table blueprint is uniform logits.

Always report raw positive-AVERAGE coverage separately from regret-backoff and
actual uniform decision fractions, by street/active count. The original raw
average coverage gates, including tight>=5%,>=200decisions,+2pp over v3parent
for BOTH seeds, REMAIN FAILED; regret fallback cannot retroactively pass them.
Even a component PASS alone does not qualify the current model for sustained
training or justify a live OpenPoker switch. No live changes are authorized here.

At most two3GiB hard address-space evaluation workers, host reserve2GiB,
disk reserve10GiB. No training writers. Each invocation requires an explicit
future deadline within12h, initially2026-11-15T02:30UTC. Freeze inputs, code and
this criteria before launch. Lock supervisor/outputs, publish reports atomically,
retain lifetime peak/attempt metrics and10second health samples. An abnormal
exit stops peers and requires investigated recovery; no automatic failure retry.
No cloud, reserved6100000 or silent parameter/gate changes. At completion verify
every report/input hash, paired vector, resource limit and source attribution;
write the separate operational/component outcome and actionable next work.
