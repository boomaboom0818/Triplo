# Sampling pilot 001

Fresh v2 learners, seeds 11/29, 250 iterations each. The recipe
`call-mixture-v1` changes only external-action sampling in average-strategy
traversal: q(a)=0.5/|A| + 0.5 I[a=call]. Every legal action has support and
inverse sampling reach is multiplied by 1/q(a). Regret traversal is unchanged.
An exhaustive small-tree test verifies expected averaging weights; exact resume
and legacy config compatibility are covered by tests. This does not establish
Hold'em convergence or control estimator variance. Increased call sampling can
make trees more expensive; budget stops require investigation.

Outputs: `runs/bot/sampling-001`. Each child has a 3 GiB address-space cap,
3,600-second training invocation allowance, 500,000-node iteration cap and the
existing 1,250,000-set cap. No retries or reset-on-failure. The new recipe is stored
in checkpoint configuration; resumes from legacy recipes are rejected. All
previous checkpoints remain untouched.

Evaluate one-copy primary results on seed 5,200,000, 1,000 independent deal blocks.
Compare the pilot with an explicitly labeled uniform candidate and the original
v2 learner at 250 iterations. All share deals, seat rotations and frozen validation.
Retain per-deal block payoffs and use paired whole-block bootstrap differences.
Also report postflop coverage by active-player count. Final seed 6,100,000 stays
untouched. The old validation agent remains weak and independently trained.

Predeclared screening gate: both seeds must achieve >=5% positive-weight postflop
coverage and >=200 postflop decisions in each of calling, mixed, and learned
populations. Against the frozen learned opponent, paired payoff differences
versus both the uniform candidate and the original v2 control must have a lower
95% bootstrap bound above zero. Other populations remain reported, including
aggressive/tight where postflop counts may be too small for a coverage gate.
Intervals are screening evidence, not a multiplicity-adjusted final certification.
Passing requires review before a bounded 1,000-iteration confirmation stage;
sustained training follows only if that confirmation retains coverage, payoff
and resource evidence. Failed/inconclusive results do not authorize automatic
scaling or relaxed resource limits.

## Authorized exploratory continuation

After reviewing the failed 250-iteration payoff screen, the user authorized
continued bounded training on 2026-11-14. `scripts/sampling_curve.py` resumes the
unchanged recipe into new 500/750/1,000 checkpoints, with a one-hour allowance
per 250-iteration increment and unchanged memory/node/table protections.
This is an exploratory learning curve, not a passed promotion gate.

At 500 and 1,000, evaluate against the retained 250-iteration parent and uniform
control using shared seed 5,300,000 and 1,000 deal blocks. Record action counts to
investigate the tight-script gap. Also evaluate the OpenPoker check/fold/call
fallback separately; paired differences for that mode measure the combined
policy/fallback change, not training alone. Live results are observational and
must not be pooled with controlled local results. The client now logs checkpoint
hash, coverage and action type, never cards or authentication tokens. Running
local experiments does not start or restart a live OpenPoker client.

## Curve recovery and report integrity

After investigating a stopped curve and confirming no active supervisor/writer:

```sh
.venv/bin/python -m scripts.sampling_curve --resume
```

The supervisor lock prevents duplicate curve writers. Resume checks manifest
parents/design, checkpoint recipe and iteration bounds, and valid saved RNG state.
Complete stages are skipped. Only interruption/running/time-budget stops permit
incomplete-stage recovery; node/table/unknown failures require investigation.
Existing reports must match candidate/opponent hashes, mode, seed and deal counts.
New evaluation output is written to `.pending`, validated, then atomically
published. A malformed legacy final report is preserved and causes an explicit
stop; inspect and quarantine it before re-evaluation rather than silently trusting
or replacing it. A checkpoint failing validation also stops recovery; independently
verify `.previous` before any manual recovery.

The updated runner evaluates all stages, including 750. A supervisor already
running the earlier code continues its loaded implementation, so overnight
monitoring supplies the missing 750-stage evaluations in that case. This change
does not restart or modify an active learner's algorithm.

## Verified stage review

Run `python -m scripts.review_curve` to validate complete curve reports and publish
`verified-review.json`. The reader leaves active/empty legacy evaluation outputs
alone and checks checkpoint/opponent hashes, evaluation modes, shared deal design
and sample counts before comparison. The review distinguishes training-versus-
parent differences from a same-checkpoint live-fallback difference, and records
tight-population action counts. This prevents an apparent live-policy change from
being attributed solely to more training. Bootstrap inputs now reject non-finite,
boolean or miscounted payoff samples.

First overnight review: candidate A's 500-iteration standard policy has positive
nominal paired learned-opponent intervals versus its 250 parent and uniform
control on development seed 5,300,000. Candidate B's evaluation was still pending;
this single-stage, selected development result is not a replicated promotion or a
competitive-strength claim. The tight-script gap persists: covered preflop
folding increases, whereas that trivial script rewards indiscriminate aggression.
The OpenPoker-style check/fold/call fallback folds many additional uncovered
preflop states and materially changes this matchup. Keep live checkpoint/fallback
changes separate from the training recipe and do not deploy a fallback change
based on this scripted result alone.
