# lean-002 sparse-support diagnosis and mitigation

The 2,002,060-iteration checkpoint has near-complete table lookup coverage but
little average mass in many multiway states. At a normalized mass cutoff of
1e-4, 45,525/50,515 four-plus-player flop rows, 46,282/46,647 turn rows and all
11,384 river rows fall below the cutoff. These are unweighted row inventories,
not on-policy decision frequencies. The existing run journal also attributes
station losses to weak postflop bets in multiway hands.

`LeanPolicy.from_checkpoint(saved, min_support=1e-4)` now uses its existing tight
rule fallback below that cutoff. The default remains zero (previous behavior).
Support means sum of the average row divided by T(T+1)/2; it is a heuristic,
NOT a visit count or effective sample size. This changes inference, not training.
Existing checkpoints and optimizer state are unchanged.

Reproduce the screening experiment with a new output filename:

```sh
.venv/bin/python -m scripts.lean_support_check \
  --checkpoint runs/bot/lean-002/snapshot-2002060.json \
  --deals 1000 --seed 17310000 --min-support 0.0001 \
  --out runs/bot/lean-002/support-screen-17310000.json
```

The script refuses to overwrite reports. It evaluates all three policies on the
same deals, rotating through all six seats; opponents retain their full raise
menu. Reports retain deal-block samples, provenance, actual learned-decision
fractions, and paired differences. Intervals are pointwise 95% normal intervals
over independent deal blocks, not six independent observations per rotation.
No multiple-comparison correction is applied; this is a screen, not qualification.
The cutoff was taken from the earlier experiment, before this fresh seed was run.

| Opponent | Original bb/100 | Sparse fallback bb/100 | Tight fallback alone bb/100 |
| --- | ---: | ---: | ---: |
| Tight | 0.5 | 14.4 | 4.1 |
| Aggressive | 4.3 | 10.2 | 27.6 |
| Station | -67.5 | 81.3 | 192.8 |
| Shover | 28.7 | 31.5 | 30.9 |

Against station, the paired sparse-fallback improvement is +148.8 bb/100
(95% interval +96.1 to +201.5). But it is still worse than fallback alone by
111.5 bb/100 (paired interval -169.4 to -53.7). Differences against the original
on the other three styles include zero. The learned strategy has not established
added value over the simple rule baseline. Neither the original nor the modified
policy passes every style's positive-lower-bound criterion on this screen.

Optional client use: add `--min-support 0.0001` to the existing
`python -m scripts.openpoker_client ...` command. This remains opt-in and was not
activated in a live session. Decision logs count sparse fallback as uncovered.

## Training direction

Do not simply extend the stopped run and expect table lookup coverage to solve
this. A next, separately versioned experiment should mix self-play with fixed
loose-calling training opponents to expose multiway decisions. Keep some self-play
and train each seat; when opponents use a fixed policy, use that policy consistently
in action sampling and regret utility estimation. Keep evaluation styles/seeds
separate from tuning; do not optimize against the gate seed or reuse this fresh
screen as a holdout after tuning on it. A separate loose-calling training policy
should vary calling thresholds and sizes rather than copy the station gate.

Before spending hours, run a bounded multi-seed pilot and measure multiway
support, weak-hand betting, paired payoff changes, and improvement over fallback
alone. Adding opponent training changes the objective and does not establish a
multiplayer equilibrium guarantee. The inference mitigation itself does not
validate that training approach; the subsequent implementation and controlled
experiment are recorded in [the mixture pilot](LEAN-MIXTURE-PILOT.md).

Validation: 20 lean/parallel/rule tests passed; 8 client tests passed, including
sparse-row selection, default compatibility, boundary behavior, matched control
reporting, checkpoint immutability and accurate fallback logging.
