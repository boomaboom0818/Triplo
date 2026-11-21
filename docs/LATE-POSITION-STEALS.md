# Late-Position Steals

Late-Position Steals is an opt-in overlay on `equity-guarded-v3` for history
checkpoints. It keeps the checkpoint, rules, equity ranges, and postflop sizing. It can replace a v3
**fold** with a **2.5 BB raise** from the cutoff, button, or small blind when:

- Six players were dealt in. A verified history and exact raw-chip ledger show
  that everyone before us folded, without limps, raises, or extra money.
- The hero and all players left to act have at least 20 BB at the start, the
  server offers the exact size, and the hand meets the position's Chen-score
  floor (5.5 CO, 4.5 BTN, 6 SB).
- The exact remaining lineup, in order, qualifies on **both** older and newer
  completed hands for the same hero, position, and size. This is the measured
  chance that **everyone** folds, not a product of individual fold estimates.

The overlay requires 30 older and 15 newer opens per lineup. The one-sided 95% Wilson
lower bound in each period must exceed `open cost / (pot + open cost) + .03`.
This is a screening rule that gives no credit for equity when called; it is not
a lower bound on total hand profit after future bets. Repeated opponents and
selection across lineups also limit what the intervals establish. The rule
leaves unknown names, sparse profiles, short tables, calls, and existing raises
alone. It does not automatically fit or refresh stats from the live process.

## Fit and shadow

Provide a JSON array of completed public OpenPoker hand-evidence records,
including wins **and losses**. Each record needs the hand ID, start/end times,
board, blinds, dealer seat, full `players` roster with `bot_name`, `seat`, and
`stack_start`, and the sequenced `action_log`. The public hand-evidence endpoint
(`GET /api/evidence/hand/<hand_id>` on `api.openpoker.ai`) returns those fields.
Choose a chronological cutoff before looking at validation results.

```sh
PYTHONPATH=. .venv/bin/python -m scripts.fit_late_position evidence.json \
  --hero Triplo --holdout-from 2026-11-21T02:33:09+00:00 \
  --out runs/bot/late-position-folds.json
```

To shadow it, use the same checkpoint, profile file, and sizing options as v3,
and add these arguments to the client invocation:

```text
--fallback equity-guarded-v3 --late-position-steals runs/bot/late-position-folds.json
--late-position-report --log runs/bot/late-position-shadow.jsonl
```

Report mode logs a `late_position_steal` diagnostic for qualified proposals
but sends the unchanged v3 action. The diagnostic records the baseline source,
counts, bounds, exact proposed amount, and whether it was applied. Without
`--late-position-report`, explicitly enabling the overlay applies qualified opens. No
default or running service has been changed.

Initial data check: among the **881** v3 hands through 2026-11-21 06:43 UTC,
there were **22** matching opens, including **two** won immediately and **20**
that drew a continuation. There were **15 older and seven newer** opens spread
across 18 lineups, so **none** qualified. This is intentionally fail-closed. The
earlier synthetic pilot used unconditional late-position opens and is not a
validation of this gated policy. Collect more eligible hands and compare both
hands won and BB/100 before deploying it.
