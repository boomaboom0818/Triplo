# Late-Position Steals

Late-Position Steals is built into `equity-guarded-v3` for OpenPoker history
checkpoints. It keeps the checkpoint, equity ranges, and postflop sizing. It can
replace a **fold** with a **2.5 BB raise** from the cutoff, button, or small blind when:

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

## Fit and run

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

Run the [OpenPoker client](openpoker.md) from the repository root with a history
checkpoint and `--fallback equity-guarded-v3`. The client reads
`runs/bot/late-position-folds.json` once at startup and applies qualifying opens
automatically. If the file is absent or the lineup does not qualify, it keeps
the original action. An invalid file is a startup error. The old
`--late-position-steals` and `--late-position-report` switches are no longer
accepted. Use `--log PATH` to record applied steals, including the baseline
source, evidence counts, fold bound, and exact raise amount.

Initial data check: among the **881** v3 hands through 2026-11-21 06:43 UTC,
there were **22** matching opens, including **two** won immediately and **20**
that drew a continuation. There were **15 older and seven newer** opens spread
across 18 lineups, so **none** qualified. This is intentionally fail-closed. The
earlier synthetic pilot used unconditional late-position opens and is not a
validation of this rule. Measure both hands won and BB/100 on more eligible
hands before drawing conclusions about live profitability.
