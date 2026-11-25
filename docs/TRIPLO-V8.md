# Triplo v8: exploiting OpenPoker's no_code fish

Triplo v8-fast is v7-fast plus `--fish-exploit`, run with `--min-weak 3` instead of 2.

```bash
.venv/bin/python -m scripts.openpoker_client --checkpoint runs/bot/pressure-fresh-001/base-guard.json \
  --min-visits 64 --buy-in 5000 --auto-rebuy --fallback equity-guarded-v5 \
  --stack-press --deep-value --no-learned-postflop --river-solver \
  --defend-3bet --flop-peel --turn-defend --fish-exploit --opponent-stats runs/bot/opponent-stats.json \
  --avoid-strong leave --strong-bb100 220 --min-weak 3 --max-strong 1 --idle-timeout 300
```

Drop `--fish-exploit` and set `--min-weak 2` to get v7-fast exactly.

## Why Triplo earned less per hand

From the public hand evidence of 16,000 live Triplo hands (2026-11-16 to 11-24):

- Most of every table's money comes from OpenPoker's hosted `no_code` preset bots: about 3.5 per
  table, each losing about 110 BB/100. Triplo's win rate depends mostly on how many sit with it:
  +147 BB/100 with 4-5, +127 with 3, +33 with 2 or fewer (23.5% of its hands).
- The `no_code` bots play one simple, measurable style (88,671 of their decisions fitted in
  `poker/nocode_model.py`): they almost never limp, open 22-46%, call 3-bets with dominated
  hands (A8o, QJo, KJo), raise flop/turn bets about a quarter of the time regardless of strength,
  and fold only 9-14% to turn/river bets up to 1.5x pot.
- Triplo played them like any other opponent: no isolation of their opens, half-pot river value
  bets, and flat calls when they raised.

Table selection alone (copying Doppio's rules) did not explain the gap, and neither did postflop
leaks such as stab defense or river bluffs.

## What `--fish-exploit` does

It applies only against seats the public leaderboard marks weak (`no_code`, or a losing
self-hosted bot: `OpponentField.is_weak`). Measured fold profiles are not used to find fish,
because nearly every bot calls river bets and most winning regulars measure as "sticky".

| Part | Rule |
|---|---|
| Isolation (`poker/fish_preflop.py`) | When a fish opens and every other live player is a fish: 3-bet the top 12% of hands (3x in position, 4x out), call the top 30% (45% when closing the action), fold the rest; facing a fish 3-bet after our open, jam the top 6% and call the top 22%. No change to unopened pots. |
| Value sizing | With top pair or better against fish only: flop 0.75, turn 1.0, river 1.25 pot. |
| Value re-raise | When a fish raises our flop/turn bet (or bets into us), re-raise top pair or better to about 0.6 pot. |
| Table selection | `--min-weak 3`: leave full tables with fewer than 3 weak seats. |

## Evidence

`scripts/fish_field_benchmark.py` seats a hero among `NoCodeFish` seats (stacks 30-113 BB, as live)
and v7-fast regulars. The model reproduces live `no_code` statistics (VPIP 30%, preflop and
per-street fold/raise rates), and v7-fast's simulated win rate (+163 BB/100 with 4 fish and
a regular) matches its live rate at such tables (+147).

v8-fast minus v7-fast, 24,576 paired deals per field, fresh seed, 95% intervals:

| Table | v7-fast | v8-fast | difference |
|---|---|---|---|
| 5 fish | +188 | +252 | **+63.9 [+37.3, +90.4]** |
| 4 fish, 1 regular | +163 | +191 | **+28.1 [+11.5, +44.8]** |
| 3 fish, 2 regulars | +115 | +139 | **+23.7 [+11.8, +35.6]** |
| 2 fish, 3 regulars | +88 | +96 | **+7.7 [+1.8, +13.7]** |

Weighted by the live table mix, that is about +23 BB/100. Table selection was estimated by
replaying Triplo's 58 real table stints since 2026-11-21 (fish count every hand, 1.1 hands lost
per switch, new tables drawn from the observed join mix) with the simulated rates above:

| Setup | BB/100 per unit time |
|---|---|
| v7-fast, `--min-weak 2` (live before v8) | +131.8 |
| v7-fast, `--min-weak 3` | +137.1 |
| v8-fast, `--min-weak 2` | +158.6 |
| **v8-fast, `--min-weak 3`** | **+166.4 (+34.6, +26%)** |
| v8-fast, leave below 4 fish | +123.3 (too many switches) |

Component tests (8,192-32,768 paired deals) that shaped it: value sizing alone +4 to +10 at
every table; isolation +11 to +29 where fish dominate; value re-raise +2 to +11. Rejected:
widening unopened opens ("steal", -11 to +5), removing turn/river bluffs against fish (-11 to
-15), thin value bets (noise), flop c-bets against fish (noise).

## Limits

- The fish model reproduces live frequencies, but its hand selection within each frequency is an
  assumption (Chen rank preflop, made-hand percentile postflop).
- Simulated regulars are v7-fast copies; real regulars differ. `--fish-exploit` only acts when
  every relevant opponent is a leaderboard fish, so regulars change nothing directly.
- The table-selection estimate assumes switching does not change which tables the server offers.
- A live confirmation needs on the order of 10,000 hands, because one hand's result varies by
  about 20 BB.

## v8-tight

v8-tight is v7-tight plus `--fish-exploit`, run with `--min-weak 3`: 50 BB buy-in, `equity-guarded-v5`
with `--river-solver`, learned postflop play, no Stack Press and no Deep Value sizing.

```bash
.venv/bin/python -m scripts.openpoker_client --checkpoint runs/bot/pressure-fresh-001/base-guard.json \
  --min-visits 64 --buy-in 1000 --auto-rebuy --fallback equity-guarded-v5 --river-solver \
  --defend-3bet --flop-peel --turn-defend --fish-exploit --opponent-stats runs/bot/opponent-stats.json \
  --avoid-strong leave --strong-bb100 220 --min-weak 3 --max-strong 1 --idle-timeout 300
```

Drop `--fish-exploit` and set `--min-weak 2` to get v7-tight exactly. Arm `v8_tight_50bb` in
`scripts/fish_field_benchmark.py`; v8-tight minus v7-tight, 32,768 paired deals per field, 95% intervals:

| Table | v7-tight | v8-tight | difference (BB/100) |
|---|---|---|---|
| 5 fish | +81 | +114 | **+32.9 [+17.8, +48.0]** |
| 4 fish, 1 regular | +71 | +96 | **+24.7 [+15.5, +33.8]** |
| 3 fish, 2 regulars | +39 | +53 | **+13.9 [+8.1, +19.6]** |
| 2 fish, 3 regulars | +30 | +32 | +2.4 [-1.2, +6.0] |

An earlier 8,192-deal run agreed in direction but was not significant at any table. At a 50 BB stack the
exploit wins less than v8-fast does (+8 to +64 BB/100 at 250 BB), and v8-tight's absolute rates stay far
below v8-fast's at fish tables. Same limits as v8-fast apply: the fish model's hand selection is an
assumption, simulated regulars are v7-fast copies, and a live check needs about 10,000 hands. v8-tight is
not deployed.
