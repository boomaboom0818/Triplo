# Triplo v7

Triplo v7 adds one opt-in switch, `--turn-defend`, to both v6 builds. A second switch,
`--turn-probe`, is in the code with tests but stays off.

| Build | Base | Command line |
|---|---|---|
| **v7-fast** (live; released as "v7") | v6-fast | v6-fast flags + `--turn-defend` |
| **v7-tight** | v6-tight | v6-tight flags (`--fallback equity-guarded-v5 --river-solver`, 50 BB buy-in, learned postflop) + `--turn-defend` |

```bash
# v7-fast
.venv/bin/python -m scripts.openpoker_client --checkpoint runs/bot/pressure-fresh-001/base-guard.json \
  --min-visits 64 --buy-in 5000 --auto-rebuy --fallback equity-guarded-v5 \
  --stack-press --deep-value --no-learned-postflop --river-solver \
  --defend-3bet --flop-peel --turn-defend --opponent-stats runs/bot/opponent-stats.json \
  --avoid-strong leave --strong-bb100 220 --min-weak 2 --max-strong 1 --idle-timeout 300

# v7-tight
.venv/bin/python -m scripts.openpoker_client --checkpoint runs/bot/pressure-fresh-001/base-guard.json \
  --min-visits 64 --buy-in 1000 --auto-rebuy --fallback equity-guarded-v5 --river-solver \
  --defend-3bet --flop-peel --turn-defend --opponent-stats runs/bot/opponent-stats.json \
  --avoid-strong leave --strong-bb100 220 --min-weak 2 --max-strong 1 --idle-timeout 300
```

Drop `--turn-defend` to get v6-fast or v6-tight exactly. In v7-fast the check runs inside the Deep
Value postflop path; in v7-tight it runs on the regular v5 fallback paths, like `--flop-peel`.

## The leak

From public OpenPoker hand evidence (`GET /api/evidence/hand/<id>`) for 11,389 live Triplo hands
(2026-11-19 to 11-24), heads-up, first bet on a street after Triplo checked:

| Spot (v5-fast and v6) | Triplo folds | Pure-bluff break-even |
|---|---|---|
| Turn, flop checked through, 60-85% pot | 78% (n=23) | 43% |
| Flop stab, 60-85% pot | 76% (n=34) | 43% |

On live v6 (1,209 hands), turns where Triplo checked and faced a bet cost -359 BB (79% folds).
Since v5-fast routes all postflop play to the value-only fallback, Triplo's flop c-bet fell from
56% (v4) to 19%, and it won 12% of the flops it saw without showdown. The Doppio regular, on
the 159 hands it shared with Triplo, c-bet 80% and won 32% without showdown. `--flop-peel` already
cut flop stab folds to 68% (net +46 BB on v6); the turn was still open.

Refitting the range-v3 bettor factors on showdowns where the checker called (so the sample does
not depend on the bettor's cards) did not beat range v3 on held-out hands. The ranges are not the
cause: Triplo's check range is weak because it bets every strong hand, so it folds too often.

## `--turn-defend`

`GuardedEquityPolicyV4._turn_defend_call`. When the fallback would fold on the turn, heads-up,
facing the only bet on the street, at most 0.85 pot, with no raise, from a player who did not bet
the flop (a probe or delayed stab, not a barrel), it calls with:

- a flush draw or open-ended straight draw (8+ clean outs) if equity covers the price within 1%;
- a gutshot or any made pair if 0.96 x equity minus the sampling error covers the price.

No-pair, no-draw hands still fold. Barrels, multiway pots, raises and the river are unchanged, and
side-pot pricing keeps the last word.

## `--turn-probe` (off)

`_turn_probe_bet`: on a heads-up turn after the flop checked through, bet about 2/3 pot with
draws and 75% of air, keep unimproved pairs as checks, skip measured sticky callers. Between
other players, live opponents folded 60% to these bets, but to Triplo's own turn probes they folded
29% (n=38, break-even 33%), and in simulation it lost 2.2 [-6.0, +1.6] BB/100 in `real_mix`. Not shipped.

## First screen (v7-fast)

`scripts/v6_field_benchmark.py --deals 4096 --arms v6_fast v7_defend v7_probe v7_defend_probe`, measured
opponent profiles, 250 BB hero, paired deltas vs v6-fast, 95% intervals. The new `stab_exploiter`
field is the Doppio-style `HudExploiter` plus the live stab pattern: it checks back 45% of unmade
flops, then bets 2/3 pot with any two cards into a checked turn (75%) or river (60%).

| Field | v6-fast BB/100 | `--turn-defend` | `--turn-probe` | both |
|---|---|---|---|---|
| sticky | +4667.9 | 0.0 | 0.0 | 0.0 |
| real_mix | +429.0 | 0.0 | -2.2 [-6.0, +1.6] | -2.2 [-6.0, +1.6] |
| pressure | +47.2 | +1.9 [-1.8, +5.6] | 0.0 | +1.9 [-1.8, +5.6] |
| turbo | -6.6 | 0.0 | 0.0 | 0.0 |
| hud_exploiter | -11.9 | 0.0 | 0.0 | 0.0 |
| stab_exploiter | -18.7 | **+8.2 [+2.9, +13.6]** | 0.0 | +8.2 [+2.9, +13.6] |

Turn fold rate against `stab_exploiter`: 65.6% -> 55.9%. The spot is rare in these scripted fields
(most bots bet every flop), so most cells are unchanged. These are simulations; the scripted stab
bot does not adapt like the real Doppio. Confirm live, and roll back by dropping `--turn-defend`.


## Extended tests: v7-fast and v7-tight vs v6-fast and v6-tight

Raw reports are in `runs/bot/v7-crossplay-001/` (not committed). All intervals are 95%.

### Duplicate head-to-head (`scripts/triplo_v7_crossplay.py --tests h2h`)

Three seats of each bot alternate, and every deal is replayed with the bots swapped, 32,768 deals
(65,536 hands) per cell. BB/100 per seat for the first bot; the second loses the same amount.
"Measured" uses fitted opponent profiles, "cold" none; "unequal" mixes 30-500 BB stacks around a
250 BB hero seat.

| Match | measured, unequal | measured, equal | cold, unequal | cold, equal |
|---|---|---|---|---|
| **v7-fast vs v6-fast** | -0.3 [-1.5, +0.8] | +0.0 [-1.9, +1.9] | +0.1 [-0.2, +0.4] | -0.4 [-0.5, -0.2] |
| **v7-tight vs v6-tight** | +0.2 [-0.2, +0.6] | -0.4 [-0.9, +0.1] | -0.1 [-0.3, -0.0] | +0.1 [-0.2, +0.4] |
| v7-fast vs v6-tight | -5.6 [-7.8, -3.5] | -1.7 [-5.3, +1.9] | -4.2 [-6.2, -2.1] | -1.8 [-5.1, +1.5] |
| v7-tight vs v6-fast | +5.9 [+3.8, +7.9] | +2.2 [-1.4, +5.8] | +4.5 [+2.4, +6.6] | +3.4 [+0.2, +6.7] |
| v7-fast vs v7-tight | -4.5 [-6.7, -2.3] | -1.2 [-4.8, +2.4] | -5.0 [-7.1, -2.9] | -2.2 [-5.4, +1.0] |
| control: v6-tight vs v6-fast | +6.4 [+4.3, +8.5] | +2.9 [-0.7, +6.5] | +4.4 [+2.4, +6.3] | +2.3 [-1.0, +5.5] |

### Substitution (`--tests sub`)

One seat among five v6 bots, 8,192 paired deals per cell; paired difference against the same
family's v6 build (same seat, same deals).

| Field | Mode | v7-fast - v6-fast | v7-tight - v6-tight |
|---|---|---|---|
| 5x v6-fast | measured, unequal | +5.2 [-1.5, +11.8] | -5.4 [-14.1, +3.3] |
| 5x v6-fast | measured, equal | +4.3 [-4.2, +12.8] | -3.4 [-9.3, +2.4] |
| 5x v6-fast | cold, unequal | -0.0 [-1.0, +1.0] | +1.2 [-2.6, +5.0] |
| 5x v6-fast | cold, equal | -0.2 [-1.2, +0.9] | +0.6 [-3.6, +4.9] |
| 5x v6-tight | measured, unequal | +1.5 [-0.1, +3.1] | +0.1 [-0.8, +1.0] |
| 5x v6-tight | measured, equal | +3.9 [-3.5, +11.3] | -1.2 [-2.9, +0.4] |
| 5x v6-tight | cold, unequal | -0.3 [-1.8, +1.2] | +0.8 [-0.6, +2.2] |
| 5x v6-tight | cold, equal | +0.1 [-0.2, +0.5] | 0.0 |
| mixed v6 | measured, unequal | -6.2 [-13.6, +1.3] | +2.8 [-5.0, +10.7] |
| mixed v6 | measured, equal | +9.6 [-3.7, +23.0] | -5.1 [-11.7, +1.5] |
| mixed v6 | cold, unequal | +2.7 [-3.4, +8.7] | -0.6 [-3.0, +1.8] |
| mixed v6 | cold, equal | -0.2 [-0.5, +0.1] | -0.7 [-1.6, +0.2] |

### Scripted fields on fresh seeds (`scripts/v6_field_benchmark.py --seed 781000000`)

16,384 paired deals per field, measured profiles / cold start, delta vs the same family's v6 build:

| Field | v7-fast - v6-fast | v7-tight - v6-tight |
|---|---|---|
| sticky | 0.0 / 0.0 | 0.0 / 0.0 |
| real_mix | 0.0 / 0.0 | 0.0 / 0.0 |
| pressure | 0.0 / 0.0 | +0.6 [-2.8, +4.0] / +0.6 [-2.8, +4.0] |
| turbo | -0.1 [-0.4, +0.1] / 0.0 | -0.4 [-1.0, +0.2] / -0.4 [-1.0, +0.2] |
| hud_exploiter | -0.4 [-0.9, +0.1] / -0.4 [-0.9, +0.1] | -0.6 [-2.3, +1.2] / -0.6 [-2.3, +1.2] |
| stab_exploiter | +1.3 [-1.0, +3.6] / +0.7 [-2.4, +3.8] | +0.7 [-2.6, +4.0] / +0.4 [-3.2, +4.1] |

Turn fold rate against `stab_exploiter`: v6-fast 66.3% -> v7-fast 59.5%; v6-tight 62.2% -> v7-tight 55.2%.

### Reading

- In simulation, each v7 build plays like its v6 parent: every same-family comparison is within
  about 1 BB/100 head-to-head and in the scripted fields. The one significant cell, -0.4
  [-0.5, -0.2] for v7-fast vs v6-fast cold at equal stacks, is negligible.
- The first screen's +8.2 [+2.9, +13.6] against `stab_exploiter` did not hold up: on 16,384 fresh
  deals it is +1.3 [-1.0, +3.6]. `--turn-defend` lowers the turn fold rate as designed, but the
  scripted bots rarely reach the spot (most bet every flop), so the simulated effect is small. The
  case for it rests on the live leak (79% turn folds to stabs), which must be confirmed live.
- The fast/tight gap carries over unchanged from v6 (control row: v6-tight beats v6-fast by the same
  2-6 BB/100 as v7-tight beats v6-fast). Tight beats fast head-to-head at 250 BB,
  while fast wins far more against weak fields (sticky +4,698 vs +3,475 BB/100, real_mix +433 vs
  +282). v7-fast stays the live build for chip rate on soft tables.
- Live, in its first 194 hands (2026-11-24 13:20-15:49 PST), `--turn-defend` did not fire: the
  heads-up turn probes v7-fast faced were either called already by the v6 logic (pairs, draws) or
  no-pair hands it still folds by design.

## v7-fast vs v7-tight at their live buy-ins

The tests above put both builds at 250 BB. Live, v7-fast buys in at 5,000 chips (250 BB) and v7-tight
at 1,000 (50 BB).

Duplicate head-to-head with each bot at its own buy-in in every seat it takes
(`scripts/triplo_v7_crossplay.py --hero v7_fast --rivals v7_tight --buyin v7_fast=250
--buyin v7_tight=50`), 32,768 deals per mode, BB/100 per seat for v7-fast:

| Mode | v7-fast (250 BB) vs v7-tight (50 BB) |
|---|---|
| measured profiles | -2.1 [-3.7, -0.5] |
| cold start | -2.9 [-4.6, -1.2] |

Scripted fields, same deals and opponents, hero at its buy-in (`scripts/v6_field_benchmark.py
--arms v7_fast_250bb v7_tight_50bb --seed 801000000`), 16,384 paired deals per field, BB/100
(measured profiles / cold start):

| Field | v7-fast at 250 BB | v7-tight at 50 BB | tight - fast |
|---|---|---|---|
| sticky | +4,796 / +3,701 | +1,368 / +1,348 | -3,428 [-3,607, -3,249] / -2,352 [-2,499, -2,206] |
| real_mix | +460 / +356 | +166 / +163 | -294 [-344, -243] / -192 [-234, -151] |
| pressure | +47 / +75 | +25 / +26 | -22.7 [-47.5, +2.1] / -48.7 [-73.1, -24.3] |
| turbo | +40 / +27 | -7 / -3 | -46.5 [-75.3, -17.7] / -29.5 [-57.1, -2.0] |
| hud_exploiter | -24 / -25 | -16 / -16 | +8.2 [-5.3, +21.7] / +9.0 [-3.3, +21.3] |
| stab_exploiter | -18 / -18 | -12 / -14 | +5.6 [-8.1, +19.4] / +4.4 [-8.7, +17.5] |

Reading: v7-tight wins the direct match, but at its live buy-ins v7-fast makes far more chips per
hand against every field with weak or mixed players, by about 2,400-3,400 BB/100 against callers.
v7-tight only does better against the two HUD-exploiter fields, and not significantly. For chip
rate on the leaderboard, v7-fast stays the live build. v7-tight is the choice for a table of
strong regulars that attack Triplo, where the 50 BB buy-in caps losses.

v7-tight's postflop bets also include far more hands without top pair (36-50% of its bets in these
fields, against 4-22% for v7-fast), because its learned policy still plays postflop. That is
closer to the betting mix Doppio uses.
