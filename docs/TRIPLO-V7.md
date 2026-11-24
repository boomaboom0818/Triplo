# Triplo v7

Triplo v7 is v6-fast plus one opt-in switch, `--turn-defend`. A second switch,
`--turn-probe`, is in the code with tests but stays off.

```bash
.venv/bin/python -m scripts.openpoker_client --checkpoint runs/bot/pressure-fresh-001/base-guard.json \
  --min-visits 64 --buy-in 5000 --auto-rebuy --fallback equity-guarded-v5 \
  --stack-press --deep-value --no-learned-postflop --river-solver \
  --defend-3bet --flop-peel --turn-defend --opponent-stats runs/bot/opponent-stats.json \
  --avoid-strong leave --strong-bb100 220 --min-weak 2 --max-strong 1 --idle-timeout 300
```

Drop `--turn-defend` to get v6-fast exactly. Both switches default off and also work on the
v6-tight path (no `--deep-value`).

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

## Evidence

`scripts/v6_field_benchmark.py --deals 4096 --arms v6_fast v7_defend v7_probe v7_fast`, measured
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
