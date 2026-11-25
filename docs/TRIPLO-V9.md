# Triplo v9: defend against bigger opens

Status: in progress. Live stays on v8-fast until a v9 change beats it in paired simulation.

## The leak

Triplo's own live play (season 17 session log, first preflop decision facing a single open):

| Open size | Decisions | Fold | Call | Raise | Source |
|---|---|---|---|---|---|
| ≤ 2.5 BB | 4,373 | 54% | 19% | 27% | 87% learned |
| 3 BB | 1,600 | 71% | 18% | 11% | 64% `preflop_rules` |
| 3.5-4 BB | 468 | 92% | 6% | 3% | 96% `preflop_rules` |
| > 4 BB | 333 | 97% | 2% | 1% | 96% `preflop_rules` |

Opens of 3 BB or more make up 35% of these spots. The learned tree only knows the trained open
sizes, so bigger opens fall through to `preflop_rules`, which folds almost everything. In
simulation, Triplo v7-fast copies facing a 3.5 BB open went from 96% learned decisions to 15%
and from 50% to 84% folds, so the bot can be exploited this way.

## A/B harness

`poker/ab_metric.py` and `scripts/ab_report.py` give per-hand capped (±25, ±50 BB) and uncapped
BB/100 with 95% intervals, the share of positive 10-minute windows, and a big-pot breakdown. They
work on live session logs, on paired simulations, and with the leaderboard tracker.

**Uncapped BB/100 is the deploy gate.** Triplo's profit is concentrated in big pots: in live v8
play (1,333 hands from 2026-11-25 03:44 UTC) it made +266 BB/100 uncapped, -3 capped at ±25 BB
and +56 capped at ±50 BB, and pots of 50 BB or more carried +312 BB/100. Capped metrics are only
for screening changes that leave big pots alone.

```bash
.venv/bin/python <tree>/scripts/ab_report.py --buy-in 5000 \
  --arm v8=2026-11-25T03:44:36Z --leaderboard runs/bot/leaderboard-track.jsonl
```

At live v8's SD (30.6 BB per hand, about 75 hands per hour), pinning one version to ±35 BB/100
takes about 29,000 hands (16 days), so small effects must be settled in simulation first.

## Rejected earlier candidates

Four fish-table flags were tried first and removed because none beat v8-fast in paired
simulation (12,288 deals per table type, uncapped BB/100): bigger opens when only fish are
behind (-0.6 to -5.0), small flop stabs into fish (-2.9 to -4.7), a no-overbet cap while a
regular remains (-8.5 to -10.2, all in big pots) and raise-instead-of-limp (+5 on the screen,
-3 to -6 on fresh deals). Opening bigger when fish are only the majority won +13.1 [+0.2, +26.0]
on fresh deals, but only by pushing the simulated regulars (Triplo copies) off their trained
tree, which is the leak above rather than something live opponents share.
