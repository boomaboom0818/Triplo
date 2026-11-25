# Triplo v9: defend against bigger opens

Status: research complete, **nothing deployed**. Live stays on v8-fast (release 9342601).
`--open-translation` is in the client, off by default and decision-identical to v8 when off,
but no setting of it beat v8 on fresh simulated deals.

## The leak

Triplo's own live play (season 17 session log, first preflop decision facing a single open):

| Open size | Decisions | Fold | Call | Raise | Source |
|---|---|---|---|---|---|
| ≤ 2.5 BB | 4,373 | 54% | 19% | 27% | 87% learned |
| 3 BB | 1,600 | 71% | 18% | 11% | 64% `preflop_rules` |
| 3.5-4 BB | 468 | 92% | 6% | 3% | 96% `preflop_rules` |
| > 4 BB | 333 | 97% | 2% | 1% | 96% `preflop_rules` |

Opens of 3 BB or more make up 35% of these spots. In simulation, Triplo v7-fast copies facing a
3.5 BB open went from 96% learned decisions to 15% and from 50% to 84% folds.

### Where it happens in the code

The lean menu only opens to half-pot or pot (`FastGame.menu`): 2 BB or 2.5 BB. The learned row
is looked up by `history_cfr.compact_features`, whose `lean_key` part buckets the price
(`facing = call * 6 // pot`) and the pot (`size`), and whose history part tokenizes the open by
its pot fraction. A 2 or 2.5 BB open in front of the hijack gives `facing=3, size=1` (`|314|`)
with thousands of visits per hand class. Every open of 3 BB or more gives `facing=4, size=2`
(`|424|`), which has **zero** visits for every hand, so `HistoryPolicy.act` finds no row and
returns `self.fallback.act(o)`: the tight-v1 preflop rules plus the guarded commitment cap
(calling more than 3 BB needs Chen ≥ 9). The client logs that as `preflop_rules`.

### Who opens big, live

`scripts/v9_open_report.py` joins the session log with the wire trace (seat names, blinds) and
the client's leaderboard table checks (weak = no_code or losing, strong ≥ 220 BB/100):

| Opens of | Mostly by | Triplo folds | Triplo net, BB/hand |
|---|---|---|---|
| 3 BB | fish (83%); fish open 3 BB 28% of the time | 71% | +1.88 |
| 3.5-4 BB | regulars: strong 45%, unrated 31%, rated 22%; fish 2% | 92% | -0.13 |
| > 4 BB | fish 80% (mostly big shoves), strong 14% | 95% | -0.49 |

The 3.5-4 BB openers are a handful of bots (the Svan family, RoRo, tamias23, Herqules,
Synapse): they open 60-70% of hands from every seat and, as the opener facing a 3-bet, call 63%,
fold 25% and 4-bet 11%. RoRo and Herqules always open 3.5 BB; the Svan bots open 2.5 BB or 4 BB,
and their shown 4 BB hands are much stronger (mean Chen rank 0.16, 72% top-20%) than their 2.5 BB
hands (0.48), so their size is a tell. They are 9% of non-fish seats over the season but only
2.6% in the v8 window, because v8's table selection leaves tables with strong regulars. Fish,
as the opener facing a 3-bet, call 91% and fold 1-2% (2,027 spots).

## The fix tried: open-size translation

`poker/open_translation.py`, enabled by `HistoryPolicy.open_translation` (client
`--open-translation MAX_BB`, 0 = any size). Only when the real spot has no usable learned row,
hero's first preflop decision facing exactly one open bigger than 2.5 BB (folds before it,
folds or calls after it, nobody all in) is replayed in `FastGame` with the open re-sized to
2.5 BB. The learned row of that translated spot is sampled, and the action is mapped back: fold
and call keep their meaning (the call pays the real price) and raises keep their place on the
menu (half-pot, pot, all-in), so a pot 3-bet over a 3.5 BB open goes to 8.5 BB. The pressure
guard still applies. Options for testing: `fish_only` (only opens by leaderboard-weak seats) and
`keep` (keep only translated calls or raises; other translated actions go to the rules).

Identity: with the flag off, the v8 release (9342601) and this tree give the same digest over
all 19,430 live preflop decisions of the season-17 log (`scripts/v9_preflop_replay.py`, RNG
reseeded per decision), and the paired simulations change 0 of 49,152 hands. On those live
spots, `--open-translation 3` translates 757 decisions and changes 253 actions, all at 3 BB
opens (fold to 3-bet 112, call to 3-bet 66, fold to call 59). With any size it changes 448.

## Simulation

`scripts/v9_open_benchmark.py`: paired deals on the fish-table benchmark (NoCodeFish with live
stacks, hero and regulars 250 BB, live table mix 5 fish 4% / 4+1 48% / 3+2 40% / 2+3 8%),
v9's A/B metrics, a per-spot breakdown (who opened, which size) and big-pot (≥ 50 BB) share.
New opponents, because the old regulars (v7-fast copies) only open to the trained sizes:

* **big openers** fitted to the live log: `fixed` (always 3.5 BB) and `svan` (4 BB with most of
  its top 20%, else 2.5 BB), both opening 60-70% from every seat and answering 3-bets as live.
* **3 BB regulars** (`--regs three`): live rated-reg open widths (16-35%) at 3 BB.
* **sticky fish** (`--fish sticky`, the default): as the opener facing a 3-bet they call 91% as
  live. Plain NoCodeFish folds 15-30% of its range there, which would overpay 3-bet bluffs.

The main setting is `--regs mix` (each regular a big opener with probability 0.25, which makes
~15% of regular opens 3.5 BB+, as in the live v8 window) with sticky fish. Live-mix weighted
paired differences against v8, BB/100 with 95% intervals:

| Run (seeds, deals per table) | Arm | Uncapped | ±50 BB cap | ±25 BB cap | Pots ≥ 50 BB |
|---|---|---|---|---|---|
| Screen (960M, 12,288) | `tr` all sizes | +3.6 [-5.7, +12.9] | -0.6 [-4.9, +3.6] | -3.2 [-5.9, -0.5] | +5.6 |
| | `tr3` ≤ 3 BB | **+6.9 [+1.8, +12.0]** | +2.3 [-0.4, +5.0] | -0.2 [-1.9, +1.5] | +7.9 |
| Confirm (970M, 12,288) | `tr3` | +2.5 [-2.8, +7.9] | +0.0 [-2.7, +2.8] | -1.4 [-3.1, +0.4] | +4.2 |
| | `off` | 0.0 (0 of 49,152 hands changed) | | | |
| Confirm, 4x (980M, 49,152) | `tr3` | +0.9 [-1.7, +3.4] | -0.3 [-1.7, +1.0] | -1.6 [-2.4, -0.8] | +2.2 |
| Screen 2 (990M, 12,288) | `tr3` | +0.1 [-6.3, +6.4] | -2.1 [-4.9, +0.8] | -2.5 [-4.3, -0.7] | +1.6 |
| | `tr3r` 3-bets only | +2.4 [-3.2, +8.0] | -0.8 [-3.4, +1.8] | -1.4 [-3.1, +0.2] | +3.4 |
| | `tr3c` calls only | -1.3 [-3.9, +1.3] | -1.0 [-2.0, +0.1] | -0.9 [-1.6, -0.3] | -0.8 |
| | `trr` 3-bets, all sizes | +3.4 [-3.4, +10.3] | -1.8 [-5.5, +1.8] | -3.0 [-5.4, -0.5] | +4.5 |

Robustness screens (two or three tables, weighted over those): with plain NoCodeFish `tr3`
+5.6 [-0.0, +11.3]; with every regular a 3 BB opener `tr3` +11.4 [-4.5, +27.3] and fish-only
`tr3f` +9.9 [+2.0, +17.7] (on 3 BB regular opens translation is about neutral, +9 and +15 BB/100
of those hands with wide intervals).

What the runs say:

* **3.5-4 BB opens: translation loses.** Against `svan` 4 BB opens it lost -430 BB/100 of those
  hands [-835, -26] (the 2.5 BB defending range is far too wide against a value-sized open);
  against `fixed` 3.5 BB opens it was -179 and +90 in two tables. That rules out `tr`.
* **3 BB opens (fish): no reliable gain.** The screen's +6.9 did not repeat: +2.5, then +0.9 on
  four times the deals, then +0.1. Pooling all four `tr3` runs (screen included) gives about
  +1.9 [-0.1, +3.9]. Small pots consistently lose (±25 cap -1.4 to -2.5, the 4x run
  significantly) and big pots win a little.
* **Translated calls bleed; translated 3-bets are not clearly better.** `tr3c` is negative and
  significantly so in small pots. The 3-bet-only arms are positive but not significant, and all
  of their gain is in big pots, so it depends on how the simulated openers answer 3-bets and
  play big pots — the kind of gain that fooled the earlier v9 work.

So the leak is real (Triplo folds far more than it should in theory), but the fix that uses the
learned 2.5 BB answers does not earn more in simulation, and the live numbers agree that the
3 BB spots are already profitable (+1.9 BB/hand, mostly fish). The 3.5-4 BB spots are mostly
regulars that v8's table selection already avoids. The deploy bar (beat v8 on fresh deals with
the interval above zero, no big-pot loss, suite green) is not met, so live stays on v8-fast.

### Commands

```bash
PY=.venv/bin/python; T=<tree>
$PY $T/scripts/v9_open_report.py --log runs/bot/openpoker-season-17-equity/session.jsonl \
  --trace runs/bot/openpoker-season-17-equity/trace.jsonl --out open-report.json
# one table type per call (~2-8 minutes each); results resume per table, then summarize
$PY $T/scripts/v9_open_benchmark.py --deals 12288 --regs mix --fish sticky --arms v8 tr3 tr3r \
  --fields fish4_reg1 --out-dir runs/bot/triplo-v9-open/screen
$PY $T/scripts/v9_open_benchmark.py --summarize --out-dir runs/bot/triplo-v9-open/screen
# identity on live spots: the digests must match
$PY -I $T/scripts/v9_preflop_replay.py LOG --tree ~/.local/share/triplo/releases/9342601 --out v8.json
$PY -I $T/scripts/v9_preflop_replay.py LOG --compare v8.json --out off.json
```

The benchmark and replay read the policy-only checkpoint cache
(`scripts/v6_field_benchmark.py` `CHECKPOINT_CACHE`, sha256 of the live checkpoint d553036c…).
Results of this round: `runs/bot/triplo-v9-open/` (not committed).

### Where to look next

The translated policy loses in small pots: hero continues preflop and then gives up postflop.
A likely reason (not tested here) is that the postflop fallback weights a preflop raiser's range
toward strong hands (`equity_policy.preflop_weight`), which a 60-70% opener is not, so wider
preflop defense may only pay together with postflop ranges for wide openers.

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
tree, which is the leak above rather than something live opponents share. Their code is on the
local branch `archive/triplo-v9-fish-flags`.
