# Triplo v6 (v6-fast and v6-tight)

Triplo v6 adds two opt-in fallback switches, `--defend-3bet` and `--flop-peel`.
They close the first two of three leaks that a HUD-driven OpenPoker regular
(Doppio) was exploiting. There are two builds:

| Build | Base | Command line |
|---|---|---|
| **v6-fast** (was "v6") | v5-fast | v5-fast flags + `--defend-3bet --flop-peel` |
| **v6-tight** | v5 | v5 flags (`--fallback equity-guarded-v5 --river-solver`, 50 BB buy-in, learned postflop) + `--defend-3bet --flop-peel` |

v6-tight is the conservative build: no Stack Press, no Deep Value sizing, the
learned policy still plays the postflop spots it covers, and it buys in at 50 BB.

| Leak (Doppio's HUD on v5-fast) | v6 switch |
|---|---|
| Fold to 3-bet 89% while opening 29% | `--defend-3bet` |
| Folds 57% to flop c-bets, then rarely folds turn/river | `--flop-peel` |
| Almost never bluffs postflop | not shipped (see below) |

```bash
# v6-fast
.venv/bin/python -m scripts.openpoker_client --checkpoint runs/bot/pressure-fresh-001/base-guard.json \
  --min-visits 64 --buy-in 5000 --fallback equity-guarded-v5 --stack-press --deep-value \
  --no-learned-postflop --river-solver --opponent-stats runs/bot/opponent-stats.json \
  --defend-3bet --flop-peel

# v6-tight
.venv/bin/python -m scripts.openpoker_client --checkpoint runs/bot/pressure-fresh-001/base-guard.json \
  --min-visits 64 --buy-in 1000 --fallback equity-guarded-v5 --river-solver \
  --opponent-stats runs/bot/opponent-stats.json \
  --defend-3bet --flop-peel
```

Dropping the last two flags gives v5-fast or v5 exactly; both default off.

## `--defend-3bet`

Before v6, once our open was 3-bet, `PressureGuard` overrode the learned row
with the tight-v1 rule. That folded everything below a Chen score of 9 to any
3-bet larger than 3 BB, and `TT/99/AQ` to anything over 10 BB.

With `--defend-3bet` (`GuardedEquityPolicy._act_preflop_defend_3bet`), the
guarded fallback answers the 3-bet itself, and `PressureGuard` defers to it:

- `AA/KK` never fold and 4-bet when the raise is still small.
- `QQ/JJ/AKs` keep the 200-chip (100 BB) commitment cap.
- After our own open, every other hand gets a total-commitment cap set by hand
  class and position (`_preflop_commit_limit`), plus a pot-odds ceiling
  (0.45 for the strongest tier, down to 0.36 out of position). Pairs, suited
  aces, suited connectors and broadways defend; offsuit junk still folds.
  Small pairs defend only with deep enough stacks for set value.
- A cold 3-bet (someone else opened) stays tight: only Chen 10+ at a fair price.
- Against a 4-bet or more, hands below `QQ/JJ/AKs` continue only with Chen 10+,
  up to 60 chips and a 0.36 price.

`--defend-3bet` works the same way in both builds: v5 and v5-fast share the
learned preflop policy and `PressureGuard`.

## `--flop-peel`

v5-fast needed about 45% raw equity to call a half-pot flop bet (0.90 realization
discount, +0.05 margin and the Monte Carlo error). With `--flop-peel`, when the
normal check fails on the flop, facing a bet of up to 0.78 pot with at most two
opponents and no raise, it also calls with:

- a flush draw or open-ended straight draw (8+ clean outs) if equity covers the price;
- a gutshot or any pair against bets up to 0.67 pot, with a milder realization discount;
- no pair and no draw only heads-up against bets up to 0.55 pot.

Turn and river decisions are unchanged.

In v6-fast the peel runs inside the Deep Value postflop path. In v6-tight
(no `--deep-value`) the same check runs on the regular v5 fallback paths,
heads-up and multiway (`GuardedEquityPolicyV4._after_postflop_baseline`): a
baseline flop fold facing a bet gets the peel check, and side-pot pricing
still has the last word, as in the Deep Value path. Flop spots that the
learned policy covers are not changed; on live v5 the fallback made 29 of the
35 flop decisions facing a bet.

## Not shipped: `semi_bluff`

The third switch (`semi_bluff`, selective flop/turn semi-bluffs only against
measured non-sticky opponents) is in the code with tests but has no CLI flag.
Alone it lost a little in real_mix (−6.2), pressure (−9.2) and turbo (−8.3)
BB/100, all inside their intervals, and all three switches together were −13.7
against turbo. It needs a better bluff selection before it can ship.

## Evidence

All runs use the fast simulator, the live checkpoint (`base-guard.json`,
`min_visits=64`) and a 250 BB hero. Intervals are 95%. Raw reports are in
`runs/bot/v6-crossplay-001/` (not committed).

### v6-fast: five fields, v6-fast minus v5-fast (`scripts/v6_field_benchmark.py`)

4,096 paired deals per field, unequal stacks, measured opponent profiles:

| Field | v5-fast BB/100 | v6-fast − v5-fast |
|---|---|---|
| sticky (5× `LooseCaller(2)`) | +4,668.6 | 0.0 (no 3-bets or c-bets to react to) |
| real_mix | +403.7 | **+29.3 [+5.3, +53.2]** |
| pressure (5× aggressive) | +10.3 | **+40.0 [+13.3, +66.7]** |
| turbo (5× Ramsay) | −1.8 | −4.8 [−31.4, +21.8] |
| hud_exploiter (Doppio-style) | −30.5 | +18.2 [−11.2, +47.5] |

Cold start (no profiles), each switch alone: `--defend-3bet` real_mix +14.5
[−5, +34], pressure +22.9 [+2, +43], turbo −7.3 [−34, +20], hud_exploiter
+16.1 [−6, +38]; `--flop-peel` real_mix +14.6 [0, +29], pressure +19.9
[+1, +39], turbo −0.6 [−2, +1], hud_exploiter −2.2 [−9, +4]; sticky 0.0 for both.

Simulated HUD stats against the Doppio-style field, v5-fast → v6-fast: fold to 3-bet
89.9% → 59.4% (pressure 87.7% → 54.6%, turbo 94.6% → 57.1%, real_mix 95.6% →
64.6%); flop fold to c-bet 48.6% → 33.8%; turn fold 49.3% → 59.5%; river fold
21.9% → 28.8%; VPIP/PFR/3-bet unchanged (45.5/30.9/22.0). Part of the flop fold
moves to the turn, which a double-barreling opponent could still target.

### v6-fast head-to-head against v4, v5 and v5-fast (`scripts/triplo_v6_crossplay.py`)

Duplicate format: three seats of each bot, alternating, and every deal replayed
with the bots swapped, 4,096 deals (8,192 hands) per cell. The table is
zero-sum between the two bots. Each bot runs as its live command line sets it:
v4 = `equity-guarded-v4`; v5 = `equity-guarded-v5 --river-solver`; v5-fast and
v6-fast as above. Result is BB/100 per v6-fast seat:

| Mode | v6-fast vs v4 | v6-fast vs v5 | v6-fast vs v5-fast |
|---|---|---|---|
| measured, unequal stacks | **−8.5 [−15.3, −1.8]** | +1.0 [−7.3, +9.2] | **+7.9 [+1.9, +13.9]** |
| measured, equal 250 BB | +0.5 [−13.2, +14.2] | −7.6 [−18.9, +3.8] | +7.6 [−3.4, +18.7] |
| cold, unequal stacks | −4.9 [−12.5, +2.6] | −1.7 [−9.1, +5.7] | −0.5 [−5.5, +4.5] |
| cold, equal 250 BB | +4.7 [−8.4, +17.8] | +8.0 [−2.0, +18.0] | +3.8 [−4.4, +12.0] |

Control: v5-fast in v6-fast's seat on the same deals (`--hero v5_fast`), and the
paired v6-fast − v5-fast difference against the same rival:

| Mode | v5-fast vs v4 | v6-fast − v5-fast (vs v4) | v5-fast vs v5 | v6-fast − v5-fast (vs v5) |
|---|---|---|---|---|
| measured, unequal | **−5.2 [−8.8, −1.6]** | −3.3 [−9.1, +2.5] | −4.9 [−10.4, +0.6] | +5.8 [−1.0, +12.6] |
| measured, equal | −4.8 [−15.2, +5.7] | +5.3 [−4.3, +14.9] | −3.3 [−10.3, +3.6] | −4.3 [−13.7, +5.2] |
| cold, unequal | **−5.3 [−10.1, −0.5]** | +0.4 [−5.8, +6.5] | −0.2 [−4.6, +4.1] | −1.5 [−7.8, +4.8] |
| cold, equal | −2.1 [−11.7, +7.5] | +6.8 [−2.8, +16.5] | +3.2 [−3.1, +9.5] | +4.8 [−3.4, +13.0] |

The loss to v4 at unequal stacks was already there in v5-fast; v6-fast does not
significantly change it in any mode. v6-fast beats v5-fast directly in the measured
unequal-stack mode, and no cell shows v6-fast significantly behind v5-fast or v5.

A larger rerun on fresh seeds (2026-11-24, 32,768 duplicate deals per cell,
`runs/bot/v6-vs-v5fast-002/`, not committed) confirmed the edge, smaller:
v6-fast vs v5-fast +3.8 [+1.9, +5.7] measured unequal, +1.9 [−1.7, +5.4]
measured equal, +2.6 [+0.7, +4.5] cold unequal, +5.1 [+1.9, +8.3] cold equal.
Two short-stack cells aimed at a live weak spot also favored v6-fast: +1.7
[+0.5, +3.0] with the hero at 250 BB against mostly 35–75 BB opponents, and
+2.7 [+1.3, +4.0] with every seat at 60 BB.

The same script also runs a substitution test (one hero seat among five copies
of v4, v5, v5-fast or a mix, 2,048 deals per cell). Its intervals are ±30–80
BB/100, too wide to separate the bots; no v6-fast − v5-fast cell was significant.

### v6-tight head-to-head

Same duplicate format, `--hero v6_tight --rivals v5 v6_fast --tests h2h
--h2h-deals 8192` (8,192 deals, 16,384 hands per cell; raw report in
`runs/bot/v6-tight-crossplay-001/`, not committed). Every seat starts at the
script's 250 BB hero stack, so these cells do not test v6-tight's 50 BB live
buy-in. BB/100 per v6-tight seat:

| Mode | v6-tight vs v5 | v6-tight vs v6-fast |
|---|---|---|
| measured, unequal stacks | **+5.0 [+1.3, +8.8]** | −0.6 [−5.4, +4.3] |
| measured, equal 250 BB | −0.5 [−6.6, +5.6] | +4.4 [−2.5, +11.4] |
| cold, unequal stacks | +0.1 [−3.9, +4.2] | **+8.1 [+4.4, +11.9]** |
| cold, equal 250 BB | +1.8 [−3.8, +7.5] | +2.3 [−3.8, +8.3] |

v6-tight beats v5 in the same mode where v6-fast beat v5-fast (measured,
unequal stacks) and is not significantly behind v5 or v6-fast in any cell.
Head-to-head between Triplo builds says little about soft live fields, where
v5-fast's Deep Value sizing earned most of its edge.

### Limits

These are simulations against scripted fields and other Triplo versions, not
live OpenPoker results. The real Doppio adapts; the scripted `hud_exploiter`
does not. Confirm live, and roll back either build by dropping
`--defend-3bet --flop-peel` (v6-fast → v5-fast, v6-tight → v5).
