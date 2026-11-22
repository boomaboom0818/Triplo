# Triplo v4 (`equity-guarded-v4`)

`equity-guarded-v4` is the successor fallback to `equity-guarded-v3` (`--fallback equity-guarded-v4`).

## Design: Fair-share multiway postflop betting thresholds

In `equity-guarded-v3`, postflop betting thresholds only distinguished heads-up
(`opponents == 1`: `0.50` baseline, `0.68` general value, `0.85` big size) from
multiway (`opponents >= 2`: flat `0.40`, `0.60`, `0.85`). In 3-to-6-player pots,
fair-share equity is `1 / (opponents + 1)` (`0.333` down to `0.167`), so `v3`
checked strong multiway favorites (including top pair and two pair+ with 30–39%
usable equity against 3–5 opponents).

`GuardedEquityPolicyV4` (`poker/equity_pricing.py`) fixes this scaling while
preserving every preflop and heads-up `v3` decision verbatim (including `v3`'s
calibrated ranges, side-pot pricing, `value_size` with `--opponent-stats`,
Late-Position Steals support, and all poker safeguards):

- **Preflop & heads-up postflop (`opponents <= 1`):** identical to
  `equity-guarded-v3`.
- **Multiway postflop (`opponents >= 2`, `N = opponents + 1`):**
  - Baseline (`strong` / `top_pair`) threshold: `1.0 / N` (`0.333`, `0.250`,
    `0.200`, `0.167` for 3–6 players). When a newly unlocked multiway baseline
    bet has `usable < 0.40`, it uses the smallest legal size (`raises[0]`,
    half-pot on the lean menu) rather than full pot/2× pot, keeping risk
    bounded before `value_size` runs.
  - General value threshold: `0.68 * (2.0 / N) ** 0.5`.
  - Big-size threshold: `0.85 * (2.0 / N) ** 0.35`.

## Untouched holdout confirmation (16 opponent scenarios)

Pre-registered criteria before spending reserved seed `910_000_000`: positive
95% lower bound on the 16-scenario aggregate paired delta vs. `v3`, and **zero**
scenarios with a statistically significant regression (95% upper bound < 0).

**Passed (`runs/bot/v4-multiway-confirm-001/holdout.json`):** across **3,000
paired deal blocks per scenario (~132,000 hands total)** using the live
checkpoint (`base-guard.json`, `min_visits=64`) and branch-on-difference paired
evaluation:

- **16-scenario aggregate:** **+6.21 ± 2.79 BB/100** (95% CI **[+3.42, +9.00]**)
- **Changed hands:** 713 (285 improved)
- **Statistically significant wins (3):**
  - `loose-100` (`LooseCaller` mix): **+70.11 ± 35.36** `[+34.75, +105.47]` (347 changed)
  - `scripted-prof` (`ScriptedOpponent` with `--opponent-stats` profiles): **+25.22 ± 14.79** `[+10.43, +40.01]` (90 changed)
  - `bluff-river` (45% river bluff frequency): **+10.69 ± 9.35** `[+1.34, +20.05]` (35 changed)
- **Statistically significant regressions:** **0 / 16** (`tight-50`: +1.77,
  `agg-50`: −0.09, `tight-100`: −0.85, `agg-100`: −0.04, `station-100`: −1.76
  ± 8.54, `shover-100`: 0.00, `unequal-mix`: +0.02, `vs-v3-100`: +1.82,
  `vs-v3-200`: −3.17 ± 6.64, `vs-turbo-100`: −1.97 ± 7.52, `vs-turbo-200`:
  −1.92 ± 15.27, `maniac-shove`: +1.21, `incumbent`: −1.69 ± 3.42; every
  negative point estimate is within 1 standard error of zero).
- **Heads-up river parity:** exact **0.00 BB** difference from `v3` (zero
  heads-up action changes).
- **Real OpenPoker coverage:** 43 multiway check-to spots in recorded live
  sessions qualified (e.g., top pair in 3–6 player pots with 33–47% raw equity
  that `v3` checked).

## Earlier v4 experiments rejected for tradeoffs

Before selecting multiway fair-share scaling, the following ideas were tested on
development screens across the same 16-scenario suite and rejected because each
introduced regressions against at least one opponent class:

- **Heads-up river DCFR subgame solver:** gained +36.8 BB/100 vs. Turbo-100 and
  +14–40 BB/100 vs. `bluff-river`, but lost to `station-100` (−21.6), `agg-100`
  (−19.0), and `loose-100` (−17.8).
- **6-class de-biased Range Model v4:** improved real OpenPoker held-out exact
  combo log-loss by −0.071 nats (n=1,979), but regressed on `tight-100` (−12.4
  to −13.9 BB/100).
- **Extreme overbet polarization / overbet fold guard:** gained vs.
  `scripted-prof` (+39.0), but lost vs. `maniac-shove` (−52.2) and `bluff-river`
  (−36.9).
- **Smaller postflop value sizing / river thin-checking:** gained vs. `tight-100`
  (+11–13), but lost vs. `loose-100` (−76 to −104).
- **Preflop pot-odds call rescue:** gained vs. `incumbent` (+7.9) and `agg-100`
  (+6.6), with minor drag vs. `tight-50` (−3.2) and `tight-100` (−1.4).
