# Rule-opponent sanity gate

`python -m scripts.rule_gate --checkpoint PATH --deals 1500 --out REPORT.json`

One checkpoint seat (with the live check/fold fallback) plays five copies of each
fixed rule opponent in `poker/rule_opponents.py`. Every deal is replayed with the
checkpoint in all six seats; the unit of analysis is the deal.

| Style | Behaviour |
|---|---|
| `tight-v1` | Plays roughly the top fifth of hands, calls all-ins only with premiums, bets made hands. |
| `aggressive-v1` | Opens wider, re-raises opens with strong hands, stabs at checked flops. |
| `station-v1` | Never raises; calls preflop with playable hands and never folds a pair. |

**Pass:** the lower end of the 95% interval is above 0 bb/100 against every style.
Always folding scores about -25 bb/100, printed for reference.

Why it exists: on 2026-11-15 the v10 checkpoint passed every sibling-checkpoint and
scripted-population gate yet lost about 388 bb/100 to `tight-v1`. Those gates use
opponents that mostly fold or mostly call, which reward shoving.

Passing is necessary, never sufficient. The rule opponents are weak and fixed;
a pass is not evidence of competitive strength. Do not tune a checkpoint against
this gate's seed (880000); use it as a check, not a training target.
