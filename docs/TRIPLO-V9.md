# Triplo v9: five candidates, none beat v8

Triplo v9 adds a lower-variance A/B harness and four opt-in improvements on top of v8-fast. Each
was screened in simulation and the promising ones were re-tested on fresh deals. **None cleared the
deploy bar**, so the live bot stays on v8-fast. Every v9 flag is off by default, and with all of them
off v9 makes exactly v8's decisions: 0.0% of hands changed on 49,152 simulated deals, and the same
action on 1,112 replayed live postflop spots.

| # | Improvement | Flag | Verdict |
|---|---|---|---|
| 1 | Capped A/B metrics (`poker/ab_metric.py`, `scripts/ab_report.py`) | (tooling) | Kept; used below and for live tracking |
| 2 | Bigger opens / isolation raises when every player left to act is a fish | `--fish-open BB` (`--fish-open-limper BB`) | Rejected: neutral to negative |
| 3 | No Deep Value overbets while a regular remains | `--fish-value-cap POT` | Rejected: loses big pots |
| 4 | Small heads-up flop stab when a fish checks to us (exact fraction) | `--fish-stab POT` (`--fish-stab-hands`) | Rejected: negative |
| 5 | Raise instead of open-limping with top hands | `--raise-not-limp SHARE` | Rejected: did not replicate |

All four flags need `--fish-exploit`.

## Deploy bar

Deploy a flag only if it beats v8 on paired simulated deals in **uncapped** BB/100 by more than
noise (the 95% interval excludes zero, live-weighted over table types) and passes three more
checks:

- it does not lose money in big pots;
- it does not shift decisions onto the fallback;
- the unit tests stay green.

Uncapped is the gate because Triplo's profit is concentrated in big pots. In live v8 play
(1,333 hands since 2026-11-25 03:44 UTC) the bot made +266 BB/100 uncapped, -3 capped at ±25 BB
and +56 capped at ±50 BB, and pots of 50 BB or more carried +312 BB/100. Capped metrics are only
for screening changes that leave big pots alone. #3 shows why: the ±25 BB cap rated it
+0.9 while it lost 10 BB/100 uncapped.

## Method

`scripts/v9_field_benchmark.py` extends the v8 fish-table benchmark:

- `NoCodeFish` seats (fitted to live `no_code` bots, live stack sizes);
- v7-fast regulars;
- hero at 250 BB.

Every hand is recorded with its final pot, its decision sources and which v9 rule fired.

Each arm plays the same deals as v8. Screens used 12,288 paired deals per table type
(seeds 930M-933M); confirmations used 12,288 fresh deals per type (seeds 950M-953M). Results are
weighted by the live table mix: 5 fish 4%, 4 fish + 1 regular 48%, 3 + 2 40%, 2 + 3 8%. The tables
below give the arm minus v8 in BB/100, with 95% intervals.

### Screens

| Arm | uncapped | cap ±50 | cap ±25 | hands changed |
|---|---|---|---|---|
| `open=3` | -0.6 [-4.8, +3.5] | -0.6 [-3.1, +1.8] | -1.0 [-2.4, +0.4] | 3.2% |
| `open=3.5` | -2.8 [-6.7, +1.0] | -1.8 [-4.2, +0.6] | -2.0 [-3.4, -0.5] | 3.2% |
| `open=4` | -5.0 [-9.2, -0.9] | -3.3 [-5.8, -0.8] | -2.4 [-3.9, -0.9] | 3.2% |
| `open=3.5,lim=0.5` | -1.8 [-5.7, +2.0] | -1.2 [-3.6, +1.2] | -1.5 [-2.9, -0.1] | 3.2% |
| `open=3.5,lim=1.5` | -3.0 [-6.9, +0.9] | -2.0 [-4.5, +0.5] | -2.1 [-3.6, -0.6] | 3.2% |
| `open=3.5,scope=majority` | +6.0 [-7.9, +19.9] | +6.6 [+0.0, +13.2] | +2.5 [-1.5, +6.5] | 14.8% |
| `stab=0.33` | -4.6 [-8.5, -0.7] | -3.4 [-6.2, -0.5] | -1.4 [-3.6, +0.7] | 3.5% |
| `stab=0.5` | -4.7 [-9.1, -0.4] | -3.9 [-7.1, -0.7] | -2.7 [-5.0, -0.4] | 3.5% |
| `stab=0.33,stabh=nopair` | -2.9 [-5.8, +0.1] | -2.1 [-4.4, +0.1] | -0.7 [-2.5, +1.0] | 2.5% |
| `stab=0.5,stabh=nopair` | -4.5 [-7.6, -1.4] | -3.9 [-6.3, -1.4] | -2.7 [-4.5, -0.8] | 2.5% |
| `cbet=0.5` (v8's unused menu-snapped c-bet, reference) | -6.4 [-12.5, -0.4] | -5.7 [-10.2, -1.2] | -3.3 [-6.5, -0.1] | 5.3% |
| `cap=0.75` | -10.2 [-18.6, -1.9] | +0.2 [-2.6, +2.9] | +0.9 [-0.5, +2.2] | 2.8% |
| `cap=1.0` | -8.5 [-15.2, -1.8] | -0.7 [-3.1, +1.7] | +0.1 [-1.1, +1.4] | 2.8% |
| `limp=0.25` | +6.3 [-3.1, +15.7] | +1.5 [-3.1, +6.1] | +0.3 [-2.4, +3.1] | 4.6% |
| `limp=0.25,limps=fish` | -0.9 [-3.6, +1.7] | -0.5 [-2.0, +1.0] | -0.3 [-1.1, +0.6] | 0.7% |
| `limp=0.1` | +5.2 [-2.5, +12.9] | +2.3 [-1.4, +6.0] | +1.6 [-0.5, +3.8] | 2.3% |

Robustness checks:

- **Value cap with live-style regulars.** Regulars given the sticky profile that live
  `--opponent-stats` measures (`--reg-profile live`, three tables with regulars): `cap=0.75`
  -8.3 [-16.1, -0.5], `cap=1.0` -6.2 [-12.6, +0.3].
- **Fish that fold more to bigger opens**, as live (`--fish sized`, 5-fish and 4+1 tables):
  `open=3` +1.5 [-6.0, +9.0], `open=3.5` -5.8 [-12.5, +1.0].

Big pots (final pot of 50 BB or more), paired contribution:

| Arm | Big-pot contribution |
|---|---|
| `stab=0.33` | -6.7 [-10.2, -3.1] |
| `cap=0.75` | -10.8 [-19.1, -2.5] |
| `cap=1.0` | -8.8 [-15.5, -2.1] |
| `limp=0.1` | +4.6 [-3.1, +12.3] |
| `limp=0.25` | +5.7 [-3.6, +15.1] |

Shift onto the fallback (postflop is always the fallback under `--no-learned-postflop`):

- No arm moved the hero's preflop non-learned share by more than 0.1 point, except `scope=majority`
  (-2 to -2.6 points: fewer 3-bets to face).
- Postflop decisions changed by at most +1.4% (`cap`); stabs end hands earlier (-3 to -5%).

### Confirmation on fresh deals

| Arm | uncapped | cap ±50 | cap ±25 | big pots |
|---|---|---|---|---|
| `limp=0.1` | -5.6 [-13.8, +2.7] | -1.1 [-4.4, +2.2] | -0.6 [-2.5, +1.4] | -6.1 [-14.3, +2.2] |
| `limp=0.25` | -2.9 [-13.0, +7.3] | -1.3 [-5.6, +3.0] | -0.6 [-3.3, +2.1] | -3.7 [-13.9, +6.4] |
| `open=3.5,scope=majority` | +13.1 [+0.2, +26.0] | +5.0 [-1.4, +11.3] | +1.6 [-2.3, +5.5] | +14.7 [+1.9, +27.5] |
| `limp=0.1` + `open=3.5,scope=majority` | +10.3 [-5.0, +25.6] | +4.3 [-2.9, +11.5] | +1.5 [-2.8, +5.9] | +12.1 [-3.1, +27.3] |

Raise-not-limp flipped sign on fresh deals: `limp=0.1` scored +7.2 on 3+2 tables in the screen
and -16.6 [-30.7, -2.6] on fresh ones. Its screen result was noise.

## Why `scope=majority` was not deployed

`open=3.5,scope=majority` is the only arm whose fresh-deal interval excludes zero. It also opens
bigger when one or two regulars are still to act, as long as fish are the majority. The gain does
not come from fish. The same open restricted to all-fish spots (`open=3.5`) is neutral to negative
at every table.

A diagnostic on 1,500 deals at 3-fish + 2-regular tables shows where it does come from. The
simulated regulars are Triplo v7-fast copies:

| | Facing a 2.5 BB open (v8) | Facing a 3.5 BB open |
|---|---|---|
| Learned-policy decisions | 96% | 15% (the rest go to `preflop_rules`) |
| Fold | 50% | 84% |
| 3-bet | 30% | 9% |

The resized hands with a regular behind added +29.9 BB/100; the all-fish ones added +3.3. The
arm wins by pushing the simulated regulars off their trained tree. Live regulars are other
people's bots, so the simulation cannot predict this gain; it measures a weakness of our own
regular model. Deploying it would be a guess about unknown opponents, so it stays off.

### The same weakness in live Triplo

Triplo's own live play (season 17 session log, first preflop decision facing a single open) shows
the same pattern:

| Open size | Decisions | Fold | Call | Raise | Source |
|---|---|---|---|---|---|
| ≤ 2.5 BB | 4,373 | 54% | 19% | 27% | 87% learned |
| 3 BB | 1,600 | 71% | 18% | 11% | 64% `preflop_rules` |
| 3.5-4 BB | 468 | 92% | 6% | 3% | 96% `preflop_rules` |
| > 4 BB | 333 | 97% | 2% | 1% | 96% `preflop_rules` |

Opens of 3 BB or more make up 35% of these spots. Mapping off-menu open sizes onto the trained
tree, or a defend range against bigger opens, is the most promising next change. Any such fix
must be tested against opponents that open bigger.

## Live tracking

`scripts/ab_report.py` reads the client's session log. It reports:

- capped and uncapped BB/100 with 95% intervals;
- the share of positive 10-minute windows;
- the big-pot breakdown;
- the leaderboard tracker's rate for the same window.

```bash
.venv/bin/python <tree>/scripts/ab_report.py --buy-in 5000 \
  --arm v8=2026-11-25T03:44:36Z --leaderboard runs/bot/leaderboard-track.jsonl
```

Live v8 so far: 1,333 hands at about 75 hands per hour. It made +266 [+102, +431] BB/100 uncapped
(SD 30.6 BB per hand), 54% of 10-minute windows were positive, and the tracker shows +232 BB/100
over 17.8 hours.

At that SD:

- Pinning one version to ±35 BB/100 takes about 29,000 hands (16 days).
- An unpaired comparison of two versions to ±35 BB/100 takes about 59,000 hands per version.

Only large live differences can be resolved; small effects must be settled in simulation first.

## Reproduce

The four table types run one after another, so a single run with seed 930000000 uses the seeds
listed above. Results are saved under `runs/bot/triplo-v9-sim/` (not committed).

```bash
python scripts/v9_field_benchmark.py --deals 12288 --workers 10 --seed 930000000 \
  --profiles-cache fish-profiles.json --out screen.json \
  --arms v8 off open=3 open=3.5 open=4 open=3.5,lim=0.5 open=3.5,lim=1.5 open=3.5,scope=majority
python scripts/v9_field_benchmark.py --deals 12288 --seed 930000000 ... \
  --arms v8 stab=0.33 stab=0.5 stab=0.33,stabh=nopair stab=0.5,stabh=nopair cbet=0.5
python scripts/v9_field_benchmark.py --deals 12288 --seed 930000000 ... \
  --arms v8 cap=0.75 cap=1.0 limp=0.25 limp=0.25,limps=fish limp=0.1
python scripts/v9_field_benchmark.py --deals 12288 --seed 950000000 ... \
  --arms v8 limp=0.1 limp=0.25 open=3.5,scope=majority limp=0.1,open=3.5,scope=majority
PYTHONPATH=. python scripts/v9_live_replay.py runs/bot/openpoker-season-17-equity/session.jsonl \
  --since 2026-11-25T03:44:36Z --out replay.json   # identity check: every v9 option off
```

## Limits

- The regulars in the simulation are Triplo v7-fast copies, which is why `scope=majority` cannot
  be judged here.
- The fish model reproduces live frequencies. Its hand selection within each frequency is an
  assumption.
- Raise-not-limp and bigger opens change which hands reach the learned tree. The simulation
  covers that for the hero, but not for real opponents' responses.
