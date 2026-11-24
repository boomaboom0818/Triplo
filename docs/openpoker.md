# OpenPoker client

Run from the repository root with a completed six-player Holdem CFR checkpoint:

```bash
uv pip install --python .venv/bin/python '.[openpoker]'
# Set OPEN_POKER_API_KEY securely in your environment, then:
.venv/bin/python -m scripts.openpoker_client --checkpoint /path/to/completed-v2.json
```

The client only reads the checkpoint once; it never writes training artifacts.
Accepts `history-public-v1`, `texture-buckets-v2` and `lean-v3` checkpoints, including the experimental
loose-caller mixture policies. Exact-observation and v1 checkpoints need historical
state reconstruction and are rejected. Training uses 1/2 blinds; live chip amounts
are rounded to this scale, while outgoing raises retain advertised chip bounds.
This is an approximation, not evidence of live playing strength. Unseen or zero-weight
states in the older CFR policy check if possible, otherwise fold, otherwise call. Coverage warnings identify
these decisions. Heads-up hands and incomplete snapshots are rejected.

Lean policies use their trained raise menu and a `tight-v1` fallback for unseen
states. Add `--min-support 0.0001` to also use that fallback for sparse average
rows; default zero preserves the original policy. This normalized average-mass
threshold is a heuristic, not a visit count. Decision logs correctly mark fallback
decisions as uncovered. See [the sparse-support experiment](LEAN-SPARSE-SUPPORT.md).
For diagnostic sessions, `--log PATH --max-minutes 5` records local decisions and
hand results and exits after the first completed hand past the time budget. This
is a hand-boundary stop, not a hard connection timeout.

Buy-in defaults to 2000 chips (`--buy-in 1000..5000`). Add `--auto-rebuy` to enable
server-managed rebuys. No live connection is made by installation or tests.
Stop with Ctrl-C; server disconnect grace rules apply.

The adapter requests an authoritative resync on each turn rather than merging
partial game events. It uses the resync's current hand ID, token, private cards,
fold flags and action menu together. Duplicate tokens reuse the same action payload.
Reconnect performs the active-game REST lookup before joining. Authentication failures
stop the client; transport failures retry with bounded backoff. Live behavior has not
been verified with an authenticated account. Public league only.

Protocol: https://docs.openpoker.ai/api-reference/websocket-protocol/
Message fields and recovery: https://docs.openpoker.ai/llms-full.txt

History checkpoints use sequenced public events to reconstruct betting history.
Learned rows require verified replay plus `--min-visits` (default64). Missing or
inconsistent history uses the rule fallback and is recorded in the decision log.
Reconnect requests now retain sequence watermarks and replay events before using
the final snapshot. See [the river/history report](RIVER-003.md) for limitations
and a short diagnostic command. Bounded river checkpoints are local-only and are
not accepted by the live client.

## Triplo v5

Use a history checkpoint and `--fallback equity-guarded-v5` for the v5 policy.
V5 defaults to a 1,000-chip buy-in (50 BB at 10/20 blinds) and uses its
calibrated heads-up river solver on eligible fallback decisions with verified
history. The solver requires the optional `river-solver` extra:

```bash
uv pip install --python .venv/bin/python '.[openpoker,river-solver]'
.venv/bin/python -m scripts.openpoker_client --checkpoint runs/bot/pressure-fresh-001/base-guard.json --fallback equity-guarded-v5
```

For a caller-heavy field, select the deeper buy-in explicitly with
`--buy-in 2500` or `--buy-in 5000`. Those presets default to the guarded v4
betting behavior. `--river-solver` enables the combined variant at deeper
buy-ins. `--no-river-solver` runs the solver-off control at any buy-in.
The switch is set when the client starts and applies throughout the session.
Eligible decisions log the solver probability, proposed action and baseline
inside `equity_fallback.river_solver`.

## Triplo v6-fast and v6-tight

Both builds add `--defend-3bet --flop-peel`. **v6-fast** (previously just "v6")
adds them to the v5-fast command line; **v6-tight** adds them to the v5 command
line (50 BB buy-in, learned postflop, no Stack Press or Deep Value):

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

`--defend-3bet` (v4/v5 fallbacks, history checkpoints) defends our opens against
3-bets by price and position instead of folding every non-premium hand.
`--flop-peel` (v4/v5 fallbacks) calls small and medium flop bets more often with
draws and pairs when the fallback would otherwise fold; it works with or without
`--deep-value`. Both are off by default. The startup log line shows
`Defend 3-Bet` and `Flop Peel`. Design and evidence: [Triplo v6](TRIPLO-V6.md).

## Leaving tables when strong opponents sit down (opt-in)

Ported from Triplo Turbo's "apex shark" rule. Off by default; OpenPoker only.

```bash
# Log what it would do, without leaving:
.venv/bin/python -m scripts.openpoker_client --checkpoint ... --log runs/bot/x/session.jsonl \
    --avoid-strong report --strong-bb100 300
# Actually leave and requeue:
.venv/bin/python -m scripts.openpoker_client --checkpoint ... --avoid-strong leave --strong-bb100 300
```

An opponent is **strong** when their public season result, shrunk toward zero
with a 500-hand prior, is at least `--strong-bb100` (default 220, Turbo's apex
bar; requires 200+ hands). An opponent is **weak** when they are a `no_code`
preset or at most −15 BB/100. The client stays while weak seats are at least as
many as strong ones, and leaves otherwise. Unrated players and bots on your own
account count as neither.

The lineup is checked when the bot is seated, when a player joins, and after each
hand. A leave sent when a player joins mid-hand is held by the server until that
hand ends. After a leave the client waits 10 s before rejoining the lobby, so it
is less likely to be seated back at the same table.

Safeguards:
- It acts only on a leaderboard read from the last 60 s, fetched in the
  background every 30 s, never on the decision path.
- It leaves only when your stack covers `--buy-in`. Leaving cashes out the whole
  stack, and the bot rejoins at `--buy-in`.
- After a leave it waits 30 s before leaving again, doubling with each
  consecutive leave up to 10 min. Four hands at one table reset that backoff.

Each check that finds a strong opponent is logged as `strong_opponent_check`, with
`leave_requested` showing whether a leave was sent.

Evidence so far (2026-11-23) is suggestive, not significant. On 12,661 live
Triplo hands from 9 tables, hands that the 300 bar would have left won −81
BB/100 (95% interval −261 to +87), while the rest won +124. Opponents were rated
with the later leaderboard, which favors the rule, and live results came mostly
from earlier models. Run `report` first. Switch to `leave` only if flagged hands
stay clearly below unflagged hands across at least 20 tables.

Turbo's other table rules (its table-profitability score and leaving with a big
stack when a deep shark sits down) were evaluated and not ported. On the same
hands, the tables they would have left were profitable.

## Live replay repair and equity fallback (2026-11-16)

Captured live traffic showed that a `player_action` closing a round can carry the
**resulting** street (for example the final preflop call is labeled `flop`).
Replay now derives the acting street from the validated ledger and accepts either
the acting or resulting label only if the action actually produces that transition.
Actor, pot, contribution, stack, fold and final-snapshot checks remain mandatory.
Short all-in calls are replayed as calls, not raises. A sanitized real capture is
retained in `tests/fixtures/openpoker-street-transition.json`.

`--trace PATH` optionally saves public events and resync snapshots, including the
bot's own hole cards, with action-authority/authentication tokens removed. Keep
these diagnostic files private. `--log` now includes acknowledgment latency,
server rejection codes and transport exception/close details. Rejected action IDs
are not resent on a repeated token; stale authority requests a resync. Connection
backoff resets on an accepted action rather than just on socket authentication.

`--fallback equity-v1` selects the new bounded Monte Carlo fallback for history or
lean checkpoints. The default remains `tight-v1`; changing the fallback does not
change trained checkpoint weights or the visit gate. Equity inference consumes
only the observation, uses broad public-history range priors, samples compatible
joint opponent holdings and board runouts, and estimates showdown pot share
(including ties). It uses pot odds and a conservative equity-realization margin,
retains supported made-hand value bets, and preserves the old preflop baseline.
It does not implement opponent learning, future betting search, or side-pot EV.
Its sampling error is not uncertainty about the correctness of its range priors.

Run the paired screen on fresh seeds (100 and 50 BB, four fixed rule styles,
six seat rotations per deal):

```sh
.venv/bin/python -m scripts.equity_fallback_check --seed 40164000 --deals 1000 \
  --out runs/bot/equity-fallback-new.json
```

Reports retain source hashes, per-deal outcomes, paired differences, pointwise
95% normal intervals, and postflop decision latency. These are weak fixed-opponent
screens, not proof of OpenPoker profitability. Preserve reports and avoid tuning
on a confirmation seed and continuing to describe it as held out.

`--fallback equity-guarded-v1` selects an experimental fallback that counts total
preflop contributions when applying the tight chart's commitment limits, and
requires hole cards to improve the board's category for strong-hand value raises.
It also suppresses raises with weak made hands facing a bet or a prior postflop
reraise. Preflop caps require a complete ledger; the live adapter discards
unverified history before invoking this fallback. Learned checkpoint decisions
are unaffected. The default remains `tight-v1` and `equity-v1` remains the frozen
control. This candidate has not established improved live profitability.

The guarded fallback treats a paired board plus one hole-card pair as one pair,
does not re-raise a postflop reraise without usable equity >= 0.80, folds river
calls above 100 BB without two pair or better, and caps JJ/QQ preflop commitment
at 100 BB.

**Stack Press** (`--stack-press`, off by default; requires `--fallback
equity-guarded-v1`) lets strong hands with usable equity >= 0.80 raise up to the
whole stack, and adds 2x-pot and half-stack sizes to the menu when the hero has
more than 100 BB. Turn it on against loose or weaker fields and off against
tight, strong ones. The client log prints `Stack Press ON/OFF` at startup.
Compare variants with `python -m scripts.selfplay_ab` (duplicate self-play) or
`python -m scripts.live_gate evaluate --hero-fallback equity-guarded-v1+stack-press`.

Reproduce a paired development screen against the frozen equity fallback with:

```bash
python -m scripts.equity_fallback_check --guarded --deals 300 --seed 71661000 --out runs/bot/guarded-equity-screen.json
```

The initial 30-deal smoke screen in `runs/bot/guarded-equity-resume-screen-v2.json`
was inconclusive. It tests six seats against four fixed rule opponents at two
stack depths, not the nine-condition learned-policy promotion gate. No promotion
or live deployment follows from that smoke screen.

Decision records include `decision_source`: `learned_policy`, `poker_safeguard`,
`pressure_guard`, `preflop_rules`, `equity_fallback`, `rule_fallback`, or
`safety_fallback`. `covered` retains its existing meaning (a learned row supplied
the action without override). A safeguard or opt-in `decision_guard`
(`equity-pressure-v1` or `equity-pressure-v2`, exported via
`scripts/export_pressure_guard.py`) can supersede learned play, so `covered: false`
alone does not imply that the uncovered fallback was used. These diagnostics are
never sent on the wire.

The equity range model now classifies top pair by the rank actually paired with
the board, rather than the highest hole-card kicker. Overpairs remain in the
same weight tier. This corrects range evidence; it does not establish a win-rate
improvement or add sequential opponent modeling.
