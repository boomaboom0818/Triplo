# OpenPoker replay repair and equity fallback

## Scope

The November 16, 2026 session had no verified postflop decisions in the inspected
125-decision postflop sample. A new redacted wire capture reproduced the cause:
OpenPoker labels a round-closing action with the resulting street. The replay
assumed every event named its acting street. A final preflop call labeled `flop`
therefore invalidated every subsequent history.

Replay now derives the acting street from its state and validates the reported
street against the before/after transition. All ledger and snapshot checks remain.
The regression fixture is an actual live capture with display metadata removed.
The client also records acknowledgment latency, rejection codes and connection
exception details, retains rejected IDs to avoid repeated stale submissions, and
only resets reconnect backoff after an accepted action. No specific root cause
for the earlier 24 transport disconnects has been established.

## Equity fallback

`poker/equity_policy.py` adds broad public-action range priors, compatible joint
holding sampling, uniform remaining-board runouts and tie-aware showdown equity.
It uses 256 deterministic samples per postflop observation, pot odds, conservative
sampling/realization margins, and supported baseline value bets. A river where the
board plays does not trigger a pointless value raise. The preflop baseline remains
unchanged. Select it with `--fallback equity-v1`; the CLI default stays `tight-v1`
for reproducibility of existing commands.

This is heuristic equity-based play, not a solver or opponent-adaptation system.
It does not model future betting or side-pot eligibility. Its Monte Carlo error
does not measure uncertainty in the assumed ranges.

## Evaluation

The first 100-deal screen exposed underbetting against calling stations. After
retaining equity-supported baseline value bets, the reused development screen
improved. A fresh 200-deal confirmation was mixed. The final unchanged policy was
then tested on another fresh seed, 40163000, with 1,000 deal blocks per case,
six seat rotations, four styles and two stack depths: **48,000 hands per policy**.

| Style | Stack | Tight fallback bb/100 | Equity fallback bb/100 | Paired difference, 95% interval |
|---|---:|---:|---:|---|
| Tight | 50 BB | 3.5 | 5.9 | +2.5 [-2.8, 7.8] |
| Tight | 100 BB | 5.1 | 18.8 | +13.7 [3.2, 24.2] |
| Aggressive | 50 BB | 1.9 | 8.7 | +6.9 [-4.6, 18.3] |
| Aggressive | 100 BB | 18.4 | 38.9 | +20.6 [-0.1, 41.2] |
| Calling station | 50 BB | 94.1 | 117.5 | +23.5 [12.0, 34.9] |
| Calling station | 100 BB | 145.8 | 180.8 | +35.0 [18.3, 51.8] |
| Shover | 50 BB | 13.8 | 13.8 | 0.0 [0.0, 0.0] |
| Shover | 100 BB | 23.8 | 23.8 | 0.0 [0.0, 0.0] |

Intervals are pointwise normal intervals over paired deal blocks, with no
multiple-comparison correction. Styles share deals and are not independent.
Shover cases had no hero postflop decisions, so equality is expected. Fixed rule
opponents are weak sanity checks; these results do not establish live profitability
or improvement for the full learned-policy/fallback combination. Postflop p95
latency was 4.7–6.4 ms; the maximum observed was under 19 ms in this experiment.

Artifacts (preserved locally under ignored `runs/bot/`):

- `equity-fallback-screen-001.json`: initial development screen.
- `equity-fallback-screen-002.json`: revised policy, reused development seed.
- `equity-fallback-confirmation-001.json`: first fresh confirmation (mixed).
- `equity-fallback-confirmation-002.json`: larger fresh confirmation, source hashes,
  per-deal samples and latency.
- `openpoker-season-17-fixed/`: live replay-fix verification with tight fallback.
- `openpoker-season-17-equity/`: deployment manifest, session, redacted trace and log.

The live deployment preserves the 876,520-iteration checkpoint, 64-visit gate,
1,000-chip buy-in, auto-rebuy and stop-on-season-end settings. Its postflop learned
coverage still depends on checkpoint support after successful history verification.

## Verification and activation

The broad regression run passed 390 tests and found one new equity edge-case
failure (raising a shared royal-flush board). That was repaired; the failed test
then passed on rerun. All 27 focused client/history/equity tests passed on final
code, including three tests added after broad collection. Final collection has
394 tests. `git diff --check` passed. Coverage includes captured live transitions,
missing/corrupt replay, short all-in calls, duplicate/rejected tokens, trace
redaction, pot-share ties, draws, nuts, multiway equity and deterministic sampling.

After loading the existing checkpoint before replacing the old connection, the
live equity deployment produced verified flop and turn histories, recorded equity
fallback diagnostics, and received accepted action acknowledgments (approximately
217–298 ms for its first three decisions). This verifies integration, not a live
win-rate improvement. Existing stop-on-season-end changes were preserved.
