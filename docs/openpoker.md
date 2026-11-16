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
