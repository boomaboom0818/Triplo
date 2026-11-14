# OpenPoker client

Run from the repository root with a completed six-player Holdem CFR checkpoint:

```bash
uv pip install --python .venv/bin/python '.[openpoker]'
# Set OPEN_POKER_API_KEY securely in your environment, then:
.venv/bin/python -m scripts.openpoker_client --checkpoint /path/to/completed-v2.json
```

The client only reads the checkpoint once; it never writes training artifacts.
Requires `texture-buckets-v2`. Exact-observation and v1 checkpoints need historical
state reconstruction and are rejected. Training uses 1/2 blinds; live chip amounts
are rounded to this scale, while outgoing raises retain advertised chip bounds.
This is an approximation, not evidence of live playing strength. Unseen or zero-weight
states check if possible, otherwise fold, otherwise call. Coverage warnings identify
these decisions. Heads-up hands and incomplete snapshots are rejected.

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
