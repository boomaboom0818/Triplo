# Faster range lookup, same `equity-guarded-v3`

This branch starts from `9347710` and does not include v4. It caches made-hand
classes by public street board instead of repeatedly evaluating equivalent hole
pairs. Per-opponent range weights still use the original v3 factors, in the same
multiplication and summation order, over the same unblocked pairs. The sampler,
seed, action rules, side-pot pricing, value sizing, and fallback name are unchanged.

The lookup groups hole cards by rank and, on boards where a flush is possible,
whether each hole card is of that suit. At most one suit can have three board
cards. Lookup memory is bounded at 64 tables of 2,704 bytes plus cache overhead.

## Replay result

**Zero differences** in 6,863 decisions (4,432 postflop), in both runs. This includes
744 integer raise actions, 37 side-pot pricing paths, and 28 value-sizing EV paths.
The comparison checked exact equities, standard errors, and the other diagnostics
as well as actions. Tests also compare the cached class to direct evaluation for
every compatible holding on 36 boards, and check exact cumulative range weights.

| Replay | Original v3 p50 / p99 / max | Cached v3 p50 / p99 / max |
|---|---:|---:|
| Run 1, all decisions | 10.18 / 44.36 / 79.11 ms | 2.87 / 6.98 / 17.02 ms |
| Run 1, postflop | 16.24 / 47.85 / 79.11 ms | 3.42 / 8.42 / 17.02 ms |
| Run 2, all decisions | 6.57 / 28.47 / 51.46 ms | 2.92 / 8.05 / 15.02 ms |
| Run 2, postflop | 10.51 / 29.64 / 51.46 ms | 3.49 / 8.98 / 15.02 ms |

Runs used the same frozen log, separate processes pinned to CPU 3, and reversed
the implementation order for the second pair. Host load varied; the slower
baseline in run 1 is not a live latency claim. Even against the faster baseline,
postflop p99 is **3.3× faster**. These are offline fallback decision times, not
end-to-end server times or a guaranteed maximum.

Full suite: **470 passed, 1 skipped** (local training artifact absent in this worktree).

## Reproduce

From this worktree, run each implementation in a separate process on the same
CPU so they do not share caches or compete with one another. The benchmark checks
every action (including exact raise-to amounts) plus the complete fallback
diagnostics, not whether an older historical action happens to agree with v3.
Timing covers the fallback's `act` call only, not API or learned lookup latency.

```bash
ROOT=$PWD
PY=$ROOT/.venv/bin/python
BENCH=$ROOT/scripts/range_v3_benchmark.py
LOG=/tmp/opencode/triplo-v4/log.jsonl  # frozen snapshot; not the append-only live file
OUT=$(mktemp -d /tmp/opencode/range-v3-bench.XXXXXX)
mkdir "$OUT/reference"
git archive 9347710 poker | tar -x -C "$OUT/reference"
(cd "$OUT/reference" && PYTHONPATH=. taskset -c 3 "$PY" "$BENCH" run "$LOG" "$OUT/baseline.json")
PYTHONPATH=. taskset -c 3 "$PY" "$BENCH" run "$LOG" "$OUT/fast.json"
PYTHONPATH=. "$PY" "$BENCH" compare "$OUT/baseline.json" "$OUT/fast.json"
$PY -m pytest -q
```

Use an available CPU number in place of `3` if necessary. The original snapshot
was the first 25,929,719 bytes of the season-17 session log; SHA-256
`edfb6336d7b52792d02e9d7547765d5015534af1f979339c4fd6a463d0377ab7`.
Nothing here contacts OpenPoker or changes the live service.
