# Lean training line

Started 2026-11-16 after the v7-v10 checkpoints all lost 160-850 bb/100 to fixed
rule opponents (see `docs/RULE-GATE.md`). It replaces the engine, the trained
game's raise menu, the abstraction and the algorithm. Old checkpoints do not
carry over; old runs stay preserved.

| Piece | File | What it is |
|---|---|---|
| Engine | `poker/fast_game.py`, `poker/fast_eval.py` | Integer cards, list state, `clone()`. Reproduces `poker.game.Game` exactly: same deals per seed, same observations, same payoffs including side pots and odd chips (differential tests). |
| Menu | `FastGame(lean=True)` | Half-pot and pot raises; the all-in only when it is at most a twice-pot raise. No bare minimum raise, no 100bb open-shove. |
| Abstraction | `poker/lean_abstraction.py` (`lean-v3`) | Preflop: 169 classes. Postflop: hand-strength percentile bucket (10), draw flag (3), board texture (6). Context: position, players in, relative seat, price to call, pot size, stack-to-pot. |
| Solver | `poker/lean_cfr.py` | External-sampling MCCFR, one deal per iteration, each seat traverses once, linear weighting, average accumulated at sampled opponent nodes. No importance weights. |
| Trainer | `scripts/lean_train.py` | Single writer, 4 GiB limit, finite deadline, atomic checkpoints, row and host-reserve guards. |
| Gate | `scripts/rule_gate.py` | Lean checkpoints are gated on the fast engine with the lean menu. |

| Multi-core | `poker/lean_parallel.py` | Forked workers traverse against tables frozen for the round and return deltas. Deterministic for a given seed, worker count and batch schedule. |
| Player | `poker.lean_cfr.LeanPolicy` | Samples the average strategy; unseen states (for example a 100bb all-in) are played by the fixed `tight-v1` rule player. |
| Live adapter | `scripts/openpoker_client.py` | Builds the lean menu and `lean-v3` keys for lean checkpoints; short-handed tables map to 6-max with early seats folded. |

Measured on this host (6 physical cores): about 55 iterations per second on one
core and about 190 with 10 workers. The old trainer managed 0.46.

`poker/cfr.py` and the other old-line sources stay byte-identical, because the
preserved v10 evidence hashes them.

Known gaps:
1. No multiplayer equilibrium guarantee; strength is judged only by play against opponents.
2. The rule gate is a floor, not a measure of strength. Stronger held-out opponents are still needed.
3. Opponent bet sizes outside half-pot/pot are seen only through the price-to-call and pot-size buckets.
