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

Measured on this host: about 55 iterations per second (old trainer: 0.46).

Known gaps, in priority order:
1. The live adapter still builds the old menu and old keys; it must mirror the lean menu and `lean-v3` before any lean checkpoint plays live.
2. Opponents can shove at any stack depth; the trained game has no such node, so those states fall back to check/fold.
3. Single core. Twelve are available.
4. No multiplayer equilibrium guarantee; strength is judged only by play against opponents.
