"""lean-v3 information abstraction for the fast engine.

Postflop cards are summarised by hand-strength percentile against every two-card
holding on the board, a draw flag and coarse board texture, so top pair and bottom
pair are no longer the same state. Works on integer cards; an Observation wrapper
gives identical keys for evaluation and live play.
"""
from bisect import bisect_left, bisect_right

from .fast_eval import BITS, CARD_ID, STRAIGHT_DRAW, evaluate

VERSION = 'lean-v3'
STRENGTH_EDGES = (.15, .30, .45, .60, .70, .80, .87, .93, .97)
_tables = {}


def board_scores(board):
    """Sorted scores of every two-card holding on this board (small per-deal cache)."""
    scores = _tables.get(board)
    if scores is None:
        if len(_tables) > 64:
            _tables.clear()
        rest = [c for c in range(52) if c not in board]
        scores = sorted(evaluate((a, b) + board) for k, a in enumerate(rest) for b in rest[k + 1:])
        _tables[board] = scores
    return scores


def card_features(hole, board):
    """Preflop: 169-class label. Postflop: (strength bucket, draw, texture) label."""
    a, b = hole[0] >> 2, hole[1] >> 2
    if not board:
        if a < b:
            a, b = b, a
        return f'{a:x}{b:x}{"s" if hole[0] & 3 == hole[1] & 3 else "o"}'
    score = evaluate(hole + board)
    scores = board_scores(board)
    strength = (bisect_left(scores, score) + bisect_right(scores, score)) / (2 * len(scores))
    bucket = bisect_right(STRENGTH_EDGES, strength)
    suit_count = [0, 0, 0, 0]
    board_suit = [0, 0, 0, 0]
    board_ranks = ranks = 0
    paired = 0
    for card in board:
        bit = 1 << (card >> 2)
        if board_ranks & bit:
            paired = 1
        board_ranks |= bit
        board_suit[card & 3] += 1
        suit_count[card & 3] += 1
    ranks = board_ranks | 1 << a | 1 << b
    suit_count[hole[0] & 3] += 1
    suit_count[hole[1] & 3] += 1
    draw = 0
    if len(board) < 5 and score >> 20 < 4:
        if any(suit_count[s] == 4 and suit_count[s] > board_suit[s] for s in range(4)):
            draw = 2
        elif STRAIGHT_DRAW[ranks] == 4 and STRAIGHT_DRAW[board_ranks] < 4:
            draw = 1
    texture = min(2, max(board_suit) - 1) * 2 + paired
    return f'{bucket}{draw}{texture}'


def lean_key(position, street, cards, stacks, active, pot, call, legal):
    """cards is card_features(hole, board); everything else is public."""
    count = sum(active)
    before = sum(active[:position])
    if street:
        seat = 0 if not before else 2 if before == count - 1 else 1
        count = min(count, 4)
    else:
        seat = before
    best = 0
    for i, stack in enumerate(stacks):
        if active[i] and i != position and stack > best:
            best = stack
    effective = min(stacks[position], best)
    ratio = effective / max(1, pot)
    spr = 0 if ratio < 1 else 1 if ratio < 2 else 2 if ratio < 4 else 3 if ratio < 8 else 4
    size = 0 if pot <= 3 else 1 if pot <= 8 else 2 if pot <= 20 else 3 if pot <= 50 else 4 if pot <= 120 else 5
    facing = min(6, call * 6 // max(1, pot))
    return f'{street}{position}{count}{seat}|{cards}|{facing}{size}{spr}|{len(legal)}{"f" if "fold" in legal else "-"}'


def observation_key(o):
    hole = tuple(CARD_ID[c] for c in o.hole)
    board = tuple(CARD_ID[c] for c in o.board)
    return lean_key(o.position, o.street, card_features(hole, board), o.stacks, o.active, o.pot, o.call, o.legal)
