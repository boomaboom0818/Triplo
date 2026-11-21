"""equity-guarded-v3: opponent ranges calibrated on real showdowns."""
from itertools import combinations
from random import Random

import pytest

from poker.equity_policy import _board_classes, _class_index, calibrated_range, estimate_equity, line_labels, preflop_weight
from poker.equity_pricing import GuardedEquityPolicyV2, GuardedEquityPolicyV3, make_fallback
from poker.fast_eval import CARD_ID
from poker.game import Observation
from poker.range_calibration import CLASSES, FACTORS


def river(history, call=20, pot=60):
    return Observation(0, 0, ('Ah', '9d'), ('Ad', '7c', '4s', 'Jh', '2c'), (180, 160, 200, 200, 200, 200),
                       (True, True, False, False, False, False), pot, call, 3,
                       ('call', 'fold', 80, 100) if call else ('call', 30, 60), tuple(history))


BET_EVERY_STREET = [(0, 0, 0, 'call', 3, 1), (1, 1, 0, 'call', 4, 0),
                    (0, 0, 1, 'call', 4, 0), (1, 1, 1, 4, 4, 0), (0, 0, 1, 'call', 8, 4),
                    (0, 0, 2, 'call', 12, 0), (1, 1, 2, 8, 12, 0), (0, 0, 2, 'call', 20, 8),
                    (0, 0, 3, 'call', 28, 0), (1, 1, 3, 20, 28, 0)]


def test_registered_and_table_complete():
    assert isinstance(make_fallback('equity-guarded-v3'), GuardedEquityPolicyV3)
    assert GuardedEquityPolicyV3.RANGE_MODEL == 'v3' and GuardedEquityPolicyV2.RANGE_MODEL == 'v1'
    for street in ('flop', 'turn', 'river'):
        for action in ('bet/raise|0', 'bet/raise|1', 'call|1', 'check|0'):
            factor = FACTORS[f'{street}|{action}']
            assert len(factor) == len(CLASSES) and all(.1 <= f <= 6 for f in factor)


def test_line_labels_read_the_public_history():
    o = river(BET_EVERY_STREET)
    assert line_labels(o, 1) == ((1, 'flop|bet/raise|0'), (2, 'turn|bet/raise|0'), (3, 'river|bet/raise|0'))
    assert line_labels(o, 0) == ((1, 'flop|call|1'), (2, 'turn|call|1'), (3, 'river|check|0'))


def test_checkers_hold_fewer_strong_hands_than_bettors():
    board = tuple(CARD_ID[c] for c in ('Ad', '7c', '4s', 'Jh', '2c'))
    known = tuple(sorted(board))

    def strong_share(lines):
        from poker.equity_policy import private_strength
        pairs, cumulative, total = calibrated_range(known, board, False, lines)
        prev, strong = 0.0, 0.0
        for hand, c in zip(pairs, cumulative):
            if private_strength(hand, board)[0]:
                strong += c - prev
            prev = c
        return strong / total

    checked = strong_share(((1, 'flop|check|0'), (2, 'turn|check|0'), (3, 'river|check|0')))
    bet = strong_share(((1, 'flop|bet/raise|0'), (2, 'turn|bet/raise|0'), (3, 'river|bet/raise|0')))
    assert checked < .5 * bet


def test_v3_estimates_differ_from_v1_only_through_the_range_model():
    o = river(BET_EVERY_STREET)
    v1_equity = estimate_equity(o, 256)[0]
    v3_equity = estimate_equity(o, 256, 'v3')[0]
    assert v1_equity != v3_equity
    # Preflop decisions do not use ranges: identical to v2.
    pre = Observation(2, 2, ('Kh', 'Kd'), (), (200,) * 6, (True,) * 6, 3, 2, 0, ('call', 'fold', 5, 7), ())
    assert GuardedEquityPolicyV3().act(pre) == GuardedEquityPolicyV2().act(pre)


def test_board_class_cache_matches_direct_evaluation():
    rng = Random(6372)
    boards = [tuple(rng.sample(range(52), n)) for n in (3, 4, 5) for _ in range(10)]
    for cards in (
        '4h 5h 6h',      # both hole cards may complete a flush or straight flush
        '4h 5h 6h 5s',   # suit and rank both matter on a paired board
        '4h 5h 6h 7h 2c',
        'Ts Js Qs Ks As',  # board-only royal flush
        '8s 8h 8c 8d 2c',
        '3s 3d 4c 4d',
    ):
        boards.append(tuple(CARD_ID[c] for c in cards.split()))
    for board in boards:
        table = _board_classes(board)
        for a, b in combinations((c for c in range(52) if c not in board), 2):
            assert table[a * 52 + b] == _class_index((a, b), board)


@pytest.mark.parametrize('raised, blockers', [(False, 'Ac 7s'), (True, '7h 2c')])
def test_lookup_preserves_exact_range_weights_and_pair_order(raised, blockers):
    board = tuple(CARD_ID[c] for c in '8h 6h 2h 8d 9h'.split())
    known = tuple(sorted((*board, *(CARD_ID[c] for c in blockers.split()))))
    lines = ((1, 'flop|bet/raise|1'), (2, 'turn|call|1'), (3, 'river|bet/raise|0'))
    pairs = tuple(combinations((c for c in range(52) if c not in known), 2))
    total, cumulative = 0., []
    for a, b in pairs:
        weight = preflop_weight(a, b, raised)
        for street, label in lines:
            weight *= FACTORS[label][_class_index((a, b), board[:street + 2])]
        total += weight
        cumulative.append(total)
    # Same sums and pair order are required for identical deterministic draws.
    assert calibrated_range(known, board, raised, lines) == (pairs, tuple(cumulative), total)
