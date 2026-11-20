"""equity-guarded-v3: opponent ranges calibrated on real showdowns."""
from poker.equity_policy import calibrated_range, estimate_equity, line_labels
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
