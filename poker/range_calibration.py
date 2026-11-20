"""Opponent range calibration for ``equity-guarded-v3`` (range model v3).

Generated from real OpenPoker showdowns (season 17 session log, 5,447 revealed
opponent hands) by fitting one multiplier per hand class for each postflop
street/action line, applied on top of the v1 preflop prior. Showdown-selection
bias was corrected per line with full-information simulation, limited to x1.5.
Regenerate with scripts.range_sim_lines (simulation) and scripts.range_fit
(real showdowns from the client's --log plus /api/me/hand-history boards).
Held-out log-loss of the shown class: v1 pressure model 1.233, calibrated 1.127.

Key: "street|action|faced" with action bet/raise, call or check and faced = 1 when
another player bet or raised on that street. Classes: two pair or better from own
cards, top pair/overpair, weaker pair, no pair (see poker.equity_policy.private_strength).
"""

CLASSES = ('two pair+', 'top pair/overpair', 'weaker pair', 'no pair')

FACTORS = {
    'flop|bet/raise|0': (2.485, 1.899, 1.272, 0.549),  # n=761
    'flop|bet/raise|1': (2.015, 1.580, 1.242, 0.678),  # n=869
    'flop|call|1': (0.956, 1.767, 1.441, 0.671),  # n=1160
    'flop|check|0': (0.592, 0.897, 1.007, 1.060),  # n=2657
    'turn|bet/raise|0': (1.663, 1.231, 1.074, 0.605),  # n=737
    'turn|bet/raise|1': (1.400, 1.352, 1.052, 0.717),  # n=602
    'turn|call|1': (1.124, 1.404, 1.271, 0.617),  # n=1014
    'turn|check|0': (0.638, 0.868, 0.995, 1.145),  # n=2627
    'river|bet/raise|0': (1.851, 0.825, 0.838, 0.648),  # n=710
    'river|bet/raise|1': (2.090, 0.804, 0.725, 0.506),  # n=271
    'river|call|1': (0.759, 1.349, 1.208, 0.777),  # n=870
    'river|check|0': (0.409, 1.125, 1.090, 1.141),  # n=2328
}
