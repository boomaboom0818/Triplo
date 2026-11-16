"""Training-only loose callers, distinct from the held-out rule gate.

Profiles vary preflop price tolerance and postflop strength/price thresholds.
They consume only public observations and their own hole cards. No gate imports.
"""
from .fast_eval import CARD_ID
from .lean_abstraction import card_features

VERSION = 'loose-callers-v1'
PROFILES = ((6, 3, 0.45), (12, 2, 0.75), (20, 1, 1.0))


class LooseCaller:
    def __init__(self, profile):
        self.price, self.bucket, self.pot_fraction = PROFILES[profile]

    def act(self, observation):
        o = observation
        if not o.call:
            return 'call'
        if not o.street:
            a, b = (CARD_ID[c] >> 2 for c in o.hole)
            premium = a == b and a >= 9 or min(a, b) >= 10
            playable = a == b or max(a, b) >= 8 or o.hole[0][1] == o.hole[1][1] or abs(a - b) <= 2
            call = premium or playable and o.call <= self.price
        else:
            bucket = int(card_features(tuple(CARD_ID[c] for c in o.hole),
                                       tuple(CARD_ID[c] for c in o.board))[0])
            call = bucket >= 8 or bucket >= self.bucket and o.call <= max(2, o.pot * self.pot_fraction)
        return 'call' if call or 'fold' not in o.legal else 'fold'
