"""Fixed deterministic rule opponents for sanity gates. Not strong players.

They see only their own Observation. A checkpoint that cannot beat them is not
ready for anything; beating them is necessary, never sufficient, strength evidence.
"""
from pokerkit import StandardHighHand

RANKS = '23456789TJQKA'
STYLES = ('tight-v1', 'aggressive-v1', 'station-v1', 'shover-v1')
STRONG = ('Straight', 'Flush', 'Full house', 'Four of a kind', 'Straight flush')


def chen(hole):
    """Chen preflop score without the half-point rounding."""
    a, b = sorted((RANKS.index(c[0]) for c in hole), reverse=True)
    base = {12: 10, 11: 8, 10: 7, 9: 6}.get(a, (a + 2) / 2)
    if a == b:
        return max(5, base * 2)
    gap = a - b - 1
    score = base + (2 if hole[0][1] == hole[1][1] else 0) - {0: 0, 1: 1, 2: 2, 3: 4}.get(gap, 5)
    return score + (1 if gap <= 1 and a < 10 else 0)


def made(o):
    """(strong, top_pair_or_overpair, any_pair) using own cards only."""
    label = str(StandardHighHand.from_game(''.join(o.hole), ''.join(o.board)).entry.label)
    board = {c[0] for c in o.board}
    top = max(RANKS.index(c[0]) for c in o.board)
    pocket = o.hole[0][0] == o.hole[1][0]
    connects = pocket or any(c[0] in board for c in o.hole)
    strong = label in STRONG or (label not in ('High card', 'One pair') and connects)
    top_pair = label == 'One pair' and (any(RANKS.index(c[0]) == top for c in o.hole if c[0] in board)
                                        or (pocket and RANKS.index(o.hole[0][0]) > top))
    return strong, top_pair, label != 'High card' and connects


class RulePolicy:
    def __init__(self, style='tight-v1'):
        if style not in STYLES:
            raise ValueError('unknown rule opponent')
        self.style = style

    def act(self, o):
        raises = [a for a in o.legal if isinstance(a, int)]
        slot = lambda i: raises[min(i, len(raises) - 1)]
        passive = 'call' if not o.call else ('fold' if 'fold' in o.legal else 'call')
        style = self.style
        if not o.street:
            s = chen(o.hole)
            if style == 'shover-v1':           # largest raise with any decent hand, else give up
                if s >= 9: return raises[-1] if raises else 'call'
                return passive
            if style == 'station-v1':
                if s >= 10: return 'call'
                return 'call' if s >= 5 and o.call <= 20 else passive
            if s >= 12:                       # AA KK QQ JJ AKs: raise pot, never fold
                return slot(len(raises) - 2) if len(raises) > 1 and o.call < 40 else 'call'
            if style == 'aggressive-v1':
                if s >= 10 and raises and 2 < o.call <= 12: return slot(1)      # re-raise opens
                if s >= 6 and o.call <= 2 and raises: return slot(1)            # open wide
                if s >= 10: return 'call' if o.call <= 50 else passive
                return 'call' if s >= 7 and o.call <= 8 else passive
            if s >= 9:                        # TT 99 AK AQ AJs KQs: open pot, call up to 10bb
                if o.call <= 2 and raises: return slot(1)
                return 'call' if o.call <= 20 else passive
            if s >= 7:                        # open min, call up to 3bb
                if o.call <= 2 and raises: return slot(0)
                return 'call' if o.call <= 6 else passive
            return passive
        strong, top_pair, pair = made(o)
        if strong:
            if style == 'station-v1': return 'call'
            return slot(2) if raises and o.call < o.pot else 'call'
        if style == 'shover-v1':
            return 'call' if top_pair else passive
        if style == 'station-v1':
            return 'call' if pair else passive
        if top_pair:
            if not o.call and raises: return slot(1)
            return 'call' if o.call <= o.pot else passive
        if style == 'aggressive-v1':
            if not o.call and raises and o.street == 1: return slot(1)          # flop stab when checked to
            if pair and o.call * 2 <= o.pot: return 'call'
        return passive
