"""Fast pure-Python high-hand evaluator for 5-7 integer cards.

Card id = rank * 4 + suit, rank 0..12 for 2..A, suit 0..3 for c d h s (pokerkit
standard deck order). Scores compare like poker hands; score >> 20 is the category.
"""
RANKS = '23456789TJQKA'
SUITS = 'cdhs'
LABELS = ('High card', 'One pair', 'Two pair', 'Three of a kind', 'Straight',
          'Flush', 'Full house', 'Four of a kind', 'Straight flush')
NAMES = tuple(r + s for r in RANKS for s in SUITS)
CARD_ID = {name: i for i, name in enumerate(NAMES)}


def _tables():
    bits, straight, top, high, draw = [], [], {k: [] for k in (1, 2, 3, 5)}, [], []
    for mask in range(8192):
        ranks = [r for r in range(12, -1, -1) if mask >> r & 1]
        bits.append(len(ranks))
        high.append(ranks[0] if ranks else -1)
        best = -1
        for top_rank in range(12, 2, -1):
            need = sum(1 << (top_rank - k) for k in range(5)) if top_rank > 3 else 0b1000000001111
            if mask & need == need:
                best = top_rank
                break
        straight.append(best)
        for k in top:
            top[k].append(sum(1 << r for r in ranks[:k]))
        # Most ranks present in any five-rank straight window, ace playing low or high.
        low = (mask << 1 | mask >> 12 & 1) & 0x3fff
        draw.append(max(bin(low >> start & 31).count('1') for start in range(10)))
    return bits, straight, top, high, draw


BITS, STRAIGHT, TOP, HIGH, STRAIGHT_DRAW = _tables()
TOP1, TOP2, TOP3, TOP5 = TOP[1], TOP[2], TOP[3], TOP[5]


def evaluate(cards):
    s0 = s1 = s2 = s3 = 0
    count = [0] * 13
    for card in cards:
        rank = card >> 2
        suit = card & 3
        count[rank] += 1
        bit = 1 << rank
        if suit == 0: s0 |= bit
        elif suit == 1: s1 |= bit
        elif suit == 2: s2 |= bit
        else: s3 |= bit
    for mask in (s0, s1, s2, s3):
        if BITS[mask] >= 5:
            top = STRAIGHT[mask]
            return 8 << 20 | top if top >= 0 else 5 << 20 | TOP5[mask]
    ranks = s0 | s1 | s2 | s3
    quads = trips = pair = second = -1
    for rank in range(12, -1, -1):
        n = count[rank]
        if n < 2: continue
        if n == 4: quads = rank
        elif n == 3:
            if trips < 0: trips = rank
            elif pair < 0: pair = rank
        elif pair < 0: pair = rank
        elif second < 0: second = rank
    if quads >= 0:
        return 7 << 20 | quads << 4 | HIGH[ranks & ~(1 << quads)]
    if trips >= 0 and pair >= 0:
        return 6 << 20 | trips << 4 | pair
    top = STRAIGHT[ranks]
    if top >= 0:
        return 4 << 20 | top
    if trips >= 0:
        return 3 << 20 | trips << 13 | TOP2[ranks & ~(1 << trips)]
    if second >= 0:
        return 2 << 20 | pair << 8 | second << 4 | HIGH[ranks & ~(1 << pair) & ~(1 << second)]
    if pair >= 0:
        return 1 << 20 | pair << 13 | TOP3[ranks & ~(1 << pair)]
    return TOP5[ranks]
