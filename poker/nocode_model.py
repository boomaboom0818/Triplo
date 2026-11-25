"""Model of OpenPoker's hosted ``no_code`` preset bots, fitted to live decisions.

Fitted to 88,671 decisions by ``no_code`` bots in 16,000 public OpenPoker hands that
Triplo played (2026-11-16 to 11-24). Frequencies are the measured ones; hand selection
inside each frequency uses the Chen score preflop and the made-hand percentile
(``strength_table``) postflop, which matches what their revealed cards show.

Measured facts the model reproduces:

* Preflop they almost never limp (2-3%), open-raise 22% (UTG) to 46% (BTN) to 2-3 BB,
  fold ~70% to a raise and 3-bet ~3%. As the opener facing a 3-bet they fold 28-45%,
  4-bet 4-7%; facing a 4-bet or more they almost always re-raise (75%) or call.
* Postflop bet sizes are fixed per street (0.66 / 0.72 / 0.76 pot).
* Facing a first bet they fold 42% (flop), 20% (turn), 13% (river), nearly
  independent of size up to 2x pot. Flop/turn raises are almost strength-independent
  (~35% / ~27% of continuing hands); river raises are strength-driven.

This is a betting-policy model for simulation, not a claim about the real templates.
"""
from functools import lru_cache
from hashlib import blake2b
from itertools import combinations

from .equity_pricing import strength_table
from .fast_eval import CARD_ID, NAMES
from .rule_opponents import chen

# Seat labels by fast-game position for six players (0 = SB ... 5 = BTN).
POSITION = {0: 'SB', 1: 'BB', 2: 'UTG', 3: 'HJ', 4: 'CO', 5: 'BTN'}
OPEN_RATE = dict(UTG=.22, HJ=.305, CO=.323, BTN=.461, SB=.367)
LIMP_RATE = .03
BB_RAISE_VS_LIMP = .21
CONTINUE_VS_RAISE = .30      # fold ~70% to one raise
THREEBET_SHARE = .11         # ~3.3% of all hands facing a raise
FOLD_VS_3BET = (.15, .30)    # small / large 3-bet, as a share of the opening range
FOURBET_SHARE = .07
BET_SIZE = {1: .66, 2: .72, 3: .76}
# Bet probability when checked to, by street and made-hand percentile bucket.
BET_RATE = {  # revealed-hand rates scaled up to the overall live rates (24% / 29% / 22%)
    1: (.10, .11, .18, .39, .37, .38),
    2: (.16, .17, .38, .39, .39, .38),
    3: (.11, .07, .12, .16, .40, .90),
}
# Made-hand percentile below which they fold to a first bet (calibrated in simulation
# to the live fold rates 42% / 20% / 13%), and to a raise (18% / 6.5% / 12.6%).
FOLD_BELOW = {1: .58, 2: .42, 3: .32}
FOLD_BELOW_RAISED = {1: .45, 2: .28, 3: .30}
# Raise share of continuing hands facing a bet, by street (river by strength bucket).
RAISE_RATE = {1: (.35,) * 6, 2: (.27,) * 6, 3: (.13, .09, .08, .14, .30, .49)}
RAISE_SIZE = .75


@lru_cache(maxsize=1)
def _chen_percentiles():
    """Share of all 1,326 holdings with a strictly better Chen score, per score."""
    scores = sorted((chen((NAMES[a], NAMES[b])) for a, b in combinations(range(52), 2)), reverse=True)
    out, total = {}, len(scores)
    for i, s in enumerate(scores):
        out.setdefault(s, i / total)
    return out


@lru_cache(maxsize=4096)
def _strength(board):
    return strength_table(board)


def _bucket(hs):
    return 0 if hs < .3 else 1 if hs < .5 else 2 if hs < .7 else 3 if hs < .85 else 4 if hs < .95 else 5


class NoCodeFish:
    """Seat policy reproducing the measured no_code preset behaviour."""

    def __init__(self, seed=0):
        self.seed = seed

    def _u(self, o, tag):
        h = blake2b(repr((self.seed, tag, o.position, o.hole, o.board, o.pot, len(o.history))).encode(),
                    digest_size=8).digest()
        return int.from_bytes(h, 'big') / (1 << 64)

    def _rank(self, o):
        """Preflop hand rank in [0, 1): 0 is the best holding; ties broken pseudo-randomly."""
        table = _chen_percentiles()
        score = chen(tuple(o.hole))
        lo = table[score]
        better = [v for v in table.values() if v > lo]
        hi = min(better) if better else 1.0
        return lo + (hi - lo) * self._u(o, 'rank')

    def act(self, o, game=None):
        raises = [a for a in o.legal if isinstance(a, int)]
        lo, hi = (raises[0], raises[-1]) if raises else (None, None)
        top = max(game.bets) if game is not None else o.call
        passive = 'call' if not o.call else 'fold'

        def to(total):
            return min(hi, max(lo, int(round(total))))

        if o.street == 0:
            return self._preflop(o, top, lo, hi, to, passive)
        return self._postflop(o, top, lo, hi, to, passive)

    def _preflop(self, o, top, lo, hi, to, passive):
        raised = [h for h in o.history if h[2] == 0 and isinstance(h[3], int)]
        pos = POSITION.get(o.position, 'UTG') if len(o.active) == 6 else POSITION.get(o.position, 'UTG')
        r = self._rank(o)
        if not raised:
            if pos == 'BB':
                return to(top * 3 + 2) if lo is not None and r < BB_RAISE_VS_LIMP else 'call'
            rate = OPEN_RATE.get(pos, .25)
            if lo is not None and r < rate:
                u = self._u(o, 'open')
                return to(5 if u < .55 else 6 if u < .85 else 4)  # 2.5 / 3 / 2 BB (chips are half-BB)
            if r < rate + LIMP_RATE:
                return 'call'
            return passive
        if len(raised) == 1:
            if r < CONTINUE_VS_RAISE:
                if lo is not None and self._u(o, '3b') < THREEBET_SHARE:
                    return to(top * 3.2)
                return 'call'
            return passive
        if len(raised) == 2:
            mine = [h for h in raised if h[1] == o.position]
            acted = any(h[1] == o.position and h[2] == 0 and h[3] != 'fold' for h in o.history
                        if o.history.index(h) > o.history.index(raised[0]))
            if not mine and not acted:  # cold to a 3-bet (blind check or limp behind)
                return 'call' if r < .15 else passive
            width = OPEN_RATE.get(pos, .3) if mine else CONTINUE_VS_RAISE
            fold_share = FOLD_VS_3BET[0] if o.call <= 12 else FOLD_VS_3BET[1]
            if r > width * (1 - fold_share):
                return passive
            if lo is not None and self._u(o, '4b') < FOURBET_SHARE:
                return to(top * 2.3)
            return 'call'
        # Facing a 4-bet or more: re-raise (shove) 75%, call 21%, fold 3%.
        u = self._u(o, '5b')
        if u < .03:
            return passive
        if lo is not None and u < .78:
            return hi
        return 'call'

    def _postflop(self, o, top, lo, hi, to, passive):
        board = tuple(sorted(CARD_ID[c] for c in o.board))
        hs = _strength(board)[tuple(sorted(CARD_ID[c] for c in o.hole))]
        b = _bucket(hs)
        street = o.street
        bets = [h for h in o.history if h[2] == street and isinstance(h[3], int)]
        if not o.call:
            if lo is not None and self._u(o, 'bet') < BET_RATE[street][b]:
                return to(top + o.pot * BET_SIZE[street])
            return 'call'
        frac = o.call / max(1, o.pot - o.call)
        floor = (FOLD_BELOW if len(bets) <= 1 else FOLD_BELOW_RAISED)[street]
        if frac > 2:
            floor = max(floor, .5)
        elif frac > 1.2 and street == 3:
            floor = max(floor, .3)
        if hs < floor and self._u(o, 'fold') < .9:
            return passive
        rate = RAISE_RATE[street][b] if len(bets) <= 1 else (.30, .24, .35)[street - 1]
        if lo is not None and self._u(o, 'raise') < rate:
            return to(top + RAISE_SIZE * (o.pot + o.call))
        return 'call'
