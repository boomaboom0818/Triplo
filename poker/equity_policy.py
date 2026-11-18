"""Bounded, deterministic Monte Carlo fallback using public range heuristics.

Ranges are deliberately broad priors, not learned opponent models. Joint holdings
are conditioned on card compatibility; unseen board cards are sampled afterwards.
No simulator state or opponent private cards are consumed. Preflop remains the
existing tight baseline, isolating the postflop change for paired evaluation.
"""
from bisect import bisect_left
from collections import Counter
from functools import lru_cache
from hashlib import blake2b
from itertools import combinations
from math import sqrt
from random import Random

from .fast_eval import CARD_ID, NAMES, evaluate
from .rule_opponents import RulePolicy, chen


@lru_cache(maxsize=16384)
def preflop_weight(a, b, raised):
    score = chen((NAMES[a], NAMES[b]))
    # Retain weak holdings: an observed raise is evidence, not proof of strength.
    return (1.0 if score >= 9 else .35 if score >= 6 else .10) if raised else (1.0 if score >= 5 else .45)


@lru_cache(maxsize=256)
def holding_range(known, board, preflop_raise, pressure):
    available = [c for c in range(52) if c not in known]
    pairs, cumulative, total = [], [], 0.0
    top = max((c >> 2 for c in board), default=0)
    for a, b in combinations(available, 2):
        weight = preflop_weight(a, b, preflop_raise)
        if pressure:
            category = evaluate((a, b) + board) >> 20
            own_pair = a >> 2 == b >> 2 or any(c >> 2 in (a >> 2, b >> 2) for c in board)
            top_pair = (a >> 2 == top or b >> 2 == top
                        or a >> 2 == b >> 2 > top)
            # A broad value/bluff mixture. Board-only hands are not private strength.
            strength = 1.0 if category >= 4 or category >= 2 and own_pair else .65 if top_pair else .25 if own_pair else .12
            weight *= strength ** min(pressure, 2)
        total += weight
        pairs.append((a, b)); cumulative.append(total)
    return tuple(pairs), tuple(cumulative), total


def estimate_equity(o, samples=256):
    """Expected showdown pot share and Monte Carlo standard error.

    No future betting or side-pot EV is modeled. History positions, not physical
    seat ids, select range evidence. Missing history keeps broad priors.
    """
    if samples < 32:
        raise ValueError('at least 32 equity samples required')
    hole = tuple(CARD_ID[c] for c in o.hole)
    board = tuple(CARD_ID[c] for c in o.board)
    known = tuple(sorted(hole + board))
    opponents = [i for i, active in enumerate(o.active) if active and i != o.position]
    if not opponents:
        return 1.0, 0.0
    ranges = []
    for pos in opponents:
        actions = [h for h in o.history if h[1] == pos]
        pre = any(h[2] == 0 and isinstance(h[3], int) for h in actions)
        pressure = sum(h[2] > 0 and isinstance(h[3], int) for h in actions)
        # Without a ledger, a current bet still signals at least one strong range.
        if not o.history and o.call and pos == opponents[0]:
            pressure = 1
        ranges.append(holding_range(known, board, pre, pressure))
    seed = int.from_bytes(blake2b(repr((known, board, o.history, o.active, o.call)).encode(), digest_size=8).digest(), 'big')
    rng = Random(seed)
    total = square = 0.0
    accepted = 0
    # Reject the whole joint draw on collisions; never bias later seats by
    # silently resampling only their holdings. Bound work even for narrow ranges.
    for _ in range(samples * 32):
        hands = [pairs[bisect_left(cumulative, rng.random() * mass)] for pairs, cumulative, mass in ranges]
        cards = [c for hand in hands for c in hand]
        if len(set(cards)) != len(cards):
            continue
        used = set(known).union(cards)
        runout = board + tuple(rng.sample([c for c in range(52) if c not in used], 5 - len(board)))
        hero = evaluate(hole + runout)
        scores = [evaluate(hand + runout) for hand in hands]
        share = 0.0 if max(scores) > hero else 1.0 / (1 + scores.count(hero))
        total += share; square += share * share; accepted += 1
        if accepted == samples:
            break
    if accepted < samples:
        raise RuntimeError('equity sampling budget exhausted')
    mean = total / samples
    return mean, sqrt(max(0.0, square / samples - mean * mean) / samples)


class EquityPolicy:
    """Conservative range-equity decisions; no profitability guarantee."""
    def __init__(self, samples=256):
        if samples < 32:
            raise ValueError('at least 32 equity samples required')
        self.samples = samples
        self.preflop = RulePolicy('tight-v1')
        self.last = None

    def act(self, o):
        self.last = None
        if not o.street:
            return self.preflop.act(o)
        equity, error = estimate_equity(o, self.samples)
        opponents = sum(o.active) - 1
        price = o.call / max(1, o.pot + o.call)
        # Account conservatively for sampling noise and unrealized future equity.
        usable = equity * (1.0 if o.street == 3 else .90) - error
        self.last = dict(equity=equity, standard_error=error, pot_odds=price, samples=self.samples)
        raises = [a for a in o.legal if isinstance(a, int)]
        baseline = self.preflop.act(o)
        board_plays = o.street == 3 and evaluate(tuple(CARD_ID[c] for c in o.hole + o.board)) == evaluate(tuple(CARD_ID[c] for c in o.board))
        if board_plays:
            raises = []
            baseline = 'call'

        # Retain the baseline's value extraction with made hands when equity
        # supports it; high multiway equity thresholds otherwise miss thin value.
        if isinstance(baseline, int) and usable >= (.50 if opponents == 1 else .40):
            return baseline
        if raises and usable >= (.68 if opponents == 1 else .60):
            # Avoid a deep shove merely because it is the largest legal size.
            bounded = [a for a in raises if a <= max(2, o.call + 2 * o.pot)]
            if bounded:
                return bounded[-1] if usable > .85 else bounded[0]
        if not o.call:
            return 'call'
        if usable >= price + (.025 if o.street == 3 else .05):
            return 'call'
        return 'fold' if 'fold' in o.legal else 'call'


def preflop_commitment(o):
    """Hero's total street contribution, in normalized chips, from the ledger.

    Integer actions are raise-to amounts. Calls add the recorded call amount;
    blind positions 0 and 1 begin with one and two chips respectively.
    Requires a complete public action history.
    """
    committed = 1 if o.position == 0 else 2 if o.position == 1 else 0
    for _, position, street, action, _, call in o.history:
        if street == 0 and position == o.position:
            if isinstance(action, int):
                committed = action
            elif action == 'call':
                committed += call
    return committed


def private_strength(hole, board):
    """Made strength contributed by hole cards, excluding board-only categories."""
    category = evaluate(hole + board) >> 20
    ranks = [c >> 2 for c in board]
    # The evaluator requires five cards; a four-card two-pair board has no
    # kicker and cannot be scored directly (its missing kicker encodes -1).
    if len(board) == 5:
        board_category = evaluate(board) >> 20
    else:
        counts = sorted(Counter(ranks).values(), reverse=True)
        board_category = (7 if counts[0] == 4 else 3 if counts[0] == 3
                          else 2 if counts.count(2) == 2 else 1 if counts[0] == 2 else 0)
    own = [c >> 2 for c in hole]
    pair = own[0] == own[1] or any(r in ranks for r in own)
    strong = category >= 2 and category > board_category
    top_pair = category == 1 and (max(ranks) in own or own[0] == own[1] > max(ranks))
    return strong, top_pair, pair


class GuardedEquityPolicy(EquityPolicy):
    """Opt-in development fallback; requires a complete ledger for preflop caps.

    Preserve EquityPolicy as a frozen control. This heuristic is not promoted by
    default: equity priors can overfold to bluffs and do not model side-pot EV.
    """
    def act(self, o):
        if not o.street:
            proposed = super().act(o)
            score = chen(o.hole)
            if proposed == 'fold' or score < 7 or score >= 12 or not o.history:
                return proposed
            limit = 20 if score >= 9 else 6
            committed = preflop_commitment(o)
            cost = proposed if isinstance(proposed, int) else committed + o.call
            if cost <= limit:
                return proposed
            return 'call' if not o.call or committed + o.call <= limit else ('fold' if 'fold' in o.legal else 'call')

        self.last = None
        equity, error = estimate_equity(o, self.samples)
        opponents = sum(o.active) - 1
        price = o.call / max(1, o.pot + o.call)
        usable = equity * (1.0 if o.street == 3 else .90) - error
        self.last = dict(equity=equity, standard_error=error, pot_odds=price, samples=self.samples)
        raises = [a for a in o.legal if isinstance(a, int)]
        hole = tuple(CARD_ID[c] for c in o.hole)
        board = tuple(CARD_ID[c] for c in o.board)
        strong, top_pair, _ = private_strength(hole, board)
        board_plays = o.street == 3 and evaluate(hole + board) == evaluate(board)
        reraised = any(h[2] > 0 and isinstance(h[3], int) and h[5] > 0 for h in o.history)
        baseline = 'call'
        if board_plays or (not strong and (o.call or reraised)):
            raises = []
        elif raises and strong and o.call < o.pot:
            baseline = raises[min(2, len(raises) - 1)]
        elif raises and top_pair and not o.call:
            baseline = raises[min(1, len(raises) - 1)]
        if isinstance(baseline, int) and usable >= (.50 if opponents == 1 else .40):
            return baseline
        if raises and usable >= (.68 if opponents == 1 else .60):
            bounded = [a for a in raises if a <= max(2, o.call + 2 * o.pot)]
            if bounded:
                return bounded[-1] if usable > .85 else bounded[0]
        if not o.call or usable >= price + (.025 if o.street == 3 else .05):
            return 'call'
        return 'fold' if 'fold' in o.legal else 'call'


class PressureGuard:
    """Experimental veto of unsupported learned continuations under pressure.

    This is a heuristic restriction, not an EV oracle. It sacrifices some bluffs
    and can overfold against opponents outside the evaluation population. Enable
    only explicitly in a candidate checkpoint and measure before promotion.
    """
    VERSION = 'equity-pressure-v1'
    V2_VERSION = 'equity-pressure-v2'
    VERSIONS = (VERSION, V2_VERSION)

    def __init__(self, version=VERSION):
        if version not in self.VERSIONS:
            raise ValueError('unknown history decision guard')
        self.version = version
        self.equity = EquityPolicy()

    def act(self, o, proposed):
        if proposed == 'fold' or not o.call or 'fold' not in o.legal:
            return proposed
        if not o.street:
            raises = sum(street == 0 and isinstance(action, int)
                         for _, _, street, action, _, _ in o.history)
            if raises < 2:
                return proposed
        if self.equity.act(o) == 'fold':
            return 'fold'
        if self.version == self.V2_VERSION and o.street and isinstance(proposed, int):
            hole = tuple(CARD_ID[c] for c in o.hole)
            board = tuple(CARD_ID[c] for c in o.board)
            strong, _, _ = private_strength(hole, board)
            board_plays = o.street == 3 and evaluate(hole + board) == evaluate(board)
            if board_plays or not strong:
                return 'call'
        return proposed

