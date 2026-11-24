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
from .rule_opponents import RANKS, RulePolicy, chen


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


def line_labels(o, pos):
    """Range-v3 evidence for one opponent: postflop (street, "street|action|faced") so far."""
    out = []
    for street in range(1, o.street + 1):
        mine = [h for h in o.history if h[1] == pos and h[2] == street]
        if not mine:
            continue
        if any(isinstance(h[3], int) for h in mine):
            action = 'bet/raise'
        elif any(h[3] == 'call' and h[5] > 0 for h in mine):
            action = 'call'
        elif any(h[3] == 'call' for h in mine):
            action = 'check'
        else:
            continue
        # A check was made before any bet (a later bet is still pending its answer).
        faced = action != 'check' and any(isinstance(h[3], int) for h in o.history if h[2] == street and h[1] != pos)
        out.append((street, '%s|%s|%d' % (('', 'flop', 'turn', 'river')[street], action, faced)))
    return tuple(out)


def _class_index(hole, board):
    strong, top, pair = private_strength(hole, board)
    return 0 if strong else 1 if top else 2 if pair else 3


@lru_cache(maxsize=64)
def _board_classes(board):
    """Class lookup by a*52+b for each compatible a<b on a postflop board.

    A flush/straight flush needs at least three board cards of one suit.
    Otherwise only hole ranks matter; when such a suit exists, also
    distinguish whether each hole card belongs to it. The table does not
    depend on hero blockers or opponent lines and can serve all ranges.
    """
    suits = [c & 3 for c in board]
    flush_suit = next((s for s in range(4) if suits.count(s) >= 3), -1)
    codes = [(c >> 2) * 2 + ((c & 3) == flush_suit) for c in range(52)]
    table = bytearray(52 * 52)
    seen = {}
    for a, b in combinations((c for c in range(52) if c not in board), 2):
        key = codes[a] * 26 + codes[b]
        if key not in seen:
            seen[key] = _class_index((a, b), board)
        table[a * 52 + b] = seen[key]
    return bytes(table)


@lru_cache(maxsize=256)
def calibrated_range(known, board, preflop_raise, lines):
    """Range v3: v1 preflop prior times fitted per-line class multipliers
    (poker.range_calibration), each evaluated on the board of its street."""
    from .range_calibration import FACTORS
    available = [c for c in range(52) if c not in known]
    steps = [(_board_classes(board[:street + 2]), FACTORS[label]) for street, label in lines if label in FACTORS]
    pairs, cumulative, total = [], [], 0.0
    for a, b in combinations(available, 2):
        weight = preflop_weight(a, b, preflop_raise)
        slot = a * 52 + b
        for classes, factor in steps:
            weight *= factor[classes[slot]]
        total += weight
        pairs.append((a, b)); cumulative.append(total)
    return tuple(pairs), tuple(cumulative), total


def opponent_ranges(o, known, board, opponents, model='v1'):
    """Public range prior per opponent position, from this hand's raise evidence."""
    if model == 'v3':
        ranges = []
        for pos in opponents:
            pre = any(h[1] == pos and h[2] == 0 and isinstance(h[3], int) for h in o.history)
            ranges.append(calibrated_range(known, board, pre, line_labels(o, pos)))
        return ranges
    ranges = []
    for pos in opponents:
        actions = [h for h in o.history if h[1] == pos]
        pre = any(h[2] == 0 and isinstance(h[3], int) for h in actions)
        pressure = sum(h[2] > 0 and isinstance(h[3], int) for h in actions)
        # Without a ledger, a current bet still signals at least one strong range.
        if not o.history and o.call and pos == opponents[0]:
            pressure = 1
        ranges.append(holding_range(known, board, pre, pressure))
    return ranges


def estimate_equity(o, samples=256, model='v1'):
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
    ranges = opponent_ranges(o, known, board, opponents, model)
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

    RANGE_MODEL = 'v1'

    def act(self, o):
        self.last = None
        if not o.street:
            return self.preflop.act(o)
        equity, error = estimate_equity(o, self.samples, self.RANGE_MODEL)
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
    # A paired board (board_category == 1) plus one hole-card pair has category == 2
    # (two pair), but hero only contributed one pair; require either an unpaired
    # board for two-pair or trips+ improving the board category.
    strong = (category >= 3 and category > board_category) or (category == 2 and board_category == 0)
    top_pair = (category == 1 or (category == 2 and board_category == 1)) and (max(ranks) in own or own[0] == own[1] > max(ranks))
    return strong, top_pair, pair


class GuardedEquityPolicy(EquityPolicy):
    """Opt-in development fallback; requires a complete ledger for preflop caps.

    Preserve EquityPolicy as a frozen control. This heuristic is not promoted by
    default: equity priors can overfold to bluffs and do not model side-pot EV.

    ``stack_press`` ("Stack Press") lets near-nut hands (strong and usable
    equity >= STACK_PRESS_EQUITY) raise up to the hero's whole stack instead
    of call + 2 * pot. It earns more against loose or weaker fields and risks
    more against strong, tight ones, so it is off by default and toggled per
    deployment (``--stack-press`` on the OpenPoker client).
    """
    STACK_PRESS_EQUITY = .80

    def __init__(self, samples=256, stack_press=False, defend_3bet=False):
        super().__init__(samples)
        self.stack_press = stack_press
        self.defend_3bet = defend_3bet

    def _preflop_commit_limit(self, o, score):
        """Maximum total preflop commitment (normalized chips) for a given hand."""
        if score >= 15:
            return 10 ** 9
        if score >= 12:
            return 200
        if not self.defend_3bet:
            return 20 if score >= 9 else 6
        raises = sum(1 for h in o.history if h[2] == 0 and isinstance(h[3], int))
        if raises < 2:
            return 20 if score >= 9 else 6
        if raises >= 3:
            # Facing a 4-bet+: stay disciplined; only TT/AQ+/AKo call moderate 4-bets.
            return 60 if score >= 10 else 0
        # Facing a 3-bet (raises == 2): price- and position-aware caps.
        last_raiser = next((h[1] for h in reversed(o.history) if h[2] == 0 and isinstance(h[3], int)), -1)
        in_pos = o.position > last_raiser
        r1, r2 = sorted((RANKS.index(c[0]) for c in o.hole), reverse=True)
        suited = o.hole[0][1] == o.hole[1][1]
        pair = (r1 == r2)
        if score >= 10:  # AKo, AQs, AJs, KQs, TT
            if (r1 == 12 and r2 >= 10) or (pair and r1 >= 8):  # AKo, AQs, TT
                return 52 if in_pos else 44
            return 42 if in_pos else 34
        if score >= 9:  # 99, AQo, KJs, QJs, JTs
            return 38 if in_pos else 30
        if suited and r1 == 12:  # ATs, A9s-A2s
            return 30 if in_pos else 26
        if pair:  # 88-22
            if r1 >= 6:  # 88
                return 30 if in_pos else 26
            if r1 >= 5:  # 77
                return 28 if in_pos else 24
            eff_behind = min(o.stacks[o.position],
                             max((o.stacks[i] for i, a in enumerate(o.active) if a and i != o.position), default=0))
            if r1 >= 3 and eff_behind >= 8 * o.call:  # 66, 55
                return 28 if in_pos else 22
            if eff_behind >= 10 * o.call:  # 44-22 deep implied odds
                return 26 if in_pos else (20 if eff_behind >= 14 * o.call else 0)
            return 0
        if suited and score >= 7.5:  # KTs, QTs, J9s, T9s, 98s
            return 30 if in_pos else 26
        if suited and score >= 6.0:  # 87s, 76s, 65s, K9s, Q9s, J8s, T8s
            return 28 if in_pos else 22
        if suited and (r1 >= 10 or (r1 - r2 <= 2 and r2 >= 2)):  # K8s-K2s, Q8s-Q2s, 54s, 75s, 64s, 97s, 86s
            return 26 if in_pos else (20 if (r1 == 11 and r2 >= 5) or (r1 - r2 == 1 and r2 >= 3) else 0)
        if score >= 8:  # AJo, KQo
            return 28 if in_pos else 22
        if (r1 == 12 and r2 >= 8) or (r1 >= 10 and r2 >= 9):  # ATo, KJo, QJo
            return 24 if in_pos else (20 if r1 == 12 else 0)
        if in_pos and ((r1 == 12 and r2 >= 6) or (r1 == 11 and r2 == 8) or (r1 == 9 and r2 == 8)):  # A9o, A8o, KTo, JTo IP vs small 3b
            return 22
        return 0

    def _act_preflop_defend_3bet(self, o):
        """Price- and position-aware facing-3-bet defense (opt-in via defend_3bet=True)."""
        raises_hist = [h for h in o.history if h[2] == 0 and isinstance(h[3], int)]
        if len(raises_hist) < 2 or not o.call or 'fold' not in o.legal:
            return None
        score = chen(o.hole)
        committed = preflop_commitment(o)
        total_cost = committed + o.call
        price = o.call / max(1, o.pot + o.call)
        raises_legal = [a for a in o.legal if isinstance(a, int)]
        # Tier 1: AA/KK never fold; 4-bet/raise when affordable.
        if score >= 15:
            if raises_legal and o.call < 40:
                return raises_legal[min(len(raises_legal) - 2, len(raises_legal) - 1)] if len(raises_legal) > 1 else raises_legal[0]
            return 'call'
        # Tier 2: QQ/JJ/AKs keep 100 BB (200-chip) commitment discipline.
        if score >= 12:
            limit = 200
            if len(raises_hist) == 2 and raises_legal and o.call < 40:
                size = raises_legal[min(len(raises_legal) - 2, len(raises_legal) - 1)] if len(raises_legal) > 1 else raises_legal[0]
                if size <= limit:
                    return size
            return 'call' if total_cost <= limit else 'fold'
        # Cold-facing a 3-bet (hero did not open): stay tight.
        hero_raised = any(h[1] == o.position for h in raises_hist)
        if not hero_raised:
            if len(raises_hist) == 2 and score >= 10 and total_cost <= 36 and price <= 0.42:
                return 'call'
            return 'fold'
        limit = self._preflop_commit_limit(o, score)
        if limit <= 0 or total_cost > limit:
            return 'fold'
        last_raiser = raises_hist[-1][1]
        in_pos = o.position > last_raiser
        if len(raises_hist) >= 3:
            # Facing 4-bet+: require reasonable price too
            return 'call' if price <= 0.36 else 'fold'
        # Facing a 3-bet: price gate scales with hand tier and position
        max_price = 0.45 if score >= 10 else 0.42 if score >= 9 else (0.40 if in_pos else 0.36)
        return 'call' if price <= max_price else 'fold'

    def act(self, o):
        if not o.street:
            if self.defend_3bet and o.history:
                defended = self._act_preflop_defend_3bet(o)
                if defended is not None:
                    return defended
            proposed = super().act(o)
            score = chen(o.hole)
            if proposed == 'fold' or score < 7 or score >= 15 or not o.history:
                return proposed
            limit = 200 if score >= 12 else 20 if score >= 9 else 6
            committed = preflop_commitment(o)
            cost = proposed if isinstance(proposed, int) else committed + o.call
            if cost <= limit:
                return proposed
            return 'call' if not o.call or committed + o.call <= limit else ('fold' if 'fold' in o.legal else 'call')

        self.last = None
        equity, error = estimate_equity(o, self.samples, self.RANGE_MODEL)
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
        press = self.stack_press and strong and usable >= self.STACK_PRESS_EQUITY
        if board_plays or (not strong and (o.call or reraised)) or (o.call and reraised and usable < .80):
            raises = []
        elif raises and strong and o.call < o.pot:
            baseline = raises[-1] if press else raises[min(2, len(raises) - 1)]
        elif raises and top_pair and not o.call:
            baseline = raises[min(1, len(raises) - 1)]
        if isinstance(baseline, int) and usable >= (.50 if opponents == 1 else .40):
            return baseline
        if raises and usable >= (.68 if opponents == 1 else .60):
            cap = max(2, o.call + 2 * o.pot)
            if press:
                cap = max(cap, o.stacks[o.position])
            bounded = [a for a in raises if a <= cap]
            if bounded:
                return bounded[-1] if press or usable > .85 else bounded[0]
        if not o.call:
            return 'call'
        if o.street == 3 and o.call > 200 and not strong:
            return 'fold' if 'fold' in o.legal else 'call'
        if usable >= price + (.025 if o.street == 3 else .05):
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

    def act(self, o, proposed, fallback=None):
        if proposed == 'fold' or not o.call or 'fold' not in o.legal:
            return proposed
        if not o.street:
            if self.version == self.V2_VERSION and o.history:
                score = chen(o.hole)
                if 12 <= score < 15:
                    committed = preflop_commitment(o)
                    cost = proposed if isinstance(proposed, int) else committed + o.call
                    if cost > 200:
                        return 'call' if committed + o.call <= 200 else 'fold'
            raises = sum(street == 0 and isinstance(action, int)
                         for _, _, street, action, _, _ in o.history)
            if raises < 2:
                return proposed
            if fallback is not None and getattr(fallback, 'defend_3bet', False):
                if fallback.act(o) == 'fold':
                    return 'fold'
                score = chen(o.hole)
                if score < 15 and o.history:
                    limit = fallback._preflop_commit_limit(o, score)
                    committed = preflop_commitment(o)
                    cost = proposed if isinstance(proposed, int) else committed + o.call
                    if cost > limit:
                        return 'call' if committed + o.call <= limit else 'fold'
                return proposed
        if self.equity.act(o) == 'fold':
            return 'fold'
        if self.version == self.V2_VERSION and o.street:
            hole = tuple(CARD_ID[c] for c in o.hole)
            board = tuple(CARD_ID[c] for c in o.board)
            strong, _, _ = private_strength(hole, board)
            board_plays = o.street == 3 and evaluate(hole + board) == evaluate(board)
            if o.street == 3 and o.call > 200 and not strong:
                return 'fold'
            if isinstance(proposed, int):
                reraised = any(h[2] > 0 and isinstance(h[3], int) and h[5] > 0 for h in o.history)
                usable = (self.equity.last['equity'] * (1.0 if o.street == 3 else .90)
                          - self.equity.last['standard_error']) if self.equity.last else 0.0
                if board_plays or not strong or (reraised and usable < .80):
                    return 'call'
        return proposed

