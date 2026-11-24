"""Pot-by-pot call pricing and opponent-aware value sizing (``equity-guarded-v2``).

Opt-in successor of ``GuardedEquityPolicy`` (``equity-guarded-v1``, unchanged).
v2 keeps every v1 decision except two:

* Facing a bet with all-in players at the table, the call/fold price is computed
  per main/side pot: each layer of the contribution ledger (including the hero's
  call) is won against only the players eligible for it. When the structure is a
  single pot contested by every active player, v1 pricing is used verbatim.
* When v1 bets or raises, the size is chosen by an explicit EV estimate among the
  legal sizes inside v1's own bound: each sampled opponent holding continues
  only if its current hand-strength percentile clears a pot-odds threshold, so
  EV(size) = P(fold) * pot + E[called share * final pot - investment].

Both use one Monte Carlo pass over the same public range priors as v1. Future
betting is not modeled (exact on the river, an approximation earlier). Opponent
calling styles are parameters, not measured frequencies: the Observation carries
no per-opponent statistics, so sizing only changes when ``profiles`` supplies a
calling style for every opponent who can still act; otherwise v1's size stands.
"""
from bisect import bisect_left
from functools import lru_cache
from hashlib import blake2b
from itertools import combinations
from math import sqrt
from random import Random

from .equity_policy import EquityPolicy, GuardedEquityPolicy, estimate_equity, opponent_ranges, private_strength
from .fast_eval import CARD_ID, STRAIGHT, evaluate

VERSION = 'equity-guarded-v2'
# Calling threshold on hand-strength percentile: base + slope * pot-odds price.
PROFILES = {
    'tight': (.45, 1.2),
    'balanced': (.30, 1.0),
    'sticky': (.10, .6),
}
SIZE_EPSILON = .02          # keep v1's size unless another beats it by 2% of pot
LEDGER_TOLERANCE = .03      # live histories are rounded to normalized chips


def ledger(o):
    """Per-position (total paid, current-street bet) from the public history.

    Blinds are 1 and 2 normalized chips; integer actions are raise-to amounts and
    calls add the recorded call. Returns None unless the replay reproduces the
    observed pot and the hero's call price, so callers can fall back to v1.
    """
    if not o.history:
        return None
    n = len(o.stacks)
    paid = [0] * n
    bets = [0] * n
    paid[0] = bets[0] = 1
    paid[1] = bets[1] = 2
    street = 0
    for _, pos, st, action, _, call in o.history:
        if st != street:
            bets = [0] * n
            street = st
        if isinstance(action, int):
            delta = action - bets[pos]
        elif action == 'call':
            delta = call
        else:
            continue
        if delta < 0:
            return None
        paid[pos] += delta
        bets[pos] += delta
    if o.street != street:
        bets = [0] * n
    tolerance = max(1, LEDGER_TOLERANCE * o.pot)
    owed = min(o.stacks[o.position], max(bets) - bets[o.position])
    if abs(sum(paid) - o.pot) > tolerance or abs(owed - o.call) > tolerance:
        return None
    return paid, bets


def pot_layers(paid, in_hand, capped, dead=0):
    """Main and side pots as [(amount, eligible positions)], smallest level first.

    ``capped`` players cannot put in more chips (all in, or final contributions);
    an uncapped in-hand player stays eligible above its current contribution.
    Folded money fills the layers it reaches but is never eligible. ``dead``
    chips are an unattributed main-pot amount open to every in-hand player.
    """
    n = len(paid)
    top = max(paid)
    levels = sorted({paid[i] for i in range(n) if in_hand[i] and capped[i] and 0 < paid[i] < top} | {top})
    out, previous = [], 0
    if dead:
        out.append([dead, frozenset(i for i in range(n) if in_hand[i])])
    for level in levels:
        amount = sum(min(p, level) - min(p, previous) for p in paid)
        eligible = frozenset(i for i in range(n) if in_hand[i] and (paid[i] >= level or not capped[i]))
        if out and out[-1][1] == eligible:
            out[-1][0] += amount
        elif amount:
            out.append([amount, eligible])
        previous = level
    return [(amount, eligible) for amount, eligible in out]


@lru_cache(maxsize=64)
def strength_table(board):
    """Percentile of every two-card holding's current made hand on ``board``."""
    available = [c for c in range(52) if c not in board]
    scored = sorted((evaluate(pair + board), pair) for pair in combinations(available, 2))
    total = len(scored)
    table, i = {}, 0
    while i < total:
        j = i
        while j < total and scored[j][0] == scored[i][0]:
            j += 1
        value = (i + .5 * (j - i)) / total
        for k in range(i, j):
            table[scored[k][1]] = value
        i = j
    return table


def share(hero, scores, subset):
    """Hero's pot share against the opponents (indices) contesting one pot."""
    best = max((scores[j] for j in subset), default=-1)
    if best > hero:
        return 0.0
    return 1.0 / (1 + sum(scores[j] == hero for j in subset)) if best == hero else 1.0


def sample_showdowns(o, samples, model='v1'):
    """One joint draw per sample: hero score, opponent scores and strength percentiles.

    Opponent holdings are drawn jointly from v1's range priors with whole-draw
    rejection on card collisions, then one runout is dealt.
    """
    hole = tuple(CARD_ID[c] for c in o.hole)
    board = tuple(CARD_ID[c] for c in o.board)
    known = tuple(sorted(hole + board))
    opponents = [i for i, active in enumerate(o.active) if active and i != o.position]
    ranges = opponent_ranges(o, known, board, opponents, model)
    table = strength_table(tuple(sorted(board)))
    seed = int.from_bytes(blake2b(repr(('v2', known, board, o.history, o.active, o.call)).encode(),
                                  digest_size=8).digest(), 'big')
    rng = Random(seed)
    out = []
    for _ in range(samples * 32):
        hands = [pairs[bisect_left(cumulative, rng.random() * mass)] for pairs, cumulative, mass in ranges]
        cards = [c for hand in hands for c in hand]
        if len(set(cards)) != len(cards):
            continue
        used = set(known).union(cards)
        runout = board + tuple(rng.sample([c for c in range(52) if c not in used], 5 - len(board)))
        out.append((evaluate(hole + runout), tuple(evaluate(h + runout) for h in hands),
                    tuple(table[h] for h in hands)))
        if len(out) == samples:
            return opponents, out
    raise RuntimeError('equity sampling budget exhausted')


def draw_outs(hole, board):
    """Exact clean flush + straight draw outs on flop/turn (0 on river or made straight+)."""
    if len(board) >= 5:
        return 0
    cards = hole + board
    if (evaluate(cards) >> 20) >= 4:
        return 0
    suits = [0, 0, 0, 0]
    bsuits = [0, 0, 0, 0]
    branks = 0
    for c in board:
        bsuits[c & 3] += 1
        suits[c & 3] += 1
        branks |= 1 << (c >> 2)
    ranks = branks
    for c in hole:
        suits[c & 3] += 1
        ranks |= 1 << (c >> 2)
    flush_suit = next((s for s in range(4) if suits[s] == 4 and suits[s] > bsuits[s]), -1)
    used = set(cards)
    outs = 0
    if flush_suit >= 0:
        outs += sum(1 for r in range(13) if (r * 4 + flush_suit) not in used)
    for r in range(13):
        if ranks & (1 << r):
            continue
        sh = STRAIGHT[ranks | (1 << r)]
        if sh >= 0 and sh > STRAIGHT[branks | (1 << r)]:
            outs += 3 if flush_suit >= 0 else 4
    return outs


class GuardedEquityPolicyV2(GuardedEquityPolicy):
    """``equity-guarded-v1`` plus side-pot call pricing and EV-based value sizing.

    ``profiles`` maps an opponent position (or ``None`` for the default) to a
    PROFILES name or an explicit ``(base, slope)`` calling threshold. Without
    profiles (the registered default) only the side-pot pricing differs from v1.
    """
    VERSION = VERSION

    def __init__(self, samples=256, stack_press=False, profiles=None, defend_3bet=False):
        super().__init__(samples, stack_press, defend_3bet=defend_3bet)
        self.profiles = dict(profiles or {})

    def threshold(self, position):
        profile = self.profiles.get(position, self.profiles.get(None, 'balanced'))
        return PROFILES[profile] if isinstance(profile, str) else tuple(profile)

    def act(self, o):
        proposed = super().act(o)
        if not o.street:
            return proposed
        self._draws = None
        if isinstance(proposed, int):
            return self.value_size(o, proposed)
        proposed = self._after_postflop_baseline(o, proposed)
        if o.call and 'fold' in o.legal:
            if o.street == 3 and o.call > 200 and not private_strength(
                    tuple(CARD_ID[c] for c in o.hole), tuple(CARD_ID[c] for c in o.board))[0]:
                return proposed
            priced = self.price_call(o)
            if priced is not None:
                return priced
        return proposed

    def _after_postflop_baseline(self, o, proposed):
        """Hook between the baseline postflop check/call/fold and side-pot pricing."""
        return proposed

    def draws(self, o):
        if self._draws is None:
            self._draws = sample_showdowns(o, self.samples, self.RANGE_MODEL)
        return self._draws

    # -- pot-by-pot call pricing -------------------------------------------
    def price_call(self, o):
        """'call'/'fold' from per-pot EV, or None when v1's single pot is exact."""
        book = ledger(o)
        if book is None:
            return None
        hero = o.position
        paid = list(book[0])
        call = min(o.call, o.stacks[hero])
        paid[hero] += call
        capped = [s == 0 for s in o.stacks]
        capped[hero] = call >= o.stacks[hero]
        layers = pot_layers(paid, o.active, capped)
        everyone = frozenset(i for i, a in enumerate(o.active) if a)
        if all(eligible == everyone for _, eligible in layers):
            return None
        opponents, draws = self.draws(o)
        index = {pos: j for j, pos in enumerate(opponents)}
        river = o.street == 3
        discount, margin = (1.0, .025) if river else (.90, .05)
        ev, contested, pots = -call, 0, []
        for amount, eligible in layers:
            if hero not in eligible:
                pots.append(dict(amount=amount, eligible=sorted(eligible), hero_share=None))
                continue
            subset = [index[p] for p in eligible if p != hero]
            shares = [share(h, scores, subset) for h, scores, _ in draws]
            mean = sum(shares) / len(shares)
            error = sqrt(max(0.0, sum(s * s for s in shares) / len(shares) - mean * mean) / len(shares))
            ev += (mean * discount - error) * amount
            contested += amount
            pots.append(dict(amount=amount, eligible=sorted(eligible), hero_share=mean, standard_error=error))
        action = 'call' if ev >= margin * contested else 'fold'
        if self.last is not None:
            self.last = dict(self.last, version=VERSION, pricing='pot-by-pot', pots=pots,
                             call_ev=ev, required_ev=margin * contested, priced_action=action)
        return action

    # -- opponent-aware value sizing ---------------------------------------
    def value_size(self, o, proposed):
        hero = o.position
        raises = [a for a in o.legal if isinstance(a, int)]
        bound = max(proposed, max(2, o.call + 2 * o.pot))
        sizes = [a for a in raises if a <= bound]
        if len(sizes) < 2 or proposed not in sizes:
            return proposed
        all_in = [i for i, a in enumerate(o.active) if a and i != hero and o.stacks[i] == 0]
        if not o.call and not all_in:
            # No bet this street and nobody capped: v1's pot is one shared main pot.
            paid, bets, dead = [0] * len(o.stacks), [0] * len(o.stacks), o.pot
        else:
            book = ledger(o)
            if book is None:
                return proposed
            (paid, bets), dead = book, 0
        acting = [i for i, a in enumerate(o.active) if a and i != hero and o.stacks[i] > 0]
        if not acting or not all(i in self.profiles or None in self.profiles for i in acting):
            # No calling-style information: keep v1's size (conservative fallback).
            return proposed
        opponents, draws = self.draws(o)
        index = {pos: j for j, pos in enumerate(opponents)}
        values = {}
        for size in sizes:
            invest = size - bets[hero]
            owed = {i: min(o.stacks[i], max(0, size - bets[i])) for i in acting}
            limits = {}
            for i in acting:
                base, slope = self.threshold(i)
                limits[i] = base + slope * owed[i] / max(1, o.pot + invest + owed[i])
            cache, total = {}, 0.0
            for h, scores, strength in draws:
                callers = tuple(i for i in acting if strength[index[i]] >= limits[i])
                layers = cache.get(callers)
                if layers is None:
                    final = list(paid)
                    final[hero] += invest
                    for i in callers:
                        final[i] += owed[i]
                    in_hand = [i == hero or i in callers or i in all_in for i in range(len(final))]
                    layers = cache[callers] = [
                        (amount, [index[p] for p in eligible if p != hero])
                        for amount, eligible in pot_layers(final, in_hand, [True] * len(final), dead)
                        if hero in eligible]
                total += sum(share(h, scores, subset) * amount for amount, subset in layers) - invest
            values[size] = total / len(draws)
        best = max(sizes, key=lambda s: (values[s], -s))
        chosen = best if values[best] - values[proposed] > SIZE_EPSILON * o.pot else proposed
        if self.last is not None:
            self.last = dict(self.last, version=VERSION, size_ev={str(s): v for s, v in values.items()},
                             v1_size=proposed, chosen_size=chosen)
        return chosen


VERSION_V3 = 'equity-guarded-v3'


class GuardedEquityPolicyV3(GuardedEquityPolicyV2):
    """``equity-guarded-v2`` with calibrated ranges and evidence-gated live opens.

    The range model (``poker.equity_policy.calibrated_range``) replaces v1's
    hand-set pressure weights with fitted hand-class multipliers by postflop
    line. Late-position opens additionally need verified live history and the
    exact wire amounts and opponent names, which are not in ``Observation``.
    """
    VERSION = VERSION_V3
    RANGE_MODEL = 'v3'

    def __init__(self, samples=256, stack_press=False, profiles=None, defend_3bet=False):
        super().__init__(samples, stack_press, profiles, defend_3bet=defend_3bet)
        self.late_position_evidence = None

    def late_position_open(self, o, action, seats, bb, pot, wire, *, history_complete):
        """Check the final action, including folds from the learned policy."""
        if history_complete is True and self.late_position_evidence is not None:
            return self.late_position_evidence.propose(o, action, seats, bb, pot, wire)
        return None


VERSION_V4 = 'equity-guarded-v4'


def multiway_thresholds(opponents):
    """Postflop betting thresholds scaled by fair-share equity 1 / (opponents + 1)."""
    players = opponents + 1
    return 1.0 / players, .68 * (2.0 / players) ** .5, .85 * (2.0 / players) ** .35


class GuardedEquityPolicyV4(GuardedEquityPolicyV3):
    """``equity-guarded-v3`` with fair-share multiway postflop betting thresholds.

    Preflop and heads-up postflop decisions stay verbatim v3. In 3-to-6-player
    postflop pots (``opponents >= 2``), v3's flat 0.40 / 0.60 / 0.85 betting
    thresholds are scaled by fair-share equity ``1 / (opponents + 1)``, using
    the smallest legal bet size (half-pot on the lean menu) when a newly
    unlocked baseline bet has ``usable < 0.40`` before ``value_size`` runs.
    All v3 ranges, side-pot pricing, opponent-aware value sizing, late-position
    steal support, and poker safeguards stay intact.
    """
    VERSION = VERSION_V4

    def __init__(self, samples=256, stack_press=False, profiles=None, deep_value=False,
                 defend_3bet=False, flop_peel=False, semi_bluff=False,
                 turn_defend=False, turn_probe=False):
        super().__init__(samples, stack_press, profiles, defend_3bet=defend_3bet)
        self.deep_value = deep_value
        self.flop_peel = flop_peel
        self.semi_bluff = semi_bluff
        self.turn_defend = turn_defend
        self.turn_probe = turn_probe

    def _all_non_sticky(self, acting):
        """True only when every live acting opponent has a measured non-sticky profile.

        Uses predicted fold rate at half-pot (base + slope / 3 >= 0.38) so
        calling stations (LooseCaller ~0.10-0.19, PROFILES['sticky'] ~0.30) and
        cold-start unknowns are strictly excluded, while price-sensitive regulars
        (high slope, moderate/low base) are recognized as foldable.
        """
        if not acting:
            return False
        for i in acting:
            if i not in self.profiles and None not in self.profiles:
                return False
            base, slope = self.threshold(i)
            if base + slope / 3.0 < 0.38:
                return False
        return True

    def _flop_peel_call(self, o, hole, board, equity, error, price, usable, pair, reraised, opponents):
        """Opt-in Fix #2: relax flop call threshold vs small/medium flop c-bets."""
        if o.street != 1 or not o.call or reraised or opponents > 2:
            return False
        pot_before = max(1, o.pot - o.call)
        bet_frac = o.call / pot_before
        if bet_frac > 0.78:
            return False
        bettor = next((h[1] for h in reversed(o.history) if h[2] == 1 and isinstance(h[3], int)), None)
        in_pos = bettor is not None and o.position > bettor
        outs = draw_outs(hole, board)
        # Implied-odds realization credit for draws and pairs vs small/medium flop c-bets
        if outs >= 8:
            # Flush draw / OESD: strong implied odds up to ~0.75x pot
            return (equity - error) >= price - (0.02 if in_pos else 0.0)
        if bet_frac <= 0.67 and (outs >= 4 or pair):
            # Gutshot or made pair (middle/bottom/underpair): relax the harsh 0.90 + 0.05 margin
            realized = equity * (0.97 if in_pos else 0.94) - error
            return realized >= price - (0.01 if in_pos else 0.0)
        # No pair, no straight/flush draw (e.g. two overcards + backdoors): peel only small HU bets
        if opponents == 1 and bet_frac <= 0.55:
            realized = equity * (0.96 if in_pos else 0.92) - error
            return realized >= price + (0.0 if in_pos else 0.02)
        return False

    TURN_DEFEND_MAX_FRAC = 0.85
    TURN_PROBE_FRAC = 0.66
    TURN_PROBE_AIR_RATE = 0.75

    @staticmethod
    def _turn_probe_line(o):
        """True when nobody bet the flop, so a first turn bet is a probe, not a barrel."""
        return not any(h[2] == 1 and isinstance(h[3], int) for h in o.history)

    def _turn_defend_call(self, o, hole, board, equity, error, price, pair, reraised, opponents):
        """Opt-in Triplo v7: stop overfolding heads-up to a first turn bet from a player who
        did not bet the flop (a probe or delayed stab after Triplo checked).

        Live v5-fast/v6 hands: Triplo folded 78% to 60-85% pot turn probes after the flop
        checked through (break-even for a pure bluff is 43%), and its calls in these spots
        won on average. Barrels (the bettor also bet the flop) and the river are unchanged.
        """
        if o.street != 2 or not o.call or reraised or opponents != 1:
            return False
        turn_bets = [h for h in o.history if h[2] == 2 and isinstance(h[3], int)]
        if len(turn_bets) != 1:
            return False
        bettor = turn_bets[0][1]
        if any(h[2] == 1 and h[1] == bettor and isinstance(h[3], int) for h in o.history):
            return False
        if o.call / max(1, o.pot - o.call) > self.TURN_DEFEND_MAX_FRAC:
            return False
        outs = draw_outs(hole, board)
        if outs >= 8:
            return equity - error >= price - 0.01
        if pair or outs >= 4:
            return equity * 0.96 - error >= price
        return False

    def _turn_probe_bet(self, o, hole, board, raises, acting, opponents, reraised, pair):
        """Opt-in Triplo v7: bet the turn heads-up when the flop checked through.

        Live OpenPoker field: 60% folds to 60-85% pot turn probes after a checked flop
        (break-even 43%). Bets draws and most air; keeps unimproved pairs as checks for
        showdown value; skips opponents measured as sticky callers.
        """
        if o.street != 2 or o.call or reraised or not raises or opponents != 1 or len(acting) != 1:
            return None
        if not self._turn_probe_line(o) or any(h[2] == 2 and isinstance(h[3], int) for h in o.history):
            return None
        villain = acting[0]
        if (villain in self.profiles or None in self.profiles) and self.threshold(villain)[0] <= .18:
            return None
        if pair:
            return None
        if draw_outs(hole, board) < 4:
            h = blake2b(repr(('turn-probe', hole, board, o.pot, o.position)).encode(), digest_size=8).digest()
            if int.from_bytes(h, 'big') / (1 << 64) >= self.TURN_PROBE_AIR_RATE:
                return None
        eff = min(o.stacks[o.position], o.stacks[villain])
        target = max(2, round(o.pot * self.TURN_PROBE_FRAC))
        cap = min(o.pot, int(0.45 * eff))
        candidates = [a for a in raises if a <= cap]
        if not candidates:
            return None
        return min(candidates, key=lambda a: (abs(a - target), a))

    def _after_postflop_baseline(self, o, proposed):
        """Flop Peel without Deep Value (Triplo v6-tight): a baseline flop fold gets the same
        peel check that ``_act_deep_value`` applies, before side-pot pricing has the last word."""
        if not self.last or o.street not in (1, 2):
            return proposed
        peel = self.flop_peel and o.street == 1 and o.call and proposed == 'fold'
        defend = self.turn_defend and o.street == 2 and o.call and proposed == 'fold'
        probe = self.turn_probe and o.street == 2 and not o.call and proposed == 'call'
        if not (peel or defend or probe):
            return proposed
        hole = tuple(CARD_ID[c] for c in o.hole)
        board = tuple(CARD_ID[c] for c in o.board)
        strong, top_pair, pair = private_strength(hole, board)
        reraised = any(h[2] > 0 and isinstance(h[3], int) and h[5] > 0 for h in o.history)
        equity, error, price = self.last['equity'], self.last['standard_error'], self.last['pot_odds']
        opponents = sum(o.active) - 1
        if peel and self._flop_peel_call(o, hole, board, equity, error, price, equity * .90 - error,
                                         pair, reraised, opponents):
            self.last = dict(self.last, flop_peel=True)
            return 'call'
        if defend and self._turn_defend_call(o, hole, board, equity, error, price, pair, reraised, opponents):
            self.last = dict(self.last, turn_defend=True)
            return 'call'
        if probe and not strong and not top_pair:
            raises = [a for a in o.legal if isinstance(a, int)]
            acting = [i for i, a in enumerate(o.active) if a and i != o.position and o.stacks[i] > 0]
            bet = self._turn_probe_bet(o, hole, board, raises, acting, opponents, reraised, pair)
            if bet is not None:
                self.last = dict(self.last, turn_probe=True, chosen_size=bet)
                return bet
        return proposed

    def _semi_bluff_bet(self, o, hole, board, equity, raises, acting, opponents, reraised):
        """Opt-in Fix #3: selective flop/turn semi-bluffs strictly vs measured non-sticky opponents."""
        if o.street not in (1, 2) or o.call or reraised or not raises:
            return None
        if opponents > 2 or not self._all_non_sticky(acting):
            return None
        eff = min(o.stacks[o.position], max((o.stacks[i] for i in acting), default=0))
        if eff <= 0:
            return None
        outs = draw_outs(hole, board)
        pre_raises = [h[1] for h in o.history if h[2] == 0 and isinstance(h[3], int)]
        pf_agg = pre_raises[-1] if pre_raises else None
        is_pfr = (pf_agg == o.position)
        hero_bet_flop = any(h[2] == 1 and h[1] == o.position and isinstance(h[3], int) for h in o.history)
        min_fold33 = min(self.threshold(i)[0] + self.threshold(i)[1] / 3.0 for i in acting)

        draws_only = (self.semi_bluff == 'draws')
        should_bet = False
        target_frac = 0.50
        if o.street == 1:
            if outs >= 8 and (opponents == 1 or (outs >= 12 and equity >= 0.35)):
                # Strong draw semi-bluff (flush draw / OESD / combo draw)
                should_bet = True
                target_frac = 0.50
            elif not draws_only and opponents == 1 and is_pfr:
                # Disciplined heads-up c-bet as preflop aggressor vs non-sticky opponent
                if outs >= 4 or equity >= 0.32:
                    should_bet = True
                    target_frac = 0.50
                elif min_fold33 >= 0.52:
                    h = blake2b(repr(('sb-f1', hole, board, o.pot, o.position)).encode(), digest_size=8).digest()
                    if (int.from_bytes(h, 'big') / (1 << 64)) < 0.25:
                        should_bet = True
                        target_frac = 0.50
        else:  # o.street == 2 (turn)
            if opponents == 1:
                if outs >= 8:
                    # Strong draw turn semi-bluff (flush draw / OESD)
                    should_bet = True
                    target_frac = 0.65
                elif not draws_only and hero_bet_flop and outs >= 4 and min_fold33 >= 0.55:
                    # Selective gutshot second barrel vs tight-folding opponents
                    h = blake2b(repr(('sb-t2', hole, board, o.pot, o.position)).encode(), digest_size=8).digest()
                    if (int.from_bytes(h, 'big') / (1 << 64)) < 0.35:
                        should_bet = True
                        target_frac = 0.65

        if not should_bet:
            return None
        # Disciplined bluff sizing: never overbet or commit > 45% of effective stack as a semi-bluff
        target = max(2, round(o.pot * target_frac))
        cap = min(o.pot, int(0.45 * eff))
        candidates = [a for a in raises if a <= cap]
        if not candidates:
            return None
        return min(candidates, key=lambda a: (abs(a - target), a))

    def _act_deep_value(self, o):
        self._draws = None
        self.last = None
        opponents = sum(o.active) - 1
        equity, error = estimate_equity(o, self.samples, self.RANGE_MODEL)
        price = o.call / max(1, o.pot + o.call)
        usable = equity * (1.0 if o.street == 3 else .90) - error
        self.last = dict(equity=equity, standard_error=error, pot_odds=price, samples=self.samples)
        raises = [a for a in o.legal if isinstance(a, int)]
        hole = tuple(CARD_ID[c] for c in o.hole)
        board = tuple(CARD_ID[c] for c in o.board)
        strong, top_pair, pair = private_strength(hole, board)
        board_plays = o.street == 3 and evaluate(hole + board) == evaluate(board)
        reraised = any(h[2] > 0 and isinstance(h[3], int) and h[5] > 0 for h in o.history)
        base_thresh, val_thresh, big_thresh = (
            multiway_thresholds(opponents) if opponents >= 2 else (.50, .68, .85))
        acting = [i for i, a in enumerate(o.active) if a and i != o.position and o.stacks[i] > 0]
        all_sticky = (bool(acting)
                      and all((i in self.profiles or None in self.profiles) and self.threshold(i)[0] <= .18
                              for i in acting))
        press = ((self.stack_press and strong and usable >= self.STACK_PRESS_EQUITY)
                 or (strong and usable >= (.72 if all_sticky else .78)))
        allow_tp_raise = (top_pair and not reraised and o.call > 0 and o.call <= o.pot
                          and usable >= (.68 if all_sticky else .76))
        baseline = 'call'
        if board_plays or (not strong and not allow_tp_raise and (o.call or reraised)) or (o.call and reraised and usable < .80):
            raises = []
        elif raises and strong and (o.call < o.pot or usable >= .78):
            baseline = raises[-1] if press else raises[min(2, len(raises) - 1)]
        elif raises and top_pair and (not o.call or allow_tp_raise):
            baseline = raises[min(1, len(raises) - 1)]
        proposed = None
        if isinstance(baseline, int) and usable >= base_thresh:
            proposed = raises[0] if opponents >= 2 and usable < .40 and raises else baseline
        elif raises and usable >= val_thresh:
            cap = max(2, o.call + 2 * o.pot)
            if press:
                cap = max(cap, o.stacks[o.position])
            bounded = [a for a in raises if a <= cap]
            if bounded:
                proposed = bounded[-1] if press or usable > big_thresh else bounded[0]
        if proposed is None:
            if not o.call:
                if self.turn_probe and not board_plays and not strong and not top_pair:
                    tp = self._turn_probe_bet(o, hole, board, raises, acting, opponents, reraised, pair)
                    if tp is not None:
                        self.last = dict(self.last, version=self.VERSION, turn_probe=True, chosen_size=tp)
                        return tp
                if self.semi_bluff and not board_plays and not strong and not top_pair:
                    sb = self._semi_bluff_bet(o, hole, board, equity, raises, acting, opponents, reraised)
                    if sb is not None:
                        self.last = dict(self.last, version=self.VERSION, semi_bluff=True, chosen_size=sb)
                        return sb
                proposed = 'call'
            elif o.street == 3 and o.call > 200 and not strong:
                proposed = 'fold' if 'fold' in o.legal else 'call'
            elif usable >= price + (.025 if o.street == 3 else .05):
                proposed = 'call'
            elif self.flop_peel and self._flop_peel_call(o, hole, board, equity, error, price, usable, pair, reraised, opponents):
                self.last = dict(self.last, flop_peel=True)
                proposed = 'call'
            elif self.turn_defend and self._turn_defend_call(o, hole, board, equity, error, price, pair, reraised, opponents):
                self.last = dict(self.last, turn_defend=True)
                proposed = 'call'
            else:
                proposed = 'fold' if 'fold' in o.legal else 'call'
        if isinstance(proposed, int):
            return self._deep_value_size(o, proposed, press, all_sticky, strong, usable)
        if o.call and 'fold' in o.legal:
            if o.street == 3 and o.call > 200 and not strong:
                return proposed
            priced = self.price_call(o)
            if priced is not None:
                return priced
        return proposed

    def _deep_value_size(self, o, proposed, press, all_sticky, strong, usable):
        hero = o.position
        raises = [a for a in o.legal if isinstance(a, int)]
        if press or (strong and all_sticky and usable >= .68):
            bound = max(proposed, o.stacks[hero] + o.pot + o.call)
        elif all_sticky and usable >= .60:
            bound = max(proposed, max(2, o.call + 3 * o.pot))
        else:
            bound = max(proposed, max(2, o.call + 2 * o.pot))
        sizes = [a for a in raises if a <= bound]
        if len(sizes) < 2 or proposed not in sizes:
            return proposed
        all_in = [i for i, a in enumerate(o.active) if a and i != hero and o.stacks[i] == 0]
        if not o.call and not all_in:
            paid, bets, dead = [0] * len(o.stacks), [0] * len(o.stacks), o.pot
        else:
            book = ledger(o)
            if book is None:
                return proposed
            (paid, bets), dead = book, 0
        acting = [i for i, a in enumerate(o.active) if a and i != hero and o.stacks[i] > 0]
        if not acting or not all(i in self.profiles or None in self.profiles for i in acting):
            return proposed
        opponents, draws = self.draws(o)
        index = {pos: j for j, pos in enumerate(opponents)}
        values = {}
        for size in sizes:
            invest = size - bets[hero]
            owed = {i: min(o.stacks[i], max(0, size - bets[i])) for i in acting}
            limits = {}
            for i in acting:
                base, slope = self.threshold(i)
                limits[i] = base + slope * owed[i] / max(1, o.pot + invest + owed[i])
            cache, total = {}, 0.0
            for h, scores, strength in draws:
                callers = tuple(i for i in acting if strength[index[i]] >= limits[i])
                layers = cache.get(callers)
                if layers is None:
                    final = list(paid)
                    final[hero] += invest
                    for i in callers:
                        final[i] += owed[i]
                    in_hand = [i == hero or i in callers or i in all_in for i in range(len(final))]
                    layers = cache[callers] = [
                        (amount, [index[p] for p in eligible if p != hero])
                        for amount, eligible in pot_layers(final, in_hand, [True] * len(final), dead)
                        if hero in eligible]
                total += sum(share(h, scores, subset) * amount for amount, subset in layers) - invest
            values[size] = total / len(draws)
        best = max(sizes, key=lambda s: (values[s], -s))
        chosen = best if values[best] - values[proposed] > SIZE_EPSILON * o.pot else proposed
        if self.last is not None:
            self.last = dict(self.last, version=self.VERSION, deep_value=True,
                             size_ev={str(s): v for s, v in values.items()},
                             v1_size=proposed, chosen_size=chosen)
        return chosen

    def _act_multiway(self, o):
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
        base_thresh, val_thresh, big_thresh = multiway_thresholds(opponents)
        if isinstance(baseline, int) and usable >= base_thresh:
            return raises[0] if usable < .40 and raises else baseline
        if raises and usable >= val_thresh:
            cap = max(2, o.call + 2 * o.pot)
            if press:
                cap = max(cap, o.stacks[o.position])
            bounded = [a for a in raises if a <= cap]
            if bounded:
                return bounded[-1] if press or usable > big_thresh else bounded[0]
        if not o.call:
            return 'call'
        if o.street == 3 and o.call > 200 and not strong:
            return 'fold' if 'fold' in o.legal else 'call'
        if usable >= price + (.025 if o.street == 3 else .05):
            return 'call'
        return 'fold' if 'fold' in o.legal else 'call'

    def act(self, o):
        if o.street and self.deep_value:
            return self._act_deep_value(o)
        if not o.street or sum(o.active) - 1 <= 1:
            return super().act(o)
        proposed = self._act_multiway(o)
        self._draws = None
        if isinstance(proposed, int):
            return self.value_size(o, proposed)
        proposed = self._after_postflop_baseline(o, proposed)
        if o.call and 'fold' in o.legal:
            if o.street == 3 and o.call > 200 and not private_strength(
                    tuple(CARD_ID[c] for c in o.hole), tuple(CARD_ID[c] for c in o.board))[0]:
                return proposed
            priced = self.price_call(o)
            if priced is not None:
                return priced
        return proposed


VERSION_V5 = 'equity-guarded-v5'
FALLBACKS = ('equity-v1', 'equity-guarded-v1', VERSION, VERSION_V3, VERSION_V4, VERSION_V5)


def make_fallback(name, stack_press=False, **options):
    """Resolve a fallback name to a policy instance (equity-guarded-v1 stays the default build)."""
    if name == 'equity-v1':
        return EquityPolicy(**options)
    if name == 'equity-guarded-v1':
        return GuardedEquityPolicy(stack_press=stack_press, **options)
    if name == VERSION:
        return GuardedEquityPolicyV2(stack_press=stack_press, **options)
    if name == VERSION_V3:
        return GuardedEquityPolicyV3(stack_press=stack_press, **options)
    if name == VERSION_V4:
        return GuardedEquityPolicyV4(stack_press=stack_press, **options)
    if name == VERSION_V5:
        from .river_solver_fallback import RiverSolverFallback
        return RiverSolverFallback(stack_press=stack_press, **options)
    raise ValueError('unknown equity fallback: %s' % name)
