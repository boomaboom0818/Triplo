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

from .equity_policy import EquityPolicy, GuardedEquityPolicy, opponent_ranges, private_strength
from .fast_eval import CARD_ID, evaluate

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


def sample_showdowns(o, samples):
    """One joint draw per sample: hero score, opponent scores and strength percentiles.

    Opponent holdings are drawn jointly from v1's range priors with whole-draw
    rejection on card collisions, then one runout is dealt.
    """
    hole = tuple(CARD_ID[c] for c in o.hole)
    board = tuple(CARD_ID[c] for c in o.board)
    known = tuple(sorted(hole + board))
    opponents = [i for i, active in enumerate(o.active) if active and i != o.position]
    ranges = opponent_ranges(o, known, board, opponents)
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


class GuardedEquityPolicyV2(GuardedEquityPolicy):
    """``equity-guarded-v1`` plus side-pot call pricing and EV-based value sizing.

    ``profiles`` maps an opponent position (or ``None`` for the default) to a
    PROFILES name or an explicit ``(base, slope)`` calling threshold. Without
    profiles (the registered default) only the side-pot pricing differs from v1.
    """
    VERSION = VERSION

    def __init__(self, samples=256, stack_press=False, profiles=None):
        super().__init__(samples, stack_press)
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
        if o.call and 'fold' in o.legal:
            if o.street == 3 and o.call > 200 and not private_strength(
                    tuple(CARD_ID[c] for c in o.hole), tuple(CARD_ID[c] for c in o.board))[0]:
                return proposed
            priced = self.price_call(o)
            if priced is not None:
                return priced
        return proposed

    def draws(self, o):
        if self._draws is None:
            self._draws = sample_showdowns(o, self.samples)
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


FALLBACKS = ('equity-v1', 'equity-guarded-v1', VERSION)


def make_fallback(name, stack_press=False, **options):
    """Resolve a fallback name to a policy instance (equity-guarded-v1 stays the default build)."""
    if name == 'equity-v1':
        return EquityPolicy(**options)
    if name == 'equity-guarded-v1':
        return GuardedEquityPolicy(stack_press=stack_press, **options)
    if name == VERSION:
        return GuardedEquityPolicyV2(stack_press=stack_press, **options)
    raise ValueError('unknown equity fallback: %s' % name)
