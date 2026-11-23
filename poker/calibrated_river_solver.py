"""Bounded heads-up river DCFR with public calibrated combo ranges.

The tree permits one bet per line and no raises. Utilities are normalized to
the pot before river betting. Card removal is exact for both modeled ranges.
"""
from dataclasses import replace
from hashlib import blake2b
from itertools import pairwise

import numpy as np

from .equity_policy import opponent_ranges
from .fast_eval import CARD_ID, evaluate
from .public_context import betting_context


ITERATIONS = 48


def _strategy(regrets):
    positive = np.maximum(regrets, 0)
    total = positive.sum(axis=0)
    return np.divide(positive, total, out=np.full_like(positive, .5), where=total > 0)


class ComboPayoffs:
    """Exact conditional payoff sums with shared-card holdings removed."""

    def __init__(self, hands, ranks):
        self.cards = np.asarray(hands).T
        self.order = np.argsort(ranks, kind='stable')
        rank_order = np.asarray(ranks)[self.order]
        self.lo = np.searchsorted(rank_order, ranks, side='left')
        self.hi = np.searchsorted(rank_order, ranks, side='right')
        # Sort the two incidences of each combo by card, then by hand rank.
        # Prefix sums over those groups enable exact blocker removal in O(n)
        # per payoff query, without materializing a combo-by-combo matrix.
        stride = len(hands)
        keys = self.cards * stride + self.lo
        incidence_order = np.argsort(keys.ravel(), kind='stable')
        self.incidence = incidence_order % stride
        sorted_keys = keys.ravel()[incidence_order]
        self.card_lo = np.searchsorted(sorted_keys, keys, side='left')
        self.card_hi = np.searchsorted(sorted_keys, keys, side='right')
        self.card_start = np.searchsorted(sorted_keys, self.cards * stride)
        self.card_end = np.searchsorted(sorted_keys, (self.cards + 1) * stride)

    def compatible(self, mass):
        a, b = self.cards
        blocked = np.bincount(a, mass, minlength=52) + np.bincount(b, mass, minlength=52)
        # The identical combo is counted twice in blocked.
        return mass.sum() - blocked[a] - blocked[b] + mass

    def showdown(self, mass):
        prefix = np.r_[0., np.cumsum(mass[self.order])]
        cards = np.r_[0., np.cumsum(mass[self.incidence])]
        blocked = cards[self.card_lo] + cards[self.card_hi] - cards[self.card_start] - cards[self.card_end]
        return prefix[self.lo] + prefix[self.hi] - prefix[-1] - blocked.sum(axis=0)


def solve_tree(payoffs, oop_mass, ip_mass, bet, iterations=ITERATIONS):
    """Return OOP open, IP response, IP check-to, and OOP check-bet strategies.

    Rows 0 and 1 are respectively check/bet or fold/call.
    """
    ranges = (oop_mass, ip_mass)
    regrets = [np.zeros((2, len(ranges[i])), dtype=np.float64) for i in (0, 1, 1, 0)]
    average = [np.zeros_like(regret) for regret in regrets]

    for step in range(1, iterations + 1):
        opening, responding, checked_to, checked_then_bet = map(_strategy, regrets)
        ip_bets = ip_mass * checked_to[1]
        ip_calls = ip_mass * responding[1]
        oop_bets = oop_mass * opening[1]
        oop_checks = oop_mass * opening[0]
        oop_calls = oop_checks * checked_then_bet[1]

        oop_fold = -.5 * payoffs.compatible(ip_bets)
        oop_call = (.5 + bet) * payoffs.showdown(ip_bets)
        oop_check = .5 * payoffs.showdown(ip_mass * checked_to[0])
        oop_check += checked_then_bet[0] * oop_fold + checked_then_bet[1] * oop_call
        oop_open = .5 * payoffs.compatible(ip_mass * responding[0])
        oop_open += (.5 + bet) * payoffs.showdown(ip_calls)

        ip_fold = -.5 * payoffs.compatible(oop_bets)
        ip_call = (.5 + bet) * payoffs.showdown(oop_bets)
        ip_check = .5 * payoffs.showdown(oop_checks)
        ip_open = .5 * payoffs.compatible(oop_checks * checked_then_bet[0])
        ip_open += (.5 + bet) * payoffs.showdown(oop_calls)

        discount = (step - 1) ** 1.5 / (1 + (step - 1) ** 1.5)
        updates = ((oop_check, oop_open), (ip_fold, ip_call),
                   (ip_check, ip_open), (oop_fold, oop_call))
        reach = (1, 1, 1, opening[0])
        strategies = (opening, responding, checked_to, checked_then_bet)
        for regret, sums, values, strategy, own_reach in zip(regrets, average, updates, strategies, reach):
            utilities = np.asarray(values)
            regret *= np.where(regret > 0, discount, .5)
            regret += utilities - (strategy * utilities).sum(axis=0)
            sums += step ** 2 * strategy * own_reach

    return tuple(_strategy(sums) for sums in average)


def _tree_input(o, baseline):
    if o.street != 3 or len(o.board) != 5 or sum(o.active) != 2 or not o.history:
        return None
    try:
        context = betting_context(o)
    except ValueError:
        return None
    hero = o.position
    opponent = next(i for i, active in enumerate(o.active) if active and i != hero)
    # Postflop acts in position order. Match contributions before river bets.
    if context.paid[hero] - context.bets[hero] != context.paid[opponent] - context.bets[opponent]:
        return None
    river = tuple(h for h in o.history if h[2] == 3)
    oop = min(hero, opponent)
    if not o.call:
        expected = () if hero == oop else ((oop, 'call', 0),)
        if tuple((h[1], h[3], h[5]) for h in river) != expected:
            return None
        effective = min(o.stacks[hero], o.stacks[opponent])
        sizes = [a for a in o.legal if type(a) is int and a <= effective]
        if not sizes:
            return None
        bet = min(sizes, key=lambda a: abs(2 * a - o.pot))
        return oop, (0 if hero == oop else 2), o.pot, bet, bet
    if baseline not in ('call', 'fold') or 'fold' not in o.legal:
        return None
    expected = (oop,) if hero != oop else (hero, opponent)
    if (tuple(h[1] for h in river) != expected or type(river[-1][3]) is not int
            or any(h[3] != 'call' or h[5] != 0 for h in river[:-1])
            or river[-1][5] != 0 or o.pot <= context.bets[opponent]):
        return None
    return oop, (3 if hero == oop else 1), o.pot - context.bets[opponent], o.call, 'call'


def propose(o, baseline):
    """Return (action, diagnostics) for an eligible fallback decision, else None."""
    inputs = _tree_input(o, baseline)
    if inputs is None:
        return None
    oop, node, pot, bet, aggressive_action = inputs
    ip = next(i for i, active in enumerate(o.active) if active and i != oop)
    board = tuple(CARD_ID[c] for c in o.board)
    hole = tuple(sorted(CARD_ID[c] for c in o.hole))
    # The solved tree itself accounts for checking and betting on the river.
    # Start from both players' public priors before those actions.
    start = replace(o, history=tuple(h for h in o.history if h[2] < 3))
    first, second = opponent_ranges(start, tuple(sorted(board)), board, (oop, ip), 'v3')
    hands = first[0]
    mass = [np.asarray([b-a for a, b in pairwise((0.,) + r[1])], dtype=np.float64)
            / r[2] for r in (first, second)]
    ranks = [evaluate(hand + board) for hand in hands]
    strategy = solve_tree(ComboPayoffs(hands, ranks), *mass, bet / pot)
    p = float(strategy[node][1, hands.index(hole)])
    # Independent deterministic draw. Evaluating the candidate never advances
    # the checkpoint's action RNG or changes the solver's public ranges.
    draw = int.from_bytes(blake2b(repr((hole, board, o.history, node)).encode(),
                                 digest_size=8).digest(), 'big') / 2**64
    passive = 'fold' if o.call else 'call'
    action = aggressive_action if draw < p else passive
    return action, dict(node=node, pot=pot, bet=bet, probability=p, baseline=baseline,
                        proposed=action, iterations=ITERATIONS)
