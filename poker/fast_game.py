"""Lean no-limit hold'em engine for training. Mirrors poker.game.Game exactly.

Same seeds deal the same cards, every decision exposes the same Observation and
menu, and payoffs match the pokerkit reference (tests replay both side by side).
State is a handful of small lists, so clone() is cheap and there is no deepcopy.
"""
from random import Random

from .fast_eval import NAMES, evaluate
from .game import Observation

SMALL_BLIND, BIG_BLIND, MIN_BET = 1, 2, 2
LEAN_MENU = 'lean-menu-v1'


class FastGame:
    __slots__ = ('n', 'order', 'deck', 'next_card', 'hole', 'board', 'stacks', 'bets', 'paid', 'active',
                 'actors', 'raise_size', 'acted', 'short', 'street', 'live', 'payoffs', 'history', 'lean')

    def __init__(self, seed, players=6, button=0, stacks=None, lean=False):
        # lean=True selects LEAN_MENU: half-pot and pot raises only, plus the all-in when
        # it is at most a twice-pot raise. No bare minimum raise (it permits ~100-raise
        # wars that explode the tree) and no 100bb open-shove.
        self.lean = lean
        if not 3 <= players <= 6:
            raise ValueError('tables must have 3–6 players')
        n = self.n = players
        self.order = tuple((button + 1 + i) % n for i in range(n))
        stacks = stacks or [200] * n
        self.stacks = [stacks[i] for i in self.order]
        if min(self.stacks) < 1:
            raise ValueError('stacks must be positive')
        deck = list(range(52))
        Random(seed).shuffle(deck)
        self.deck = deck
        self.hole = [(deck[i], deck[n + i]) for i in range(n)]
        self.next_card = 2 * n
        self.board = []
        self.bets = [0] * n
        self.active = [True] * n
        self.history = []
        self.payoffs = None
        self.live = True
        self.street = 0
        for i, blind in ((0, SMALL_BLIND), (1, BIG_BLIND)):
            amount = min(blind, self.stacks[i])
            self.bets[i] = amount
            self.stacks[i] -= amount
        self.paid = self.bets[:]
        self._begin(2 % n)

    def clone(self):
        g = object.__new__(FastGame)
        g.n = self.n; g.order = self.order; g.deck = self.deck; g.next_card = self.next_card
        g.hole = self.hole; g.board = self.board[:]; g.stacks = self.stacks[:]; g.bets = self.bets[:]
        g.paid = self.paid[:]; g.active = self.active[:]; g.actors = self.actors[:]
        g.raise_size = self.raise_size; g.acted = self.acted; g.short = self.short
        g.lean = self.lean; g.street = self.street; g.live = self.live; g.payoffs = self.payoffs; g.history = self.history[:]
        return g

    # --- betting rounds -------------------------------------------------
    def _begin(self, opener):
        n, stacks, bets, active = self.n, self.stacks, self.bets, self.active
        self.raise_size = 0
        self.acted = 0
        self.short = 0
        while True:
            # A player nobody can make put in more chips does not act (pokerkit effective stack).
            totals = sorted(stacks[j] + bets[j] for j in range(n) if active[j])
            second = totals[-2]
            actors = [i for i in ((opener + k) % n for k in range(n))
                      if active[i] and stacks[i] and second > bets[i]]
            self.actors = actors
            if actors and not (len(actors) == 1 and bets[actors[0]] >= max(bets)):
                return
            if not self._next_street():
                return
            opener = 0

    def _next_street(self):
        """Collect bets and deal; False once the hand is over."""
        self.bets = [0] * self.n
        if self.street == 3:
            self._showdown()
            return False
        self.street += 1
        self.next_card += 1                      # burn
        count = 3 if self.street == 1 else 1
        self.board.extend(self.deck[self.next_card:self.next_card + count])
        self.next_card += count
        self.raise_size = 0
        self.acted = 0
        self.short = 0
        return True

    def _end_round(self):
        if sum(self.active) <= 1:
            winner = self.active.index(True)
            total = sum(self.paid)
            self.payoffs = [-p for p in self.paid]
            self.payoffs[winner] += total
            self.live = False
            self.actors = []
        elif self._next_street():
            self._begin(0)

    def _showdown(self):
        n, paid, active = self.n, self.paid, self.active
        board = self.board
        score = [evaluate(self.hole[i] + tuple(board)) if active[i] else -1 for i in range(n)]
        levels = sorted(set(p for p in paid if p))

        def pots(alive):
            out, previous = [], 0                 # [amount, eligible tuple]
            for level in levels:
                amount = sum(min(p, level) - min(p, previous) for p in paid)
                eligible = tuple(i for i in range(n) if alive[i] and paid[i] >= level)
                if out and out[-1][1] == eligible:
                    out[-1][0] += amount
                elif amount:
                    out.append([amount, eligible])
                previous = level
            return out

        # Like the reference, discard hands that cannot win any pot before the final
        # pots are formed; this decides which tied winner receives an odd chip.
        alive = [False] * n
        for amount, eligible in pots(active):
            best = max(score[i] for i in eligible)
            for i in eligible:
                if score[i] == best:
                    alive[i] = True
        won = [0] * n
        for amount, eligible in pots(alive):
            best = max(score[i] for i in eligible)
            winners = [i for i in eligible if score[i] == best]
            share, extra = divmod(amount, len(winners))
            for i in winners:
                won[i] += share
            won[winners[0]] += extra
        self.payoffs = [won[i] - paid[i] for i in range(n)]
        self.live = False
        self.actors = []

    # --- interface matching poker.game.Game -------------------------------
    def can_raise(self):
        i = self.actors[0]
        bets, stacks = self.bets, self.stacks
        top = max(bets)
        owed = top - bets[i]
        if min(stacks[i], owed) < self.raise_size:
            return False
        if self.short and self.short < self.raise_size and self.acted >> i & 1:
            return False
        if stacks[i] <= owed:
            return False
        active = self.active
        return any(active[j] and stacks[j] + bets[j] > top for j in range(self.n) if j != i)

    def menu(self):
        """(pot, call, legal) for the actor, identical to Game.observe()."""
        i = self.actors[0]
        bets, stack = self.bets, self.stacks[i]
        top = max(bets)
        pot = sum(self.paid)
        legal = ['call']
        if bets[i] < top:
            legal.append('fold')
        if self.can_raise():
            hi = stack + bets[i]
            lo = min(hi, max(self.raise_size, MIN_BET) + top)
            sizes = {min(hi, max(lo, top + pot // 2)), min(hi, max(lo, top + pot))}
            if not self.lean:
                sizes.update((lo, hi))
            elif hi <= top + 2 * pot:
                sizes.add(hi)
            legal.extend(sorted(sizes))
        return pot, min(stack, top - bets[i]), tuple(legal)

    def observe(self):
        i = self.actors[0]
        pot, call, legal = self.menu()
        return Observation(self.order[i], i, tuple(NAMES[c] for c in self.hole[i]),
                           tuple(NAMES[c] for c in self.board), tuple(self.stacks), tuple(self.active),
                           pot, call, self.street, legal, tuple(self.history))

    def act(self, action, checked=True):
        i = self.actors[0]
        if checked:
            pot, call, legal = self.menu()
            if action not in legal:
                raise ValueError('illegal action')
            self.history.append((self.order[i], i, self.street, action, pot, call))
        bets, stacks = self.bets, self.stacks
        self.acted |= 1 << i
        if action == 'fold':
            self.active[i] = False
            del self.actors[0]
        elif action == 'call':
            amount = min(stacks[i], max(bets) - bets[i])
            bets[i] += amount; stacks[i] -= amount; self.paid[i] += amount
            del self.actors[0]
        else:
            top = max(bets)
            increase = action - top
            delta = action - bets[i]
            bets[i] = action; stacks[i] -= delta; self.paid[i] += delta
            n, active = self.n, self.active
            self.actors = [j for j in ((i + k) % n for k in range(1, n)) if active[j] and stacks[j]]
            if increase >= self.raise_size:
                self.acted = 1 << i
                self.raise_size = increase
            self.short = self.short + increase if not stacks[i] else 0
            if self.short >= self.raise_size:
                self.short = 0
            return
        if not self.actors or sum(self.active) <= 1:
            self._end_round()

    def payoff(self):
        out = [0] * self.n
        for i, p in enumerate(self.order):
            out[p] = self.payoffs[i] / 2
        return out
