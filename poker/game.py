"""Trusted simulator. Policies receive immutable, detached observations only."""
from dataclasses import dataclass
from random import Random
from collections import deque
from time import perf_counter
from pokerkit import Automation, Deck, NoLimitTexasHoldem

@dataclass(frozen=True)
class Observation:
    player: int
    position: int
    hole: tuple
    board: tuple
    stacks: tuple
    active: tuple
    pot: int
    call: int
    street: int
    legal: tuple
    history: tuple

class Game:
    def __init__(self, seed, players=6, button=0, stacks=None):
        if not 3 <= players <= 6:
            raise ValueError('tables must have 3–6 players')
        self.order = tuple((button + 1 + i) % players for i in range(players))
        self.history = []
        auto = tuple(a for a in Automation if a not in (Automation.HOLE_DEALING, Automation.BOARD_DEALING))
        stacks = stacks or [200] * players
        self.state = NoLimitTexasHoldem.create_state(auto, True, 0, (1,2), 2,
                                                    tuple(stacks[i] for i in self.order), players)
        deck = list(Deck.STANDARD)
        Random(seed).shuffle(deck)
        self.state.deck_cards = deque(deck)
        self.advance()

    def advance(self):
        s = self.state
        while s.status and s.actor_index is None:
            if s.can_deal_hole(): s.deal_hole()
            elif s.can_deal_board(): s.deal_board()
            else: raise RuntimeError('unexpected engine phase')

    def observe(self):
        s = self.state
        i = s.actor_index
        legal = ['call']
        if s.can_fold(): legal.append('fold')
        if s.can_complete_bet_or_raise_to():
            lo = s.min_completion_betting_or_raising_to_amount
            hi = s.max_completion_betting_or_raising_to_amount
            for amount in sorted({lo, hi, min(hi,max(lo,max(s.bets)+s.total_pot_amount//2)),
                                  min(hi,max(lo,max(s.bets)+s.total_pot_amount))}):
                if s.can_complete_bet_or_raise_to(amount): legal.append(amount)
        return Observation(self.order[i], i, tuple(map(repr,s.hole_cards[i])),
                           tuple(map(repr,s.get_board_cards(0))), tuple(s.stacks),tuple(s.statuses),
                           s.total_pot_amount,s.checking_or_calling_amount,s.street_index,
                           tuple(legal),tuple(self.history))

    def act(self, action):
        o = self.observe()
        if action not in o.legal: raise ValueError('illegal action')
        self.history.append((o.player,o.position,o.street,action,o.pot,o.call))
        if action == 'fold': self.state.fold()
        elif action == 'call': self.state.check_or_call()
        else: self.state.complete_bet_or_raise_to(action)
        self.advance()

    def payoff(self):
        out = [0]*len(self.order)
        for i,p in enumerate(self.order): out[p] = self.state.payoffs[i]/2
        return out

def play(policies, seed, button=0):
    g = Game(seed,len(policies),button)
    traces = [[] for _ in policies]
    latency=[]
    while g.state.status:
        o=g.observe(); start=perf_counter()
        a=policies[o.player].act(o)
        latency.append(perf_counter()-start)
        traces[o.player].append((o,a)); g.act(a)
    return g.payoff(), traces, latency
