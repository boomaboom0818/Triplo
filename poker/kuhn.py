"""Three-card, two-player Kuhn poker adapter and exhaustive best-response audit."""
from itertools import permutations, product


class Kuhn:
    def __init__(self, cards, history=''):
        self.cards, self.history = cards, history

    @property
    def terminal(self):
        return self.history in ('cc', 'bc', 'bf', 'cbc', 'cbf')

    @property
    def actor(self):
        return len(self.history) % 2

    @property
    def actions(self):
        return ('f', 'c') if self.history.endswith('b') else ('c', 'b')

    def key(self):
        return f'{self.actor}:{self.cards[self.actor]}:{self.history}'

    def child(self, action):
        return Kuhn(self.cards, self.history + action)

    def utility(self, player):
        if self.history in ('bf', 'cbf'):
            winner = 0 if self.history == 'bf' else 1
            stake = 1
        else:
            winner = int(self.cards[1] > self.cards[0])
            stake = 1 if self.history == 'cc' else 2
        return stake if winner == player else -stake


def value(policy, responder=None, choices=None):
    def walk(state):
        if state.terminal:
            return state.utility(0)
        if state.actor == responder:
            return walk(state.child(choices[state.key()]))
        probs = policy(state.key(), len(state.actions))
        return sum(p * walk(state.child(a)) for p, a in zip(probs, state.actions))
    return sum(walk(Kuhn(cards)) for cards in permutations(range(3), 2)) / 6


def audit(policy):
    best = []
    for player, histories in ((0, ('', 'cb')), (1, ('c', 'b'))):
        keys = [(f'{player}:{card}:{h}', ('f','c') if h.endswith('b') else ('c','b'))
                for card in range(3) for h in histories]
        values = [value(policy, player, dict(zip((k for k, _ in keys), actions)))
                  for actions in product(*(a for _, a in keys))]
        best.append(max(values) if player == 0 else -min(values))
    return dict(value_player0=value(policy), nash_conv=sum(best), exploitability=sum(best)/2)
