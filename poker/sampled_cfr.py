"""Shared external-sampling kernel for small-game validation and poker training.

States provide terminal, actor, actions, key(), child(action), utility(player).
Simple averaging uses only the next player's pass, as in OpenSpiel. This is a
practical multiplayer estimator, not an unbiased full-average estimator.
"""
from random import Random


class SampledCFR:
    def __init__(self, players, seed=1):
        self.players = players
        self.rng = Random(seed)
        self.regrets, self.sums, self.visits = {}, {}, {}
        self.iterations = self.nodes = 0

    def strategy(self, key, count):
        row = self.regrets.get(key, [0.] * count)
        if len(row) != count:
            raise ValueError('information-set action mismatch')
        positive = [max(0., x) for x in row]
        total = sum(positive)
        return [x / total for x in positive] if total else [1 / count] * count

    def iterate(self, root):
        weight = self.iterations + 1
        for player in range(self.players):
            self.walk(root, player, weight)
        self.iterations += 1

    def walk(self, state, player, weight):
        self.nodes += 1
        if state.terminal:
            return state.utility(player)
        key, actions = state.key(), state.actions
        strategy = self.strategy(key, len(actions))
        if state.actor != player:
            if state.actor == (player + 1) % self.players:
                row = self.sums.setdefault(key, [0.] * len(actions))
                for j, probability in enumerate(strategy):
                    row[j] += weight * probability
                self.visits[key] = self.visits.get(key, 0) + 1
            return self.walk(state.child(self.rng.choices(actions, strategy)[0]), player, weight)
        values = [self.walk(state.child(a), player, weight) for a in actions]
        value = sum(p * v for p, v in zip(strategy, values))
        # Fetch AFTER descent: a lossy abstraction may revisit a coordinate.
        row = self.regrets.setdefault(key, [0.] * len(actions))
        for j, v in enumerate(values):
            row[j] += weight * (v - value)
        return value

    def average(self, key, count):
        row = self.sums.get(key, [0.] * count)
        total = sum(row)
        return [x / total for x in row] if total else [1 / count] * count
