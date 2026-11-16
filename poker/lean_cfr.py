"""Plain external-sampling MCCFR with linear weighting on the fast engine.

One deal per iteration; each seat traverses it once, enumerating its own actions
and sampling everyone else from the current strategy. The average strategy is
accumulated at sampled opponent nodes. No importance weights anywhere, so no
update can dominate the average. No multiplayer equilibrium guarantee.
"""
import json
import math
import os
from pathlib import Path
from random import Random

from .fast_game import FastGame, LEAN_MENU
from .lean_abstraction import VERSION, card_features, lean_key

ALGORITHM = 'lean-linear-es-mccfr-v1'
MIXTURE_ALGORITHM = 'lean-loose-mixture-es-mccfr-v1'
ALGORITHMS = (ALGORITHM, MIXTURE_ALGORITHM)
PLAYERS = 6


class LeanCFR:
    def __init__(self, seed=1):
        self.rng = Random(seed)
        self.seed = seed
        self.regrets = {}
        self.sums = {}
        self.iterations = 0
        self.nodes = 0

    def iterate(self):
        weight = self.iterations + 1
        root = FastGame(self.rng.randrange(2 ** 63), PLAYERS, lean=True)
        for traverser in range(PLAYERS):
            self.cards = {}
            self.walk(root.clone(), traverser, weight)
        self.iterations = weight

    def walk(self, game, traverser, weight):
        self.nodes += 1
        if not game.live:
            return game.payoffs[traverser] / 2
        actor = game.actors[0]
        pot, call, legal = game.menu()
        street = game.street
        cards = self.cards.get((actor, street))
        if cards is None:
            cards = self.cards[actor, street] = card_features(game.hole[actor], tuple(game.board))
        key = lean_key(actor, street, cards, game.stacks, game.active, pot, call, legal)
        count = len(legal)
        row = self.regrets.get(key)
        if row is None:
            strategy = [1 / count] * count
        else:
            total = 0.
            for x in row:
                if x > 0: total += x
            strategy = [x / total if x > 0 else 0. for x in row] if total else [1 / count] * count
        if actor != traverser:
            sums = self.sums.get(key)
            if sums is None:
                sums = self.sums[key] = [0.] * count
            for j in range(count):
                sums[j] += weight * strategy[j]
            game.act(self.rng.choices(legal, strategy)[0], False)
            return self.walk(game, traverser, weight)
        values = [0.] * count
        value = 0.
        last = count - 1
        for j in range(count):
            child = game if j == last else game.clone()
            child.act(legal[j], False)
            values[j] = v = self.walk(child, traverser, weight)
            value += strategy[j] * v
        if row is None:
            row = self.regrets[key] = [0.] * count
        for j in range(count):
            row[j] += weight * (values[j] - value)
        return value

    def save(self, path, **extra):
        path = Path(path)
        temporary = path.with_name(path.name + '.tmp')
        temporary.write_text(json.dumps(dict(
            schema=1, algorithm=ALGORITHM, abstraction=VERSION, menu=LEAN_MENU, players=PLAYERS, seed=self.seed,
            iterations=self.iterations, nodes=self.nodes, rng=self.rng.getstate(),
            regrets=self.regrets, sums=self.sums, **extra)))
        os.replace(temporary, path)


def load(path):
    saved = json.loads(Path(path).read_text())
    if (saved.get('schema') != 1 or saved.get('algorithm') not in ALGORITHMS or saved.get('abstraction') != VERSION
            or saved.get('menu') != LEAN_MENU or saved.get('players') != PLAYERS):
        raise ValueError('incompatible lean checkpoint')
    return saved


def resume(path):
    saved = load(path)
    if saved['algorithm'] != ALGORITHM:
        raise ValueError('mixture checkpoints require the parallel trainer')
    solver = LeanCFR(saved['seed'])
    state = saved['rng']
    solver.rng.setstate((state[0], tuple(state[1]), state[2]))
    solver.regrets, solver.sums = saved['regrets'], saved['sums']
    solver.iterations, solver.nodes = saved['iterations'], saved['nodes']
    return solver


class LeanPolicy:
    """Average-strategy player for lean checkpoints.

    States the training never produced (for example a 100bb all-in, which the lean
    menu does not contain) are played by a fixed tight rule player instead of
    folding everything: it calls big bets with premium hands and folds the rest.
    """
    abstraction = VERSION
    menu = LEAN_MENU

    def __init__(self, sums, seed=0, fallback='tight-v1', min_average_mass=0.):
        from .lean_abstraction import observation_key
        from .rule_opponents import RulePolicy
        self.sums, self.rng, self.key = sums, Random(seed), observation_key
        self.fallback = RulePolicy(fallback)
        if not math.isfinite(min_average_mass) or min_average_mass < 0:
            raise ValueError('minimum average mass must be finite and nonnegative')
        self.min_average_mass = min_average_mass
        self.decisions = self.covered = 0
        self.thin = 0

    @classmethod
    def from_checkpoint(cls, saved, seed=0, min_support=0.):
        """Optional fallback for sparse rows, preserving the default policy.

        Support is linear average mass / sum(1..iterations), NOT a visit count
        or effective sample size. Multiple traversals can contribute per deal.
        This heuristic must be evaluated against both raw and fallback policies.
        """
        if not math.isfinite(min_support) or min_support < 0:
            raise ValueError('minimum support must be finite and nonnegative')
        iterations = saved['iterations']
        if type(iterations) is not int or iterations < 0:
            raise ValueError('invalid checkpoint iterations')
        return cls(saved['sums'], seed, min_average_mass=min_support * iterations * (iterations + 1) / 2)

    def act(self, o):
        row = self.sums.get(self.key(o))
        self.decisions += 1
        if row and len(row) == len(o.legal) and any(x > 0 for x in row):
            if sum(row) < self.min_average_mass:
                self.thin += 1
                return self.fallback.act(o)
            self.covered += 1
            return self.rng.choices(o.legal, row)[0]
        return self.fallback.act(o)
