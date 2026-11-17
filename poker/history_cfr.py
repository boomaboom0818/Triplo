"""Versioned history-aware Hold'em adapter and multi-core sustained trainer.

Supports both exact public history keys (history-public-v1) and compact
pot-relative history keys (history-compact-v1) with multi-core shared-memory
training (ParallelHistory) and loose-caller population mixing.
"""
from array import array
import gc
import json
import math
import multiprocessing
import os
import shutil
from pathlib import Path
from random import Random

from .fast_game import FastGame, LEAN_MENU
from .lean_abstraction import card_features, lean_key
from .fast_eval import CARD_ID, evaluate
from .loose_opponents import LooseCaller, PROFILES, VERSION as POPULATION_VERSION
from .rule_opponents import RulePolicy
from .sampled_cfr import SampledCFR

ALGORITHM = 'history-linear-es-v1'
PARALLEL_ALGORITHM = 'history-parallel-es-v1'
MIXTURE_ALGORITHM = 'history-loose-mixture-es-v1'
LIVE_ALGORITHM = 'history-live-mixture-es-v1'
ALGORITHMS = (ALGORITHM, PARALLEL_ALGORITHM, MIXTURE_ALGORITHM, LIVE_ALGORITHM)
VERSION = 'history-public-v1'
COMPACT_VERSION = 'history-compact-v1'
VERSIONS = (VERSION, COMPACT_VERSION)
STATE_SCHEMA = 'history-parallel-state-v1'
PLAYERS = 6
_SOLVER = None


def observation_key(o):
    hole = tuple(CARD_ID[c] for c in o.hole)
    board = tuple(CARD_ID[c] for c in o.board)
    history = [(position, street, action) for _, position, street, action, _, _ in o.history]
    # Exact stacks, prices and action menu preserve action-coordinate meaning.
    return json.dumps([o.position, o.active, card_features(hole, board), o.pot, o.call,
                       o.stacks, o.legal, history], separators=(',', ':'))


def _raise_token(action, pot, call):
    if action == 'fold':
        return 'f'
    if action == 'call':
        return 'k' if not call else 'c'
    ratio = (action - call) / max(1, pot)
    return 's' if ratio < 0.7 else 'p' if ratio < 1.4 else 'o'


def compact_features(position, street, cards, stacks, active, pot, call, legal, history):
    base = lean_key(position, street, cards, stacks, active, pot, call, legal)
    raises = [0, 0, 0, 0]
    calls = 0
    last_agg = -1
    last_tok = '-'
    for _, pos, st, act, before_pot, owed in history:
        tok = _raise_token(act, before_pot, owed)
        if tok in ('s', 'p', 'o'):
            raises[st] = min(4, raises[st] + 1)
            last_agg = pos
            if st == street:
                calls = 0
                last_tok = tok
        elif st == street and tok in ('c', 'k'):
            calls = min(3, calls + 1)
    if last_agg < 0:
        agg_rel = '-'
    elif last_agg == position:
        agg_rel = 'h'
    elif not active[last_agg]:
        agg_rel = 'x'
    else:
        agg_rel = str(sum(active[:last_agg]))
    p0 = min(2, raises[0]) if street > 0 else raises[0]
    p1 = min(2, raises[1]) if street > 1 else raises[1]
    p2 = min(2, raises[2]) if street > 2 else raises[2]
    p3 = raises[3]
    menu = ''.join('s' if (a - call) / max(1, pot) < 0.7 else 'p' if (a - call) / max(1, pot) < 1.4 else 'o'
                   for a in legal if isinstance(a, int))
    return f'{base}|{p0}{p1}{p2}{p3}{agg_rel}{last_tok}{calls}|{menu}'


def compact_observation_key(o):
    hole = tuple(CARD_ID[c] for c in o.hole)
    board = tuple(CARD_ID[c] for c in o.board)
    return compact_features(o.position, o.street, card_features(hole, board),
                            o.stacks, o.active, o.pot, o.call, o.legal, o.history)


class PokerState:
    def __init__(self, game):
        self.game = game

    @property
    def terminal(self): return not self.game.live

    @property
    def actor(self): return self.game.actors[0]

    @property
    def actions(self): return self.game.menu()[2]

    def key(self): return observation_key(self.game.observe())

    def child(self, action):
        game = self.game.clone()
        game.act(action)  # checked=True also records the public betting history
        return PokerState(game)

    def utility(self, player): return self.game.payoffs[player] / 2


class _HistoryWorker:
    """Multi-core worker traversing history-aware games against frozen shared arrays."""
    def __init__(self, index, base, rng, stack_bb=100, population_probability=0., abstraction=COMPACT_VERSION, curriculum=None):
        self.index, self.base, self.rng = index, base, rng
        self.curriculum = curriculum
        self.stack_bb = stack_bb
        self.population_probability = population_probability
        self.abstraction = abstraction
        self.regrets = {}
        self.sums = {}
        self.visits = {}
        self.nodes = 0
        self.fixed = {}
        self.population_iterations = 0

    def iterate(self, weight):
        stacks = [2 * self.stack_bb] * PLAYERS
        if self.curriculum:
            from .live_curriculum import sample_root
            root = sample_root(self.rng, self.curriculum)
        else:
            root = FastGame(self.rng.randrange(2 ** 63), PLAYERS, stacks=stacks, lean=True)
        population = self.population_probability > 0 and self.rng.random() < self.population_probability
        self.population_iterations += population
        for traverser in range(PLAYERS):
            self.fixed = {}
            if population:
                eligible = [i for i in range(PLAYERS) if i != traverser and root.active[i]]
                seats = self.rng.sample(eligible, min(3, len(eligible)))
                if self.curriculum:
                    from .live_curriculum import opponent
                    self.fixed = {i: opponent(self.rng, self.curriculum) for i in seats}
                else:
                    self.fixed = {i: LooseCaller(self.rng.randrange(len(PROFILES))) for i in seats}
            self.cards = {}
            game = root.clone()
            if self.curriculum:
                game.lean = 63 & ~sum(1 << i for i in self.fixed)
            self.walk(game, traverser, weight)

    def walk(self, game, traverser, weight):
        self.nodes += 1
        if not game.live:
            return game.payoffs[traverser] / 2
        actor = game.actors[0]
        if actor in self.fixed:
            game.act(self.fixed[actor].act(game.observe()))
            return self.walk(game, traverser, weight)
        pot, call, legal = game.menu()
        street = game.street
        cards = self.cards.get((actor, street))
        if cards is None:
            cards = self.cards[actor, street] = card_features(game.hole[actor], tuple(game.board))
        if self.abstraction == COMPACT_VERSION:
            key = compact_features(actor, street, cards, game.stacks, game.active, pot, call, legal, game.history)
        else:
            key = observation_key(game.observe())
        count = len(legal)
        row = self.regrets.get(key)
        slot = self.index.get(key)
        if slot is None:
            current = row
        else:
            start = slot >> 3
            current = self.base[start:start + count]
            if row is not None:
                current = [a + b for a, b in zip(current, row)]
        if current is None:
            strategy = [1 / count] * count
        else:
            total = 0.
            for x in current:
                if x > 0:
                    total += x
            strategy = [x / total if x > 0 else 0. for x in current] if total else [1 / count] * count
        if actor != traverser:
            if actor == (traverser + 1) % PLAYERS:
                sums = self.sums.get(key)
                if sums is None:
                    sums = self.sums[key] = [0.] * count
                for j in range(count):
                    sums[j] += weight * strategy[j]
                self.visits[key] = self.visits.get(key, 0) + 1
            game.act(self.rng.choices(legal, strategy)[0])
            return self.walk(game, traverser, weight)
        values = [0.] * count
        value = 0.
        last = count - 1
        for j in range(count):
            child = game if j == last else game.clone()
            child.act(legal[j])
            values[j] = v = self.walk(child, traverser, weight)
            value += strategy[j] * v
        row = self.regrets.setdefault(key, [0.] * count)
        for j in range(count):
            row[j] += weight * (values[j] - value)
        return value


def _work(job):
    seed, round_index, worker, start, workers, batch = job
    gc.disable()
    solver = _SOLVER
    task = _HistoryWorker(solver.index, solver.regrets, Random(f'{seed}:{round_index}:{worker}'),
                          solver.stack_bb, solver.population_probability, solver.abstraction, solver.curriculum)
    for j in range(batch):
        task.iterate(start + j * workers + worker + 1)
    return task.regrets, task.sums, task.visits, task.nodes, task.population_iterations


class ParallelHistory:
    def __init__(self, seed=1, stack_bb=100, population_probability=0., abstraction=COMPACT_VERSION, curriculum=None):
        if type(stack_bb) is not int or not 2 <= stack_bb <= 200:
            raise ValueError('stack_bb must be an integer in [2, 200]')
        if not math.isfinite(population_probability) or not 0 <= population_probability <= 1:
            raise ValueError('population probability must be between zero and one')
        if abstraction not in VERSIONS:
            raise ValueError('unsupported history abstraction')
        if curriculum not in (None, 'live-conditions-v1', 'live-aggressive-v2') or curriculum and abstraction != COMPACT_VERSION:
            raise ValueError('live curriculum requires compact history abstraction')
        self.seed = seed
        self.curriculum = curriculum
        self.stack_bb = stack_bb
        self.population_probability = population_probability
        self.abstraction = abstraction
        self.population_iterations = 0
        self.index = {}
        self.regrets = array('d')
        self.sums = array('d')
        self.visits = array('Q')
        self.iterations = self.nodes = self.rounds = 0

    def round(self, workers, batch):
        global _SOLVER
        jobs = [(self.seed, self.rounds, k, self.iterations, workers, batch) for k in range(workers)]
        _SOLVER = self
        try:
            if workers == 1:
                results = [_work(jobs[0])]
                gc.enable()
            else:
                gc.freeze()
                with multiprocessing.get_context('fork').Pool(workers) as pool:
                    results = pool.map(_work, jobs, chunksize=1)
                gc.unfreeze()
        finally:
            _SOLVER = None
        for regrets, sums, visits, nodes, pop_iters in results:
            self.nodes += nodes
            self.population_iterations += pop_iters
            for table, delta in ((self.regrets, regrets), (self.sums, sums)):
                for key, row in delta.items():
                    start = self._slot(key, len(row))
                    for j, x in enumerate(row):
                        table[start + j] += x
            for key, count in visits.items():
                slot = self.index[key]
                self.visits[slot >> 3] += count
        self.iterations += workers * batch
        self.rounds += 1

    def _slot(self, key, count):
        slot = self.index.get(key)
        if slot is None:
            slot = self.index[key] = len(self.regrets) * 8 + count
            zeros = [0.] * count
            self.regrets.extend(zeros)
            self.sums.extend(zeros)
            self.visits.extend([0] * count)
        elif slot & 7 != count:
            raise ValueError('information-set action mismatch')
        return slot >> 3

    def rows(self, table):
        for key, slot in self.index.items():
            start = slot >> 3
            yield key, table[start:start + (slot & 7)]

    def meta(self):
        return dict(schema=1,
                    algorithm=LIVE_ALGORITHM if self.curriculum else MIXTURE_ALGORITHM if self.population_probability else PARALLEL_ALGORITHM,
                    curriculum=self.curriculum,
                    abstraction=self.abstraction, menu=LEAN_MENU, players=PLAYERS,
                    seed=self.seed, stack_bb=self.stack_bb, iterations=self.iterations,
                    nodes=self.nodes, rounds=self.rounds,
                    population_probability=self.population_probability,
                    population_version=POPULATION_VERSION,
                    population_iterations=self.population_iterations)

    def save_state(self, directory):
        directory = Path(directory)
        fresh = directory.with_name(directory.name + '.tmp')
        if fresh.exists():
            shutil.rmtree(fresh)
        fresh.mkdir(parents=True)
        with (fresh / 'keys.txt').open('w') as handle:
            for key, slot in self.index.items():
                handle.write(f'{key}\t{slot}\n')
        for name, table in (('regrets.f64', self.regrets), ('sums.f64', self.sums), ('visits.u64', self.visits)):
            with (fresh / name).open('wb') as handle:
                table.tofile(handle)
        (fresh / 'meta.json').write_text(json.dumps(dict(self.meta(), state_schema=STATE_SCHEMA,
                                                         rows=len(self.index), values=len(self.regrets))))
        old = directory.with_name(directory.name + '.old')
        if old.exists():
            shutil.rmtree(old)
        if directory.exists():
            os.replace(directory, old)
        os.replace(fresh, directory)
        if old.exists():
            shutil.rmtree(old)

    @classmethod
    def load_state(cls, directory):
        directory = Path(directory)
        meta = json.loads((directory / 'meta.json').read_text())
        if (meta.get('state_schema') != STATE_SCHEMA or meta.get('algorithm') not in ALGORITHMS
                or meta.get('abstraction') not in VERSIONS or meta.get('menu') != LEAN_MENU):
            raise ValueError('incompatible parallel history state')
        solver = cls(meta['seed'], meta['stack_bb'], meta.get('population_probability', 0.), meta['abstraction'], meta.get('curriculum'))
        if meta['algorithm'] != solver.meta()['algorithm']:
            raise ValueError('incompatible population configuration')
        solver.population_iterations = meta.get('population_iterations', 0)
        solver.iterations, solver.nodes, solver.rounds = meta['iterations'], meta['nodes'], meta['rounds']
        with (directory / 'keys.txt').open() as handle:
            for line in handle:
                key, slot = line.rstrip('\n').split('\t')
                solver.index[key] = int(slot)
        for name, table in (('regrets.f64', solver.regrets), ('sums.f64', solver.sums), ('visits.u64', solver.visits)):
            with (directory / name).open('rb') as handle:
                table.fromfile(handle, meta['values'])
        if len(solver.index) != meta['rows']:
            raise ValueError('truncated parallel history state')
        return solver

    def export_policy(self, path):
        path = Path(path)
        temporary = path.with_name(path.name + '.tmp')
        sums = {}
        visits = {}
        for key, slot in self.index.items():
            start = slot >> 3
            count = slot & 7
            row = self.sums[start:start + count]
            if any(row):
                sums[key] = [round(x, 3) for x in row]
                visits[key] = int(self.visits[start])
        temporary.write_text(json.dumps(dict(self.meta(), policy_only=True, regrets={}, sums=sums, visits=visits)))
        os.replace(temporary, path)


def save(path, solver, seed, stack_bb, abstraction=VERSION):
    data = dict(schema=1, algorithm=ALGORITHM, abstraction=abstraction, menu=LEAN_MENU,
                players=6, seed=seed, stack_bb=stack_bb, iterations=solver.iterations,
                nodes=solver.nodes, rng=solver.rng.getstate(), regrets=solver.regrets,
                sums=solver.sums, visits=solver.visits)
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data))
    temporary.replace(path)


def load(path):
    data = json.loads(Path(path).read_text())
    if (data.get('schema') != 1 or data.get('algorithm') not in ALGORITHMS
            or data.get('abstraction') not in VERSIONS
            or data.get('menu') != LEAN_MENU or data.get('players') != 6
            or type(data.get('stack_bb')) is not int or data['stack_bb'] < 2):
        raise ValueError('incompatible history checkpoint')
    return data


def resume(path):
    data = load(path)
    if data['algorithm'] != ALGORITHM:
        raise ValueError('parallel history checkpoints resume via ParallelHistory.load_state')
    solver = SampledCFR(6, data['seed'])
    state = data['rng']
    solver.rng.setstate((state[0], tuple(state[1]), state[2]))
    solver.regrets, solver.sums, solver.visits = data['regrets'], data['sums'], data['visits']
    solver.iterations, solver.nodes = data['iterations'], data['nodes']
    return solver, data


class HistoryPolicy:
    abstraction = VERSION
    menu = LEAN_MENU
    key = staticmethod(observation_key)

    def __init__(self, saved, seed=0, min_visits=64, learned=True, preflop_commitment=None):
        if type(min_visits) is not int or min_visits < 1:
            raise ValueError('positive integer minimum visits required')
        self.sums, self.visits = saved['sums'], saved['visits']
        self.stack_bb = saved['stack_bb']
        self.abstraction = saved.get('abstraction', VERSION)
        self.key = compact_observation_key if self.abstraction == COMPACT_VERSION else observation_key
        self.rng = Random(seed)
        self.min_visits, self.learned = min_visits, learned
        self.fallback = RulePolicy('tight-v1')
        self.decisions = self.covered = self.safeguards = self.thin = 0
        self.pressure_guard = None
        self.pressure_overrides = 0
        self.commitment_overrides = 0
        # Opt-in. A partial ledger undercounts chips already invested, so the
        # caller must attest that the public history is complete.
        self.preflop_commitment = bool(preflop_commitment if preflop_commitment is not None
                                       else saved.get('preflop_commitment', False))
        if saved.get('decision_guard') is not None:
            from .equity_policy import PressureGuard
            self.pressure_guard = PressureGuard(saved['decision_guard'])

    def _apply_commitment(self, o, action, history_complete):
        if not self.preflop_commitment or history_complete is not True or o.street or action == 'fold':
            return action
        from .preflop_commitment import guard_preflop
        limited = guard_preflop(o, action)
        if limited != action:
            self.commitment_overrides += 1
        return limited

    def act(self, o, history_complete=None):
        self.decisions += 1
        # Hard poker invariants take precedence over both learned and rule play.
        if o.street == 3:
            hole = tuple(CARD_ID[c] for c in o.hole)
            board = tuple(CARD_ID[c] for c in o.board)
            score = evaluate(hole + board)
            rest = [c for c in range(52) if c not in hole + board]
            best = max(evaluate((a,b) + board) for j,a in enumerate(rest) for b in rest[j+1:])
            if score >= best:
                # Never fold the nuts; avoid pointless raises when the board plays.
                self.safeguards += 1
                raises = [a for a in o.legal if isinstance(a, int)]
                return self._apply_commitment(o, raises[-1] if score > best and raises else 'call', history_complete)
        if not o.street and o.hole[0][0] == o.hole[1][0] == 'A':
            self.safeguards += 1
            return self._apply_commitment(o, self.fallback.act(o), history_complete)
        # Live callers explicitly attest replay completeness. A partial history
        # must never reuse a root-like or otherwise populated table coordinate.
        history_present = bool(o.history) or (not o.street and o.position == 2 and o.pot == 3)
        if history_complete is not None:
            history_present = history_complete
        key = self.key(o)
        row = self.sums.get(key)
        if self.learned and history_present and row and len(row) == len(o.legal) and sum(row) > 0:
            if self.visits.get(key, 0) >= self.min_visits:
                action = self.rng.choices(o.legal, row)[0]
                # Folding a free check is dominated.
                if action == 'fold' and not o.call:
                    self.safeguards += 1
                    return self._apply_commitment(o, 'call', history_complete)
                if self.pressure_guard is not None:
                    guarded = self.pressure_guard.act(o, action)
                    if guarded != action:
                        self.pressure_overrides += 1
                        return self._apply_commitment(o, guarded, history_complete)
                limited = self._apply_commitment(o, action, history_complete)
                if limited == action:
                    self.covered += 1
                return limited
            self.thin += 1
        return self._apply_commitment(o, self.fallback.act(o), history_complete)

