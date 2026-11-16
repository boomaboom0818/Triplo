"""Multi-core version of the lean solver.

Each round forks `workers` processes. Every worker runs `batch` iterations of the
same traversal as poker.lean_cfr against the tables frozen at the start of the
round (plus its own updates) and returns only its deltas; the parent adds them.
Tables live in two flat double arrays that children only read, so forked
workers share them without copying. Results depend only on the seed, worker
count and batch sizes, never on timing.
"""
from array import array
import gc
import json
import multiprocessing
import os
from pathlib import Path
from random import Random
import shutil

from .fast_game import FastGame, LEAN_MENU
from .lean_abstraction import VERSION, card_features, lean_key
from .lean_cfr import ALGORITHM, PLAYERS

STATE_SCHEMA = 'lean-parallel-state-v1'
_SOLVER = None


class Worker:
    """LeanCFR.walk reading frozen base tables and writing local deltas."""
    def __init__(self, index, base, rng):
        self.index, self.base, self.rng = index, base, rng
        self.regrets = {}
        self.sums = {}
        self.nodes = 0

    def iterate(self, weight):
        root = FastGame(self.rng.randrange(2 ** 63), PLAYERS, lean=True)
        for traverser in range(PLAYERS):
            self.cards = {}
            self.walk(root.clone(), traverser, weight)

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
                if x > 0: total += x
            strategy = [x / total if x > 0 else 0. for x in current] if total else [1 / count] * count
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


def _work(job):
    seed, round_index, worker, start, workers, batch = job
    gc.disable()
    solver = _SOLVER
    task = Worker(solver.index, solver.regrets, Random(f'{seed}:{round_index}:{worker}'))
    for j in range(batch):
        task.iterate(start + j * workers + worker + 1)
    return task.regrets, task.sums, task.nodes


class ParallelLean:
    def __init__(self, seed=1):
        self.seed = seed
        self.index = {}                 # key -> offset * 8 + action count
        self.regrets = array('d')
        self.sums = array('d')
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
                gc.freeze()             # keep the collector from dirtying shared pages in children
                with multiprocessing.get_context('fork').Pool(workers) as pool:
                    results = pool.map(_work, jobs, chunksize=1)
                gc.unfreeze()
        finally:
            _SOLVER = None
        for regrets, sums, nodes in results:
            self.nodes += nodes
            for table, delta in ((self.regrets, regrets), (self.sums, sums)):
                for key, row in delta.items():
                    start = self._slot(key, len(row))
                    for j, x in enumerate(row):
                        table[start + j] += x
        self.iterations += workers * batch
        self.rounds += 1

    def _slot(self, key, count):
        slot = self.index.get(key)
        if slot is None:
            slot = self.index[key] = len(self.regrets) * 8 + count
            zeros = [0.] * count
            self.regrets.extend(zeros)
            self.sums.extend(zeros)
        elif slot & 7 != count:
            raise ValueError('information-set action mismatch')
        return slot >> 3

    def rows(self, table):
        for key, slot in self.index.items():
            start = slot >> 3
            yield key, table[start:start + (slot & 7)]

    # --- persistence --------------------------------------------------------
    def meta(self):
        return dict(schema=1, algorithm=ALGORITHM, abstraction=VERSION, menu=LEAN_MENU, players=PLAYERS,
                    seed=self.seed, iterations=self.iterations, nodes=self.nodes, rounds=self.rounds)

    def save_state(self, directory):
        """Compact exact state for resuming; replaced atomically."""
        directory = Path(directory)
        fresh = directory.with_name(directory.name + '.tmp')
        if fresh.exists():
            shutil.rmtree(fresh)
        fresh.mkdir(parents=True)
        with (fresh / 'keys.txt').open('w') as handle:
            for key, slot in self.index.items():
                handle.write(f'{key}\t{slot}\n')
        for name, table in (('regrets.f64', self.regrets), ('sums.f64', self.sums)):
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
        if (meta.get('state_schema') != STATE_SCHEMA or meta.get('algorithm') != ALGORITHM
                or meta.get('abstraction') != VERSION or meta.get('menu') != LEAN_MENU):
            raise ValueError('incompatible lean state')
        solver = cls(meta['seed'])
        solver.iterations, solver.nodes, solver.rounds = meta['iterations'], meta['nodes'], meta['rounds']
        with (directory / 'keys.txt').open() as handle:
            for line in handle:
                key, slot = line.rstrip('\n').split('\t')
                solver.index[key] = int(slot)
        for name, table in (('regrets.f64', solver.regrets), ('sums.f64', solver.sums)):
            with (directory / name).open('rb') as handle:
                table.fromfile(handle, meta['values'])
        if len(solver.index) != meta['rows']:
            raise ValueError('truncated lean state')
        return solver

    def export_policy(self, path):
        """Average strategy only, in the JSON layout poker.lean_cfr.load() reads."""
        path = Path(path)
        temporary = path.with_name(path.name + '.tmp')
        with temporary.open('w') as handle:
            handle.write(json.dumps(dict(self.meta(), policy_only=True, regrets={}))[:-1] + ', "sums": {')
            first = True
            for key, row in self.rows(self.sums):
                if any(row):
                    handle.write(('' if first else ',') + json.dumps(key) + ':' + json.dumps([round(x, 3) for x in row]))
                    first = False
            handle.write('}}')
        os.replace(temporary, path)
