import json
from random import Random

import pytest

from poker.fast_game import FastGame
from poker.lean_cfr import MIXTURE_ALGORITHM, load, resume
from poker.lean_parallel import ParallelLean, Worker
from poker.loose_opponents import LooseCaller, PROFILES


def test_loose_callers_are_legal_and_allow_multiway_flops():
    flops = 0
    for seed in range(30):
        game = FastGame(seed, lean=True)
        while game.live:
            if game.street == 1 and sum(game.active) >= 4:
                flops += 1
            game.act(LooseCaller(game.actors[0] % len(PROFILES)).act(game.observe()))
    assert flops > 30


def test_fixed_opponent_utility_without_policy_contamination(monkeypatch):
    import poker.lean_parallel as module
    class Game:
        live = True
        actors = [0]
        street = 0
        hole = [(0, 1), (2, 3)]
        board = ()
        stacks = [20, 20]
        active = [True, True]
        def observe(self): return 'private observation'
        def menu(self): return 3, 1, ('call', 'fold')
        def clone(self):
            result = Game()
            result.actors = self.actors[:]
            return result
        def act(self, action, checked):
            if self.actors == [0]:
                assert action == 'call'
                self.actors = [1]
            else:
                self.live = False
                self.payoffs = [0, 2 if action == 'call' else -2]
    class Fixed:
        def act(self, o):
            assert o == 'private observation'
            return 'call'
    monkeypatch.setattr(module, 'card_features', lambda *args: 'cards')
    monkeypatch.setattr(module, 'lean_key', lambda actor, *args: str(actor))
    worker = Worker({}, [], Random(1), 1.)
    worker.fixed = {0: Fixed()}
    worker.cards = {}
    assert worker.walk(Game(), 1, 2) == 0
    assert worker.regrets == {'1': [2., -2.]}
    assert worker.sums == {}


def test_population_resume_and_export(tmp_path):
    a = ParallelLean(31, .5)
    a.round(2, 5)
    a.save_state(tmp_path / 'state')
    b = ParallelLean.load_state(tmp_path / 'state')
    a.round(2, 5)
    b.round(2, 5)
    assert a.index == b.index and a.regrets == b.regrets and a.sums == b.sums
    assert a.meta() == b.meta() and 0 < a.population_iterations < a.iterations
    a.export_policy(tmp_path / 'policy.json')
    assert load(tmp_path / 'policy.json')['algorithm'] == MIXTURE_ALGORITHM
    with pytest.raises(ValueError, match='parallel'):
        resume(tmp_path / 'policy.json')
    meta = json.loads((tmp_path / 'state/meta.json').read_text())
    meta['population_probability'] = 0
    (tmp_path / 'state/meta.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='population'):
        ParallelLean.load_state(tmp_path / 'state')


def test_population_rotates_all_learners():
    worker = Worker({}, [], Random(22), 1.)
    for weight in range(1, 6):
        worker.iterate(weight)
    assert worker.population_iterations == 5
    assert {int(k[1]) for k in worker.regrets} == set(range(6))
    assert {int(k[1]) for k in worker.sums} == set(range(6))


def test_pilot_keeps_parent_and_records_paired_controls(tmp_path):
    import subprocess
    import sys
    parent = tmp_path / 'parent'
    root = tmp_path / 'pilot'
    solver = ParallelLean(1)
    solver.round(1, 2)
    solver.save_state(parent)
    before = {p.name: p.read_bytes() for p in parent.iterdir()}
    # Tiny synthetic run: /tmp may be a tmpfs smaller than the production reserve.
    launch = ("import runpy; import scripts.lean_train as limits; "
              "limits.DISK_RESERVE = 0; limits.HOST_RESERVE = 0; "
              "runpy.run_module('scripts.lean_mixture_pilot', run_name='__main__')")
    subprocess.run([sys.executable, '-c', launch,
                    '--parent', str(parent), '--root', str(root), '--iterations', '4',
                    '--workers', '1', '--batch', '2', '--deals', '2', '--arm-seconds', '30'],
                   check=True, capture_output=True, timeout=60)
    assert {p.name: p.read_bytes() for p in parent.iterdir()} == before
    report = json.loads((root / 'summary.json').read_text())
    assert report['parent_unchanged'] and len(report['training']) == 4
    assert set(report['paired']['mixture-71-raw']) == {'control-71-raw', 'parent-raw', 'fallback'}
    assert ParallelLean.load_state(root / 'mixture-71-state').population_probability == .5


def test_pilot_rejects_zero_workers_before_creating_output():
    from argparse import Namespace
    from scripts.lean_mixture_pilot import run
    with pytest.raises(ValueError, match='pilot limits'):
        run(Namespace(workers=0, batch=1, deals=2, arm_seconds=30, iterations=4))
