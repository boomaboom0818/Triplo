from random import Random

import pytest

from poker import lean_parallel
from poker.lean_cfr import LeanCFR, LeanPolicy, load
from poker.lean_parallel import ParallelLean, Worker
from scripts.lean_train import batch_size


def tables(solver):
    return ({k: list(v) for k, v in solver.rows(solver.regrets)},
            {k: list(v) for k, v in solver.rows(solver.sums) if any(v)})


def test_worker_from_empty_tables_equals_single_process_solver():
    reference = LeanCFR(9)
    for _ in range(25): reference.iterate()
    worker = Worker({}, [], Random(9))
    for j in range(25): worker.iterate(j + 1)
    assert worker.regrets == reference.regrets and worker.sums == reference.sums and worker.nodes == reference.nodes


def test_worker_reads_frozen_base_plus_own_delta():
    # Splitting a run into a merged base and a continuing worker changes nothing.
    reference = LeanCFR(9)
    for _ in range(30): reference.iterate()
    first = Worker({}, [], Random(9))
    for j in range(12): first.iterate(j + 1)
    solver = ParallelLean(9)
    for key, row in first.regrets.items():
        start = solver._slot(key, len(row))
        for j, x in enumerate(row): solver.regrets[start + j] = x
    second = Worker(solver.index, solver.regrets, first.rng)
    for j in range(12, 30): second.iterate(j + 1)
    merged = {k: list(v) for k, v in first.regrets.items()}
    for key, row in second.regrets.items():
        base = merged.setdefault(key, [0.] * len(row))
        for j, x in enumerate(row): base[j] += x
    assert merged.keys() == reference.regrets.keys()
    assert all(abs(a - b) < 1e-6 for key in merged for a, b in zip(merged[key], reference.regrets[key]))


def test_rounds_are_deterministic_across_processes_and_merge_all_workers():
    a, b = ParallelLean(4), ParallelLean(4)
    for solver in (a, b):
        solver.round(3, 6)
        solver.round(3, 6)
    assert tables(a) == tables(b) and a.nodes == b.nodes and a.iterations == 36 and a.rounds == 2
    assert len(a.regrets) == len(a.sums) == sum(slot & 7 for slot in a.index.values())
    assert lean_parallel._SOLVER is None


def test_state_round_trip_and_policy_export(tmp_path):
    solver = ParallelLean(6)
    solver.round(2, 8)
    solver.save_state(tmp_path / 'state')
    solver.save_state(tmp_path / 'state')          # replacing an existing state
    again = ParallelLean.load_state(tmp_path / 'state')
    assert tables(again) == tables(solver) and again.iterations == 16 and again.rounds == 1
    solver.round(2, 4); again.round(2, 4)
    assert tables(again) == tables(solver)
    solver.export_policy(tmp_path / 'policy.json')
    saved = load(tmp_path / 'policy.json')
    assert saved['iterations'] == 24 and saved['policy_only'] and len(saved['sums']) > 100
    assert LeanPolicy(saved['sums'], 1).sums.keys() == tables(solver)[1].keys()
    (tmp_path / 'state' / 'keys.txt').write_text('')
    with pytest.raises(ValueError):
        ParallelLean.load_state(tmp_path / 'state')


def test_batch_grows_with_experience_and_is_capped():
    assert batch_size(0, 10) == 10 and batch_size(40000, 10) == 1000 and batch_size(10 ** 7, 10) == 2000
