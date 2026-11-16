"""Train/resume the versioned history model with finite local resource limits.

Supports both single-process training (workers=1, default for backwards-compatible
tests) and multi-core shared-memory sustained training (--workers N --population P).
"""
import argparse
import json
import os
import resource
import signal
import shutil
import time
from pathlib import Path

from poker.fast_game import FastGame
from poker.history_cfr import COMPACT_VERSION, VERSION, VERSIONS, ParallelHistory, PokerState, save, resume
from poker.sampled_cfr import SampledCFR
from poker.train_cfr import run_lock
from scripts.scaling_validation import available_memory

MAX_BATCH = 500


def batch_size(iterations, workers):
    return max(10, min(MAX_BATCH, max(10, iterations // (4 * max(1, workers)))))


def train(root, minutes, seed=1, stack_bb=20, resuming=False, max_rows=500000,
          workers=1, population=0., abstraction=VERSION, save_minutes=5, snapshots=(), curriculum=None, initialize_from=None):
    if not 0 < minutes <= 24 * 60 or type(stack_bb) is not int or not 2 <= stack_bb <= 200:
        raise ValueError('minutes must be in (0,1440], stack_bb an integer in [2,200]')
    if type(max_rows) is not int or max_rows < 1:
        raise ValueError('positive row limit required')
    if type(workers) is not int or not 1 <= workers <= max(1, (os.cpu_count() or 1) - 1):
        raise ValueError('workers must be between 1 and cpu_count - 1')
    if curriculum not in (None, 'live-conditions-v1', 'live-aggressive-v2') or curriculum and abstraction != COMPACT_VERSION:
        raise ValueError('live curriculum requires compact abstraction')
    if initialize_from and (resuming or curriculum is None):
        raise ValueError('initialization requires a new curriculum run')
    if abstraction not in VERSIONS:
        raise ValueError('unsupported history abstraction')
    mem_limit = (10 if workers > 1 else 4) * 1024 ** 3
    resource.setrlimit(resource.RLIMIT_AS, (mem_limit, mem_limit))
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with run_lock(root / 'trainer'):
        if workers == 1 and population == 0. and abstraction == VERSION:
            path = root / 'checkpoint.json'
            if path.exists() != resuming:
                raise ValueError('use --resume exactly when checkpoint exists')
            if resuming:
                solver, saved = resume(path)
                if saved['seed'] != seed or saved['stack_bb'] != stack_bb:
                    raise ValueError('resume seed/stack configuration mismatch')
            else:
                solver = SampledCFR(6, seed)
            start = time.monotonic()
            next_save = start + 30
            reason = 'deadline'
            stopping = False
            completed = True

            def interrupted(signum, frame):
                nonlocal stopping
                stopping = True

            previous = {s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGINT)}
            try:
                while time.monotonic() - start < minutes * 60 and not stopping:
                    if len(solver.regrets) >= max_rows or len(solver.sums) >= max_rows:
                        reason = 'row limit'
                        break
                    if available_memory() < 2 * 1024 ** 3 or shutil.disk_usage(root).free < 10 * 1024 ** 3:
                        reason = 'host reserve'
                        break
                    completed = False
                    solver.iterate(PokerState(FastGame(solver.rng.randrange(2 ** 63), stacks=[2 * stack_bb] * 6, lean=True)))
                    completed = True
                    if time.monotonic() >= next_save:
                        save(path, solver, seed, stack_bb)
                        next_save = time.monotonic() + 30
                if stopping:
                    reason = 'interrupted'
            finally:
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
                if completed:
                    save(path, solver, seed, stack_bb)
                else:
                    reason = 'failed mid-iteration; last complete checkpoint retained'
                (root / 'progress.json').write_text(json.dumps(dict(
                    iterations=solver.iterations, nodes=solver.nodes,
                    rows=len(solver.regrets), average_rows=len(solver.sums),
                    seconds=time.monotonic() - start, stop_reason=reason, stack_bb=stack_bb), indent=2))
            return solver

        # Multi-core or compact/mixture sustained training mode
        state_dir = root / 'state'
        if state_dir.exists() != resuming:
            raise ValueError('use --resume exactly when state directory exists')
        if resuming:
            solver = ParallelHistory.load_state(state_dir)
            if (solver.seed != seed or solver.stack_bb != stack_bb
                    or solver.population_probability != population or solver.abstraction != abstraction or solver.curriculum != curriculum):
                raise ValueError('resume configuration mismatch')
        elif initialize_from:
            solver = ParallelHistory.load_state(initialize_from)
            if solver.abstraction != abstraction:
                raise ValueError('initialization abstraction mismatch')
            solver.seed, solver.stack_bb = seed, stack_bb
            solver.population_probability, solver.curriculum = population, curriculum
        else:
            solver = ParallelHistory(seed, stack_bb, population, abstraction, curriculum)
        start_iterations, start_nodes = solver.iterations, solver.nodes
        start = time.monotonic()
        end = start + 60 * minutes
        next_save = start + 60 * save_minutes
        pending = sorted(s for s in snapshots if s > solver.iterations)
        reason = 'deadline'
        stopping = False
        completed = True

        def interrupted(signum, frame):
            nonlocal stopping
            stopping = True

        previous = {s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGINT)}

        def write_progress(status):
            elapsed = time.monotonic() - start
            (root / 'progress.json').write_text(json.dumps(dict(
                status=status, stop_reason=reason, iterations=solver.iterations, nodes=solver.nodes,
                rows=len(solver.index), rounds=solver.rounds, workers=workers, stack_bb=stack_bb,
                abstraction=abstraction, population_probability=population, curriculum=curriculum,
                population_iterations=solver.population_iterations,
                seconds=elapsed, pid=os.getpid(),
                iterations_per_second=(solver.iterations - start_iterations) / max(elapsed, 1e-9),
                nodes_per_second=(solver.nodes - start_nodes) / max(elapsed, 1e-9),
                host_available_bytes=available_memory()), indent=2))

        try:
            while time.monotonic() < end and not stopping:
                if len(solver.index) >= max_rows:
                    reason = 'row limit'
                    break
                if available_memory() < 2 * 1024 ** 3 or shutil.disk_usage(root).free < 10 * 1024 ** 3:
                    reason = 'host reserve'
                    break
                completed = False
                solver.round(workers, batch_size(solver.iterations, workers))
                completed = True
                if pending and solver.iterations >= pending[0]:
                    solver.export_policy(root / f'snapshot-{solver.iterations}.json')
                    pending = [s for s in pending if s > solver.iterations]
                if time.monotonic() >= next_save:
                    solver.save_state(state_dir)
                    solver.export_policy(root / 'checkpoint.json')
                    next_save = time.monotonic() + 60 * save_minutes
                write_progress('running')
            if stopping:
                reason = 'interrupted'
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
            if completed:
                solver.save_state(state_dir)
                solver.export_policy(root / 'checkpoint.json')
            else:
                reason = 'failed mid-round; last complete state retained'
            write_progress('stopped: ' + reason)
        return solver


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--minutes', required=True, type=float)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--stack-bb', type=int, default=20)
    p.add_argument('--max-rows', type=int, default=5_000_000)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--population', type=float, default=0.,
                   help='fraction of deals with 3 loose callers (0..1)')
    p.add_argument('--abstraction', choices=VERSIONS, default=VERSION)
    p.add_argument('--save-minutes', type=float, default=5)
    p.add_argument('--snapshots', type=int, nargs='*', default=[])
    p.add_argument('--curriculum', choices=['live-conditions-v1', 'live-aggressive-v2'])
    p.add_argument('--initialize-from', type=Path, help='fork optimizer state into a new curriculum run; source is never written')
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    train(a.root, a.minutes, a.seed, a.stack_bb, a.resume, a.max_rows,
          a.workers, a.population, a.abstraction, a.save_minutes, a.snapshots, a.curriculum, a.initialize_from)

