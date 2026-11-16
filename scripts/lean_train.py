"""Train the lean line on several cores.

python -m scripts.lean_train --root runs/bot/lean-002 --minutes 120 --workers 10 [--resume]

One writer per root (lock file), finite deadline, 10 GiB address-space limit per
process, atomic state directory, policy snapshots at requested iteration marks.
Stops early, keeping the last state, when table or host limits are reached.
"""
import argparse
import json
import os
import resource
import shutil
import time
from pathlib import Path

from poker.lean_parallel import ParallelLean
from poker.train_cfr import run_lock
from scripts.scaling_validation import available_memory

MEMORY = 10 * 1024 ** 3
HOST_RESERVE = 2 * 1024 ** 3
DISK_RESERVE = 10 * 1024 ** 3
MAX_BATCH = 2000


def batch_size(iterations, workers):
    """Short rounds early so no round adds more than a quarter of all experience."""
    return max(10, min(MAX_BATCH, iterations // (4 * workers)))


def train(root, minutes, workers, seed=1, resuming=False, save_minutes=20, max_rows=20_000_000, snapshots=()):
    if not 0 < minutes <= 24 * 60:
        raise ValueError('finite run length up to 24 hours required')
    if not 1 <= workers <= max(1, (os.cpu_count() or 1) - 1):
        raise ValueError('leave at least one core free')
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY, MEMORY))
    os.nice(10)
    state = root / 'state'
    with run_lock(root / 'trainer'):
        if state.exists() != resuming:
            raise ValueError('use --resume exactly when a saved state exists')
        solver = ParallelLean.load_state(state) if resuming else ParallelLean(seed)
        start_iterations, start_nodes = solver.iterations, solver.nodes
        start = time.monotonic()
        end = start + 60 * minutes
        next_save = start + 60 * save_minutes
        pending = sorted(s for s in snapshots if s > solver.iterations)
        reason = 'deadline'

        def write(status):
            elapsed = time.monotonic() - start
            (root / 'progress.json').write_text(json.dumps(dict(
                status=status, iterations=solver.iterations, nodes=solver.nodes, rows=len(solver.index),
                rounds=solver.rounds, workers=workers, session_seconds=elapsed, pid=os.getpid(),
                iterations_per_second=(solver.iterations - start_iterations) / max(elapsed, 1e-9),
                nodes_per_second=(solver.nodes - start_nodes) / max(elapsed, 1e-9),
                host_available_bytes=available_memory(),
                deadline_unix=time.time() + max(0, end - time.monotonic())), indent=1))

        try:
            while time.monotonic() < end:
                solver.round(workers, batch_size(solver.iterations, workers))
                if pending and solver.iterations >= pending[0]:
                    solver.export_policy(root / f'snapshot-{solver.iterations}.json')
                    pending = [s for s in pending if s > solver.iterations]
                if len(solver.index) > max_rows:
                    reason = 'row limit'; break
                if available_memory() < HOST_RESERVE or shutil.disk_usage(root).free < DISK_RESERVE:
                    reason = 'host reserve'; break
                if time.monotonic() >= next_save:
                    solver.save_state(state)
                    next_save = time.monotonic() + 60 * save_minutes
                write('running')
        finally:
            solver.save_state(state)
            solver.export_policy(root / 'latest-policy.json')
            write('stopped: ' + reason)
    return solver


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--minutes', required=True, type=float)
    parser.add_argument('--workers', required=True, type=int)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--save-minutes', type=float, default=20)
    parser.add_argument('--snapshots', type=int, nargs='*', default=[])
    args = parser.parse_args()
    train(args.root, args.minutes, args.workers, args.seed, args.resume, args.save_minutes, snapshots=args.snapshots)
