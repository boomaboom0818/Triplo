"""Train the lean line: plain linear external-sampling MCCFR on the fast engine.

python -m scripts.lean_train --root runs/bot/lean-001 --minutes 60 [--resume]

Single process, 4 GiB address-space limit, finite deadline, atomic checkpoints.
Stops early (keeping the last checkpoint) if the table or host memory limits are hit.
"""
import argparse
import json
import os
import resource
import shutil
import time
from pathlib import Path

from poker.lean_cfr import LeanCFR, resume
from poker.train_cfr import run_lock
from scripts.scaling_validation import available_memory

MEMORY = 4 * 1024 ** 3
HOST_RESERVE = 2 * 1024 ** 3
DISK_RESERVE = 10 * 1024 ** 3


def train(root, minutes, seed=1, resuming=False, save_minutes=15, max_rows=6_000_000, snapshots=()):
    if not 0 < minutes <= 24 * 60:
        raise ValueError('finite run length up to 24 hours required')
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY, MEMORY))
    os.nice(10)
    latest = root / 'latest.json'
    with run_lock(root / 'trainer'):
        if latest.exists() != resuming:
            raise ValueError('use --resume exactly when latest.json exists')
        solver = resume(latest) if resuming else LeanCFR(seed)
        start_iterations, start_nodes = solver.iterations, solver.nodes
        start = time.monotonic()
        end = start + 60 * minutes
        next_save = start + 60 * save_minutes
        pending = sorted(s for s in snapshots if s > solver.iterations)
        reason = 'deadline'

        def write(status):
            elapsed = time.monotonic() - start
            (root / 'progress.json').write_text(json.dumps(dict(
                status=status, iterations=solver.iterations, nodes=solver.nodes, regret_rows=len(solver.regrets),
                average_rows=len(solver.sums), session_seconds=elapsed, pid=os.getpid(),
                iterations_per_second=(solver.iterations - start_iterations) / max(elapsed, 1e-9),
                nodes_per_second=(solver.nodes - start_nodes) / max(elapsed, 1e-9),
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                deadline_unix=time.time() + max(0, end - time.monotonic())), indent=1))

        try:
            while time.monotonic() < end:
                for _ in range(200):
                    solver.iterate()
                if pending and solver.iterations >= pending[0]:
                    solver.save(root / f'snapshot-{solver.iterations}.json')
                    pending = [s for s in pending if s > solver.iterations]
                if len(solver.sums) > max_rows:
                    reason = 'row limit'; break
                if available_memory() < HOST_RESERVE or shutil.disk_usage(root).free < DISK_RESERVE:
                    reason = 'host reserve'; break
                if time.monotonic() >= next_save:
                    solver.save(latest); write('running')
                    next_save = time.monotonic() + 60 * save_minutes
        except MemoryError:
            reason = 'memory limit'
            raise
        finally:
            if reason != 'memory limit':
                solver.save(latest)
            write('stopped: ' + reason)
    return solver


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--minutes', required=True, type=float)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--save-minutes', type=float, default=15)
    parser.add_argument('--snapshots', type=int, nargs='*', default=[])
    args = parser.parse_args()
    train(args.root, args.minutes, args.seed, args.resume, args.save_minutes, snapshots=args.snapshots)
