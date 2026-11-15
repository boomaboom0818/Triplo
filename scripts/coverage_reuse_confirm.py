"""Prospective v10 engineering confirmation under the user-approved amended metric.

Approved 2026-11-15: the 3x gate counts UNIQUE NEW positive-average keys per node
(absent/zero in the immutable v8 parent, active 2/3 postflop) instead of all
positive updates, and total positive updates per node must not regress. Every
other frozen engineering gate is unchanged. The earlier v10 probe remains failed.
"""
import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
from random import Random
import resource
import shutil
import signal
import time

from poker.cfr import Holdem
from poker.config import load_checkpoint
from poker.train import atomic_save
from poker.train_cfr import run_lock
from scripts.coverage_reuse_probe import Baseline, Candidate, digest, full_batch, selected
from scripts.scaling_validation import available_memory
from scripts.stratified_average_probe import summarize
from scripts.vector_variance import VectorVariance, SCHEMA

ROOT = Path('runs/bot/coverage-reuse-confirm-001')
SEED_BASE = 12700000
PARENT_ITERATIONS = (8906, 9150)


def gate_report(totals, summaries, new_cells, peaks, max_ratio, role_batches):
    """Pure amended gate arithmetic; index 0 is v8 baseline, 1 is v10."""
    def ratio(key):
        return totals[1][key] / totals[0][key] if totals[0][key] > 0 else None

    def per_node(field):
        rates = [summaries[i][field] / totals[i]['nodes'] if totals[i]['nodes'] else 0. for i in range(2)]
        return rates[1] / rates[0] if rates[0] else None

    new_key_gain = per_node('active23_new_positive_keys')
    update_gain = per_node('active23_postflop_updates')
    work, wall, seconds = ratio('variance_nodes'), ratio('variance_seconds'), ratio('seconds')
    gates = dict(
        exact_regret_rng_nodes=True,
        exact_covered_updates=True,
        variance_work=work is not None and work <= 1.5,
        variance_time=wall is not None and wall <= 2,
        total_time=seconds is not None and seconds <= 2,
        new_unique_key_rate=new_key_gain is not None and new_key_gain >= 3,
        total_update_rate_not_regressed=update_gain is not None and update_gain >= 1,
        all_role_street_new_keys=len(new_cells) == 18 and all(v >= 20 for v in new_cells.values()),
        defensive_bound=max_ratio <= 1 + 1e-12,
        finite_actual_weights=all(p > 0 and math.isfinite(p * p) for p in peaks),
        bounded_batches=all(r['nodes'] <= 10000 and r['seconds'] <= 2 for r in role_batches))
    return dict(gates=gates, new_unique_keys_per_node_gain=new_key_gain, raw_update_rate_gain=update_gain,
                variance_work_ratio=work, variance_time_ratio=wall, total_time_ratio=seconds)


def run(candidate, deadline):
    parsed = datetime.fromisoformat(deadline)
    if parsed.tzinfo is None or not time.time() < parsed.timestamp() <= time.time() + 2 * 3600:
        raise ValueError('finite future deadline within two hours required')
    end = parsed.timestamp()
    resource.setrlimit(resource.RLIMIT_AS, (3 * 1024**3, 3 * 1024**3))
    os.nice(10)

    def expired(signum, frame):
        raise TimeoutError('coverage-reuse confirmation deadline reached')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(max(1, math.ceil(end - time.time())))
    with run_lock(ROOT / f'candidate-{candidate}'):
        out = ROOT / f'candidate-{candidate}.json'
        metrics = ROOT / f'candidate-{candidate}.metrics.json'
        if out.exists() or metrics.exists():
            raise ValueError('preserve completed/failed attempts; no unchanged retry')
        manifest = json.loads((ROOT / 'manifest.json').read_text())
        assert manifest['deadline'] == deadline

        def verify():
            for path, sha in manifest['sources'].items():
                assert digest(path) == sha, path
            for row in manifest['inputs']:
                assert digest(row['path']) == row['sha256']
        verify()
        status = 'failed'
        error = None
        current = None
        start = time.monotonic()
        minimum = available_memory()
        try:
            if minimum < 8 * 1024**3 or shutil.disk_usage(ROOT).free < 10 * 1024**3:
                raise RuntimeError('coverage-reuse confirmation preflight reserves')
            saved = load_checkpoint(manifest['inputs'][candidate]['path'])
            assert saved['iterations'] == PARENT_ITERATIONS[candidate]
            assert saved['future_average_completed_iterations'] == 2000
            solvers = [Baseline(average_origin=saved['future_average_origin']),
                       Candidate(average_origin=saved['future_average_origin'])]
            for solver in solvers:
                solver.regrets = saved['regrets']
                solver.iterations = saved['iterations']
                solver.setup(saved)
                solver.max_ratio = 0.
            fixtures = []
            totals = [dict(nodes=0, seconds=0., variance_nodes=0., variance_seconds=0.) for _ in solvers]
            for pop_index, kind in enumerate(manifest['populations']):
                for deal_index, deal in enumerate(manifest['deals']):
                    estimators = [VectorVariance(), VectorVariance()]
                    pairs = []
                    nodes = [[], []]
                    walls = [[], []]
                    state = Holdem(deal, 6, saved['abstraction'])
                    for replicate in range(manifest['replicates']):
                        seed = SEED_BASE + candidate * 100000 + pop_index * 10000 + deal_index * 100 + replicate
                        signatures = []
                        batches = []
                        for variant, solver in enumerate(solvers):
                            current = dict(population=kind, deal=deal, replicate=replicate, variant=variant)
                            minimum = min(minimum, available_memory())
                            if minimum < 2 * 1024**3 or shutil.disk_usage(ROOT).free < 10 * 1024**3:
                                raise RuntimeError('coverage-reuse confirmation host reserve breached')
                            solver.rng = Random(seed)
                            solver.sums = {}
                            solver.nodes = 0
                            solver.opponents = selected(saved, seed, kind)
                            signature, roles = full_batch(solver, state, end)
                            signatures.append(signature)
                            batches.append(roles)
                            estimators[variant].add(solver.sums)
                            nodes[variant].append(solver.nodes)
                            walls[variant].append(sum(r['seconds'] for r in roles))
                        assert signatures[0] == signatures[1], 'regret/RNG/node mismatch'
                        covered = {key for solver in solvers for key in solver.sums
                                   if any(saved['sums'].get(key, ()))}
                        assert all(solvers[0].sums.get(key, []) == solvers[1].sums.get(key, [])
                                   for key in covered), 'covered-coordinate update differs from v8'
                        pairs.append(dict(replicate=replicate, seed=seed, signature=signatures[0],
                                          exact_regret_rng_nodes=True, exact_covered_updates=True,
                                          covered_update_keys=len(covered), roles=batches))
                        atomic_save(ROOT / f'candidate-{candidate}.progress.json',
                                    dict(completed_fixtures=len(fixtures), current=current,
                                         completed_replicates=replicate + 1, deadline=deadline))
                    variants = []
                    for j, estimator in enumerate(estimators):
                        variance = estimator.trace()
                        mean_nodes = sum(nodes[j]) / len(nodes[j])
                        mean_seconds = sum(walls[j]) / len(walls[j])
                        variants.append(dict(variance_schema=SCHEMA, variance_trace=variance,
                                             mean_nodes=mean_nodes, mean_seconds=mean_seconds))
                        totals[j]['nodes'] += sum(nodes[j])
                        totals[j]['seconds'] += sum(walls[j])
                        totals[j]['variance_nodes'] += variance * mean_nodes
                        totals[j]['variance_seconds'] += variance * mean_seconds
                    fixtures.append(dict(population=kind, deal=deal, variants=variants, pairs=pairs))
            verify()
            summaries = [summarize(s) for s in solvers]
            new_cells = {}
            for owner in range(6):
                for street in (1, 2, 3):
                    keys = set()
                    for active in (2, 3):
                        keys.update(solvers[1].new_keys.get(f'owner={owner}:street={street}:active={active}', set()))
                    new_cells[f'owner={owner}:street={street}'] = len(keys)
            peaks = [max((v['maximum'] for v in s.average_weight_moments.values()), default=0.) for s in solvers]
            report = gate_report(totals, summaries, new_cells, peaks, solvers[1].max_ratio,
                                 [r for f in fixtures for p in f['pairs'] for v in p['roles'] for r in v])
            atomic_save(out, dict(
                candidate=candidate, manifest_sha256=digest(ROOT / 'manifest.json'), status='completed',
                read_only=True, fixtures=fixtures, totals=totals, summaries=summaries,
                new_keys_by_role_street=new_cells, peak_actual_weights=peaks, **report,
                eligible_for_registration=all(report['gates'].values()),
                note='Amended preregistered new-unique-key metric, user-approved before observations. '
                     'Full equal-count sources, unchanged future-linear target. '
                     'No training/payoff/strength evidence.'))
            status = 'completed'
        except BaseException as exc:
            error = repr(exc)
            raise
        finally:
            peaks = {line.split(':')[0]: int(line.split()[1]) * 1024
                     for line in Path('/proc/self/status').read_text().splitlines()
                     if line.startswith(('VmPeak:', 'VmHWM:'))}
            atomic_save(metrics, dict(
                status=status, error=error, last_case=current, pid=os.getpid(), deadline=deadline,
                address_space_limit_bytes=resource.getrlimit(resource.RLIMIT_AS)[0],
                minimum_host_available_bytes=minimum, wall_seconds=time.monotonic() - start, **peaks))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', type=int, choices=(0, 1), required=True)
    parser.add_argument('--deadline', required=True)
    args = parser.parse_args()
    run(args.candidate, args.deadline)
