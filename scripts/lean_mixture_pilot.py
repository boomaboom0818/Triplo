"""Bounded two-seed loose-caller experiment, with matched self-play continuations.

Outputs are immutable. Resuming an interrupted experiment requires a new root;
the original parent state is read-only. Normal intervals are screening only.
"""
import argparse
import hashlib
import json
import resource
import time
from collections import defaultdict
from pathlib import Path

from poker.lean_cfr import LeanPolicy, load
from poker.lean_parallel import ParallelLean
from poker.rule_opponents import STYLES
from poker.train_cfr import run_lock
from scripts.lean_train import HOST_RESERVE, DISK_RESERVE
from scripts.scaling_validation import available_memory
from scripts.rule_gate import play_style, summarize
import shutil


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(solver):
    mass = defaultdict(float)
    for key, row in solver.rows(solver.sums):
        if key[0] != '0':
            mass[f'street{key[0]}-active{key[2]}'] += sum(row)
    return dict(mass)


def save(path, obj):
    with path.open('x') as handle:
        json.dump(obj, handle, indent=2)


def evaluate(saved, deals, seed, cutoff):
    result = {}
    for style in STYLES:
        hero = LeanPolicy.from_checkpoint(saved, 5, cutoff)
        values, shoves = play_style(hero, style, deals, seed=seed, lean=True)
        result[style] = dict(summarize(values, shoves), deal_block_bb_per_100=values,
                             learned_decision_fraction=hero.covered / hero.decisions)
    return result


def difference(a, b):
    result = {}
    for style in STYLES:
        values = [x-y for x, y in zip(a[style]['deal_block_bb_per_100'], b[style]['deal_block_bb_per_100'])]
        summary = summarize(values, 0)
        result[style] = {k: summary[k] for k in ('bb_per_100', 'interval_95')}
    return result


def run(args):
    if args.workers < 1 or args.batch < 1 or args.deals < 2 or not 0 < args.arm_seconds <= 1800:
        raise ValueError('invalid pilot limits')
    if args.iterations <= 0 or args.iterations % (args.workers * args.batch):
        raise ValueError('iterations must be a positive multiple of workers * batch')
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024 ** 3, 4 * 1024 ** 3))
    args.root.mkdir(parents=True, exist_ok=False)
    parent_hashes = {p.name: digest(p) for p in args.parent.iterdir() if p.is_file()}
    sources = [Path(p) for p in ('poker/lean_parallel.py', 'poker/lean_cfr.py', 'poker/loose_opponents.py',
                                  'poker/fast_game.py', 'poker/fast_eval.py', 'poker/lean_abstraction.py',
                                  'poker/rule_opponents.py', 'scripts/rule_gate.py', 'scripts/lean_mixture_pilot.py')]
    manifest = dict(parent=str(args.parent.resolve()), parent_hashes=parent_hashes,
                    sources={str(p): digest(p) for p in sources}, training_seeds=[71, 72],
                    population_probability=.5, iterations_per_arm=args.iterations,
                    workers=args.workers, batch=args.batch, arm_seconds=args.arm_seconds,
                    evaluation_seed=18310000, deals=args.deals, min_support=1e-4,
                    note='Continuation seeds share one parent; not independent from-scratch replications. Pointwise normal 95% screening intervals.')
    save(args.root / 'manifest.json', manifest)
    for path in sources:
        target = args.root / 'source' / path
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as handle:
            handle.write(path.read_bytes())
    with run_lock(args.root / 'pilot'):
        parent = ParallelLean.load_state(args.parent)
        parent.export_policy(args.root / 'parent.json')
        start_iterations = parent.iterations
        before = inventory(parent)
        del parent
        training = {}
        for seed in manifest['training_seeds']:
            for arm, probability in (('control', 0.), ('mixture', .5)):
                name = f'{arm}-{seed}'
                solver = ParallelLean.load_state(args.parent)
                solver.seed = seed
                solver.population_probability = probability
                solver.population_iterations = 0
                start = time.monotonic()
                while solver.iterations < start_iterations + args.iterations:
                    if time.monotonic() - start > args.arm_seconds:
                        raise TimeoutError(f'{name}: arm deadline, no incomplete arm qualified')
                    if available_memory() < HOST_RESERVE or shutil.disk_usage(args.root).free < DISK_RESERVE:
                        raise RuntimeError('host reserve reached')
                    solver.round(args.workers, args.batch)
                after = inventory(solver)
                added_weight = (solver.iterations * (solver.iterations + 1) - start_iterations * (start_iterations + 1)) / 2
                training[name] = dict(iterations=solver.iterations, elapsed_seconds=time.monotonic() - start,
                    population_iterations=solver.population_iterations, rows=len(solver.index),
                    added_average_mass_per_weight={k:(v-before.get(k, 0.))/added_weight for k,v in after.items()})
                solver.save_state(args.root / f'{name}-state')
                solver.export_policy(args.root / f'{name}.json')
                save(args.root / f'{name}-training.json', training[name])
                print(json.dumps(dict(completed=name, **training[name])), flush=True)
                del solver
        evaluations = {}
        for name in ('parent', 'control-71', 'mixture-71', 'control-72', 'mixture-72'):
            saved = load(args.root / f'{name}.json')
            for mode, cutoff in (('raw', 0.), ('supported', 1e-4)):
                label = f'{name}-{mode}'
                evaluations[label] = evaluate(saved, args.deals, manifest['evaluation_seed'], cutoff)
                save(args.root / f'{label}-evaluation.json', evaluations[label])
                print('Evaluated ' + label, flush=True)
            del saved
        evaluations['fallback'] = evaluate(dict(iterations=0, sums={}), args.deals, manifest['evaluation_seed'], 0.)
        save(args.root / 'fallback-evaluation.json', evaluations['fallback'])
        paired = {}
        for seed in manifest['training_seeds']:
            for mode in ('raw', 'supported'):
                name = f'mixture-{seed}-{mode}'
                paired[name] = {control:difference(evaluations[name], evaluations[control])
                               for control in (f'control-{seed}-{mode}', f'parent-{mode}', 'fallback')}
        if {p.name: digest(p) for p in args.parent.iterdir() if p.is_file()} != parent_hashes:
            raise RuntimeError('parent state changed during pilot')
        if any(digest(p) != manifest['sources'][str(p)] for p in sources):
            raise RuntimeError('source changed during pilot')
        save(args.root / 'summary.json', dict(training=training, paired=paired, parent_unchanged=True))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', required=True, type=Path)
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--iterations', type=int, default=10000)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--batch', type=int, default=100)
    p.add_argument('--arm-seconds', type=float, default=180)
    p.add_argument('--deals', type=int, default=500)
    run(p.parse_args())
