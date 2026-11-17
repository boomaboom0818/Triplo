"""Bounded, reproducible guard ablation on development deals; never deploys."""
import argparse
import json
from pathlib import Path
from statistics import mean
import subprocess
import sys

from scripts.aggressive_campaign import readiness
from scripts.live_gate import digest

SOURCES = ('scripts/guard_ablation.py', 'scripts/export_pressure_guard.py',
           'scripts/live_gate.py', 'poker/history_cfr.py', 'poker/equity_policy.py',
           'poker/preflop_commitment.py', 'poker/live_curriculum.py', 'poker/fast_game.py',
           'poker/rule_opponents.py')


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2)


def run(root, base, incumbent, both):
    root = Path(root)
    root.mkdir(exist_ok=False)
    manifest = dict(seed=61161000, deals=300, family_alpha=.025,
                    checkpoints={str(p): digest(p) for p in (base, incumbent, both)},
                    sources={p: digest(p) for p in SOURCES},
                    variants=['original', 'preflop', 'postflop', 'both'],
                    confirmation_consumed=False)
    write(root/'manifest.json', manifest)

    def command(args, tag):
        with (root/(tag+'.log')).open('x') as f:
            subprocess.run([sys.executable, '-m', *args], stdout=f,
                           stderr=subprocess.STDOUT, check=True)

    paths = dict(original=base, both=both, incumbent=incumbent, fallback=incumbent)
    for name in ('preflop', 'postflop'):
        paths[name] = root/(name+'.json')
        command(['scripts.export_pressure_guard', '--checkpoint', str(base),
                 '--out', str(paths[name]), '--guard', f'equity-pressure-{name}-v1'], 'export-'+name)
    reports = {}
    for name in ('preflop', 'postflop', 'both', 'original', 'incumbent', 'fallback'):
        out = root/('development-'+name+'.json')
        args = ['scripts.live_gate', 'evaluate', '--checkpoint', str(paths[name]),
                '--incumbent', str(incumbent), '--out', str(out),
                '--seed', str(manifest['seed']), '--deals', str(manifest['deals'])]
        if name == 'fallback':
            args.append('--fallback-only')
        command(args, name)
        reports[name] = json.loads(out.read_text())
        print(name, mean(c['bb_per_100'] for c in reports[name]['cases'].values()), flush=True)
    assessments = {name: readiness(reports[name], reports['incumbent'], reports['fallback'], .025)
                   for name in manifest['variants']}
    for path, sha in {**manifest['sources'], **manifest['checkpoints']}.items():
        if digest(path) != sha:
            raise RuntimeError('frozen input changed: '+path)
    result = dict(assessments=assessments,
                  eligible=[name for name, r in assessments.items() if r['ready_for_confirmation']],
                  point_estimates={name: {case: c['bb_per_100'] for case, c in r['cases'].items()}
                                   for name, r in reports.items()},
                  confirmation_consumed=False, promote=False)
    write(root/'result.json', result)
    print('eligible', result['eligible'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--base', required=True, type=Path)
    p.add_argument('--incumbent', required=True, type=Path)
    p.add_argument('--both', required=True, type=Path)
    a = p.parse_args()
    run(a.root, a.base, a.incumbent, a.both)
