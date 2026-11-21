"""Offline v3 parity and timing on a frozen client log. No API access.

    python -m scripts.range_v3_benchmark run SESSION.jsonl OUTPUT.json
    python -m scripts.range_v3_benchmark compare BASELINE.json CANDIDATE.json

Run the same script in separate processes importing the respective checkouts.
Pass the script by absolute path and set PYTHONPATH=. in the baseline checkout.
"""
import argparse
import collections
import hashlib
import json
import math
import time
from pathlib import Path

from poker.equity_pricing import make_fallback
from poker.game import Observation


def source(r):
    if r['type'] != 'decision' or not r.get('observation'):
        return None
    src = r.get('decision_source')
    if src in ('equity_fallback', 'preflop_rules'):
        return src
    if src is None and r.get('covered') is False:
        return 'legacy-equity' if r.get('equity_fallback') else 'legacy-unclassified'


def run(session, output):
    if output.exists():
        raise ValueError(f'output already exists: {output}')
    policy = make_fallback('equity-guarded-v3')
    digest = hashlib.sha256()
    rows = []
    with session.open('rb') as f:
        for i, line in enumerate(f):
            digest.update(line)
            r = json.loads(line)
            src = source(r)
            if not src:
                continue
            if r.get('stack_press_sizes'):
                raise ValueError('Stack Press snapshots require a different action menu')
            o = Observation(**{k: tuple(tuple(x) if isinstance(x, list) else x for x in v)
                               if isinstance(v, list) else v for k, v in r['observation'].items()})
            policy.profiles = {int(k): v for k, v in (r.get('opponent_profiles') or {}).items()}
            t0 = time.perf_counter_ns()
            action = policy.act(o)
            elapsed = time.perf_counter_ns() - t0
            rows.append([i, o.street, src, action, policy.last, elapsed])
    if not rows:
        raise ValueError('no eligible fallback decisions')
    with output.open('x') as f:
        json.dump(dict(session_sha256=digest.hexdigest(), rows=rows), f, separators=(',', ':'))


def timing(rows):
    times = sorted(row[-1] for row in rows)
    return tuple(times[math.ceil(q * len(times)) - 1] / 1e6 for q in (.5, .99, 1.))


def compare(baseline, candidate):
    a, b = (json.loads(p.read_text()) for p in (baseline, candidate))
    if a['session_sha256'] != b['session_sha256']:
        raise ValueError('different log inputs')
    control, fast = a['rows'], b['rows']
    for old, new in zip(control, fast, strict=True):
        if old[:3] != new[:3]:
            raise ValueError('different decision snapshots')
        if old[3:-1] != new[3:-1]:
            raise ValueError(f'action or diagnostics differ on log line {old[0] + 1}')
    print(f'Exact action and diagnostics parity: {len(control):,} decisions')
    print('Sources:', dict(collections.Counter(row[2] for row in control)))
    for name, pred in [('all', lambda _: True), ('postflop', lambda r: r[1] > 0)]:
        old = [r for r in control if pred(r)]
        new = [r for r in fast if pred(r)]
        if old:
            print(f'{name} ({len(old):,}): p50 / p99 / max (ms)')
            print('  baseline: ', '/'.join(f'{t:6.2f}' for t in timing(old)))
            print('  fast:     ', '/'.join(f'{t:6.2f}' for t in timing(new)))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest='command', required=True)
    for name, paths in [('run', ('session', 'output')), ('compare', ('baseline', 'candidate'))]:
        p = sub.add_parser(name)
        for path in paths:
            p.add_argument(path, type=Path)
    args = vars(cli.parse_args())
    {'run': run, 'compare': compare}[args.pop('command')](**args)
