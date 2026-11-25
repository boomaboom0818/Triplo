"""Replay live preflop decisions through a Triplo build (v9 --open-translation identity check).

Reads ``decision`` rows from the client's session log (each carries the full observation,
history verification, leaderboard fish seats and opponent profiles) and re-runs every
preflop one through the live v8-fast policy stack (history checkpoint, equity-guarded-v5
fallback with the v8 flags, --fish-exploit preflop layer), built from the code tree given
by ``--tree``. The learned policy samples its action, so the RNG is reseeded per decision
from (hand, history length): two trees that decide alike give identical outputs.

    # v8 release vs this tree with the flag off: the digests must match
    python scripts/v9_preflop_replay.py LOG --tree ~/.local/share/triplo/releases/9342601 --out v8.json
    python scripts/v9_preflop_replay.py LOG --out off.json
    # how often translation fires on real spots, and what it changes
    python scripts/v9_preflop_replay.py LOG --open-translation 3 --out tr3.json --compare v8.json
"""
import argparse
from collections import Counter
from hashlib import blake2b
import importlib
import json
from pathlib import Path
import sys

V8_FLAGS = dict(stack_press=True, deep_value=True, solve_river=True, defend_3bet=True, flop_peel=True,
                turn_defend=True)
CHECKPOINT_CACHE = Path('/tmp/triplo-factorial-playable.json')


def size_bucket(o):
    raises = [h[3] for h in o.history if isinstance(h[3], int)]
    if len(raises) != 1:
        return 'other'
    bb = raises[0] / 2
    return '<=2.5' if bb <= 2.5 else '3' if bb <= 3.25 else '3.5-4' if bb <= 4.25 else '>4'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('log', type=Path)
    p.add_argument('--tree', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--since')
    p.add_argument('--checkpoint-cache', type=Path, default=CHECKPOINT_CACHE,
                   help='policy-only JSON of the live checkpoint ({"policy": saved}; '
                        'scripts/v6_field_benchmark.py CHECKPOINT_CACHE)')
    p.add_argument('--open-translation', type=float, metavar='MAX_BB',
                   help='enable v9 --open-translation (0 = any size); omitted = off')
    p.add_argument('--compare', type=Path, help='an earlier --out to diff against, row by row')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.tree.resolve()))
    from poker.equity_pricing import VERSION_V5, make_fallback
    from poker.fish_preflop import FishPreflop
    from poker.game import Observation
    from poker.history_cfr import HistoryPolicy
    client = importlib.import_module('scripts.openpoker_client')
    from scripts.live_report_depth import parse_time, read_events
    since = parse_time(a.since)

    saved = json.loads(a.checkpoint_cache.read_text())['policy']
    inner = HistoryPolicy(saved, 0, 64, learned_postflop=False)
    inner.fallback = make_fallback(VERSION_V5, **V8_FLAGS, **client.FISH_EXPLOIT)
    if a.open_translation is not None:
        inner.open_translation = dict(max_open=int(round(2 * a.open_translation)) if a.open_translation else None)
    policy = FishPreflop(inner, steal=False)

    rows = []
    for e in read_events(a.log):
        if e.get('type') != 'decision' or e.get('street') != 0 or not e.get('observation'):
            continue
        if since is not None and e['time'] < since:
            continue
        o = dict(e['observation'])
        o['history'] = tuple(tuple(h) for h in o['history'])
        for k in ('hole', 'board', 'stacks', 'active', 'legal'):
            o[k] = tuple(o[k])
        o = Observation(**o)
        inner.fallback.fish_seats = set(e['fish_seats']) if e.get('fish_seats') is not None else None
        inner.fallback.profiles = {int(k): tuple(v) for k, v in (e.get('opponent_profiles') or {}).items()}
        inner.fallback.last = None
        inner.rng.seed(f"{e['hand_id']}:{len(o.history)}")
        before = (inner.covered, inner.safeguards, inner.pressure_overrides, policy.overrides,
                  getattr(inner, 'translated', 0))
        action = policy.act(o, history_complete=e.get('history_verified') is True)
        after = (inner.covered, inner.safeguards, inner.pressure_overrides, policy.overrides,
                 getattr(inner, 'translated', 0))
        source = ('fish_preflop' if after[3] > before[3] else 'poker_safeguard' if after[1] > before[1]
                  else 'pressure_guard' if after[2] > before[2] else 'open_translation' if after[4] > before[4]
                  else 'learned_policy' if after[0] > before[0] else 'preflop_rules')
        rows.append(dict(hand_id=e['hand_id'], n=len(o.history), action=action, source=source,
                         live_source=e.get('decision_source'), live_action=e.get('action'), spot=size_bucket(o)))
    digest = blake2b(json.dumps([(r['hand_id'], r['n'], r['action']) for r in rows]).encode(),
                     digest_size=16).hexdigest()
    sources = Counter(r['source'] for r in rows)
    live_match = Counter((r['source'] == r['live_source']) for r in rows if r['live_source'])
    rep = dict(tree=str(a.tree), open_translation=a.open_translation, decisions=len(rows), digest=digest,
               sources=dict(sources), source_matches_live=dict(live_match))
    if a.compare:
        def keyed(rs):
            # Unverified histories are logged empty, so (hand, length) can repeat within a hand.
            seen = Counter()
            for r in rs:
                seen[r['hand_id'], r['n']] += 1
                yield (r['hand_id'], r['n'], seen[r['hand_id'], r['n']]), r
        old = dict(keyed(json.loads(a.compare.read_text())['rows']))
        changed = Counter()
        kinds = Counter()
        for k, r in keyed(rows):
            b = old.get(k)
            if b is None:
                changed['missing'] += 1
            elif b['action'] != r['action']:
                changed[r['spot']] += 1
                kind = lambda x: 'raise' if isinstance(x, int) else x
                kinds[f"{kind(b['action'])}->{kind(r['action'])}"] += 1
        rep['compare'] = dict(against=str(a.compare), changed=sum(v for k, v in changed.items() if k != 'missing'),
                              changed_by_spot=dict(changed), transitions=dict(kinds),
                              spots=dict(Counter(r['spot'] for r in rows)))
    print(json.dumps(rep, indent=1))
    rep['rows'] = rows
    a.out.write_text(json.dumps(rep))


if __name__ == '__main__':
    main()
