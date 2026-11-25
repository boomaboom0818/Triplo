"""Replay live postflop decisions through the v5 fallback (identity and firing-rate check).

Reads ``decision`` rows from the client's session log (they carry the full observation,
the leaderboard fish seats, measured opponent profiles and Stack Press sizes) and re-runs
each postflop decision through the fallback built with the given options. With every v9
option off it must reproduce the logged live action; with options on it reports how often
each rule fires on real spots. Preflop rows need the learned checkpoint and are skipped.

    PYTHONPATH=<tree> python <tree>/scripts/v9_live_replay.py runs/bot/.../session.jsonl \\
        --since 2026-11-25T03:44:36Z --options '{"fish_stab": [0.33, "all"]}' --out replay.json
"""
import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poker.equity_pricing import VERSION_V5, make_fallback
from poker.game import Observation
from scripts.live_report_depth import parse_time, read_events

V8 = dict(stack_press=True, deep_value=True, solve_river=True, defend_3bet=True, flop_peel=True, turn_defend=True,
          fish_size={1: .75, 2: 1.0, 3: 1.25}, fish_raise=('top', .6))


def observation(row):
    o = dict(row['observation'])
    o['history'] = tuple(tuple(h) for h in o['history'])
    for k in ('hole', 'board', 'stacks', 'active', 'legal'):
        o[k] = tuple(o[k])
    obs = Observation(**o)
    extra = tuple(row.get('stack_press_sizes') or ())
    if extra:
        raises = sorted({a for a in obs.legal if isinstance(a, int)} | set(extra))
        obs = replace(obs, legal=tuple(a for a in obs.legal if not isinstance(a, int)) + tuple(raises))
    return obs


def wire_kind(action, obs):
    if action == 'fold':
        return 'fold'
    if action == 'call':
        return 'call' if obs.call else 'check'
    return 'raise'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('log', type=Path)
    p.add_argument('--since')
    p.add_argument('--until')
    p.add_argument('--options', default='{}', help='JSON fallback options added to v8 (lists become tuples)')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    since, until = parse_time(a.since), parse_time(a.until)
    extra = {k: tuple(v) if isinstance(v, list) else v for k, v in json.loads(a.options).items()}
    fb = make_fallback(VERSION_V5, **V8, **extra)
    out, fired, match = [], Counter(), Counter()
    for e in read_events(a.log):
        if e.get('type') != 'decision' or not e.get('observation') or e.get('street', 0) == 0:
            continue
        if since is not None and e['time'] < since or until is not None and e['time'] >= until:
            continue
        if e.get('decision_source') not in ('equity_fallback', 'poker_safeguard'):
            continue
        if e.get('decision_source') == 'poker_safeguard':
            continue                       # nut safeguard decided before the fallback
        obs = observation(e)
        fb.profiles = {int(k): tuple(v) for k, v in (e.get('opponent_profiles') or {}).items()}
        fb.fish_seats = set(e['fish_seats']) if e.get('fish_seats') is not None else None
        fb.last = None
        action = fb.act(obs)
        amount = None if not isinstance(action, int) else action * 10
        same_kind = wire_kind(action, obs) == e['action'] or (e['action'] == 'all_in' and isinstance(action, int))
        logged = e.get('amount')
        same = same_kind and (amount is None or logged is None or abs(logged - amount) <= 10)
        match['same' if same else 'different'] += 1
        tags = [t for t in ('fish_stab', 'fish_value_cap', 'fish_size', 'fish_raise') if (fb.last or {}).get(t)]
        fired.update(tags)
        out.append(dict(hand_id=e['hand_id'], street=obs.street, logged=[e['action'], logged],
                        replay=[wire_kind(action, obs), amount], tags=tags))
        if a.limit and len(out) >= a.limit:
            break
    hands = len({r['hand_id'] for r in out})
    rep = dict(options=a.options, decisions=len(out), hands=hands, match=dict(match), fired=dict(fired), rows=out)
    a.out.write_text(json.dumps(rep))
    print(json.dumps({k: v for k, v in rep.items() if k != 'rows'}))


if __name__ == '__main__':
    main()
