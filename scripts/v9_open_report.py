"""Who opens big on OpenPoker, and how Triplo's hands against those opens go (Triplo v9).

Reads the live client's session log (decisions, hand results, leaderboard table checks)
and the redacted wire trace (seat names, blinds) and reports, for every preflop hand
whose first raise is an open (no earlier raise):

* open sizes by opener class: ``weak`` (leaderboard no_code / losing bots, as
  ``--fish-exploit`` sees them), ``strong`` (>= the strong threshold) or ``reg`` (rated
  but neither), from the client's ``strong_opponent_check`` rows;
* the opener's shown hands at showdown, as Chen-score percentiles, by size bucket;
* how the rest of the table answers each size;
* Triplo's first decision facing a single open: action shares, decision source, and
  Triplo's net in those hands (uncapped BB per hand).

    PYTHONPATH=<tree> python <tree>/scripts/v9_open_report.py \\
        --log runs/bot/openpoker-season-17-equity/session.jsonl \\
        --trace runs/bot/openpoker-season-17-equity/trace.jsonl --out open-report.json
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poker.fish_preflop import hand_rank
from scripts.live_report_depth import hands, parse_time, read_events

BUCKETS = ((2.25, '<=2.25'), (2.75, '2.5'), (3.25, '3'), (4.25, '3.5-4'), (float('inf'), '>4'))


def bucket(size_bb):
    return next(name for edge, name in BUCKETS if size_bb <= edge)


def classes(events):
    """name -> 'weak' | 'strong' | 'reg' by majority over the client's table checks."""
    votes = defaultdict(Counter)
    for e in events:
        if e.get('type') != 'strong_opponent_check':
            continue
        strong, weak = set(e.get('strong') or ()), set(e.get('weak') or ())
        for name in e.get('opponents') or ():
            votes[name]['strong' if name in strong else 'weak' if name in weak else 'reg'] += 1
    return {name: c.most_common(1)[0][0] for name, c in votes.items()}


def wire_info(trace):
    """hand_id -> dict(bb, dealer, names{seat: name}) from the redacted wire trace."""
    info = defaultdict(lambda: dict(bb=None, dealer=None, names={}))
    for e in read_events(trace):
        m = e.get('message') or {}
        kind, hand = m.get('type'), m.get('hand_id')
        if not hand:
            continue
        if kind == 'hand_start':
            info[hand]['bb'] = (m.get('blinds') or {}).get('big_blind')
            info[hand]['dealer'] = m.get('dealer_seat')
        elif kind == 'player_action' and m.get('name') is not None:
            info[hand]['names'][m['seat']] = m['name']
        elif kind == 'resync_response':
            snap = m.get('snapshot') or {}
            for s in snap.get('seats') or ():
                if s.get('name'):
                    info[hand]['names'].setdefault(s['seat'], s['name'])
            if snap.get('big_blind'):
                info[hand]['bb'] = info[hand]['bb'] or snap['big_blind']
    return info


def first_open(actions):
    """(index, seat, raise-to chips, limpers) of the first preflop raise, or None."""
    limpers = 0
    for i, a in enumerate(actions):
        if a.get('street') != 'preflop':
            return None
        if a['action'] in ('raise', 'bet', 'all_in') and a.get('amount'):
            return i, a['seat'], a['amount'], limpers
        if a['action'] == 'call':
            limpers += 1
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--log', type=Path, required=True)
    p.add_argument('--trace', type=Path, required=True)
    p.add_argument('--since')
    p.add_argument('--buy-in', type=int, default=5000)
    p.add_argument('--hero', default='Triplo')
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    since = parse_time(a.since)
    events = [e for e in read_events(a.log) if since is None or e.get('time', 0) >= since]
    kind = classes(events)
    wire = wire_info(a.trace)
    nets = {r['hand_id']: r for r in hands(events, 20, a.buy_in)}
    first_decision = {}
    for e in events:
        if e.get('type') == 'decision' and e.get('street') == 0 and e.get('hand_id') not in first_decision:
            first_decision[e['hand_id']] = e

    sizes = defaultdict(Counter)                # class -> bucket counts
    opener_sizes = defaultdict(list)            # name -> sizes
    shown = defaultdict(list)                   # (class, bucket) -> opener hand ranks at showdown
    response = defaultdict(Counter)             # bucket -> first response of each later player
    hero_rows = defaultdict(lambda: dict(n=0, action=Counter(), source=Counter(), net=[], opener=Counter()))
    for e in events:
        if e.get('type') != 'hand_result' or not isinstance(e.get('result'), dict):
            continue
        r = e['result']
        hand = r.get('hand_id')
        w = wire.get(hand)
        bb = (w or {}).get('bb') or 20
        acts = r.get('actions') or []
        opened = first_open(acts)
        if opened is None:
            continue
        i, seat, amount, limpers = opened
        names = (w or {}).get('names', {})
        name = names.get(seat)
        size = amount / bb
        b = bucket(size)
        cls = 'hero' if name == a.hero else kind.get(name, 'unknown')
        sizes[cls][b] += 1
        if name and name != a.hero:
            opener_sizes[name].append(size)
        cards = (r.get('shown_cards') or {}).get(str(seat))
        if cards and cls != 'hero':
            shown[cls, b].append(hand_rank(tuple(cards)))
        seen = set()
        for x in acts[i + 1:]:
            if x.get('street') != 'preflop' or x['seat'] in seen or x['seat'] == seat:
                continue
            seen.add(x['seat'])
            response[b]['raise' if x['action'] in ('raise', 'all_in', 'bet') else x['action']] += 1
            if x['action'] in ('raise', 'all_in', 'bet'):
                break
        # Triplo facing this open as its first decision (no limpers, single raise).
        d = first_decision.get(hand)
        if cls == 'hero' or d is None or limpers or e.get('seat') is None:
            continue
        hist = (d.get('observation') or {}).get('history') or []
        if sum(isinstance(h[3], int) for h in hist) != 1:
            continue
        row = hero_rows[b]
        row['n'] += 1
        row['action'][d['action']] += 1
        row['source'][d.get('decision_source')] += 1
        row['opener'][cls] += 1
        net = nets.get(hand, {}).get('net')
        if net is not None:
            row['net'].append(net / bb)

    def pct(c):
        t = sum(c.values())
        return {k: round(v / t, 3) for k, v in c.most_common()} if t else {}

    out = dict(
        open_sizes={cls: dict(n=sum(c.values()), share=pct(c)) for cls, c in sizes.items()},
        big_openers=sorted(((n, kind.get(n, 'unknown'), len(s), round(mean(s), 2),
                             round(sum(x >= 3.25 for x in s) / len(s), 3))
                            for n, s in opener_sizes.items() if len(s) >= 30 and mean(s) >= 3.0),
                           key=lambda t: -t[2]),
        opener_showdown_rank={f'{c}:{b}': dict(n=len(v), mean=round(mean(v), 3),
                                               top10=round(sum(x < .1 for x in v) / len(v), 3),
                                               top25=round(sum(x < .25 for x in v) / len(v), 3))
                              for (c, b), v in sorted(shown.items()) if len(v) >= 10},
        table_response=pct_rows if (pct_rows := {b: pct(c) for b, c in response.items()}) else {},
        hero_facing_single_open={b: dict(n=r['n'], action=pct(r['action']), source=pct(r['source']),
                                         opener=pct(r['opener']),
                                         net_bb_per_hand=round(mean(r['net']), 3) if r['net'] else None,
                                         hands_with_net=len(r['net']))
                                 for b, r in sorted(hero_rows.items())},
    )
    text = json.dumps(out, indent=1)
    print(text)
    if a.out:
        a.out.write_text(text)


if __name__ == '__main__':
    main()
