"""Live OpenPoker win rate split by hero stack AND by effective depth.

Reads the client's session logs (``--log`` of scripts.openpoker_client; e.g.
runs/bot/openpoker-season-17-equity/session.jsonl) and reports, per bucket:
net BB/100 (with a per-hand normal 95% interval), hands/hour, net BB/hour and
the share of hero decisions by ``decision_source``.

Hero net per hand is the change in the hero seat's ``final_stacks`` between
consecutive ``hand_result`` events at the same table (all blinds, rake and side
pots included). A hand is excluded from the net when its starting stack is
unknown (first hand at a table, seat change) or when chips appear that the hand
cannot explain (an automatic rebuy/top-up between results); a bust (start 0)
is treated as an automatic rebuy for ``--buy-in`` chips. Effective depth is the
hero start stack capped by the deepest OTHER seat that took part in the hand
(acted or was paid), using those seats' previous final stacks; this is the
depth the policy key sees at the first decision. "Typical" effective depth is
the median over those opponents of min(hero, opponent). Hands/hour uses
the gap since the previous result, ignoring gaps above ``--max-gap`` seconds
(reconnects, lobby waits).

  .venv/bin/python -m scripts.live_report_depth runs/bot/openpoker-season-17-equity/session.jsonl
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
import math
from pathlib import Path
from statistics import mean, stdev

DEFAULT_LOG = 'runs/bot/openpoker-season-17-equity/session.jsonl'
HERO_EDGES = (40, 75, 150, 300, 750, 1500)
# Same edges as scripts.depth_ladder.DEPTH_BUCKETS.
EFFECTIVE_EDGES = (15, 35, 75, 150, 300, 750)
Z95 = 1.959963984540054


def bucket(value, edges):
    for lo, hi in zip((0,) + tuple(edges), edges):
        if value < hi:
            return f'{lo}-{hi}' if lo else f'<{hi}'
    return f'>={edges[-1]}'


def bucket_order(edges):
    return [bucket(e - 1e-9, edges) for e in edges] + [f'>={edges[-1]}']


def parse_time(text):
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return datetime.fromisoformat(text.replace('Z', '+00:00')).timestamp()


def read_events(path):
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue  # a partially written trailing line of a live log


def hands(events, big_blind=20, buy_in=1000, max_gap=300.):
    """One row per hand_result with hero net (chips, or None when unknown) and context."""
    decisions = defaultdict(list)
    results = []
    for e in events:
        if e.get('type') == 'decision' and e.get('hand_id'):
            decisions[e['hand_id']].append(e)
        elif e.get('type') == 'hand_result' and isinstance(e.get('result'), dict):
            results.append(e)
    previous = {}          # table_id -> (final stacks by seat, hero seat)
    last_time = None
    rows = []
    for e in results:
        r = e['result']
        table, seat = r.get('table_id'), e.get('seat')
        final = {int(k): v for k, v in (r.get('final_stacks') or {}).items()}
        gap = None if last_time is None else e['time'] - last_time
        last_time = e['time']
        row = dict(time=e['time'], hand_id=r.get('hand_id'), table_id=table, seat=seat, net=None,
                   hero_start=None, effective=None, typical_effective=None, excluded=None,
                   duration=gap if gap is not None and 0 <= gap <= max_gap else None,
                   sources=Counter(d.get('decision_source', 'unlogged') for d in decisions.get(r.get('hand_id'), ())))
        prior = previous.get(table)
        previous[table] = (final, seat)
        if seat is None or seat not in final:
            row['excluded'] = 'hero seat unknown'
        elif prior is None:
            row['excluded'] = 'first hand at table'
        elif prior[1] != seat or seat not in prior[0]:
            row['excluded'] = 'seat changed'
        else:
            start = prior[0][seat]
            if start == 0:
                start = buy_in
            paid = sum(p.get('amount', 0) for p in r.get('payouts', ()) if p.get('seat') == seat)
            net = final[seat] - start
            if net > paid:
                row['excluded'] = 'unexplained chips (rebuy/top-up or missing result)'
            else:
                players = {a.get('seat') for a in r.get('actions', ())} | {p.get('seat') for p in r.get('payouts', ())}
                others = [prior[0].get(s, final.get(s, 0)) for s in players if s is not None and s != seat]
                live = sorted(min(start, s) for s in others if s > 0) or [start]
                row.update(net=net, hero_start=start, effective=min(start, max(live)),
                           typical_effective=live[len(live) // 2])
        for key in ('net', 'hero_start', 'effective', 'typical_effective'):
            if row[key] is not None:
                row[key + '_bb'] = row[key] / big_blind
        rows.append(row)
    return rows


def summarize(rows, key, edges):
    groups = defaultdict(list)
    for row in rows:
        if row.get(key + '_bb') is not None:
            groups[bucket(row[key + '_bb'], edges)].append(row)
    out = {}
    for name in bucket_order(edges):
        group = groups.get(name)
        if not group:
            continue
        nets = [r['net_bb'] for r in group]
        timed = [r['duration'] for r in group if r['duration'] is not None]
        se = stdev(nets) / len(nets) ** .5 * 100 if len(nets) > 1 else math.nan
        bb100 = mean(nets) * 100
        per_hour = 3600 * len(timed) / sum(timed) if timed and sum(timed) > 0 else math.nan
        sources = Counter()
        for r in group:
            sources.update(r['sources'])
        total = sum(sources.values())
        out[name] = dict(hands=len(nets), bb_per_100=bb100, lower=bb100 - Z95 * se, upper=bb100 + Z95 * se,
                         hands_per_hour=per_hour, bb_per_hour=bb100 * per_hour / 100,
                         median_hero_bb=sorted(r['hero_start_bb'] for r in group)[len(group) // 2],
                         median_effective_bb=sorted(r['effective_bb'] for r in group)[len(group) // 2],
                         median_typical_bb=sorted(r['typical_effective_bb'] for r in group)[len(group) // 2],
                         decisions=total, source_shares={k: v / total for k, v in sources.most_common()} if total else {})
    return out


def cross(rows, hero_edges, effective_edges):
    table = defaultdict(list)
    for r in rows:
        if r.get('net_bb') is not None:
            table[(bucket(r['hero_start_bb'], hero_edges), bucket(r['effective_bb'], effective_edges))].append(r['net_bb'])
    return {f'{h}|{e}': dict(hands=len(v), bb_per_100=mean(v) * 100) for (h, e), v in table.items()}


def report(paths, since=None, until=None, big_blind=20, buy_in=1000, max_gap=300.,
           hero_edges=HERO_EDGES, effective_edges=EFFECTIVE_EDGES):
    rows = []
    for path in paths:
        rows.extend(hands(read_events(path), big_blind, buy_in, max_gap))
    rows = [r for r in rows if (since is None or r['time'] >= since) and (until is None or r['time'] < until)]
    counted = [r for r in rows if r['net'] is not None]
    timed = [r['duration'] for r in rows if r['duration'] is not None]
    nets = [r['net_bb'] for r in counted]
    return dict(logs=[str(p) for p in paths], since=since, until=until, big_blind=big_blind,
                hand_results=len(rows), hands_with_net=len(counted),
                excluded=dict(Counter(r['excluded'] for r in rows if r['excluded'])),
                overall=dict(bb_per_100=mean(nets) * 100 if nets else math.nan,
                             hands_per_hour=3600 * len(timed) / sum(timed) if timed else math.nan),
                by_hero_stack=summarize(counted, 'hero_start', hero_edges),
                by_effective_depth=summarize(counted, 'effective', effective_edges),
                by_typical_effective_depth=summarize(counted, 'typical_effective', effective_edges),
                hero_x_effective=cross(counted, hero_edges, effective_edges))


SHOW = ('learned_policy', 'preflop_rules', 'equity_fallback', 'pressure_guard', 'poker_safeguard', 'unlogged')


def render(rep):
    lines = [f'{rep["hands_with_net"]} hands with known net of {rep["hand_results"]} results; '
             f'overall {rep["overall"]["bb_per_100"]:+.1f} BB/100 at {rep["overall"]["hands_per_hour"]:.0f} hands/hour',
             f'excluded: {rep["excluded"]}', '']
    head = (f'{"bucket (BB)":<12}{"hands":>7}{"BB/100 [95% CI]":>28}{"hands/h":>9}{"BB/h":>8}{"med hero":>10}{"med eff":>9}{"med typ":>9}'
            + ''.join(f'{s[:7]:>9}' for s in SHOW))
    for title, key in (('By hero starting stack', 'by_hero_stack'),
                       ('By effective depth (hero vs deepest other player in the hand)', 'by_effective_depth'),
                       ('By typical effective depth (median over opponents of min(hero, opponent))',
                        'by_typical_effective_depth')):
        lines += [title + ':', head]
        for name, b in rep[key].items():
            ci = f'{b["bb_per_100"]:+.1f} [{b["lower"]:+.1f}, {b["upper"]:+.1f}]'
            lines.append(f'{name:<12}{b["hands"]:>7}{ci:>28}{b["hands_per_hour"]:>9.0f}{b["bb_per_hour"]:>+8.2f}'
                         f'{b["median_hero_bb"]:>10.0f}{b["median_effective_bb"]:>9.0f}{b["median_typical_bb"]:>9.0f}'
                         + ''.join(f'{b["source_shares"].get(s, 0):>9.0%}' for s in SHOW))
        lines.append('')
    lines.append('Hero stack x effective depth (hands, BB/100):')
    for k, v in sorted(rep['hero_x_effective'].items(), key=lambda kv: -kv[1]['hands']):
        lines.append(f'  {k:<22}{v["hands"]:>6}{v["bb_per_100"]:>+10.1f}')
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('logs', nargs='*', type=Path, default=[Path(DEFAULT_LOG)])
    p.add_argument('--since', help='epoch seconds or ISO time (e.g. 2026-11-18T19:55:47Z)')
    p.add_argument('--until')
    p.add_argument('--big-blind', type=float, default=20)
    p.add_argument('--buy-in', type=float, default=1000)
    p.add_argument('--max-gap', type=float, default=300., help='seconds; longer gaps do not count as playing time')
    p.add_argument('--json', type=Path, help='also write the full report here')
    a = p.parse_args()
    rep = report(a.logs, parse_time(a.since), parse_time(a.until), a.big_blind, a.buy_in, a.max_gap)
    print(render(rep))
    if a.json:
        a.json.write_text(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
