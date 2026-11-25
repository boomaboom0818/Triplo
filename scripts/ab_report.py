"""Live A/B report for Triplo builds: capped and uncapped BB/100, 10-minute windows, big pots.

Reads the client's session log(s) (``--log`` of scripts.openpoker_client) and compares
version windows given as ``NAME=SINCE[..UNTIL]`` (epoch seconds or ISO times). Each
hand's net comes from ``scripts.live_report_depth.hands`` (final-stack differences,
rebuys excluded). The first window is the baseline; later windows get an unpaired
(Welch) difference. Optionally adds the leaderboard tracker's rate for the same windows.

    PYTHONPATH=<release> .venv/bin/python <release>/scripts/ab_report.py --buy-in 5000 \\
        --arm v8=2026-11-25T03:44:36Z..2026-11-26T00:00:00Z --arm v9=2026-11-26T00:00:00Z \\
        --leaderboard runs/bot/leaderboard-track.jsonl

Caps: +-25 BB and +-50 BB per hand are low-variance views that are only valid for
changes that leave big pots alone; always read them next to the uncapped number and the
big-pot breakdown (see poker/ab_metric.py).
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poker import ab_metric as M
from scripts.live_report_depth import DEFAULT_LOG, hands, parse_time, read_events

HALF_WIDTHS = (35, 50, 100)


def parse_arm(text):
    name, _, span = text.partition('=')
    if not name or not span:
        raise argparse.ArgumentTypeError('expected NAME=SINCE[..UNTIL]')
    since, _, until = span.partition('..')
    return name, parse_time(since), parse_time(until) if until else None


def load(paths, big_blind, buy_in):
    rows, pots = [], {}
    for path in paths:
        events = list(read_events(path))
        for e in events:
            if e.get('type') == 'hand_result' and isinstance(e.get('result'), dict):
                r = e['result']
                pots[r.get('hand_id')] = (r.get('total_pot') or r.get('pot') or 0) / big_blind
        rows.extend(hands(events, big_blind, buy_in))
    for r in rows:
        r['pot_bb'] = pots.get(r['hand_id'], 0.)
    return rows


def leaderboard_rate(path, name, since, until, big_blind):
    """Chips won per 100 hands from the 10-minute leaderboard tracker, between the first and
    last snapshots inside the window (score = balance + chips at the table)."""
    snaps = []
    for line in Path(path).read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        bot = row.get('bots', {}).get(name)
        if bot and row['t'] >= since and (until is None or row['t'] < until):
            snaps.append((row['t'], bot['score'], bot['hands_played'], bot.get('rebuys', 0)))
    if len(snaps) < 2 or snaps[-1][2] == snaps[0][2]:
        return dict(snapshots=len(snaps))
    (t0, s0, h0, r0), (t1, s1, h1, r1) = snaps[0], snaps[-1]
    return dict(snapshots=len(snaps), hands=h1 - h0, chips=s1 - s0, rebuys=r1 - r0,
                bb100=round(100 * (s1 - s0) / (h1 - h0) / big_blind, 1), hours=round((t1 - t0) / 3600, 1))


def report(rows, arms, window_s=600., leaderboard=None, hero='Triplo', big_blind=20):
    out = dict(arms={}, differences={})
    data = {}
    for name, since, until in arms:
        sel = [r for r in rows if r['net'] is not None and r['time'] >= since and (until is None or r['time'] < until)]
        nets = [r['net_bb'] for r in sel]
        pots = [r['pot_bb'] for r in sel]
        data[name] = nets
        arm = M.summary(nets, pots)
        arm.pop('positive_blocks')
        arm['positive_10min_windows'] = M.window_share([(r['time'], r['net_bb']) for r in sel], window_s)
        arm['hands_needed'] = {M.cap_label(c): {f'+-{w}': M.hands_for(arm[M.cap_label(c)]['sd_bb'], w)
                                                for w in HALF_WIDTHS}
                               for c in M.CAPS if len(nets) > 1}
        if leaderboard:
            arm['leaderboard'] = leaderboard_rate(leaderboard, hero, since, until, big_blind)
        out['arms'][name] = arm
    base = arms[0][0]
    for name, _, _ in arms[1:]:
        a, b = data[base], data[name]
        diff = {M.cap_label(c): M.welch(a, b, c) for c in M.CAPS}
        if len(a) > 1 and len(b) > 1:
            sd = {M.cap_label(c): (out['arms'][base][M.cap_label(c)]['sd_bb'], out['arms'][name][M.cap_label(c)]['sd_bb'])
                  for c in M.CAPS}
            diff['hands_per_arm_needed_for_+-35'] = {k: M.hands_for_difference(*v, 35) for k, v in sd.items()}
        out['differences'][f'{name}-{base}'] = diff
    return out


def render(rep):
    lines = []
    for name, a in rep['arms'].items():
        w = a['positive_10min_windows']
        lines.append(f'== {name}: {a["uncapped"]["n"]} hands, {w.get("windows", 0)} ten-minute windows, '
                     f'{w.get("positive_share", float("nan")):.0%} positive')
        for c in M.CAPS:
            e = a[M.cap_label(c)]
            lines.append(f'   {M.cap_label(c):9} {e["bb100"]:+8.1f} BB/100 [{e["ci95"][0]:+.1f}, {e["ci95"][1]:+.1f}]'
                         f'  sd {e["sd_bb"]:.1f} BB/hand')
        b = a['big_pots']
        lines.append(f'   pots >= {b["threshold_bb"]:.0f} BB: {b["hands"]} hands ({b["rate"]:.1%}), '
                     f'{b["contribution_bb100"]:+.1f} BB/100 of the total, won {b["won_share"]:.0%}')
        lines.append('   by final pot: ' + ', '.join(f'{k}: {v["hands"]}h {v["contribution_bb100"]:+.1f}'
                                                    for k, v in a['by_pot'].items()))
        if 'leaderboard' in a:
            lines.append(f'   leaderboard tracker: {a["leaderboard"]}')
    for name, d in rep['differences'].items():
        lines.append(f'== {name} (Welch, unpaired)')
        for c in M.CAPS:
            e = d[M.cap_label(c)]
            lines.append(f'   {M.cap_label(c):9} {e["bb100"]:+8.1f} [{e["ci95"][0]:+.1f}, {e["ci95"][1]:+.1f}]')
        if 'hands_per_arm_needed_for_+-35' in d:
            lines.append(f'   hands per arm for +-35 BB/100: {d["hands_per_arm_needed_for_+-35"]}')
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('logs', nargs='*', type=Path, default=[Path(DEFAULT_LOG)])
    p.add_argument('--arm', type=parse_arm, action='append', required=True, help='NAME=SINCE[..UNTIL]')
    p.add_argument('--big-blind', type=float, default=20)
    p.add_argument('--buy-in', type=float, default=5000)
    p.add_argument('--window', type=float, default=600., help='seconds per window (default 10 minutes)')
    p.add_argument('--leaderboard', type=Path, help='runs/bot/leaderboard-track.jsonl')
    p.add_argument('--hero', default='Triplo')
    p.add_argument('--json', type=Path)
    a = p.parse_args()
    rep = report(load(a.logs, a.big_blind, a.buy_in), a.arm, a.window, a.leaderboard, a.hero, a.big_blind)
    print(render(rep))
    if a.json:
        a.json.write_text(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
