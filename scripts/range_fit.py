"""Fit per-(street, action, faced) class multipliers for the v3 opponent range model.

Range weight of a holding = v1 preflop prior x product over the opponent's postflop
streets of F[street|action|faced][class of the holding on that street's board].
Targets are real OpenPoker showdown class frequencies, de-biased per line with a
showdown-selection ratio measured in full-information simulation.

    python -m scripts.range_fit <session.jsonl> <openpoker-hands-full.json> <sim_lines.json> range_factors.json
"""
import collections
import hashlib
import json
import math
import random
import sys
from itertools import combinations

from poker.equity_policy import preflop_weight, private_strength
from poker.fast_eval import CARD_ID

CLASSES = ('two pair+', 'top pair/overpair', 'weaker pair', 'no pair')
STREETS = ('preflop', 'flop', 'turn', 'river')
PSEUDO = 20.0
CLAMP = (0.15, 6.0)
BIAS_CLAMP = (1 / 1.5, 1.5)   # simulation bots set the direction, not the size, of showdown bias


def cls_index(hole, board):
    strong, top, pair = private_strength(hole, board)
    return 0 if strong else 1 if top else 2 if pair else 3


_tables = {}


def class_table(board):
    if board not in _tables:
        av = [c for c in range(52) if c not in board]
        _tables[board] = {(a, b): cls_index((a, b), board) for a, b in combinations(av, 2)}
    return _tables[board]


def label(street, did, faced):
    return f'{street}|{did}|{int(bool(faced))}'


def real_events(session, boards):
    """Per revealed opponent: (hand_id, pre, [(street, label, board_s, class)])."""
    out = []
    for line in open(session):
        r = json.loads(line)
        if r.get('type') != 'hand_result':
            continue
        res = r['result']
        shown = res.get('shown_cards') or {}
        board = boards.get(res['hand_id'])
        if not shown or not board or len(board) < 3:
            continue
        hero = str(r.get('seat'))
        acts = res['actions']
        for seat, cards in shown.items():
            if seat == hero:
                continue
            seat = int(seat)
            hole = tuple(sorted(CARD_ID[c] for c in cards))
            pre = any(a['seat'] == seat and a['street'] == 'preflop' and a['action'] in ('raise', 'all_in') for a in acts)
            path = []
            for s in (1, 2, 3):
                if len(board) < s + 2:
                    break
                name = STREETS[s]
                mine = [a['action'] for a in acts if a['seat'] == seat and a['street'] == name]
                if not mine:
                    continue
                agg = any(a in ('raise', 'all_in') for a in mine)
                faced = any(a['street'] == name and a['seat'] != seat and a['action'] in ('raise', 'all_in') for a in acts)
                did = 'bet/raise' if agg else 'call' if 'call' in mine else 'check' if 'check' in mine else 'fold'
                if did == 'fold':
                    break
                b = tuple(CARD_ID[c] for c in board[:s + 2])
                path.append((s, label(name, did, faced), b, cls_index(hole, b)))
            if path:
                out.append((res['hand_id'], pre, hole, path))
    return out


def bias_ratios(sim_rows):
    full = collections.defaultdict(lambda: [0] * 4)
    sd = collections.defaultdict(lambda: [0] * 4)
    for street, did, faced, k, reached, _ in sim_rows:
        if did == 'fold':
            continue
        lab = label(street, did, faced)
        c = CLASSES.index(k)
        full[lab][c] += 1
        if reached:
            sd[lab][c] += 1
    ratios = {}
    for lab in full:
        f, s = full[lab], sd[lab]
        if sum(s) < 40:
            continue
        ratios[lab] = [min(BIAS_CLAMP[1], max(BIAS_CLAMP[0], ((f[c] + 1) / (sum(f) + 4)) / ((s[c] + 1) / (sum(s) + 4))))
                       for c in range(4)]
    return ratios, {lab: (sum(full[lab]), sum(sd[lab])) for lab in full}


def range_dist(pre, path_upto, board, factors):
    """Class distribution (4) of the modeled range on ``board`` after ``path_upto`` factors."""
    table = class_table(board)
    earlier = [(class_table(b), factors.get(lab)) for _, lab, b, _ in path_upto]
    dist = [0.0] * 4
    for (a, b), k in table.items():
        w = preflop_weight(a, b, pre)
        for t, f in earlier:
            if f is not None:
                w *= f[t[(a, b)]] if (a, b) in t else 1.0
        dist[k] += w
    z = sum(dist)
    return [d / z for d in dist]


def fit(events, ratios, sample_per_label=500, seed=3):
    rng = random.Random(seed)
    factors = {}
    report = {}
    for s in (1, 2, 3):
        by = collections.defaultdict(list)
        for _, pre, _, path in events:
            for i, (st, lab, b, k) in enumerate(path):
                if st == s:
                    by[lab].append((pre, path[:i], b, k))
        for lab, rows in by.items():
            if len(rows) < 30:
                continue
            obs = [0] * 4
            for r in rows:
                obs[r[3]] += 1
            bias = ratios.get(lab, [1.0] * 4)
            target = [obs[c] / len(rows) * bias[c] for c in range(4)]
            z = sum(target)
            target = [t / z for t in target]
            sample = rows if len(rows) <= sample_per_label else rng.sample(rows, sample_per_label)
            pred = [0.0] * 4
            for pre, upto, b, _ in sample:
                d = range_dist(pre, upto, b, factors)
                for c in range(4):
                    pred[c] += d[c] / len(sample)
            n = len(rows)
            f = []
            for c in range(4):
                raw = (target[c] * n + PSEUDO * pred[c]) / ((n + PSEUDO) * max(pred[c], 1e-6))
                f.append(min(CLAMP[1], max(CLAMP[0], raw)))
            factors[lab] = f
            report[lab] = dict(n=n, observed=[o / n for o in obs], target=target, prior=pred, factor=f)
            print(f'{lab:28} n={n:5}  obs {fmt(report[lab]["observed"])}  target {fmt(target)}  prior {fmt(pred)}  F {fmt(f, 2)}',
                  flush=True)
    return factors, report


def fmt(v, d=0):
    return '[' + ' '.join(f'{x:4.2f}' if d else f'{100 * x:3.0f}' for x in v) + ']'


def holdout_loss(events, factors, ratios, limit=1500, seed=5):
    """Mean log-loss of the shown class on held-out showdown events: v1-style prior
    (no postflop factors) vs fitted factors, both mapped to showdown frequencies."""
    rng = random.Random(seed)
    rows = [(pre, path[:i], b, k, lab, sum(1 for _, l, _, _ in path[:i + 1] if '|bet/raise|' in l))
            for _, pre, _, path in events for i, (_, lab, b, k) in enumerate(path)]
    rows = rows if len(rows) <= limit else rng.sample(rows, limit)
    loss = {'prior only': 0.0, 'deployed v1 (pressure)': 0.0, 'fitted v3': 0.0}
    from poker.equity_policy import holding_range
    for pre, upto, b, k, lab, pressure in rows:
        bias = ratios.get(lab, [1.0] * 4)
        pairs, cum, total = holding_range(tuple(sorted(b)), b, pre, min(pressure, 2))
        table, d, prev = class_table(b), [0.0] * 4, 0.0
        for hand, c in zip(pairs, cum):
            d[table[hand]] += c - prev
            prev = c
        sd = [d[c] / bias[c] for c in range(4)]
        loss['deployed v1 (pressure)'] -= math.log(max(sd[k] / sum(sd), 1e-6)) / len(rows)
        for name, f in (('prior only', {}), ('fitted v3', factors)):
            d = range_dist(pre, upto + [(None, lab, b, None)] if name == 'fitted v3' else [], b, f)
            sd = [d[c] / bias[c] for c in range(4)]
            z = sum(sd)
            loss[name] -= math.log(max(sd[k] / z, 1e-6)) / len(rows)
    return loss, len(rows)


def main(session, boards_file, sim_file, out):
    boards = {k: v['board'] for k, v in json.load(open(boards_file)).items()}
    events = real_events(session, boards)
    ratios, counts = bias_ratios(json.load(open(sim_file)))
    print('showdown-bias ratios from simulation (full-info / showdown), classes', CLASSES)
    for lab in sorted(ratios):
        print(f'  {lab:28} sim n={counts[lab][0]:6} showdown n={counts[lab][1]:5}  {fmt(ratios[lab], 2)}')
    train = [e for e in events if int(hashlib.md5(e[0].encode()).hexdigest(), 16) % 10 < 7]
    test = [e for e in events if int(hashlib.md5(e[0].encode()).hexdigest(), 16) % 10 >= 7]
    print(f'\nreal revealed opponents: {len(events)} (train {len(train)}, holdout {len(test)})\n')
    factors, report = fit(train, ratios)
    loss, n = holdout_loss(test, factors, ratios)
    print(f'\nholdout log-loss over {n} street decisions: ' + ', '.join(f'{k} {v:.3f}' for k, v in loss.items()))
    print('(lower is better; a uniform guess over 4 classes scores 1.386)')
    print('\nrefit on all data for the shipped table:')
    factors_all, report_all = fit(events, ratios)
    json.dump(dict(classes=CLASSES, factors=factors_all, report=report_all, bias=ratios, holdout=loss), open(out, 'w'), indent=1)


if __name__ == '__main__':
    main(*sys.argv[1:5])
