"""Showdown range report: real OpenPoker holdings by line vs the fallback range model.

For every revealed opponent hand, rebuild the line it took (preflop raise? number
of postflop bets/raises so far, what it did on the street) and the hand's class on
that street's board. Predicted class frequencies come from the model's own
weighted range (poker.equity_policy.holding_range) for the same board and line.

    python -m scripts.range_report runs/bot/openpoker-season-17-equity/session.jsonl runs/bot/openpoker-hands-full.json
"""
import collections
import json
import random
import sys
from itertools import combinations

from poker.equity_policy import holding_range, private_strength
from poker.fast_eval import CARD_ID

STREETS = ('preflop', 'flop', 'turn', 'river')
CLASSES = ('two pair+', 'top pair/overpair', 'weaker pair', 'no pair')


def klass(hole, board):
    strong, top, pair = private_strength(hole, board)
    return CLASSES[0] if strong else CLASSES[1] if top else CLASSES[2] if pair else CLASSES[3]


_pred = {}


def predicted(board, pre, pressure):
    key = (board, pre, min(pressure, 2))
    if key not in _pred:
        pairs, cumulative, total = holding_range(tuple(sorted(board)), board, pre, min(pressure, 2))
        dist = collections.Counter()
        prev = 0.0
        for hand, cum in zip(pairs, cumulative):
            dist[klass(hand, board)] += cum - prev
            prev = cum
        _pred[key] = {k: v / total for k, v in dist.items()}
    return _pred[key]


def events(session, boards):
    """Yield (line label, street, board, pre, pressure, class) per revealed opponent and street."""
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
            hole = tuple(CARD_ID[c] for c in cards)
            pre = any(a['seat'] == int(seat) and a['street'] == 'preflop' and a['action'] in ('raise', 'all_in')
                      for a in acts)
            pressure = 0
            for s in (1, 2, 3):
                name = STREETS[s]
                if len(board) < s + 2:
                    break
                mine = [a['action'] for a in acts if a['seat'] == int(seat) and a['street'] == name]
                if not mine:
                    continue
                aggressive = any(a in ('raise', 'all_in') for a in mine)
                faced = any(a['street'] == name and a['seat'] != int(seat) and a['action'] in ('raise', 'all_in')
                            for a in acts)
                did = ('bet/raise' if aggressive else 'call' if 'call' in mine else 'check' if 'check' in mine
                       else 'fold')
                b = tuple(CARD_ID[c] for c in board[:s + 2])
                # the model at a decision on this street sees pressure from earlier streets plus this one
                p = pressure + (1 if aggressive else 0)
                yield (name, did, faced), b, pre, p, klass(hole, b)
                pressure = p


def main(session, boards_file, limit=4000):
    boards = {k: v['board'] for k, v in json.load(open(boards_file)).items()}
    groups = collections.defaultdict(list)
    for label, b, pre, p, cls in events(session, boards):
        groups[label].append((b, pre, p, cls))
    rng = random.Random(7)
    print('%-34s %5s  %s' % ('line (street, action, faced aggression)', 'n',
                             '   '.join('%-17s' % c for c in CLASSES)))
    print('%-34s %5s  %s' % ('', '', '   '.join('%-17s' % 'real / model' for _ in CLASSES)))
    for label in sorted(groups, key=lambda k: (STREETS.index(k[0]), k[1], k[2])):
        rows = groups[label]
        if len(rows) < 25:
            continue
        sample = rows if len(rows) <= limit else rng.sample(rows, limit)
        real = collections.Counter(r[3] for r in rows)
        model = collections.Counter()
        for b, pre, p, _ in sample:
            for k, v in predicted(b, pre, p).items():
                model[k] += v
        n, m = len(rows), len(sample)
        cells = '   '.join('%5.0f%% / %5.0f%%   ' % (100 * real[c] / n, 100 * model[c] / m) for c in CLASSES)
        print('%-34s %5d  %s' % ('%s %s%s' % (label[0], label[1], ' (facing aggression)' if label[2] else ''), n, cells),
              flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
