"""Full-information line/class records from simulated tables, to measure showdown bias.

Each record: (street, action, faced aggression, class on that street's board,
reached showdown). Same line definitions as calibrate.py.

    python -m scripts.range_sim_lines sim_lines.json 30000
"""
import gc
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from random import Random

from poker.equity_policy import GuardedEquityPolicy
from poker.fast_eval import NAMES
from poker.history_cfr import HistoryPolicy, load
from poker.live_curriculum import root_game
from poker.rule_opponents import RulePolicy
from scripts.range_report import klass

SEATS = ('deployed', 'deployed', 'tight-v1', 'aggressive-v1', 'station-v1', 'tight-v1')
_W = None


def _init(saved):
    global _W
    pols = []
    for style in SEATS:
        if style == 'deployed':
            p = HistoryPolicy(saved, min_visits=64)
            p.fallback = GuardedEquityPolicy()
        else:
            p = RulePolicy(style)
        pols.append(p)
    _W = pols


def play(seed):
    rng = Random(seed)
    players = rng.choices((3, 4, 5, 6), (1, 2, 3, 6))[0]
    stacks = [rng.choice((40, 60, 100, 150, 250, 400)) for _ in range(6)]
    game = root_game(seed, players, stacks, lean=True)
    order = list(range(6))
    rng.shuffle(order)                  # which policy sits at which position
    for j, p in enumerate(_W):
        if hasattr(p, 'rng'):
            p.rng = Random(f'{seed}:{j}')
    while game.live:
        o = game.observe()
        pol = _W[order[o.position]]
        action = pol.act(o, history_complete=True) if isinstance(pol, HistoryPolicy) else pol.act(o)
        game.act(action)
    board = tuple(game.board)
    showdown = sum(game.active) > 1 and len(board) == 5
    out = []
    for i in range(6):
        hole = tuple(game.hole[i])
        style = SEATS[order[i]]
        for s in (1, 2, 3):
            mine = [h for h in game.history if h[1] == i and h[2] == s]
            if not mine or len(board) < s + 2:
                continue
            aggressive = any(isinstance(h[3], int) for h in mine)
            faced = any(isinstance(h[3], int) for h in game.history if h[2] == s and h[1] != i)
            if aggressive:
                did = 'bet/raise'
            elif any(h[3] == 'call' and h[5] > 0 for h in mine):
                did = 'call'
            elif any(h[3] == 'fold' for h in mine):
                did = 'fold'
            else:
                did = 'check'
            reached = showdown and game.active[i]
            out.append((('', 'flop', 'turn', 'river')[s], did, faced, klass(hole, board[:s + 2]), reached, style))
    return out


def chunk(job):
    start, n = job
    rows = []
    for k in range(start, start + n):
        rows.extend(play(k))
    return rows


def main(out, hands, workers=6):
    saved = load('runs/bot/pressure-fresh-001/base-guard.json')
    gc.collect(); gc.freeze()
    ctx = multiprocessing.get_context('fork')
    jobs = [(7_700_000 + k, min(500, hands - k)) for k in range(0, hands, 500)]
    rows = []
    with ProcessPoolExecutor(workers, mp_context=ctx, initializer=_init, initargs=(saved,)) as pool:
        for i, r in enumerate(pool.map(chunk, jobs), 1):
            rows.extend(r)
            print(f'{i}/{len(jobs)}', flush=True)
    json.dump(rows, open(out, 'w'))
    print('records', len(rows))


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]))
