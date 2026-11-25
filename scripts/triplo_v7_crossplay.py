"""Paired 6-max crossplay: Triplo v7-fast and v7-tight against v6-fast, v6-tight and each other.

Two tests, both on the fast simulator with the guarded history checkpoint:

* duplicate head-to-head: three hero seats (``--hero``, default v7_fast) alternate
  with three seats of the other bot; every deal is replayed with the bots swapped,
  so cards, seats and stacks cancel.  The table is zero-sum between the two bots,
  so the hero result is exactly what the other bot loses (per hero seat, BB/100).
* substitution: one hero seat among five copies of a field bot (or a mixed
  v6-fast/v6-tight field).  Every bot plays the same deals, and each v7 build is
  reported as a paired difference against each v6 build.

Each test runs with measured opponent profiles (fitted on a warm-up of mixed
tables) and cold-start (no profiles), at equal 250 BB stacks and at unequal
stacks with the hero at 250 BB.

    python scripts/triplo_v7_crossplay.py --playable /tmp/triplo-factorial-playable.json \\
        --out runs/bot/v7-crossplay-001/report.json
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from math import sqrt
from multiprocessing import get_context
from pathlib import Path
from random import Random
from statistics import mean, stdev
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poker import stack_press
from poker.equity_pricing import VERSION_V5, make_fallback
from poker.fast_game import BIG_BLIND, FastGame
from poker.history_cfr import HistoryPolicy
from poker.opponent_stats import OpponentStats

STACK_PROFILES = (
    (30, 50, 90, 125, 250),
    (50, 75, 125, 200, 350),
    (35, 100, 150, 250, 500),
)
HERO_BB = 250
BOTS = ('v6_fast', 'v6_tight', 'v7_fast', 'v7_tight')
V7 = ('v7_fast', 'v7_tight')
RIVALS = ('v6_fast', 'v6_tight')
SUB_FIELDS = {
    '5x_v6_fast': ('v6_fast',) * 5,
    '5x_v6_tight': ('v6_tight',) * 5,
    'mixed': ('v6_fast', 'v6_fast', 'v6_tight', 'v6_tight', 'v6_fast'),
}
MODES = (  # name, unequal stacks, measured profiles
    ('measured_unequal', True, True),
    ('measured_equal', False, True),
    ('cold_unequal', True, False),
    ('cold_equal', False, False),
)


def build(tag, saved):
    """Each bot exactly as its live command line configures it."""
    if tag not in BOTS:
        raise ValueError(tag)
    v7 = tag.startswith('v7')  # + --turn-defend
    if tag.endswith('tight'):  # --fallback equity-guarded-v5 --river-solver --defend-3bet --flop-peel
        policy = HistoryPolicy(saved, seed=0, min_visits=64)
        policy.fallback = make_fallback(VERSION_V5, solve_river=True, defend_3bet=True, flop_peel=True,
                                        turn_defend=v7)
    else:  # + --stack-press --deep-value --no-learned-postflop
        policy = HistoryPolicy(saved, seed=0, min_visits=64, learned_postflop=False)
        policy.fallback = make_fallback(VERSION_V5, stack_press=True, deep_value=True, solve_river=True,
                                        defend_3bet=True, flop_peel=True, turn_defend=v7)
    return policy


def act(policy, game, o, seed):
    pos = o.position
    press = ()
    if (stack_press.deep_enabled(policy) or stack_press.enabled(policy)) and game.can_raise():
        top = max(game.bets)
        hi = game.stacks[pos] + game.bets[pos]
        lo = min(hi, max(game.raise_size, 2) + top)
        opponents = [(game.bets[j], game.stacks[j]) for j in range(6) if j != pos and game.active[j]]
        sizer = stack_press.deep_sizes if stack_press.deep_enabled(policy) else stack_press.sizes
        press = sizer(o.street, top, o.pot, lo, hi, game.bets[pos], game.stacks[pos], opponents, o.legal)
    policy.rng.seed(f'v7-xplay:{seed}:{pos}:{len(o.history)}')
    policy.fallback.last = None
    return stack_press.act(policy, o, press, history_complete=True)


def play(seats, tags, seed, stacks, profiles, stats=None):
    """One hand; seats[i] is the policy at seat i, tags[i] its bot name. Returns BB per seat."""
    for i, policy in enumerate(seats):
        policy.fallback.profiles = {} if profiles is None else {
            j: profiles[t] for j, t in enumerate(tags) if j != i and t in profiles}
    game = FastGame(seed, button=5, stacks=[s * BIG_BLIND for s in stacks], lean=63)
    while game.live:
        o = game.observe()
        action = act(seats[o.position], game, o, seed)
        if stats is not None and o.street >= 1 and o.call > 0:
            stats.observe(tags[o.position], o.street, 'raise' if isinstance(action, int) else action,
                          o.call, o.pot - o.call)
        game.history.append((o.player, o.position, o.street, action, o.pot, o.call))
        game.act(action, checked=False)
    assert sum(game.payoffs) == 0
    return [p / BIG_BLIND for p in game.payoffs]


def deal_stacks(rng, unequal, hero=None):
    if not unequal:
        return [HERO_BB] * 6
    lineup = list(rng.choice(STACK_PROFILES))
    rng.shuffle(lineup)
    lineup.insert(rng.randrange(6) if hero is None else hero, HERO_BB)
    return lineup


def calibrate(saved, hands):
    """Fit each bot's postflop calling profile on mixed tables, as --opponent-stats would live."""
    pools = {t: [build(t, saved) for _ in range(6)] for t in BOTS}
    stats = OpponentStats(min_faced=10)
    for k in range(hands):
        seed = 881_000_000 + k
        rng = Random(f'v7-xplay-warm:{seed}')
        tags = [rng.choice(BOTS) for _ in range(6)]
        play([pools[t][i] for i, t in enumerate(tags)], tags, seed, deal_stacks(rng, True), None, stats)
    fitted = {t: stats.fit(t) for t in BOTS}
    return {t: (round(b, 4), round(k, 4)) for t, (b, k, _) in fitted.items()}, \
        {t: [round(b, 4), round(k, 4), n] for t, (b, k, n) in fitted.items()}


_SAVED = None
_PROFILES = None
_HERO = 'v7_fast'
_BUYINS = {}  # bot tag -> fixed stack in BB (each bot at its own buy-in)


def _h2h(task):
    rival, first, count, unequal, measured = task
    pools = {t: [build(t, _SAVED) for _ in range(6)] for t in (_HERO, rival)}
    profiles = _PROFILES if measured else None
    out = []
    for seed in range(first, first + count):
        stacks = deal_stacks(Random(f'v7-h2h:{seed}'), unequal)
        total = 0.
        for parity in (0, 1):
            tags = [_HERO if i % 2 == parity else rival for i in range(6)]
            seat_stacks = [_BUYINS.get(t, s) for t, s in zip(tags, stacks)]
            pay = play([pools[t][i] for i, t in enumerate(tags)], tags, seed, seat_stacks, profiles)
            total += sum(p for p, t in zip(pay, tags) if t == _HERO)
        out.append(total / 6)  # per hero seat per hand, averaged over both rotations
    return out


def _sub(task):
    field, first, count, unequal, measured = task
    pools = {t: [build(t, _SAVED) for _ in range(6)] for t in BOTS}
    profiles = _PROFILES if measured else None
    out = {t: [] for t in BOTS}
    for seed in range(first, first + count):
        rng = Random(f'v7-sub:{seed}')
        hero = rng.randrange(6)
        stacks = deal_stacks(rng, unequal, hero)
        others = list(SUB_FIELDS[field])
        rng.shuffle(others)
        for h in BOTS:
            tags = others[:hero] + [h] + others[hero:]
            pay = play([pools[t][i] for i, t in enumerate(tags)], tags, seed, stacks, profiles)
            out[h].append(pay[hero])
    return out


def est(values):
    m = mean(values) * 100
    se = stdev(values) * 100 / sqrt(len(values))
    return dict(bb100=round(m, 1), ci95=[round(m - 1.96 * se, 1), round(m + 1.96 * se, 1)], n=len(values))


def fmt(e):
    return f"{e['bb100']:+7.1f} [{e['ci95'][0]:+.1f}, {e['ci95'][1]:+.1f}]"


def chunks(first, total, workers):
    size = max(32, total // (workers * 3))
    return [(first + i, min(size, total - i)) for i in range(0, total, size)]


def main():
    global _SAVED, _PROFILES, _HERO, _BUYINS
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--playable', type=Path, default=Path('/tmp/triplo-factorial-playable.json'),
                        help="cached playable checkpoint JSON with a 'policy' key")
    parser.add_argument('--h2h-deals', type=int, default=4096, help='duplicate deals per rival and mode')
    parser.add_argument('--sub-deals', type=int, default=2048, help='paired deals per field and mode')
    parser.add_argument('--warmup', type=int, default=1500)
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--seed', type=int, default=951_000_000)
    parser.add_argument('--hero', choices=BOTS, default='v7_fast', help='head-to-head hero')
    parser.add_argument('--tests', nargs='+', choices=('h2h', 'sub'), default=['h2h', 'sub'])
    parser.add_argument('--rivals', nargs='+', choices=BOTS, help='head-to-head rivals (default: every other bot)')
    parser.add_argument('--buyin', action='append', default=[], metavar='BOT=BB',
                        help='head-to-head: give a bot a fixed stack in every seat it takes, e.g. v7_tight=50')
    parser.add_argument('--modes', nargs='+', choices=[m[0] for m in MODES], help='subset of modes to run')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    _HERO = args.hero
    _BUYINS = {tag: int(bb) for tag, bb in (item.split('=') for item in args.buyin)}
    if set(_BUYINS) - set(BOTS):
        parser.error(f'--buyin bot must be one of {BOTS}')
    modes = [m for m in MODES if not args.modes or m[0] in args.modes]

    t0 = perf_counter()
    _SAVED = json.loads(args.playable.read_text())['policy']
    _PROFILES, raw = calibrate(_SAVED, args.warmup)
    print('Fitted profiles (base, slope, faced):', json.dumps(raw), flush=True)
    report = dict(args={k: str(v) for k, v in vars(args).items()}, profiles=raw, h2h={}, substitution={})
    with ProcessPoolExecutor(args.workers, mp_context=get_context('fork')) as pool:
        if 'h2h' in args.tests:
            print(f'\n== Duplicate head-to-head: {_HERO} BB/100 per seat (rival loses the same) ==', flush=True)
        for m, (mode, unequal, measured) in enumerate(modes if 'h2h' in args.tests else ()):
            report['h2h'][mode] = {}
            for r, rival in enumerate(t for t in BOTS if t != _HERO):
                if args.rivals and rival not in args.rivals:
                    continue
                first = args.seed + m * 1_000_000 + r * 100_000
                tasks = [(rival, a, n, unequal, measured) for a, n in chunks(first, args.h2h_deals, args.workers)]
                values = [v for part in pool.map(_h2h, tasks) for v in part]
                report['h2h'][mode][rival] = dict(est(values), per_deal=[round(v, 4) for v in values])
                print(f'  {mode:17s} {_HERO} vs {rival:8s} {fmt(report["h2h"][mode][rival])}', flush=True)
        if 'sub' in args.tests:
            print('\n== Substitution: hero seat among a field; each v7 minus each v6 build, paired ==', flush=True)
        for m, (mode, unequal, measured) in enumerate(modes if 'sub' in args.tests else ()):
            report['substitution'][mode] = {}
            for f, field in enumerate(SUB_FIELDS):
                first = args.seed + 500_000 + m * 1_000_000 + f * 100_000
                tasks = [(field, a, n, unequal, measured) for a, n in chunks(first, args.sub_deals, args.workers)]
                merged = {t: [] for t in BOTS}
                for part in pool.map(_sub, tasks):
                    for t in BOTS:
                        merged[t].extend(part[t])
                row = {t: dict(abs=est(merged[t])) for t in BOTS}
                for v in V7:
                    for t in RIVALS:
                        row[v]['minus_' + t] = est([a - b for a, b in zip(merged[v], merged[t])])
                    diffs = '  '.join(f'-{t}: {fmt(row[v]["minus_" + t])}' for t in RIVALS)
                    print(f'  {mode:17s} {field:11s} {v:8s} {fmt(row[v]["abs"])} | {diffs}', flush=True)
                report['substitution'][mode][field] = row
    report['elapsed_s'] = round(perf_counter() - t0, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f'\nDone in {report["elapsed_s"]}s -> {args.out}', flush=True)


if __name__ == '__main__':
    main()
