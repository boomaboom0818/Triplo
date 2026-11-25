"""Paired benchmark of Triplo builds at simulated OpenPoker tables of no_code fish.

Fish use ``poker.nocode_model.NoCodeFish`` (fitted to live no_code decisions); regulars
are v7-fast copies. Every arm plays the same deals, seats, stacks and opponents, so
arm differences are paired. Stacks follow live data: fish 30-113 BB (median 80),
regulars and hero 250 BB unless an arm sets ``hero_bb``.

    python scripts/fish_field_benchmark.py --deals 8192 --arms v7_fast v7_tight_50bb --out report.json
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

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from poker.fast_game import BIG_BLIND, FastGame
from poker.fish_preflop import FishPreflop
from poker.nocode_model import NoCodeFish
from poker.opponent_stats import OpponentStats
from scripts.v6_field_benchmark import ARM_PRESETS, CHECKPOINT_CACHE, build_policy, hero_act_v5fast

FISH_STACKS = (30, 45, 60, 70, 80, 80, 85, 90, 95, 100, 113)
REG_BB = 250
FIELDS = {
    'fish5': ('fish',) * 5,
    'fish4_reg1': ('fish',) * 4 + ('reg',),
    'fish3_reg2': ('fish',) * 3 + ('reg',) * 2,
    'fish2_reg3': ('fish',) * 2 + ('reg',) * 3,
}
V7 = {'defend_3bet': True, 'flop_peel': True, 'turn_defend': True}
EXTRA_ARMS = {
    'fx_nobluff': {'fallback_kwargs': dict(V7, fish_nobluff=True)},
    'fx_size_mid': {'fallback_kwargs': dict(V7, fish_size={1: .66, 2: .75, 3: 1.0})},
    'fx_size_big': {'fallback_kwargs': dict(V7, fish_size={1: .75, 2: 1.0, 3: 1.25})},
    'fx_thin': {'fallback_kwargs': dict(V7, fish_thin=(.45, .6))},
    'fx_thin_loose': {'fallback_kwargs': dict(V7, fish_thin=(.35, .6))},
    'fx_all': {'fallback_kwargs': dict(V7, fish_nobluff=True, fish_size={1: .66, 2: .75, 3: 1.0}, fish_thin=(.45, .6))},
    'pf_default': {'fallback_kwargs': V7, 'preflop': {}},
    'pf_tight': {'fallback_kwargs': V7, 'preflop': dict(open_width=.40, steal_width=.55, iso_width=.08, call_width=.22)},
    'pf_wide': {'fallback_kwargs': V7, 'preflop': dict(open_width=.70, steal_width=.85, iso_width=.18, call_width=.40)},
    'pf_steal_only': {'fallback_kwargs': V7, 'preflop': dict(iso=False)},
    'pf_iso_only': {'fallback_kwargs': V7, 'preflop': dict(steal=False)},
    'fx_cbet': {'fallback_kwargs': dict(V7, fish_cbet=.5)},
    'fx_cbet_small': {'fallback_kwargs': dict(V7, fish_cbet=.33)},
    'combo_a': {'fallback_kwargs': dict(V7, fish_size={1: .75, 2: 1.0, 3: 1.25}, fish_cbet=.5), 'preflop': dict(steal=False)},
    'combo_b': {'fallback_kwargs': dict(V7, fish_size={1: .75, 2: 1.0, 3: 1.25}, fish_cbet=.5),
                'preflop': dict(steal=False, iso_width=.18, call_width=.40)},
    'v8_size': {'fallback_kwargs': dict(V7, fish_size={1: .75, 2: 1.0, 3: 1.25})},
    'v8_size_iso': {'fallback_kwargs': dict(V7, fish_size={1: .75, 2: 1.0, 3: 1.25}), 'preflop': dict(steal=False)},
    'v8_size_isowide': {'fallback_kwargs': dict(V7, fish_size={1: .75, 2: 1.0, 3: 1.25}),
                        'preflop': dict(steal=False, iso_width=.18, call_width=.40)},
    'v8_raise_top': {'fallback_kwargs': dict(V7, fish_size={1: .75, 2: 1.0, 3: 1.25}, fish_raise=('top', 1.0))},
    'v8_raise_two': {'fallback_kwargs': dict(V7, fish_size={1: .75, 2: 1.0, 3: 1.25}, fish_raise=('two', 1.0))},
    'v8_raise_top_small': {'fallback_kwargs': dict(V7, fish_size={1: .75, 2: 1.0, 3: 1.25}, fish_raise=('top', .6))},
    'v8_full_top': {'fallback_kwargs': dict(V7, fish_size={1: .75, 2: 1.0, 3: 1.25}, fish_raise=('top', 1.0)),
                    'preflop': dict(steal=False)},
    'v8_full_two': {'fallback_kwargs': dict(V7, fish_size={1: .75, 2: 1.0, 3: 1.25}, fish_raise=('two', 1.0)),
                    'preflop': dict(steal=False)},
    'v8_full_top_small': {'fallback_kwargs': dict(V7, fish_size={1: .75, 2: 1.0, 3: 1.25}, fish_raise=('top', .6)),
                          'preflop': dict(steal=False)},
    # Triplo v8-fast exactly as --fish-exploit configures it (scripts.openpoker_client.FISH_EXPLOIT).
    'v8_fast': {'fallback_kwargs': dict(V7, fish_size={1: .75, 2: 1.0, 3: 1.25}, fish_raise=('top', .6)),
                'preflop': dict(steal=False)},
    'pf_default_size': {'fallback_kwargs': dict(V7, fish_size={1: .75, 2: 1.0, 3: 1.25}), 'preflop': {}},
}


def arm_config(name):
    return EXTRA_ARMS.get(name) or ARM_PRESETS[name]


def deal(field, seed):
    rng = Random(f'fish-lineup:{field}:{seed}')
    hero = rng.randrange(6)
    others = list(FIELDS[field])
    rng.shuffle(others)
    stacks = [rng.choice(FISH_STACKS) if t == 'fish' else REG_BB for t in others]
    return hero, others, stacks


def play(hero_policy, hero_bb, regs, fish, seed, hero, others, stacks, profiles):
    tags = others[:hero] + ['hero'] + others[hero:]
    st = stacks[:hero] + [hero_bb] + stacks[hero:]
    seats = {}
    reg_iter = iter(regs)
    for i, t in enumerate(tags):
        seats[i] = hero_policy if t == 'hero' else next(reg_iter) if t == 'reg' else fish
    for i, t in enumerate(tags):
        if t in ('hero', 'reg'):
            seats[i].rng.seed(f'fish-bench:{seed}:{i}')
    kind = {i: 'reg' if t == 'hero' else t for i, t in enumerate(tags)}
    game = FastGame(seed, button=5, stacks=[s * BIG_BLIND for s in st],
                    lean=sum(1 << i for i, t in enumerate(tags) if t != 'fish'))
    while game.live:
        o = game.observe()
        i = o.position
        if tags[i] == 'fish':
            action = fish.act(o, game)
        else:
            seats[i].fallback.profiles = {} if profiles is None else {
                j: profiles[kind[j]] for j in range(6) if j != i}
            seats[i].fallback.fish_seats = {j for j in range(6) if tags[j] == 'fish'}
            action, _ = hero_act_v5fast(seats[i], game, o)
        game.history.append((o.player, o.position, o.street, action, o.pot, o.call))
        game.act(action, checked=False)
    return game.payoffs[hero] / BIG_BLIND


_SAVED = None
_ARMS = None
_PROFILES = None


def _batch(task):
    field, first, count, names, measured = task
    hero_pols = {}
    for n in names:
        cfg = arm_config(n)
        pol = build_policy(_SAVED, cfg)
        hero_pols[n] = FishPreflop(pol, **cfg['preflop']) if 'preflop' in cfg else pol
    regs = [build_policy(_SAVED, ARM_PRESETS['v7_fast']) for _ in range(3)]
    fish = NoCodeFish(seed=7)
    out = {n: [] for n in names}
    for seed in range(first, first + count):
        hero, others, stacks = deal(field, seed)
        for n in names:
            out[n].append(play(hero_pols[n], arm_config(n).get('hero_bb', REG_BB), regs, fish, seed,
                               hero, others, stacks, _PROFILES if measured else None))
    return out


def calibrate(saved, deals=400):
    """Fit OpponentStats (base, slope) for fish and regulars, as --opponent-stats does live."""
    stats = OpponentStats(min_faced=10)
    regs = [build_policy(saved, ARM_PRESETS['v7_fast']) for _ in range(3)]
    fish = NoCodeFish(seed=11)
    for seed in range(deals):
        hero, others, stacks = deal('fish4_reg1', 990_000 + seed)
        tags = others[:hero] + ['reg'] + others[hero:]
        st = stacks[:hero] + [REG_BB] + stacks[hero:]
        pols = {}
        it = iter(regs)
        for i, t in enumerate(tags):
            pols[i] = next(it) if t == 'reg' else fish
        game = FastGame(990_000 + seed, button=5, stacks=[s * BIG_BLIND for s in st],
                        lean=sum(1 << i for i, t in enumerate(tags) if t == 'reg'))
        while game.live:
            o = game.observe()
            t = tags[o.position]
            if t == 'fish':
                action = fish.act(o, game)
            else:
                pols[o.position].fallback.profiles = {}
                action, _ = hero_act_v5fast(pols[o.position], game, o)
            if o.street >= 1 and o.call > 0:
                stats.observe(t, o.street, 'raise' if isinstance(action, int) else action, o.call, o.pot - o.call)
            game.history.append((o.player, o.position, o.street, action, o.pot, o.call))
            game.act(action, checked=False)
    return {t: tuple(round(x, 4) for x in stats.fit(t)[:2]) for t in ('fish', 'reg')}


def est(vals):
    m = mean(vals) * 100
    se = stdev(vals) * 100 / sqrt(len(vals))
    return dict(bb100=round(m, 1), ci95=[round(m - 1.96 * se, 1), round(m + 1.96 * se, 1)], n=len(vals))


def run(args, names, measured):
    out = {}
    with ProcessPoolExecutor(args.workers, mp_context=get_context('fork')) as pool:
        for f_idx, field in enumerate(args.fields):
            first = args.seed + f_idx * 1_000_000
            size = max(32, args.deals // (args.workers * 3))
            tasks = [(field, first + i, min(size, args.deals - i), names, measured) for i in range(0, args.deals, size)]
            merged = {n: [] for n in names}
            for part in pool.map(_batch, tasks):
                for n in names:
                    merged[n].extend(part[n])
            base = names[0]
            out[field] = {n: dict(abs=est(merged[n]),
                                  delta=None if n == base else est([a - b for a, b in zip(merged[n], merged[base])]))
                          for n in names}
            for n in names:
                r = out[field][n]
                d = r['delta']
                print(f"  {field:11} {n:22} abs {r['abs']['bb100']:+8.1f} {r['abs']['ci95']}"
                      + (f"   delta {d['bb100']:+7.1f} {d['ci95']}" if d else '   (baseline)'), flush=True)
    return out


def main():
    global _SAVED, _PROFILES
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--deals', type=int, default=4096)
    ap.add_argument('--workers', type=int, default=9)
    ap.add_argument('--seed', type=int, default=610_000_000)
    ap.add_argument('--fields', nargs='+', default=list(FIELDS))
    ap.add_argument('--arms', nargs='+', required=True)
    ap.add_argument('--cold', action='store_true', help='also run without opponent profiles')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    t0 = perf_counter()
    _SAVED = json.loads(CHECKPOINT_CACHE.read_text())['policy']
    _PROFILES = calibrate(_SAVED)
    print('profiles', _PROFILES, flush=True)
    report = dict(args=vars(args) | dict(out=str(args.out)), profiles=_PROFILES)
    print('== measured profiles ==', flush=True)
    report['measured'] = run(args, args.arms, True)
    if args.cold:
        print('== cold start ==', flush=True)
        report['cold'] = run(args, args.arms, False)
    report['elapsed_s'] = round(perf_counter() - t0, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f'done in {report["elapsed_s"]}s', flush=True)


if __name__ == '__main__':
    main()
