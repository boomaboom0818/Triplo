"""Paired Triplo v9 benchmark at simulated OpenPoker fish tables, with v9's A/B metrics.

Same tables as scripts/fish_field_benchmark.py (NoCodeFish seats with live stacks, v7-fast
regulars, hero at 250 BB), but every hand is recorded: net, final pot, hero decision
sources, which v9 rule fired, and a digest of the hero's actions. The report gives, for each
arm against the first arm on the same deals: paired capped (+-25, +-50 BB) and uncapped
BB/100 differences with 95% intervals (poker/ab_metric.py), the big-pot breakdown, the
share of 14-hand blocks (about 10 live minutes) the arm won, preflop learned-policy
coverage, and the share of hands whose hero actions differ from the baseline.

Arms are ``v8`` (Triplo v8 exactly as --fish-exploit configures it) or comma lists of
v9 options on top of v8:

    off            every v9 option explicitly off (must be decision-identical to v8)
    open=3.5       --fish-open 3.5          lim=1.0   --fish-open-limper
    scope=majority open sizing when at least half the others are fish (default all)
    stab=0.33      --fish-stab 0.33         stabh=nopair  --fish-stab-hands nopair
    cap=0.75       --fish-value-cap 0.75
    limp=0.25      --raise-not-limp 0.25    limps=fish  only when every other player is a fish
    cbet=0.5       v8's unused fish_cbet (menu-snapped), for reference

    python scripts/v9_field_benchmark.py --deals 24576 --arms v8 open=3.5 stab=0.33 --out r.json
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from hashlib import blake2b
import json
from math import sqrt
from multiprocessing import get_context
from pathlib import Path
from random import Random
import sys
from time import perf_counter

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from poker import ab_metric as M
from poker.fast_game import BIG_BLIND, FastGame
from poker.fish_preflop import FishPreflop
from poker.nocode_model import NoCodeFish
from scripts.fish_field_benchmark import FIELDS, REG_BB, V7, calibrate, deal
from scripts.openpoker_client import FISH_EXPLOIT
from scripts.v6_field_benchmark import ARM_PRESETS, CHECKPOINT_CACHE, build_policy, hero_act_v5fast

# Share of Triplo's live full-table hands by fish count among the five opponents (v8 live
# window 2026-11-25, first decisions under the gun: 5 fish 4%, 4 fish 48%, 3 fish 40%,
# 2 or fewer 8%). Used only for the weighted summary; every field is also reported.
LIVE_WEIGHTS = {'fish5': .04, 'fish4_reg1': .48, 'fish3_reg2': .40, 'fish2_reg3': .08}
LIVE_REG_PROFILE = (0.06, 1.9)   # how --opponent-stats measures live winning regulars (base, slope)
TAGS = ('fish_stab', 'fish_value_cap', 'fish_size', 'fish_raise', 'fish_cbet')


def parse_arm(spec):
    """Arm spec -> (fallback kwargs, FishPreflop kwargs)."""
    fallback = dict(V7, **FISH_EXPLOIT)
    preflop = dict(steal=False)
    if spec == 'v8':
        return fallback, preflop
    for part in spec.split(','):
        key, _, value = part.partition('=')
        if key == 'off':
            fallback.update(fish_stab=None, fish_value_cap=None)
            preflop.update(open_bb=None, limp_raise=None)
        elif key == 'open':
            preflop['open_bb'] = float(value)
        elif key == 'lim':
            preflop['open_limper_bb'] = float(value)
        elif key == 'scope':
            preflop['open_scope'] = value
        elif key == 'stab':
            fallback['fish_stab'] = (float(value), 'all')
        elif key == 'stabh':
            fallback['fish_stab'] = (fallback['fish_stab'][0], value)
        elif key == 'cap':
            fallback['fish_value_cap'] = float(value)
        elif key == 'limp':
            preflop['limp_raise'] = float(value)
        elif key == 'limps':
            preflop['limp_raise_scope'] = value
        elif key == 'cbet':
            fallback['fish_cbet'] = float(value)
        else:
            raise ValueError(f'unknown arm option {part!r}')
    return fallback, preflop


class SizedFish(NoCodeFish):
    """Robustness variant: fish continue less often against bigger opens, as live
    (fold 65-69% to 2-3 BB, 71% to 3.5-4 BB, 74% to 4+ BB)."""

    def _preflop(self, o, top, lo, hi, to, passive):
        raised = [h for h in o.history if h[2] == 0 and isinstance(h[3], int)]
        if len(raised) == 1:
            size_bb = raised[0][3] / 2
            mult = 1.0 if size_bb <= 3 else .90 if size_bb <= 4 else .81
            if self._rank(o) >= .30 * mult and self._rank(o) < .30:
                return passive
        return super()._preflop(o, top, lo, hi, to, passive)


def build_hero(saved, spec):
    fb_kwargs, pf_kwargs = parse_arm(spec)
    pol = build_policy(saved, {'fallback_kwargs': fb_kwargs})
    return FishPreflop(pol, **pf_kwargs)


def play(hero_policy, regs, fish, seed, hero, others, stacks, profiles):
    tags = others[:hero] + ['hero'] + others[hero:]
    st = stacks[:hero] + [REG_BB] + stacks[hero:]
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
    rec = dict(src=Counter(), reg_src=Counter(), tags=Counter())
    trail = []
    hero_raised_pre = False
    while game.live:
        o = game.observe()
        i = o.position
        if tags[i] == 'fish':
            action = fish.act(o, game)
        else:
            pol = seats[i]
            pol.fallback.profiles = {} if profiles is None else {j: profiles[kind[j]] for j in range(6) if j != i}
            pol.fallback.fish_seats = {j for j in range(6) if tags[j] == 'fish'}
            if tags[i] == 'hero':
                before = (pol.overrides, pol.resized, pol.limp_raises)
                pol.fallback.last = None
                action, src = hero_act_v5fast(pol, game, o)
                after = (pol.overrides, pol.resized, pol.limp_raises)
                if after[0] > before[0]:
                    src = 'fish_preflop'
                if not o.street:
                    rec['src']['pre_after_raise:' + src if hero_raised_pre else 'pre:' + src] += 1
                    if isinstance(action, int):
                        hero_raised_pre = True
                else:
                    rec['src']['post:' + src] += 1
                if after[1] > before[1]:
                    rec['tags']['fish_open'] += 1
                if after[2] > before[2]:
                    rec['tags']['limp_raise'] += 1
                for t in TAGS:
                    if (pol.fallback.last or {}).get(t):
                        rec['tags'][t] += 1
                trail.append((o.street, len(o.history), action))
            else:
                action, src = hero_act_v5fast(pol, game, o)
                if not o.street:
                    rec['reg_src'][src] += 1
        game.history.append((o.player, o.position, o.street, action, o.pot, o.call))
        game.act(action, checked=False)
    rec['net'] = game.payoffs[hero] / BIG_BLIND
    rec['pot'] = sum(game.paid) / BIG_BLIND
    rec['digest'] = blake2b(repr(trail).encode(), digest_size=8).hexdigest()
    return rec


_SAVED = None
_PROFILES = None


def _batch(task):
    field, first, count, specs, fish_kind = task
    heroes = {s: build_hero(_SAVED, s) for s in specs}
    regs = [build_policy(_SAVED, ARM_PRESETS['v7_fast']) for _ in range(3)]
    fish = (SizedFish if fish_kind == 'sized' else NoCodeFish)(seed=7)
    out = {s: [] for s in specs}
    for seed in range(first, first + count):
        hero, others, stacks = deal(field, seed)
        for s in specs:
            out[s].append(play(heroes[s], regs, fish, seed, hero, others, stacks, _PROFILES))
    return out


def arm_stats(recs):
    nets = [r['net'] for r in recs]
    pots = [r['pot'] for r in recs]
    src, reg, tags = Counter(), Counter(), Counter()
    for r in recs:
        src.update(r['src'])
        reg.update(r['reg_src'])
        tags.update(r['tags'])

    def share(counter, prefix, key='learned_policy'):
        total = sum(v for k, v in counter.items() if k.startswith(prefix))
        return dict(decisions=total, learned=round(counter.get(prefix + key, 0) / total, 4) if total else None)

    out = M.summary(nets, pots)
    out['coverage'] = dict(hero_pre_first=share(src, 'pre:'), hero_pre_after_own_raise=share(src, 'pre_after_raise:'),
                           reg_pre=dict(decisions=sum(reg.values()),
                                        learned=round(reg['learned_policy'] / max(1, sum(reg.values())), 4)),
                           hero_sources=dict(src))
    out['rules_fired_per_100'] = {k: round(100 * v / len(recs), 2) for k, v in tags.items()}
    return out


def run(args, specs, fish_kind):
    out = {}
    with ProcessPoolExecutor(args.workers, mp_context=get_context('fork')) as pool:
        for f_idx, field in enumerate(args.fields):
            first = args.seed + f_idx * 1_000_000
            size = max(32, args.deals // (args.workers * 3))
            tasks = [(field, first + i, min(size, args.deals - i), specs, fish_kind) for i in range(0, args.deals, size)]
            merged = {s: [] for s in specs}
            for part in pool.map(_batch, tasks):
                for s in specs:
                    merged[s].extend(part[s])
            base = specs[0]
            bnets, bpots = [r['net'] for r in merged[base]], [r['pot'] for r in merged[base]]
            field_out = {}
            for s in specs:
                st = arm_stats(merged[s])
                if s != base:
                    nets, pots = [r['net'] for r in merged[s]], [r['pot'] for r in merged[s]]
                    st['delta'] = M.compare(bnets, nets, bpots, pots)
                    st['delta']['changed_hands'] = round(
                        sum(a['digest'] != b['digest'] for a, b in zip(merged[base], merged[s])) / len(bnets), 4)
                    st['delta']['big_pots_arm_minus_base'] = round(
                        st['big_pots']['rate'] - M.big_pot_summary(bnets, bpots)['rate'], 4)
                field_out[s] = st
                d = st.get('delta')
                cov = st['coverage']
                line = (f"  {field:11} {s:28} abs {st['uncapped']['bb100']:+7.1f}"
                        f"  learned pre {cov['hero_pre_first']['learned']}/{cov['hero_pre_after_own_raise']['learned']}"
                        f" reg {cov['reg_pre']['learned']}")
                if d:
                    line += (f"  d.unc {d['uncapped']['bb100']:+6.1f} {d['uncapped']['ci95']}"
                             f"  d.c50 {d['cap50']['bb100']:+6.1f} {d['cap50']['ci95']}"
                             f"  d.c25 {d['cap25']['bb100']:+6.1f} {d['cap25']['ci95']}"
                             f"  ahead {d['arm_ahead_blocks']['arm_ahead_share']}  changed {d['changed_hands']}")
                print(line, flush=True)
            out[field] = field_out
    return out


def weighted(report, specs, fields):
    """Live-mix weighted paired differences (fields are independent samples)."""
    w = {f: LIVE_WEIGHTS[f] for f in fields}
    total = sum(w.values())
    out = {}
    for s in specs[1:]:
        row = {}
        for cap in M.CAPS:
            label = M.cap_label(cap)
            mean = sum(w[f] / total * report[f][s]['delta'][label]['bb100'] for f in fields)
            se = sqrt(sum((w[f] / total * (report[f][s]['delta'][label]['ci95'][1]
                                           - report[f][s]['delta'][label]['ci95'][0]) / (2 * M.Z95)) ** 2
                          for f in fields))
            row[label] = dict(bb100=round(mean, 2), ci95=[round(mean - M.Z95 * se, 2), round(mean + M.Z95 * se, 2)])
        out[s] = row
    return out


def main():
    global _SAVED, _PROFILES
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--deals', type=int, default=4096)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--seed', type=int, default=930_000_000)
    ap.add_argument('--fields', nargs='+', default=list(FIELDS))
    ap.add_argument('--arms', nargs='+', required=True)
    ap.add_argument('--fish', choices=('model', 'sized'), default='model',
                    help='sized: fish fold more to bigger opens, as measured live (robustness check)')
    ap.add_argument('--reg-profile', choices=('measured', 'live'), default='measured',
                    help='live: give regulars the sticky (base, slope) profile live --opponent-stats measures')
    ap.add_argument('--profiles-cache', type=Path, default=Path('/tmp/triplo-v9-fish-profiles.json'))
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    for s in args.arms:
        parse_arm(s)
    t0 = perf_counter()
    _SAVED = json.loads(CHECKPOINT_CACHE.read_text())['policy']
    if args.profiles_cache.exists():
        _PROFILES = {k: tuple(v) for k, v in json.loads(args.profiles_cache.read_text()).items()}
    else:
        _PROFILES = calibrate(_SAVED)
        args.profiles_cache.write_text(json.dumps(_PROFILES))
    if args.reg_profile == 'live':
        _PROFILES = dict(_PROFILES, reg=LIVE_REG_PROFILE)
    print('profiles', _PROFILES, flush=True)
    report = dict(args=vars(args) | dict(out=str(args.out), profiles_cache=str(args.profiles_cache)), profiles=_PROFILES)
    report['fields'] = run(args, args.arms, args.fish)
    if len(args.arms) > 1:
        report['weighted'] = weighted(report['fields'], args.arms, args.fields)
        for s, row in report['weighted'].items():
            print(f'  weighted {s:28} ' + '  '.join(f"{k} {v['bb100']:+6.1f} {v['ci95']}" for k, v in row.items()),
                  flush=True)
    report['elapsed_s'] = round(perf_counter() - t0, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print(f'done in {report["elapsed_s"]}s', flush=True)


if __name__ == '__main__':
    main()
