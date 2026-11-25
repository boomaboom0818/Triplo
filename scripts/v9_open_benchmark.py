"""Paired Triplo v9 benchmark for --open-translation against opponents who open big.

Tables follow scripts/fish_field_benchmark.py (NoCodeFish seats with live stacks, hero
and regulars at 250 BB, four fish/regular mixes) with v9's A/B metrics. Two things are
new, because the simulated regulars (v7-fast copies) only open to the trained 2-2.5 BB
and so never test the leak:

* ``--regs``: what sits in the regular seats.
    v7    v7-fast copies (the old benchmark)
    big   every regular is a big opener (half ``fixed``, half ``svan``; stress test)
    three every regular opens 3 BB with a live 'reg'-class range (16-35% by seat) and
          answers 3-bets as those bots do (fold 45%, 4-bet 12%); stress test for tr3
    mix   each regular is a big opener with probability 0.25, else v7-fast: about 15%
          of regular opens are 3.5 BB+, as in the live v8 window (13-17%)
  Big openers are fitted to the live season-17 log (scripts/v9_open_report.py): they
  open 60-70% of hands from every seat, never limp, and as the opener facing a 3-bet
  call 63%, fold 25%, 4-bet 11%. ``fixed`` (RoRo/Herqules-like) always opens 3.5 BB
  (+1 BB per limper). ``svan`` (the Svan bots) opens 4 BB with most of its top 20% and
  2.5 BB otherwise, as their shown hands say (4 BB opens are much stronger). Every
  other decision is a v7-fast copy's.
* ``--fish sticky``: fish as the opener facing a 3-bet call 91%, fold 2% and 4-bet 7%,
  as live (season 17, 2,027 spots); NoCodeFish folds 15-30% of its range there, which
  would overpay 3-bet bluffs. NoCodeFish already opens 3 BB 30% of the time (live 28%).

Arms: ``v8`` (Triplo v8-fast as --fish-exploit configures it), ``off`` (v9 code, flag
off: must equal v8), ``tr`` (--open-translation, all sizes), ``trN`` (translate opens up
to N BB only, e.g. tr3), with suffixes ``f`` (opens by fish seats only) and ``r`` / ``c``
(keep only the translated 3-bets / calls; other translated actions go to the rules).

Each run writes one JSON per field to --out-dir and skips fields already there, so long
runs go in short resumable chunks; ``--summarize`` merges them with live-mix weights.

    python scripts/v9_open_benchmark.py --deals 12288 --regs mix --fish sticky \\
        --arms v8 tr tr4 --fields fish4_reg1 --out-dir runs/v9-open/screen-mix
    python scripts/v9_open_benchmark.py --summarize --out-dir runs/v9-open/screen-mix
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
from poker.nocode_model import NoCodeFish, _chen_percentiles
from poker.rule_opponents import chen
from scripts.fish_field_benchmark import FIELDS, REG_BB, V7, calibrate, deal
from scripts.openpoker_client import FISH_EXPLOIT
from scripts.v6_field_benchmark import ARM_PRESETS, CHECKPOINT_CACHE, build_policy, hero_act_v5fast

LIVE_WEIGHTS = {'fish5': .04, 'fish4_reg1': .48, 'fish3_reg2': .40, 'fish2_reg3': .08}
BIG_SHARE = {'v7': 0., 'mix': .25, 'big': 1., 'three': 1.}
# Open width by position and, as the opener facing a 3-bet, (4-bet below, call below, else fold)
# as shares of the opening range: live big openers, and live 'reg'-class bots for ``three``.
OPEN_WIDTH = {0: .62, 1: .50, 2: .60, 3: .65, 4: .67, 5: .70}
VS_3BET = (.11, .74)
REG_WIDTH = {0: .27, 1: .15, 2: .16, 3: .19, 4: .27, 5: .35}
REG_VS_3BET = (.12, .55)


def _u(*parts):
    h = blake2b(repr(parts).encode(), digest_size=8).digest()
    return int.from_bytes(h, 'big') / (1 << 64)


def _rank(hole, *salt):
    """Preflop rank in [0, 1) (0 = best), Chen ties broken by a deterministic hash."""
    table = _chen_percentiles()
    lo = table[chen(tuple(hole))]
    hi = min((v for v in table.values() if v > lo), default=1.)
    return lo + (hi - lo) * _u('rank', hole, *salt)


class StickyFish(NoCodeFish):
    """NoCodeFish, but as the opener facing a 3-bet it answers as live fish do."""

    def _preflop(self, o, top, lo, hi, to, passive):
        raised = [h for h in o.history if h[2] == 0 and isinstance(h[3], int)]
        if len(raised) == 2 and raised[0][1] == o.position:
            u = self._u(o, 'sticky3b')
            if u < .02:
                return passive
            if lo is not None and u < .09:
                return to(top * 2.3)
            return 'call'
        return super()._preflop(o, top, lo, hi, to, passive)


class BigOpener:
    """Opens most hands to 3.5-4 BB; everything else is the wrapped v7-fast policy's."""

    def __init__(self, inner, style):
        self.inner, self.style = inner, style

    def __getattr__(self, name):
        return getattr(self.inner, name)

    @staticmethod
    def _raise_to(game, target):
        """A legal raise-to amount off the lean menu (these seats keep it for learned play)."""
        i = game.actors[0]
        top = max(game.bets)
        hi = game.stacks[i] + game.bets[i]
        lo = min(hi, max(game.raise_size, 2) + top)
        return min(hi, max(lo, target))

    def preflop(self, o, game):
        raises = [h for h in o.history if h[2] == 0 and isinstance(h[3], int)]
        legal = [a for a in o.legal if isinstance(a, int)]
        r = _rank(o.hole, o.position, self.style)
        width = (REG_WIDTH if self.style == 'three' else OPEN_WIDTH).get(o.position, .6)
        vs3bet = REG_VS_3BET if self.style == 'three' else VS_3BET
        passive = 'call' if not o.call else 'fold'
        if not raises:
            limpers = sum(1 for h in o.history if h[2] == 0 and h[3] == 'call')
            if r >= width or not legal:
                return passive
            if self.style == 'fixed':
                bb = 3.5 + limpers
            elif self.style == 'three':
                bb = 3. + limpers
            else:
                strong = _u('svan', o.hole, o.position) < (.75 if r < .2 else .15)
                bb = (4. if strong else 2.5) if not limpers else (3.5 if not strong else 5.) + (limpers - 1)
            return self._raise_to(game, int(round(2 * bb)))
        if len(raises) == 2 and raises[0][1] == o.position:
            q = r / width
            if q < vs3bet[0] and legal:
                return self._raise_to(game, int(round(max(game.bets) * 2.3)))
            return 'call' if q < vs3bet[1] else passive
        return None


def parse_arm(spec):
    """'v8' | 'off' | 'tr[N][f][r|c]' -> open_translation setting (None = off).

    N limits the translated open size (BB); ``f`` translates only opens by fish
    (leaderboard-weak seats); ``r`` / ``c`` keep only translated raises / calls."""
    if spec in ('v8', 'off'):
        return None
    out = dict(max_open=None, fish_only=False, keep=None)
    while spec != 'tr' and spec[-1] in 'frc':
        if spec[-1] == 'f':
            out['fish_only'] = True
        else:
            out['keep'] = ('raise',) if spec[-1] == 'r' else ('call',)
        spec = spec[:-1]
    if spec != 'tr':
        if not spec.startswith('tr'):
            raise ValueError(f'unknown arm {spec!r}')
        out['max_open'] = int(round(2 * float(spec[2:])))
    return out
    raise ValueError(f'unknown arm {spec!r}')


def build_hero(saved, spec):
    pol = build_policy(saved, {'fallback_kwargs': dict(V7, **FISH_EXPLOIT)})
    if spec != 'v8':
        pol.open_translation = parse_arm(spec)
    return FishPreflop(pol, steal=False)


def lineup(field, seed, regs_mode):
    hero, others, stacks = deal(field, seed)
    tags = others[:hero] + ['hero'] + others[hero:]
    st = stacks[:hero] + [REG_BB] + stacks[hero:]
    rng = Random(f'v9-open-regs:{field}:{seed}')
    for i, t in enumerate(tags):
        if t == 'reg' and rng.random() < BIG_SHARE[regs_mode]:
            tags[i] = 'three' if regs_mode == 'three' else 'fixed' if rng.random() < .5 else 'svan'
    return hero, tags, st


def play(hero_policy, pools, fish, seed, hero, tags, stacks, profiles):
    seats = {}
    used = Counter()
    for i, t in enumerate(tags):
        if t == 'hero':
            seats[i] = hero_policy
        elif t != 'fish':
            seats[i] = pools[t][used[t]]
            used[t] += 1
        if t != 'fish':
            seats[i].rng.seed(f'fish-bench:{seed}:{i}')
    game = FastGame(seed, button=5, stacks=[s * BIG_BLIND for s in stacks],
                    lean=sum(1 << i for i, t in enumerate(tags) if t != 'fish'))
    rec = dict(src=Counter(), spot=None)
    trail = []
    while game.live:
        o = game.observe()
        i = o.position
        t = tags[i]
        if t == 'fish':
            action = fish.act(o, game)
        else:
            pol = seats[i]
            pol.fallback.profiles = {} if profiles is None else {
                j: profiles['fish' if tags[j] == 'fish' else 'reg'] for j in range(6) if j != i}
            pol.fallback.fish_seats = {j for j in range(6) if tags[j] == 'fish'}
            action = pol.preflop(o, game) if t in ('fixed', 'svan', 'three') and not o.street else None
            if action is None:
                if t == 'hero':
                    before = (pol.overrides, pol.translated)
                    action, src = hero_act_v5fast(pol, game, o)
                    if pol.translated > before[1]:
                        src = 'open_translation'
                    elif pol.overrides > before[0]:
                        src = 'fish_preflop'
                    rec['src'][('pre:' if not o.street else 'post:') + src] += 1
                else:
                    action, _ = hero_act_v5fast(pol, game, o)
            if t == 'hero':
                trail.append((o.street, len(o.history), action))
                if not o.street and rec['spot'] is None:
                    raises = [h for h in o.history if isinstance(h[3], int)]
                    if len(raises) == 1 and not any(h[1] == i for h in o.history):
                        rec['spot'] = f'{tags[raises[0][1]]}:{raises[0][3] / 2:g}'
        game.history.append((o.player, o.position, o.street, action, o.pot, o.call))
        game.act(action, checked=False)
    rec['net'] = game.payoffs[hero] / BIG_BLIND
    rec['pot'] = sum(game.paid) / BIG_BLIND
    rec['digest'] = blake2b(repr(trail).encode(), digest_size=8).hexdigest()
    return rec


_SAVED = None
_PROFILES = None


def _batch(task):
    field, first, count, specs, fish_kind, regs_mode = task
    heroes = {s: build_hero(_SAVED, s) for s in specs}
    pools = {'reg': [build_policy(_SAVED, ARM_PRESETS['v7_fast']) for _ in range(3)],
             'fixed': [BigOpener(build_policy(_SAVED, ARM_PRESETS['v7_fast']), 'fixed') for _ in range(3)],
             'svan': [BigOpener(build_policy(_SAVED, ARM_PRESETS['v7_fast']), 'svan') for _ in range(3)],
             'three': [BigOpener(build_policy(_SAVED, ARM_PRESETS['v7_fast']), 'three') for _ in range(3)]}
    fish = (StickyFish if fish_kind == 'sticky' else NoCodeFish)(seed=7)
    out = {s: [] for s in specs}
    for seed in range(first, first + count):
        hero, tags, stacks = lineup(field, seed, regs_mode)
        for s in specs:
            out[s].append(play(heroes[s], pools, fish, seed, hero, tags, stacks, _PROFILES))
    return out


def spot_rows(base, arm):
    """Paired uncapped difference by the first single-open spot hero faced (opener:size)."""
    groups = {}
    for a, b in zip(base, arm):
        spot = a['spot'] or '-'
        if spot != '-':
            opener, size = spot.split(':')
            size = float(size)
            spot = f"{opener}:{'<=2.5' if size <= 2.5 else '3' if size <= 3 else '3.5-4' if size <= 4 else '>4'}"
        groups.setdefault(spot, ([], []))
        groups[spot][0].append(a['net'])
        groups[spot][1].append(b['net'])
    total = len(base)
    out = {}
    for spot, (x, y) in sorted(groups.items()):
        d = M.paired(x, y)
        # Contribution to the overall BB/100: the spot's mean difference times its share.
        out[spot] = dict(hands=len(x), share=round(len(x) / total, 4), delta_bb100=d['bb100'], ci95=d['ci95'],
                         contribution_bb100=round(d['bb100'] * len(x) / total, 2))
    return out


def arm_stats(recs):
    src = Counter()
    for r in recs:
        src.update(r['src'])
    out = M.summary([r['net'] for r in recs], [r['pot'] for r in recs])
    pre = {k[4:]: v for k, v in src.items() if k.startswith('pre:')}
    total = sum(pre.values())
    out['hero_preflop_sources'] = {k: round(v / total, 4) for k, v in sorted(pre.items())}
    post = {k[5:]: v for k, v in src.items() if k.startswith('post:')}
    total = sum(post.values())
    out['hero_postflop_sources'] = {k: round(v / total, 4) for k, v in sorted(post.items())}
    return out


def run_field(args, field):
    first = args.seed + list(LIVE_WEIGHTS).index(field) * 1_000_000
    size = max(32, args.deals // (args.workers * 3))
    tasks = [(field, first + i, min(size, args.deals - i), args.arms, args.fish, args.regs)
             for i in range(0, args.deals, size)]
    merged = {s: [] for s in args.arms}
    with ProcessPoolExecutor(args.workers, mp_context=get_context('fork')) as pool:
        for part in pool.map(_batch, tasks):
            for s in args.arms:
                merged[s].extend(part[s])
    base = args.arms[0]
    bnets, bpots = [r['net'] for r in merged[base]], [r['pot'] for r in merged[base]]
    out = {}
    for s in args.arms:
        st = arm_stats(merged[s])
        if s != base:
            nets, pots = [r['net'] for r in merged[s]], [r['pot'] for r in merged[s]]
            st['delta'] = M.compare(bnets, nets, bpots, pots)
            st['delta']['changed_hands'] = round(
                sum(a['digest'] != b['digest'] for a, b in zip(merged[base], merged[s])) / len(bnets), 4)
            big = [(a['net'], b['net']) for a, b in zip(merged[base], merged[s]) if max(a['pot'], b['pot']) >= 50]
            st['delta']['big_pots_50bb'] = dict(M.paired([x for x, _ in big], [y for _, y in big]),
                                                contribution_bb100=round(
                                                    100 * sum(y - x for x, y in big) / len(bnets), 2))
            st['delta']['by_spot'] = spot_rows(merged[base], merged[s])
        out[s] = st
        d = st.get('delta')
        line = f"  {field:11} {s:6} abs {st['uncapped']['bb100']:+7.1f}  pre {st['hero_preflop_sources']}"
        if d:
            line += (f"\n      d.unc {d['uncapped']['bb100']:+6.1f} {d['uncapped']['ci95']}"
                     f"  d.c50 {d['cap50']['bb100']:+6.1f} {d['cap50']['ci95']}"
                     f"  big>=50 {d['big_pots_50bb']['contribution_bb100']:+6.1f}"
                     f"  changed {d['changed_hands']}")
        print(line, flush=True)
    return out


def weighted(fields, specs):
    """Live-mix weighted paired differences (fields are independent samples)."""
    w = {f: LIVE_WEIGHTS[f] for f in fields}
    total = sum(w.values())
    out = {}
    for s in specs[1:]:
        row = {}
        for label in [M.cap_label(c) for c in M.CAPS] + ['big_pots_50bb']:
            def get(f):
                d = fields[f][s]['delta']
                if label == 'big_pots_50bb':
                    return d[label]['contribution_bb100'], None
                return d[label]['bb100'], d[label]['ci95']
            mean = sum(w[f] / total * get(f)[0] for f in fields)
            if label == 'big_pots_50bb':
                row[label] = dict(contribution_bb100=round(mean, 2))
                continue
            se = sqrt(sum((w[f] / total * (get(f)[1][1] - get(f)[1][0]) / (2 * M.Z95)) ** 2 for f in fields))
            row[label] = dict(bb100=round(mean, 2), ci95=[round(mean - M.Z95 * se, 2), round(mean + M.Z95 * se, 2)])
        out[s] = row
    return out


def summarize(out_dir):
    fields, meta = {}, None
    for f in LIVE_WEIGHTS:
        path = out_dir / f'{f}.json'
        if path.exists():
            rep = json.loads(path.read_text())
            fields[f] = rep['result']
            meta = rep['args']
    if not fields:
        raise SystemExit(f'no field results in {out_dir}')
    specs = meta['arms']
    rep = dict(args=meta, fields_done=list(fields), weighted=weighted(fields, specs))
    for s, row in rep['weighted'].items():
        print(f'  weighted {s:6} ' + '  '.join(
            f"{k} {v['bb100']:+6.1f} {v['ci95']}" if 'bb100' in v else f"{k} {v['contribution_bb100']:+6.1f}"
            for k, v in row.items()), flush=True)
    (out_dir / 'summary.json').write_text(json.dumps(rep, indent=1))
    return rep


def main():
    global _SAVED, _PROFILES
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--deals', type=int, default=4096)
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--seed', type=int, default=960_000_000)
    ap.add_argument('--fields', nargs='+', default=list(LIVE_WEIGHTS))
    ap.add_argument('--arms', nargs='+', default=['v8', 'tr'])
    ap.add_argument('--regs', choices=tuple(BIG_SHARE), default='mix')
    ap.add_argument('--fish', choices=('model', 'sticky'), default='sticky')
    ap.add_argument('--profiles-cache', type=Path,
                    default=Path('runs/bot/triplo-v9-sim/fish-profiles.json'))
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--summarize', action='store_true')
    args = ap.parse_args()
    if args.summarize:
        summarize(args.out_dir)
        return
    for s in args.arms:
        parse_arm(s)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _SAVED = json.loads(CHECKPOINT_CACHE.read_text())['policy']
    if args.profiles_cache.exists():
        _PROFILES = {k: tuple(v) for k, v in json.loads(args.profiles_cache.read_text()).items()}
    else:
        _PROFILES = calibrate(_SAVED)
        args.profiles_cache.write_text(json.dumps(_PROFILES))
    for field in args.fields:
        path = args.out_dir / f'{field}.json'
        if path.exists():
            print(f'  {field}: done already ({path})', flush=True)
            continue
        t0 = perf_counter()
        result = run_field(args, field)
        meta = vars(args) | dict(out_dir=str(args.out_dir), profiles_cache=str(args.profiles_cache))
        path.write_text(json.dumps(dict(args=meta, profiles=_PROFILES, result=result,
                                        elapsed_s=round(perf_counter() - t0, 1)), indent=1))
        print(f'  {field}: {perf_counter() - t0:.0f}s -> {path}', flush=True)


if __name__ == '__main__':
    main()
