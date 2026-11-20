"""Duplicate head-to-head: candidate bot vs. the currently deployed OpenPoker bot.

The deployed bot (release dd0ac44) is: stage-10 checkpoint + PressureGuard v1 +
``equity-guarded-v1``, Stack Press off, min-visits 64, and the old lean menu whose
all-in is offered from the actor's *own* stack. The candidate shares the same
checkpoint and guard and differs only in what changed since:

* the effective all-in (deepest live opponent's total) on the lean menu, and
* ``equity-guarded-v2`` (pot-by-pot calls; value sizing from measured calling
  stats of the opponents, learned in a warm-up phase on deployed-bot tables).

For every deal and every seat p, two games are played with identical cards and
per-seat RNG streams: (A) the candidate at p with five deployed bots, and (B) six
deployed bots. The score is payoff_p(A) - payoff_p(B), so variance from the cards
and from the field cancels and only the candidate's decisions remain.

    python -m scripts.h2h_live --deals 2000 --workers 6 --out runs/bot/h2h-001/report.json
"""
import argparse
import gc
import json
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from random import Random
from statistics import mean, stdev
import time

from poker.equity_policy import GuardedEquityPolicy
from poker.equity_pricing import GuardedEquityPolicyV2
from poker.fast_game import FastGame
from poker.history_cfr import HistoryPolicy, load
from poker.live_curriculum import root_game
from poker.opponent_stats import OpponentStats

CHECKPOINT = 'runs/bot/pressure-fresh-001/base-guard.json'
FIELD_NAME = 'deployed-bot'
# name, players, stacks in BB (per position; HERO marks the candidate seat's depth)
CONDITIONS = (
    ('six-equal-50', 6, lambda p, rng: [50] * 6),
    ('six-equal-100', 6, lambda p, rng: [100] * 6),
    ('six-hero-800-vs-50', 6, lambda p, rng: [800 if i == p else 50 for i in range(6)]),
    ('six-mixed-live', 6, lambda p, rng: shuffled(rng, [30, 50, 75, 150, 300, 1000])),
    ('four-mixed-live', 4, lambda p, rng: shuffled(rng, [40, 50, 100, 250, 600, 1500])),
)


def shuffled(rng, values):
    values = list(values)
    rng.shuffle(values)
    return values


class LegacyMenuGame(FastGame):
    """FastGame whose lean all-in follows dd0ac44 for seats in ``LEGACY`` (bit mask)."""
    __slots__ = ()
    LEGACY = 0

    def cover(self, i):
        if LegacyMenuGame.LEGACY >> i & 1:
            return self.stacks[i] + self.bets[i]        # own stack: the old rule
        return FastGame.cover(self, i)


def make(saved, fallback):
    policy = HistoryPolicy(saved, min_visits=64)
    policy.fallback = fallback
    return policy


_W = None


def _init(saved, stats_json):
    global _W
    stats = OpponentStats(min_faced=20)
    if stats_json is not None:
        stats.players = {k: [list(r) for r in v] for k, v in stats_json.items()}
    candidate_fallback = GuardedEquityPolicyV2()
    candidate_fallback.opponent_stats = stats
    _W = dict(old=[make(saved, GuardedEquityPolicy()) for _ in range(6)],
              new=make(saved, candidate_fallback), stats=stats)


def play(game, seat_policy, seed, hero, positions_record=None):
    """Play out ``game``; ``seat_policy[i]`` acts for position i. Returns payoffs (BB)."""
    for i, policy in enumerate(seat_policy):
        policy.rng = Random(f'{"hero" if i == hero else "seat"}:{seed}:{hero}:{i}')
    while game.live:
        o = game.observe()
        policy = seat_policy[o.position]
        fallback = policy.fallback
        stats = getattr(fallback, 'opponent_stats', None)
        if stats is not None:
            # Same rule as the live client: measured profiles for live opponents.
            fallback.profiles = stats.profiles({j: FIELD_NAME for j, a in enumerate(o.active)
                                               if a and j != o.position})
        start = time.perf_counter()
        action = policy.act(o, history_complete=True)
        if positions_record is not None and o.position == hero:
            positions_record.append((o.street, time.perf_counter() - start))
        game.act(action)
    return [x / 2 for x in game.payoffs]


def build(seed, players, stacks, legacy_mask):
    game = root_game(seed, players, stacks, lean=True)
    game.__class__ = LegacyMenuGame
    LegacyMenuGame.LEGACY = legacy_mask
    return game


def _warmup(job):
    """Deployed-bot-only tables; return calling observations of every seat."""
    seed, count = job
    old, out = _W['old'], []
    for deal in range(count):
        rng = Random(seed + deal)
        name, players, stacks_fn = CONDITIONS[deal % len(CONDITIONS)]
        game = build(seed + deal, players, stacks_fn(0, rng), 63)
        play(game, old, seed + deal, -1)
        for _, pos, street, action, pot, call in game.history:
            if street >= 1 and call > 0:
                out.append((street, 'fold' if action == 'fold' else 'call', call, pot))
    return out


def _match(job):
    cond_index, seed, first, count = job
    name, players, stacks_fn = CONDITIONS[cond_index]
    old, new = _W['old'], _W['new']
    positions = [0, 1] + list(range(8 - players, 6))
    rows, timings = [], []
    for deal in range(first, first + count):
        deal_seed = seed + 1_000_000 * cond_index + deal
        diffs, actions_same = [], 0
        for p in positions:
            stacks = stacks_fn(p, Random(deal_seed))
            base = build(deal_seed, players, list(stacks), 63)
            b = play(base, old, deal_seed, p)
            seats = list(old)
            seats[p] = new
            cand = build(deal_seed, players, list(stacks), 63 & ~(1 << p))
            a = play(cand, seats, deal_seed, p, timings)
            diffs.append(a[p] - b[p])
            actions_same += cand.history == base.history
        rows.append(dict(deal=deal, diff=mean(diffs), same=actions_same / len(positions),
                         seats=len(positions)))
    return name, rows, timings


def summarize(results, timings):
    out = {}
    for name, rows in results.items():
        values = [r['diff'] * 100 for r in rows]
        m = mean(values)
        se = stdev(values) / math.sqrt(len(values)) if len(values) > 1 else float('nan')
        out[name] = dict(deals=len(rows), bb100_vs_deployed=m, ci95=[m - 1.96 * se, m + 1.96 * se],
                         same_play=mean(r['same'] for r in rows))
    timings = sorted(t for _, t in timings)
    if timings:
        out['_latency'] = dict(decisions=len(timings), p50_ms=1000 * timings[len(timings) // 2],
                               p99_ms=1000 * timings[int(.99 * (len(timings) - 1))], max_ms=1000 * timings[-1])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', default=CHECKPOINT)
    ap.add_argument('--deals', type=int, default=2000, help='duplicate deals per condition')
    ap.add_argument('--warmup', type=int, default=600, help='deployed-bot-only deals to learn calling stats')
    ap.add_argument('--seed', type=int, default=91_000_000)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--shard', type=int, default=100)
    ap.add_argument('--conditions', nargs='+', default=[c[0] for c in CONDITIONS])
    ap.add_argument('--no-stats', action='store_true', help='candidate without opponent stats (v1-size sizing)')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit('refusing to overwrite %s' % args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    saved = load(args.checkpoint)
    gc.collect(); gc.freeze()
    ctx = multiprocessing.get_context('fork')

    stats_json = None
    if not args.no_stats:
        _init(saved, None)
        with ProcessPoolExecutor(args.workers, mp_context=ctx, initializer=_init, initargs=(saved, None)) as pool:
            jobs = [(args.seed + 50_000_000 + k, min(args.shard, args.warmup - k))
                    for k in range(0, args.warmup, args.shard)]
            stats = OpponentStats(min_faced=20)
            for obs in pool.map(_warmup, jobs):
                for street, action, call, pot in obs:
                    stats.observe(FIELD_NAME, street, action, call, pot)
        base, slope, faced = stats.fit(FIELD_NAME)
        print('warm-up: deployed bot faced %d postflop bets; fold ~ %.2f + %.2f * price (fold@.25 = %.0f%%)'
              % (faced, base, slope, 100 * min(1, base + slope * .25)), flush=True)
        stats_json = stats.players

    jobs = [(i, args.seed, k, min(args.shard, args.deals - k))
            for i, c in enumerate(CONDITIONS) if c[0] in args.conditions
            for k in range(0, args.deals, args.shard)]
    results, timings = {}, []
    with ProcessPoolExecutor(args.workers, mp_context=ctx, initializer=_init,
                             initargs=(saved, stats_json)) as pool:
        for done, (name, rows, t) in enumerate(pool.map(_match, jobs), 1):
            results.setdefault(name, []).extend(rows)
            timings.extend(t)
            if done % max(1, len(jobs) // 10) == 0:
                print('%d/%d shards, %.0fs' % (done, len(jobs), time.time() - t0), flush=True)
    summary = summarize(results, timings)
    report = dict(schema=1, checkpoint=args.checkpoint, deals=args.deals, warmup=args.warmup, seed=args.seed,
                  candidate='effective all-in + equity-guarded-v2' + ('' if args.no_stats else ' + opponent stats'),
                  deployed='dd0ac44: own-stack all-in + equity-guarded-v1, Stack Press off',
                  field_stats=None if stats_json is None else dict(zip(('base', 'slope', 'faced'),
                                                                        stats.fit(FIELD_NAME))),
                  summary=summary, rows=results, seconds=time.time() - t0)
    args.out.write_text(json.dumps(report))
    print('\n%-22s %6s %28s %8s' % ('condition', 'deals', 'BB/100 vs deployed [95% CI]', 'same'))
    for name in args.conditions:
        s = summary.get(name)
        if s:
            print('%-22s %6d %+9.2f [%+7.2f, %+7.2f] %7.0f%%' % (name, s['deals'], s['bb100_vs_deployed'],
                                                                s['ci95'][0], s['ci95'][1], 100 * s['same_play']))
    if '_latency' in summary:
        print('candidate latency: p50 %.1f ms, p99 %.1f ms, max %.0f ms' % tuple(
            summary['_latency'][k] for k in ('p50_ms', 'p99_ms', 'max_ms')))
    print('wrote %s (%.0fs)' % (args.out, time.time() - t0))


if __name__ == '__main__':
    main()
