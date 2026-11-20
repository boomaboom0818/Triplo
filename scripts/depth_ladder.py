"""Paired effective-depth experiment: is "our stack is big" different from "stacks are deep"?

Replays the SAME deal seeds and SAME opponent lineups (common random numbers,
identical per-player RNG resets) under two stack ladders, using the deployed
hero (checkpoint + PressureGuard from the checkpoint + equity-guarded-v1
fallback, min-visits 64, Stack Press OFF unless asked):

  baseline  equal-50          hero 50 BB, every opponent 50 BB
  ladder A  hero-D-vs-50      hero D BB, every opponent fixed at 50 BB
                               (effective depth stays 50 BB; only OUR stack grows)
  ladder B  equal-D           hero and opponents all D BB (effective depth grows)

Because ladder A holds effective depth at the baseline, any ladder-A gap is
caused purely by the hero's own stack size. Card-free mechanisms that can do
that: the lean action menu (here and in the live adapter) only offers the
all-in when the hero's OWN stack is within call + 2 pots, so a deep hero loses
the effective shove and gets a shorter menu; the menu length is part of the
learned key, so coverage can shift; fallback size choices index into the menu.
Ladder B measures genuine deep play (effective depth grows with the stacks).

Every hand is a whole duplicate deal block: the hero takes each of the six seats
on the same cards. Paired per-deal differences against the baseline give low
variance 95% intervals. Decisions are broken down by decision source (named as
the live adapter logs them) and by remaining effective depth at the decision.

Run (about 10 minutes with the defaults on 6 workers):

  .venv/bin/python -m scripts.depth_ladder run --out runs/bot/depth-ladder-001/ladder.json
  .venv/bin/python -m scripts.depth_ladder summarize runs/bot/depth-ladder-001/ladder.json
"""
import argparse
from collections import Counter
import gc
import json
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from random import Random
from statistics import mean, stdev
import time
import zlib

from poker.history_cfr import load
from poker.live_curriculum import root_game
from poker.rule_opponents import RulePolicy
from scripts import live_gate
from scripts.live_gate import digest, make_policy, HERO_FALLBACKS

DEPLOYMENT = 'runs/bot/openpoker-season-17-equity/deployment.json'
CHECKPOINT = 'runs/bot/pressure-fresh-001/base-guard.json'
INCUMBENT = 'runs/bot/history-sustained-001/checkpoint.json'
HERO_FALLBACK = 'equity-guarded-v1'
DEPTHS = (50, 100, 200, 400, 1500)
SHORT = 50
LINEUPS = ('tight-v1', 'aggressive-v1', 'station-v1', 'mixed')
ALL_LINEUPS = LINEUPS + ('incumbent',)
SOURCES = ('learned_policy', 'preflop_rules', 'equity_fallback', 'pressure_guard', 'poker_safeguard')
# Geometric-midpoint edges so the ladder depths (and 25 BB) sit inside a bucket.
DEPTH_BUCKETS = ((15, '<15'), (35, '15-35'), (75, '35-75'), (150, '75-150'), (300, '150-300'),
                 (750, '300-750'), (math.inf, '>750'))
Z95 = 1.959963984540054


def conditions(depths=DEPTHS, short=SHORT):
    """[(name, ladder, hero_bb, opponent_bb)]; the shared baseline comes first."""
    if short not in depths or any(type(d) is not int or d < 2 for d in depths):
        raise ValueError('depths must be integers >=2 and include the short baseline depth')
    out = [(f'equal-{short}', 'baseline', short, short)]
    for d in sorted(set(depths)):
        if d != short:
            out.append((f'hero-{d}-vs-{short}', 'A', d, short))
            out.append((f'equal-{d}', 'B', d, d))
    return out


def table_stacks(position, hero_bb, opponent_bb):
    stacks = [opponent_bb] * 6
    stacks[position] = hero_bb
    return stacks


def depth_bucket(bb):
    return next(label for limit, label in DEPTH_BUCKETS if bb < limit)


def effective_bb(o):
    """Remaining-behind effective depth vs the deepest active opponent, in BB."""
    others = [s for i, s in enumerate(o.stacks) if o.active[i] and i != o.position]
    return min(o.stacks[o.position], max(others, default=0)) / 2


def classify(hero, before, street):
    """Mirror scripts/openpoker_client.decision source naming for a HistoryPolicy hero."""
    safeguards, overrides, covered, thin = before
    if hero.safeguards > safeguards:
        return 'poker_safeguard', None
    if hero.pressure_overrides > overrides:
        return 'pressure_guard', None
    if hero.covered > covered:
        return 'learned_policy', None
    return ('preflop_rules' if not street else 'equity_fallback'), ('thin' if hero.thin > thin else 'unseen')


def lineup_opponents(style, opponent_saved=None):
    if style == 'incumbent':
        return [make_policy(opponent_saved) for _ in range(6)]
    return [RulePolicy(('tight-v1', 'aggressive-v1', 'station-v1')[i % 3] if style == 'mixed' else style)
            for i in range(6)]


_STATE = None


def _decide(hero, game, o):
    # Follow live_gate's adapter path when present (Stack Press); with Stack
    # Press off it is exactly hero.act(o, history_complete=True).
    if hasattr(live_gate, 'hero_decision'):
        return live_gate.hero_decision(hero, game, o)
    return hero.act(o, history_complete=True)


def _play(game, o, action):
    if hasattr(live_gate, 'play'):
        return live_gate.play(game, o, action)
    return game.act(action)


def run_shard(job):
    """Play deals [start, stop) of one (condition, lineup) cell. Pure function of the job."""
    hero, opponent_saved, seed = _STATE
    (name, ladder, hero_bb, opponent_bb), lineup_index, style, start, stop = job
    opponents = lineup_opponents(style, opponent_saved)
    blocks, signatures, latencies = [], [], []
    sources, reasons, by_depth, by_street = Counter(), Counter(), Counter(), Counter()
    flops = hands = 0
    for deal in range(start, stop):
        deal_seed = seed + lineup_index * 100000 + deal
        outcomes, signature = [], []
        for position in range(6):
            game = root_game(deal_seed, 6, table_stacks(position, hero_bb, opponent_bb),
                             lean=True if style == 'incumbent' else 1 << position)
            hero.rng = Random(f'hero:{deal_seed}:{position}')
            for j, opponent in enumerate(opponents):
                if hasattr(opponent, 'rng'):
                    opponent.rng = Random(f'opponent:{deal_seed}:{position}:{j}')
            actions, saw_flop = [], False
            while game.live:
                o = game.observe()
                if o.position == position:
                    before = (hero.safeguards, hero.pressure_overrides, hero.covered, hero.thin)
                    t = time.perf_counter()
                    action = _decide(hero, game, o)
                    latencies.append(time.perf_counter() - t)
                    source, reason = classify(hero, before, o.street)
                    bucket = depth_bucket(effective_bb(o))
                    sources[source] += 1
                    by_depth[f'{bucket}|{source}'] += 1
                    by_street[f'{"pre" if not o.street else "post"}|{source}'] += 1
                    if reason:
                        reasons[f'{"pre" if not o.street else "post"}|{reason}'] += 1
                    actions.append((o.street, action))
                    saw_flop |= o.street > 0
                else:
                    action = opponents[o.position].act(o)
                _play(game, o, action)
            hands += 1; flops += saw_flop
            outcomes.append(game.payoffs[position] / 2)
            # Hero's own action sequence (amounts in BB) for divergence vs the baseline.
            signature.append(zlib.crc32(repr(actions).encode()))
        blocks.append(mean(outcomes) * 100)
        signatures.append(signature)
    latencies.sort()
    return dict(name=name, lineup=style, start=start, blocks=blocks, signatures=signatures,
                sources=dict(sources), fallback_reasons=dict(reasons), by_depth=dict(by_depth),
                by_street=dict(by_street), hands=hands, hero_saw_flop=flops,
                p99_seconds=latencies[int(.99 * (len(latencies) - 1))] if latencies else 0.,
                maximum_seconds=latencies[-1] if latencies else 0.)


def shards(conds, lineups, deals, shard):
    jobs = []
    for li, style in enumerate(lineups):
        for cond in conds:
            for start in range(0, deals, shard):
                jobs.append((cond, li, style, start, min(deals, start + shard)))
    # Slow (deep, station) cells first for better packing.
    return sorted(jobs, key=lambda j: (-j[0][2] * (2 if j[1] == 'station-v1' else 1), j[1], j[3]))


def merge(conds, lineups, deals, results):
    cells = {}
    for cond in conds:
        for style in lineups:
            cells[(cond[0], style)] = dict(blocks=[None] * deals, signatures=[None] * deals, sources=Counter(),
                                           fallback_reasons=Counter(), by_depth=Counter(), by_street=Counter(),
                                           hands=0, hero_saw_flop=0, p99_seconds=0., maximum_seconds=0.)
    for r in results:
        cell = cells[(r['name'], r['lineup'])]
        for k, (b, s) in enumerate(zip(r['blocks'], r['signatures'])):
            cell['blocks'][r['start'] + k] = b
            cell['signatures'][r['start'] + k] = s
        for key in ('sources', 'fallback_reasons', 'by_depth', 'by_street'):
            cell[key].update(r[key])
        cell['hands'] += r['hands']; cell['hero_saw_flop'] += r['hero_saw_flop']
        cell['p99_seconds'] = max(cell['p99_seconds'], r['p99_seconds'])
        cell['maximum_seconds'] = max(cell['maximum_seconds'], r['maximum_seconds'])
    for cell in cells.values():
        if any(b is None for b in cell['blocks']):
            raise RuntimeError('missing shard')
        for key in ('sources', 'fallback_reasons', 'by_depth', 'by_street'):
            cell[key] = dict(cell[key])
    return cells


def interval(values, z=Z95):
    if len(values) < 2:
        raise ValueError('need >=2 paired blocks')
    m = mean(values); se = stdev(values) / len(values) ** .5
    return dict(mean=m, standard_error=se, lower=m - z * se, upper=m + z * se, n=len(values))


def shares(counter):
    total = sum(counter.values())
    return {k: v / total for k, v in sorted(counter.items(), key=lambda kv: -kv[1])} if total else {}


def depth_table(by_depth):
    """{bucket: {decisions, shares}} from 'bucket|source' counts."""
    grouped = {}
    for key, n in by_depth.items():
        bucket, source = key.split('|')
        grouped.setdefault(bucket, Counter())[source] += n
    order = [label for _, label in DEPTH_BUCKETS]
    return {b: dict(decisions=sum(grouped[b].values()), shares=shares(grouped[b]))
            for b in order if b in grouped}


def analyze(report):
    """Paired statistics vs the baseline, per lineup and pooled across lineups."""
    conds = [tuple(c) for c in report['conditions']]
    lineups = report['lineups']; cells = report['cells']
    base = conds[0][0]
    n = report['deals']
    out = dict(baseline=base, conditions={}, ladder_gap={})
    pooled = lambda name: [mean(cells[f'{name}|{s}']['blocks'][i] for s in lineups) for i in range(n)]
    base_pooled = pooled(base)
    for name, ladder, hero_bb, opp_bb in conds:
        mine = pooled(name)
        sources, by_depth, by_street = Counter(), Counter(), Counter()
        same = total = 0
        per_lineup = {}
        for s in lineups:
            cell, ref = cells[f'{name}|{s}'], cells[f'{base}|{s}']
            sources.update(cell['sources']); by_depth.update(cell['by_depth']); by_street.update(cell['by_street'])
            for a, b in zip(cell['signatures'], ref['signatures']):
                same += sum(x == y for x, y in zip(a, b)); total += len(a)
            per_lineup[s] = dict(bb_per_100=interval(cell['blocks']),
                                 vs_baseline=None if name == base else
                                 interval([a - b for a, b in zip(cell['blocks'], ref['blocks'])]),
                                 source_shares=shares(Counter(cell['sources'])))
        street = {}
        for key, v in by_street.items():
            g, src = key.split('|'); street.setdefault(g, Counter())[src] += v
        decisions = sum(sources.values())
        out['conditions'][name] = dict(
            ladder=ladder, hero_bb=hero_bb, opponent_bb=opp_bb, effective_bb=min(hero_bb, opp_bb),
            bb_per_100=interval(mine),
            vs_baseline=None if name == base else interval([a - b for a, b in zip(mine, base_pooled)]),
            hero_action_identical_to_baseline=same / total if total else None,
            decisions=decisions, decisions_per_hand=decisions / max(1, sum(cells[f'{name}|{s}']['hands'] for s in lineups)),
            source_shares=shares(sources),
            street_source_shares={g: shares(c) for g, c in sorted(street.items(), reverse=True)},
            by_effective_depth=depth_table(by_depth), per_lineup=per_lineup,
            p99_seconds=max(cells[f'{name}|{s}']['p99_seconds'] for s in lineups))
    # Same depth D: "only we are deep" (A) minus "everyone is deep" (B).
    for name, ladder, hero_bb, _ in conds:
        if ladder == 'A':
            other = f'equal-{hero_bb}'
            out['ladder_gap'][f'{name} minus {other}'] = interval([a - b for a, b in zip(pooled(name), pooled(other))])
    return out


def fmt_ci(i):
    return f'{i["mean"]:+8.1f} [{i["lower"]:+7.1f}, {i["upper"]:+7.1f}]'


def render(report, analysis):
    lines = [f'Depth ladder: {report["deals"]} duplicate deals x 6 seats x {len(report["lineups"])} lineups '
             f'({", ".join(report["lineups"])}); seed {report["seed"]}; hero {report["checkpoint"]} '
             f'+ {report["hero_fallback"]}', '',
             f'{"condition":<18}{"lad":>4}{"hero":>6}{"opp":>6}  {"BB/100 [95% CI]":<27}  {"vs equal-50 (paired)":<27}'
             f'{"same":>6}{"learn":>7}{"pre-rl":>7}{"eq-fb":>7}{"guard":>7}{"safe":>6}']
    for name, c in analysis['conditions'].items():
        s = c['source_shares']
        lines.append(f'{name:<18}{c["ladder"][:4]:>4}{c["hero_bb"]:>6}{c["opponent_bb"]:>6}  {fmt_ci(c["bb_per_100"]):<27}  '
                     f'{"-" if c["vs_baseline"] is None else fmt_ci(c["vs_baseline"]):<27}'
                     f'{c["hero_action_identical_to_baseline"]:>6.0%}'
                     + ''.join(f'{s.get(k, 0):>7.0%}' for k in SOURCES[:4]) + f'{s.get("poker_safeguard", 0):>6.0%}')
    lines += ['', '"same" = share of (deal, seat) hands where the hero action sequence equals the baseline.',
              'Source shares are over all hero decisions (learn = learned_policy, pre-rl = preflop_rules, '
              'eq-fb = equity_fallback, guard = pressure_guard, safe = poker_safeguard).', '']
    if analysis['ladder_gap']:
        lines.append('Same depth, A minus B ("only we are deep" minus "everyone is deep"):')
        for k, v in analysis['ladder_gap'].items():
            lines.append(f'  {k:<34}{fmt_ci(v)}')
        lines.append('')
    lines.append('Per-lineup paired difference vs baseline (BB/100):')
    lineups = report['lineups']
    lines.append(f'{"condition":<18}' + ''.join(f'{s:>26}' for s in lineups))
    for name, c in analysis['conditions'].items():
        if c['vs_baseline'] is None:
            lines.append(f'{name:<18}' + ''.join(f'{c["per_lineup"][s]["bb_per_100"]["mean"]:>+26.1f}' for s in lineups)
                         + '   (absolute)')
        else:
            lines.append(f'{name:<18}' + ''.join(f'{fmt_ci(c["per_lineup"][s]["vs_baseline"]):>26}' for s in lineups))
    lines += ['', 'Decision sources by remaining effective depth at the decision (share learned / fallback):']
    for name, c in analysis['conditions'].items():
        parts = []
        for b, d in c['by_effective_depth'].items():
            sh = d['shares']
            fb = sh.get('preflop_rules', 0) + sh.get('equity_fallback', 0)
            parts.append(f'{b}: n={d["decisions"]} L{sh.get("learned_policy", 0):.0%}/F{fb:.0%}')
        lines.append(f'  {name:<18}' + '; '.join(parts))
    lines += ['', 'Preflop vs postflop learned share:']
    for name, c in analysis['conditions'].items():
        ss = c['street_source_shares']
        lines.append(f'  {name:<18}pre {ss.get("pre", {}).get("learned_policy", 0):.0%}  '
                     f'post {ss.get("post", {}).get("learned_policy", 0):.0%}  '
                     f'decisions/hand {c["decisions_per_hand"]:.2f}')
    return '\n'.join(lines)


def run(out, checkpoint=CHECKPOINT, seed=72411000, deals=1500, depths=DEPTHS, short=SHORT, lineups=LINEUPS,
        workers=6, shard=125, hero_fallback=HERO_FALLBACK, incumbent=INCUMBENT, saved=None):
    out = Path(out)
    if deals < 2 or out.exists():
        raise ValueError('need >=2 deals and a new output path')
    if hero_fallback not in HERO_FALLBACKS or any(s not in ALL_LINEUPS for s in lineups) or not lineups:
        raise ValueError('unknown fallback or lineup')
    conds = conditions(depths, short)
    started = time.time()
    if saved is None:
        saved = load(checkpoint)
    hero = make_policy(saved, True, hero_fallback)
    opponent_saved = load(incumbent) if 'incumbent' in lineups else None
    report = dict(schema='depth-ladder-v1', checkpoint=str(checkpoint),
                  checkpoint_sha256=digest(checkpoint) if Path(checkpoint).exists() else None,
                  hero_fallback=hero_fallback, min_visits=hero.min_visits,
                  pressure_guard=getattr(hero.pressure_guard, 'version', None),
                  stack_press=hero_fallback.endswith('stack-press'), seed=seed, deals=deals,
                  lineups=list(lineups), conditions=[list(c) for c in conds], players=6,
                  pairing='same deal seed, lineup and per-player RNG resets across conditions; hero rotates all 6 seats',
                  source_sha256={p: digest(p) for p in ('scripts/depth_ladder.py', 'scripts/live_gate.py',
                                 'poker/history_cfr.py', 'poker/equity_policy.py', 'poker/fast_game.py',
                                 'poker/rule_opponents.py') if Path(p).exists()})
    global _STATE
    _STATE = (hero, opponent_saved, seed)
    jobs = shards(conds, lineups, deals, shard)
    results = []
    try:
        gc.collect(); gc.freeze()  # keep forked workers from copying the checkpoint pages
        if workers <= 1:
            results = [run_shard(j) for j in jobs]
        else:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('fork')) as pool:
                for k, r in enumerate(pool.map(run_shard, jobs), 1):
                    results.append(r)
                    if k % max(1, len(jobs) // 20) == 0:
                        print(f'{k}/{len(jobs)} shards  {time.time() - started:.0f}s', flush=True)
    finally:
        _STATE = None
        gc.unfreeze()
    cells = merge(conds, lineups, deals, results)
    report['cells'] = {f'{n}|{s}': c for (n, s), c in cells.items()}
    report['seconds'] = time.time() - started
    report['analysis'] = analyze(report)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as f:
        json.dump(report, f, indent=1)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run')
    r.add_argument('--out', type=Path, required=True)
    r.add_argument('--checkpoint', type=Path, default=Path(CHECKPOINT))
    r.add_argument('--hero-fallback', choices=HERO_FALLBACKS, default=HERO_FALLBACK)
    r.add_argument('--seed', type=int, default=72411000)
    r.add_argument('--deals', type=int, default=1500)
    r.add_argument('--depths', type=int, nargs='+', default=list(DEPTHS))
    r.add_argument('--short', type=int, default=SHORT, help='fixed opponent depth for ladder A and the baseline')
    r.add_argument('--lineups', nargs='+', choices=ALL_LINEUPS, default=list(LINEUPS))
    r.add_argument('--workers', type=int, default=6)
    r.add_argument('--shard', type=int, default=125, help='deals per parallel job')
    s = sub.add_parser('summarize')
    s.add_argument('report', type=Path)
    args = p.parse_args()
    if args.command == 'run':
        report = run(args.out, args.checkpoint, args.seed, args.deals, tuple(args.depths), args.short,
                     tuple(args.lineups), args.workers, args.shard, args.hero_fallback)
    else:
        report = json.loads(args.report.read_text())
    print(render(report, report['analysis']))
    if args.command == 'run':
        text = args.out.with_suffix('.txt')
        if not text.exists():
            text.write_text(render(report, report['analysis']) + '\n')


if __name__ == '__main__':
    main()
