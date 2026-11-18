"""Frozen live-condition evaluation and fail-closed checkpoint promotion decisions.

Separate evaluation from selection: run screens, select ONE candidate, then spend
an untouched confirmation seed once. A passing report is a recommendation; it
never changes the running bot or overwrites a champion checkpoint.
"""
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
from pathlib import Path
from random import Random
from statistics import mean, stdev, NormalDist
import time

from poker.history_cfr import HistoryPolicy, load, COMPACT_VERSION
from poker.equity_policy import EquityPolicy
from poker.live_curriculum import root_game
from poker.rule_opponents import RulePolicy

SCENARIOS = (
    ('six-50-tight',6,(50,)*6,'tight-v1'),
    ('six-50-aggressive',6,(50,)*6,'aggressive-v1'),
    ('six-100-tight',6,(100,)*6,'tight-v1'),
    ('six-100-aggressive',6,(100,)*6,'aggressive-v1'),
    ('six-100-station',6,(100,)*6,'station-v1'),
    ('three-unequal-mixed',3,(25,75,100,100,100,200),'mixed'),
    ('four-unequal-aggressive',4,(50,100,100,100,200,500),'aggressive-v1'),
    ('five-unequal-mixed',5,(20,50,100,75,200,1000),'mixed'),
    ('six-100-incumbent',6,(100,)*6,'incumbent'),
)
FAMILY_ALPHA=.05
MARGIN=10.


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


HERO_FALLBACKS=('equity-v1','equity-guarded-v1','equity-guarded-v1+stack-press')


def make_policy(saved, learned=True, fallback='equity-v1'):
    p=HistoryPolicy(saved, min_visits=64, learned=learned)
    if fallback=='equity-v1':p.fallback=EquityPolicy()
    elif fallback in HERO_FALLBACKS:
        from poker.equity_policy import GuardedEquityPolicy
        p.fallback=GuardedEquityPolicy(stack_press=fallback.endswith('stack-press'))
    else:raise ValueError('unknown fallback')
    return p


_EVAL = None


def _evaluate_case(job):
    hero, opponent_saved, seed, deals = _EVAL
    initial_overrides = hero.pressure_overrides
    case_index, (name, players, stacks, style) = job
    blocks=[];latencies=[];learned=decisions=post_learned=post_decisions=0
    positions=[0,1]+list(range(8-players,6))
    opponents=[make_policy(opponent_saved) if style=='incumbent' else
               RulePolicy(('tight-v1','aggressive-v1','station-v1')[i%3] if style=='mixed' else style)
               for i in range(6)]
    for deal in range(deals):
        outcomes=[]
        for position in positions:
            deal_seed=seed+case_index*100000+deal
            game=root_game(deal_seed,players,list(stacks),lean=True if style=='incumbent' else 1<<position)
            # Independent RNG per player, reset per duplicate deal/hero role.
            hero.rng=Random(f'hero:{deal_seed}:{position}')
            for j,opponent in enumerate(opponents):
                if hasattr(opponent,'rng'):opponent.rng=Random(f'opponent:{deal_seed}:{position}:{j}')
            while game.live:
                o=game.observe()
                if o.position==position:
                    before=hero.covered;start=time.perf_counter()
                    action=hero.act(o,history_complete=True)
                    latencies.append(time.perf_counter()-start)
                    used=hero.covered-before
                    decisions+=1;learned+=used
                    if o.street:post_decisions+=1;post_learned+=used
                else:action=opponents[o.position].act(o)
                game.act(action)
            outcomes.append(game.payoffs[position]/2)
        blocks.append(mean(outcomes)*100)
    latencies.sort()
    result=dict(blocks=blocks,bb_per_100=mean(blocks),hands=players*deals,
        decisions=decisions,learned=learned,postflop_decisions=post_decisions,postflop_learned=post_learned,
        p99_seconds=latencies[int(.99*(len(latencies)-1))] if latencies else 0,
        maximum_seconds=max(latencies,default=0))
    result['pressure_overrides'] = hero.pressure_overrides - initial_overrides
    return name, result


def evaluate(checkpoint, incumbent, out, seed, deals, fallback=False, hero_fallback='equity-v1'):
    if deals<2 or Path(out).exists():raise ValueError('need >=2 deals and a new output')
    incumbent=Path(incumbent); checkpoint=Path(checkpoint)
    saved=load(checkpoint) if not fallback else dict(sums={},visits={},stack_bb=100,abstraction=COMPACT_VERSION)
    hero=make_policy(saved,not fallback,hero_fallback)
    opponent_saved=saved if checkpoint.resolve()==incumbent.resolve() and not fallback else load(incumbent)
    report=dict(schema=1,checkpoint=str(checkpoint),checkpoint_sha256=None if fallback else digest(checkpoint),
                incumbent_sha256=digest(incumbent),fallback_only=fallback,seed=seed,deals=deals,min_visits=64,
                fallback='equity-v1',hero_fallback=hero_fallback,scenarios=SCENARIOS,cases={})
    report['source_sha256']={p:digest(p) for p in ('scripts/live_gate.py','poker/history_cfr.py',
        'poker/equity_policy.py','poker/live_curriculum.py','poker/fast_game.py','poker/rule_opponents.py')}
    global _EVAL
    _EVAL=(hero,opponent_saved,seed,deals)
    try:
        with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('fork')) as pool:
            for name, result in pool.map(_evaluate_case,enumerate(SCENARIOS)):
                report['cases'][name]=result
                print(name,round(result['bb_per_100'],2),flush=True)
    finally:
        _EVAL=None
    if any(digest(p)!=h for p,h in report['source_sha256'].items()):raise RuntimeError('evaluation source changed')
    with Path(out).open('x') as f:json.dump(report,f,indent=2)
    return report


def interval(values,z):
    if len(values)<2 or any(not math.isfinite(v) for v in values):raise ValueError('invalid samples')
    average=mean(values);se=stdev(values)/len(values)**.5
    return dict(mean=average,standard_error=se,lower=average-z*se)


def assess(candidate, incumbent, fallback, confirmation=True, family_alpha=FAMILY_ALPHA):
    for control in (incumbent,fallback):
        for field in ('seed','deals','min_visits','fallback','incumbent_sha256','scenarios','source_sha256'):
            if candidate[field]!=control[field]:raise ValueError('incompatible paired reports: '+field)
    if incumbent['checkpoint_sha256']!=candidate['incumbent_sha256'] or not fallback['fallback_only'] or candidate['fallback_only']:
        raise ValueError('incorrect champion/fallback controls')
    names=list(candidate['cases'])
    if set(names)!=set(incumbent['cases']) or set(names)!=set(fallback['cases']) or not names:
        raise ValueError('case mismatch')
    n=candidate['deals']
    for report in (candidate,incumbent,fallback):
        if any(len(report['cases'][name]['blocks'])!=n for name in names):raise ValueError('sample count mismatch')
    # Three aggregate claims plus two noninferiority claims per condition.
    if not 0 < family_alpha <= .05:raise ValueError('family alpha must be in (0,.05]')
    comparisons=3+2*len(names)
    z=NormalDist().inv_cdf(1-family_alpha/comparisons)
    aggregate=lambda report:[mean(report['cases'][name]['blocks'][i] for name in names) for i in range(n)]
    c=aggregate(candidate);i=aggregate(incumbent);f=aggregate(fallback)
    claims=dict(profit=interval(c,z),vs_incumbent=interval([a-b for a,b in zip(c,i)],z),
                vs_fallback=interval([a-b for a,b in zip(c,f)],z))
    cases={name:{label:interval([a-b for a,b in zip(candidate['cases'][name]['blocks'],report['cases'][name]['blocks'])],z)
                for label,report in (('incumbent',incumbent),('fallback',fallback))} for name in names}
    total=lambda field:sum(case[field] for case in candidate['cases'].values())
    coverage=total('learned')/max(1,total('decisions'))
    post_coverage=total('postflop_learned')/max(1,total('postflop_decisions'))
    checks=dict(aggregate_profit=claims['profit']['lower']>0,
                beats_incumbent=claims['vs_incumbent']['lower']>0,
                beats_fallback=claims['vs_fallback']['lower']>0,
                no_material_case_regression=all(c['lower']>=-MARGIN for case in cases.values() for c in case.values()),
                learned_coverage=coverage>=.05,postflop_coverage=post_coverage>=.01,
                latency=max(case['p99_seconds'] for case in candidate['cases'].values())<=.25)
    return dict(promote=confirmation and all(checks.values()),confirmation=confirmation,checks=checks,
                family_alpha=family_alpha,one_sided_z=z,noninferiority_margin_bb100=MARGIN,
                aggregate=claims,cases=cases,learned_fraction=coverage,postflop_learned_fraction=post_coverage,
                candidate_sha256=candidate['checkpoint_sha256'],incumbent_sha256=candidate['incumbent_sha256'],
                note='Bonferroni one-sided normal bounds over independent whole-deal blocks. Fixed condition suite; not a universal strength claim.')


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    e=sub.add_parser('evaluate');e.add_argument('--checkpoint',type=Path,required=True);e.add_argument('--incumbent',type=Path,required=True)
    e.add_argument('--out',type=Path,required=True);e.add_argument('--seed',type=int,required=True);e.add_argument('--deals',type=int,required=True);e.add_argument('--fallback-only',action='store_true')
    e.add_argument('--hero-fallback',choices=HERO_FALLBACKS,default='equity-v1',help='fallback for the evaluated hero only; opponents keep equity-v1')
    a=sub.add_parser('assess');a.add_argument('--candidate',type=Path,required=True);a.add_argument('--incumbent',type=Path,required=True);a.add_argument('--fallback',type=Path,required=True);a.add_argument('--out',type=Path,required=True);a.add_argument('--screen',action='store_true')
    args=p.parse_args()
    if args.command=='evaluate':evaluate(args.checkpoint,args.incumbent,args.out,args.seed,args.deals,args.fallback_only,args.hero_fallback)
    else:
        reports=[json.loads(p.read_text()) for p in (args.candidate,args.incumbent,args.fallback)]
        result=assess(*reports,confirmation=not args.screen)
        with args.out.open('x') as f:json.dump(result,f,indent=2)
        print(json.dumps(result['checks'],indent=2))

if __name__=='__main__':main()
