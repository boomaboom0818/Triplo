"""Paired seat-balanced screen of the equity fallback versus tight-v1.

Each independent deal is one bootstrap/standard-error block, not six independent
hands. Fixed rule opponents are weak sanity checks, not a competitive benchmark.
"""
import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor
import json
import statistics
import time
from pathlib import Path
from poker.fast_game import FastGame
from poker.equity_policy import EquityPolicy
from poker.rule_opponents import RulePolicy, STYLES
from scripts.rule_gate import summarize


def case(args):
    style, stack_bb, deals, seed, samples = args
    opponent = RulePolicy(style)
    policies = {'tight': RulePolicy('tight-v1'), 'equity': EquityPolicy(samples)}
    results = {}; timings = []
    for name, hero in policies.items():
        values = []
        for deal in range(deals):
            block = []
            for seat in range(6):
                game = FastGame(seed+deal, stacks=[2*stack_bb]*6, lean=1 << ((seat-1)%6))
                while game.live:
                    o = game.observe()
                    start = time.perf_counter()
                    action = (hero if o.player==seat else opponent).act(o)
                    if o.player==seat and name=='equity' and o.street:
                        timings.append(time.perf_counter()-start)
                    game.act(action)
                block.append(game.payoff()[seat])
            values.append(statistics.mean(block)*100)
        results[name] = dict(summarize(values,0), samples=values)
    difference = [a-b for a,b in zip(results['equity']['samples'],results['tight']['samples'])]
    timings.sort()
    return f'{style}/{stack_bb}bb', dict(policies=results, paired_difference=summarize(difference,0),
        latency_seconds=dict(p95=timings[int(.95*(len(timings)-1))] if timings else 0, maximum=max(timings,default=0)))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--deals',type=int,default=100)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--samples',type=int,default=256)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.deals<2 or args.out.exists(): p.error('need at least two deals and a new output path')
    jobs=[(style,stack,args.deals,args.seed,args.samples) for style in STYLES for stack in (50,100)]
    source_hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                   ('poker/equity_policy.py','poker/rule_opponents.py','poker/fast_game.py','scripts/equity_fallback_check.py')}
    report=dict(source_sha256=source_hashes, seed=args.seed,deals=args.deals,samples=args.samples,
                note='Screen only. Paired pointwise 95% normal intervals over deal blocks; shared deals across styles.',cases={})
    with ProcessPoolExecutor(max_workers=4) as pool:
        for key,result in pool.map(case,jobs):
            report['cases'][key]=result
            print(key,json.dumps(result['paired_difference']),flush=True)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest for p,digest in source_hashes.items()), 'source changed during evaluation'
    with args.out.open('x') as f: json.dump(report,f,indent=2)

if __name__=='__main__': main()
