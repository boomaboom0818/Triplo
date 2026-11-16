"""Bounded matched history-vs-historyless poker experiment; no live connection.

Same shared kernel, card abstraction, seeds and menus in both arms. 20bb stacks
keep this first test small. Evaluate raw and guarded policies against rules and
a guarded fallback-only control on fresh paired deal blocks.
"""
import argparse
import hashlib
import json
import resource
import shutil
import time
from itertools import permutations
from pathlib import Path
from random import Random

from poker.fast_game import FastGame
from poker.history_cfr import PokerState, HistoryPolicy, save
from poker.lean_abstraction import observation_key as old_key
from poker.kuhn import Kuhn, audit
from poker.rule_opponents import STYLES, RulePolicy
from poker.sampled_cfr import SampledCFR
from scripts.rule_gate import summarize
from scripts.scaling_validation import available_memory


class HistorylessState(PokerState):
    def key(self): return old_key(self.game.observe())
    def child(self, action):
        game = self.game.clone()
        game.act(action)
        return HistorylessState(game)


class RawPolicy:
    def __init__(self, saved, key, seed=5):
        self.saved, self.key, self.rng = saved, key, Random(seed)
        self.covered = self.decisions = 0
    def act(self, o):
        self.decisions += 1
        row = self.saved['sums'].get(self.key(o))
        if row and sum(row) > 0:
            self.covered += 1
            return self.rng.choices(o.legal,row)[0]
        return self.rng.choice(o.legal)


def evaluate(policy, style, deals, stack_bb=20, seed=19310000):
    opponent = RulePolicy(style)
    blocks = []
    for deal in range(deals):
        pay = []
        for seat in range(6):
            game = FastGame(seed+deal,stacks=[2*stack_bb]*6,lean=1 << ((seat-1)%6))
            while game.live:
                o = game.observe()
                game.act((policy if o.player == seat else opponent).act(o))
            pay.append(game.payoff()[seat])
        blocks.append(sum(pay)/6*100)
    result = summarize(blocks,0)
    # This evaluator does not instrument shove frequency.
    result.pop('preflop_allins_per_100_hands')
    return dict(result,deal_block_bb_per_100=blocks,
                learned_fraction=policy.covered/max(1,policy.decisions))


def run(args):
    if not 1 <= args.iterations <= 10000 or not 2 <= args.deals <= 2000:
        raise ValueError('bounded iterations/deals required')
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
    args.root.mkdir(parents=True,exist_ok=False)
    def write(name,value):
        with (args.root/name).open('x') as f: json.dump(value,f,indent=2)
    sources = [Path(p) for p in ('poker/sampled_cfr.py','poker/kuhn.py','poker/history_cfr.py',
         'poker/lean_abstraction.py','poker/fast_game.py','poker/fast_eval.py','poker/rule_opponents.py',
         'scripts/history_pilot.py')]
    manifest = dict(seeds=[91,92],iterations=args.iterations,deals=args.deals,stack_bb=20,
        evaluation_seed=19310000,min_visits=64,per_arm_seconds=180,
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        note='From-scratch bounded game. Pointwise 95% normal intervals over paired deals; no multiple-test correction.')
    write('manifest.json',manifest)
    for path in sources:
        target=args.root/'source'/path
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(path.read_bytes())
    # Same kernel used in poker, audited against EVERY pure Kuhn response.
    kuhn=[]
    for seed in (81,82,83):
        solver=SampledCFR(2,seed); deals=list(permutations(range(3),2)); rng=Random(seed+10)
        before=audit(solver.average)
        for _ in range(30000): solver.iterate(Kuhn(rng.choice(deals)))
        after=audit(solver.average)
        kuhn.append(dict(seed=seed,before=before,after=after))
        if after['nash_conv'] >= .06 or abs(after['value_player0']+1/18) >= .015:
            raise RuntimeError('known-game correctness gate failed')
    write('kuhn.json',kuhn)
    evaluations={}; metrics={}
    for seed in (91,92):
        for arm,adapter in (('historyless',HistorylessState),('history',PokerState)):
            solver=SampledCFR(6,seed); start=time.monotonic()
            for _ in range(args.iterations):
                if time.monotonic()-start>180 or available_memory()<2*1024**3 or shutil.disk_usage(args.root).free<10*1024**3:
                    raise RuntimeError('pilot resource limit')
                solver.iterate(adapter(FastGame(solver.rng.randrange(2**63),stacks=[40]*6,lean=True)))
            name=f'{arm}-{seed}'
            # Only history checkpoints are compatible with production HistoryPolicy.
            if arm=='history': save(args.root/f'{name}.json',solver,seed,20)
            else: write(f'{name}-control.json',dict(algorithm='historyless-control-only',sums=solver.sums,visits=solver.visits))
            saved=dict(sums=solver.sums,visits=solver.visits,stack_bb=20)
            key=old_key if arm=='historyless' else HistoryPolicy.key
            metrics[name]=dict(seconds=time.monotonic()-start,iterations=solver.iterations,
                               nodes=solver.nodes,regret_rows=len(solver.regrets),average_rows=len(solver.sums))
            write(f'{name}-training.json',metrics[name]); print('Trained '+name,flush=True)
            for mode in ('raw','guarded'):
                label=f'{name}-{mode}'
                rows={}
                for style in STYLES:
                    policy=RawPolicy(saved,key) if mode=='raw' else HistoryPolicy(saved)
                    policy.key=key
                    rows[style]=evaluate(policy,style,args.deals)
                evaluations[label]=rows
                write(f'{label}-evaluation.json',rows); print('Evaluated '+label,flush=True)
            del solver
    rows={}
    for style in STYLES:
        rows[style]=evaluate(HistoryPolicy(dict(sums={},visits={},stack_bb=20),learned=False),style,args.deals)
    evaluations['fallback']=rows; write('fallback-evaluation.json',rows)
    paired={}
    for seed in (91,92):
        for mode in ('raw','guarded'):
            name=f'history-{seed}-{mode}'; paired[name]={}
            for control in (f'historyless-{seed}-{mode}','fallback'):
                paired[name][control]={}
                for style in STYLES:
                    a=evaluations[name][style]['deal_block_bb_per_100']
                    b=evaluations[control][style]['deal_block_bb_per_100']
                    r=summarize([x-y for x,y in zip(a,b)],0)
                    paired[name][control][style]={k:r[k] for k in ('bb_per_100','interval_95')}
    if any(hashlib.sha256(p.read_bytes()).hexdigest()!=manifest['sources'][str(p)] for p in sources):
        raise RuntimeError('source changed during experiment')
    write('summary.json',dict(training=metrics,paired=paired,source_verified=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--iterations',type=int,default=1000)
    p.add_argument('--deals',type=int,default=200)
    run(p.parse_args())
