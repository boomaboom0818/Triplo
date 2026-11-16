"""Read-only replay/averaging audit on consumed development deals; no training.

Current regret policy and average deltas are diagnostics only, never deployable
candidates. Heuristic equity disagreements are not proof of a strategic error.
"""
import argparse
from array import array
from collections import Counter, defaultdict
import json
from pathlib import Path
from random import Random
import statistics

from poker.history_cfr import HistoryPolicy, load
from poker.equity_policy import EquityPolicy
from poker.live_curriculum import root_game
from poker.rule_opponents import RulePolicy
from scripts.live_gate import digest


def normalize(row):
    mass=sum(row)
    return [x/mass for x in row] if mass>0 else [0.]*len(row)


def tv(a,b):return sum(abs(x-y) for x,y in zip(a,b))/2


def summarize(rows):
    if not rows:return dict(decisions=0)
    return dict(decisions=len(rows),mean_parent_mass_fraction=statistics.mean(r['parent_fraction'] for r in rows),
                median_parent_mass_fraction=statistics.median(r['parent_fraction'] for r in rows),
                mean_incumbent_mass_fraction=statistics.mean(r['incumbent_fraction'] for r in rows),
                mean_policy_tv_from_parent=statistics.mean(r['parent_tv'] for r in rows),
                mean_average_vs_regret_tv=statistics.mean(r['regret_tv'] for r in rows if r['regret_tv'] is not None),
                actions=dict(Counter(r['action_kind'] for r in rows)),
                equity_disagreements=dict(Counter(r['equity_disagreement'] for r in rows if r.get('equity_disagreement'))))


def run(candidate,parent,incumbent,state,out,deals=300,seed=61461000):
    out=Path(out)
    if out.exists() or not 2<=deals<=300:raise ValueError('new report and 2..300 consumed development blocks required')
    paths=[Path(candidate),Path(parent),Path(incumbent)]
    hashes={str(p):digest(p) for p in paths}
    saved,old,champion=[load(p) for p in paths]
    policy=HistoryPolicy(saved,min_visits=64);policy.fallback=EquityPolicy()
    baseline=EquityPolicy();opponent=RulePolicy('aggressive-v1')
    decisions=[];hands=[];keys=set()
    for deal in range(deals):
        for position in range(6):
            deal_seed=seed+deal
            game=root_game(deal_seed,6,[100]*6,lean=1<<position)
            policy.rng=Random(f'hero:{deal_seed}:{position}')
            records=[]
            while game.live:
                o=game.observe()
                if o.position!=position:game.act(opponent.act(o));continue
                key=policy.key(o);before=policy.covered;action=policy.act(o,history_complete=True)
                if policy.covered>before:
                    row=saved['sums'][key];p=old['sums'].get(key,[0.]*len(row));c=champion['sums'].get(key,[0.]*len(row))
                    fraction=sum(p)/sum(row);incumbent_fraction=sum(c)/sum(row)
                    probs=normalize(row);parent_probs=normalize(p)
                    record=dict(deal=deal,position=position,street=o.street,key=key,hole=o.hole,board=o.board,
                                pot=o.pot,call=o.call,legal=o.legal,history=o.history,active=sum(o.active),
                                action=action,action_kind='raise' if isinstance(action,int) else 'check' if action=='call' and not o.call else action,
                                probabilities=probs,parent_probabilities=parent_probs,parent_fraction=fraction,
                                incumbent_fraction=incumbent_fraction,parent_tv=tv(probs,parent_probs),
                                visits=saved['visits'][key],new_visits=saved['visits'][key]-old['visits'].get(key,0),regret_tv=None)
                    if o.street and o.call:
                        fallback_action=baseline.act(o);record['equity_diagnostic']=baseline.last
                        record['fallback_action']=fallback_action
                        if action=='fold' and fallback_action!='fold':record['equity_disagreement']='learned_fold_fallback_continue'
                        elif action!='fold' and fallback_action=='fold':record['equity_disagreement']='learned_continue_fallback_fold'
                    decisions.append(record);records.append(record);keys.add(key)
                game.act(action)
            payoff=game.payoffs[position]/2
            hands.append(dict(deal=deal,position=position,payoff_bb=payoff))
            for r in records:r['hand_payoff_bb']=payoff
    # Read only visited optimizer rows, without materializing millions of keys.
    state=Path(state);slots={}
    with (state/'keys.txt').open() as f:
        for line in f:
            key,slot=line.rstrip('\n').split('\t')
            if key in keys:slots[key]=int(slot)
    regrets={}
    with (state/'regrets.f64').open('rb') as f:
        for key,slot in slots.items():
            count=slot&7;f.seek((slot>>3)*8);row=array('d');row.fromfile(f,count)
            positive=[max(0.,v) for v in row];regrets[key]=normalize(positive) if sum(positive)>0 else [1/count]*count
    for r in decisions:
        if r['key'] in regrets:r['regret_probabilities']=regrets[r['key']];r['regret_tv']=tv(r['probabilities'],regrets[r['key']])
    postflop=[r for r in decisions if r['street']]
    exposure=defaultdict(set)
    for r in decisions:exposure[r['key']].add(tuple(r['legal']))
    worst=sorted(hands,key=lambda h:h['payoff_bb'])[:12]
    worst_ids={(h['deal'],h['position']) for h in worst}
    report=dict(note='Consumed development replay, not a new holdout. Correlated decisions; no causal attribution of hand losses. Equity priors are heuristic.',
                seed=seed,deals=deals,hands=len(hands),checkpoint_sha256=hashes,
                source_sha256={p:digest(p) for p in ('scripts/diagnose_aggressive_history.py','poker/history_cfr.py','poker/equity_policy.py','poker/fast_game.py')},
                optimizer_meta=json.loads((state/'meta.json').read_text()),
                payoff_bb100=statistics.mean(h['payoff_bb'] for h in hands)*100,
                all_learned=summarize(decisions),postflop_learned=summarize(postflop),
                by_street={str(s):summarize([r for r in decisions if r['street']==s]) for s in range(4)},
                multi_menu_keys=sum(len(v)>1 for v in exposure.values()),unique_keys=len(exposure),
                decision_weighted_fraction_parent_over_90pct=statistics.mean(r['parent_fraction']>.9 for r in decisions),
                worst_hands=worst,worst_hand_decisions=[r for r in decisions if (r['deal'],r['position']) in worst_ids],
                consumed_development_decisions=decisions)
    if any(digest(p)!=h for p,h in hashes.items()):raise RuntimeError('checkpoint changed')
    with out.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps({k:report[k] for k in ('payoff_bb100','all_learned','postflop_learned','multi_menu_keys','unique_keys','decision_weighted_fraction_parent_over_90pct')},indent=2))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--candidate',required=True,type=Path);p.add_argument('--parent',required=True,type=Path)
    p.add_argument('--incumbent',required=True,type=Path);p.add_argument('--state',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--deals',default=300,type=int);p.add_argument('--seed',default=61461000,type=int)
    a=p.parse_args();run(a.candidate,a.parent,a.incumbent,a.state,a.out,a.deals,a.seed)
