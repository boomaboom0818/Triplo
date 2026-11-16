"""Evaluate a history checkpoint and its guarded rule-only baseline locally."""
import argparse
import hashlib
import json
from pathlib import Path
from poker.history_cfr import load, HistoryPolicy
from poker.rule_opponents import STYLES
from scripts.history_pilot import evaluate
from scripts.rule_gate import summarize


def run(checkpoint, out, deals=500, seed=20310000, min_visits=64):
    if deals < 2 or out.exists():
        raise ValueError('at least two deals and a new report path required')
    saved=load(checkpoint)
    result=dict(checkpoint=str(checkpoint),sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                seed=seed,deals=deals,stack_bb=saved['stack_bb'],min_visits=min_visits,styles={},
                note='Pointwise 95% normal intervals over whole-deal blocks; no multiple-test correction.')
    for style in STYLES:
        policy=HistoryPolicy(saved,min_visits=min_visits)
        fallback=HistoryPolicy(saved,learned=False)
        candidate=evaluate(policy,style,deals,saved['stack_bb'],seed)
        control=evaluate(fallback,style,deals,saved['stack_bb'],seed)
        diff=summarize([a-b for a,b in zip(candidate['deal_block_bb_per_100'],control['deal_block_bb_per_100'])],0)
        result['styles'][style]=dict(candidate=candidate,fallback=control,
            paired_difference={k:diff[k] for k in ('bb_per_100','interval_95')})
    with out.open('x') as f: json.dump(result,f,indent=2)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--deals',type=int,default=500)
    p.add_argument('--seed',type=int,default=20310000)
    p.add_argument('--min-visits',type=int,default=64)
    a=p.parse_args()
    print(json.dumps(run(a.checkpoint,a.out,a.deals,a.seed,a.min_visits),indent=2))
