"""Measured bot-CFR pilot. Report readiness without inventing strength evidence."""
import argparse
import json
from pathlib import Path
from .train_cfr import train
from .cfr import CFRPolicy,information_key
from .abstraction import VERSION
from .game import play
from .train import atomic_save


def preflight(output,iterations=10):
    directory=Path(output);directory.mkdir(parents=True,exist_ok=True)
    checkpoint=directory/'checkpoint.json'
    result=train(iterations,checkpoint,game='holdem',checkpoint_every=1,max_seconds=120,abstraction=VERSION,population_probability=.25,archive_every=10)
    seen=total=0
    for deal in range(10):
        policies=[CFRPolicy(result['sums'],i,result['abstraction']) for i in range(6)]
        _,traces,_=play(policies,1000000+deal,deal%6)
        for trajectory in traces:
            for observation,_ in trajectory:
                total+=1;seen+=information_key(observation,result['abstraction']) in result['sums']
    report=dict(completed_iterations=result['iterations'],nodes=result['nodes'],
                seconds=result['seconds'],checkpoint_bytes=checkpoint.stat().st_size,
                held_out_decisions=total,covered_decisions=seen,
                average_policy_coverage=seen/total if total else 0,
                operational_pilot_complete=result['iterations']==iterations,
                competitive_full_training_ready=False,
                blockers=['Coarse abstraction needs learning-curve and loss validation',
                          'No strong held-out population integrated',
                          'No range-aware limited-depth search',
                          'No measured learning curve establishing benefit from scale'])
    atomic_save(directory/'report.json',report)
    print(json.dumps(report,indent=2))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='runs/bot/preflight');p.add_argument('--iterations',type=int,default=10)
    a=p.parse_args();preflight(a.output,a.iterations)
