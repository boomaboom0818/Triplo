"""Monte Carlo policy-gradient population self-play baseline (not CFR)."""
import argparse,json,copy,time
from random import Random
from .game import play
from .policy import Policy,key

def train(hands,seed,path):
    rng=Random(seed); table={}; archive=[{}]; start=time.perf_counter()
    for h in range(hands):
        learner=h%6
        policies=[Policy(rng.randrange(2**32),copy.deepcopy(rng.choice(archive)),
                         rng.choice(['blueprint','blueprint','calling','aggressive'])) for _ in range(6)]
        policies[learner]=Policy(rng.randrange(2**32),table)
        rewards,traces,_=play(policies,rng.randrange(2**63),h%6)
        # Bounded terminal-return REINFORCE; deliberately simple high-variance baseline.
        for o,a in traces[learner]:
            probs=policies[learner].probabilities(o)
            row=table.setdefault(key(o),[0.0]*len(o.legal))
            for j in range(len(row)):
                row[j]+=.02*(rewards[learner]/100)*((o.legal[j]==a)-probs[j])
        if (h+1)%100==0: archive.append(copy.deepcopy(table)); archive=archive[-8:]
    result=dict(schema=1,algorithm='tabular-reinforce-v1',engine='pokerkit-0.7.7',
                seed=seed,hands=hands,seconds=time.perf_counter()-start,table=table,
                abstraction='rank-position-board-active-pot-v1',stacks=200,players=6)
    with open(path,'w') as f: json.dump(result,f)
    print(json.dumps({k:v for k,v in result.items() if k!='table'}))
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--hands',type=int,default=1000)
    p.add_argument('--seed',type=int,default=1); p.add_argument('--output',default='artifacts/baseline.json')
    a=p.parse_args(); train(a.hands,a.seed,a.output)
