import argparse,json,time,statistics
from random import Random
from .game import play
from .policy import Policy
from .config import load_checkpoint, load_config

def evaluate(path,blocks,seed,copies,config_path="configs/bot.json"):
    config=load_config(config_path)
    if not 1 <= copies <= 6: raise ValueError("copies must be between 1 and 6")
    if blocks<2: raise ValueError('at least two independent blocks required')
    checkpoint=load_checkpoint(path); table=checkpoint['table']; reports={}
    for population in ['calling','tight','aggressive','blueprint']:
        start=time.perf_counter(); values=[]; latency=[]
        for block in range(blocks):
            results=[]
            # A block reuses one deal across all six candidate seat rotations.
            for rotation in range(6):
                candidates={(rotation+i)%6 for i in range(copies)}
                policies=[Policy(seed+block*100+rotation*6+i,table if i in candidates else {},
                                 'blueprint' if i in candidates else population, mode=config['mode']) for i in range(6)]
                payoff,traces,times=play(policies,seed+block,button=0)
                results.extend(payoff[i] for i in candidates); latency.extend(times)
            values.append(statistics.mean(results)*100)
        rng=Random(seed); boot=sorted(statistics.mean(rng.choices(values,k=blocks)) for _ in range(2000))
        reports[population]=dict(bb_per_100=statistics.mean(values),ci95_cluster_bootstrap=[boot[50],boot[1949]],
            independent_deal_blocks=blocks,table_hands=6*blocks,candidate_hands=6*blocks*copies,
            decision_mean_ms=1000*statistics.mean(latency),wall_seconds=time.perf_counter()-start)
    return dict(config=config,seed=seed,copies=copies,populations=reports,warning='Smoke benchmark only; no strong held-out opponent integrated.')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='artifacts/baseline.json')
    p.add_argument('--blocks',type=int,default=20);p.add_argument('--seed',type=int,default=100000)
    p.add_argument('--copies',type=int,choices=range(1,7),default=1)
    p.add_argument('--config',default='configs/bot.json')
    a=p.parse_args(); print(json.dumps(evaluate(a.checkpoint,a.blocks,a.seed,a.copies,a.config),indent=2))
