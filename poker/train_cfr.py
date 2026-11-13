"""CPU reference trainer. Hold'em iterations can be expensive; pilot first."""
import argparse
import json
import time
from pathlib import Path
from .cfr import MCCFR,Kuhn,Holdem
from .train import atomic_save,tuples


def train(iterations,path,seed=1,game='kuhn',players=6,resume=None):
    if iterations<1 or game not in ('kuhn','holdem') or not 3<=players<=6:
        raise ValueError('invalid training configuration')
    config=dict(game=game,players=2 if game=='kuhn' else players,seed=seed)
    if Path(path).exists() and (resume is None or Path(path).resolve()!=Path(resume).resolve()):
        raise ValueError('output exists; use resume or a new path')
    solver=MCCFR(seed);seconds=0
    if resume:
        saved=json.loads(Path(resume).read_text())
        if (saved.get('schema')!=1 or saved.get('algorithm')!='external-sampling-mccfr-v1'
                or saved.get('config')!=config or saved.get('model_id')!='bot-opponent-v1'):
            raise ValueError('incompatible CFR checkpoint')
        solver.regrets=saved['regrets'];solver.sums=saved['sums']
        solver.rng.setstate(tuples(saved['rng']));solver.iterations=saved['iterations']
        solver.nodes=saved['nodes'];seconds=saved['seconds']
    if iterations<=solver.iterations:raise ValueError('target must exceed completed iterations')
    start=time.perf_counter()
    factory=Kuhn if game=='kuhn' else lambda seed:Holdem(seed,players)
    for _ in range(solver.iterations,iterations):
        solver.step(factory,config['players'])
        result=dict(schema=1,algorithm='external-sampling-mccfr-v1',model_id='bot-opponent-v1',
                    config=config,engine='pokerkit-0.7.7',abstraction='exact-observation-v1',
                    iterations=solver.iterations,regrets=solver.regrets,sums=solver.sums,
                    rng=solver.rng.getstate(),nodes=solver.nodes,seconds=seconds+time.perf_counter()-start)
        if solver.iterations%100==0 or solver.iterations==iterations:
            atomic_save(path,result)
            print(json.dumps({k:result[k] for k in ('iterations','nodes','seconds')}),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--game',choices=['kuhn','holdem'],default='kuhn')
    p.add_argument('--players',type=int,default=6);p.add_argument('--iterations',type=int,default=1000)
    p.add_argument('--seed',type=int,default=1);p.add_argument('--resume')
    p.add_argument('--output',default='runs/bot/cfr.json')
    a=p.parse_args();train(a.iterations,a.output,a.seed,a.game,a.players,a.resume)
