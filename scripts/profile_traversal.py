"""Read-only, bounded reproduction of one checkpoint's next CFR iteration."""
import argparse,json,resource,time
from collections import Counter
from pathlib import Path
from poker.cfr import MCCFR,Holdem,TraversalLimit
from poker.config import load_checkpoint
from poker.train import atomic_save,tuples
from poker.train_cfr import run_lock
from scripts.sustained_train import digest


class ProfiledMCCFR(MCCFR):
    def __init__(self,seed,output):
        super().__init__(seed);self.output=output;self.profile=Counter();self.phase='setup';self.owner=None
        self.started=time.monotonic();self.visited=0;self.last_state={}
    def regret(self,state,updater,delta):
        self.phase='regret';self.owner=updater
        return super().regret(state,updater,delta)
    def average(self,state,owner,own_reach=1.0,inverse_sample=1.0):
        self.phase='average';self.owner=owner
        return super().average(state,owner,own_reach,inverse_sample)
    def visit(self):
        self.visited+=1;self.profile[f'{self.phase}:owner={self.owner}:nodes']+=1
        if self.visited%10000==0:self.write('running')
        super().visit()
    def record_visit(self,state,traversal,owner):
        o=state.game.observe()
        key=f'{traversal}:owner={owner}:street={o.street}:active={sum(o.active)}'
        self.profile[key]+=1
        self.last_state=dict(traversal=traversal,owner=owner,actor=o.player,street=o.street,active=sum(o.active),history_length=len(o.history),legal_actions=len(o.legal))
        super().record_visit(state,traversal,owner)
    def write(self,status):
        atomic_save(self.output,dict(status=status,visited=self.visited,seconds=time.monotonic()-self.started,phase=self.phase,owner=self.owner,counts=dict(self.profile),last_state=self.last_state))


def main(checkpoint,output,max_seconds=1200,max_nodes=500000):
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
    with run_lock(output):
        if output.exists():raise ValueError('profile exists; inspect instead of retrying')
        before=digest(checkpoint);saved=load_checkpoint(checkpoint);cfg=saved['config']
        solver=ProfiledMCCFR(cfg['seed'],output);solver.averaging_sampler=cfg.get('averaging_sampler','uniform')
        solver.regrets=saved['regrets'];solver.sums=saved['sums'];solver.rng.setstate(tuples(saved['rng']))
        solver.iterations=saved['iterations'];solver.nodes=saved['nodes']
        opponents={}
        for player in range(cfg['players']):
            if saved['archive'] and solver.rng.random()<cfg['population_probability']:
                opponents[player]=solver.rng.choice(saved['archive'])
        if cfg.get('opponent_population') not in (None,'historical-v1'):raise ValueError('profile supports historical population only')
        status='completed'
        try:solver.step(lambda seed:Holdem(seed,cfg['players'],cfg['abstraction']),cfg['players'],max_nodes,time.monotonic()+max_seconds,opponents)
        except TraversalLimit as exc:status=str(exc)
        finally:solver.write(status)
        assert digest(checkpoint)==before
        report=json.loads(output.read_text());report.update(checkpoint=str(checkpoint.resolve()),checkpoint_sha256=before,source_iterations=saved['iterations'],max_nodes=max_nodes,max_seconds=max_seconds,read_only=True)
        atomic_save(output,report)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-seconds',type=float,default=1200);p.add_argument('--max-nodes',type=int,default=500000)
    a=p.parse_args();main(a.checkpoint,a.output,a.max_seconds,a.max_nodes)
