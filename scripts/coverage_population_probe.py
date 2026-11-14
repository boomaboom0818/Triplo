"""Read-only reach/cost probe for a softened check/fold population hypothesis."""
import argparse,json,os,resource,time
from pathlib import Path
from random import Random
from poker.prefix import PrefixMCCFR
from poker.cfr import Holdem,normalize
from poker.config import load_checkpoint
from poker.train import atomic_save
from poker.train_cfr import run_lock
from scripts.sustained_train import digest
from scripts.scaling_validation import available_memory


def guarded_strategy(o):
    raises=sum(isinstance(a,int) for a in o.legal);has_fold='fold' in o.legal
    if o.call:call,fold,raise_mass=.13+(.85 if not has_fold else 0),.85,.02
    else:call,fold,raise_mass=.95,0.,.05
    if not raises:call+=raise_mass
    return normalize([call if a=='call' else fold if a=='fold' else raise_mass/raises for a in o.legal])


class ReachProbe(PrefixMCCFR):
    def __init__(self,guarded=False):
        super().__init__(depth=2);self.guarded=guarded;self.observed={};self.guard_context=None
    def outcome(self,state,updater,delta,*args,**kwargs):
        self.guard_context=state.game.observe() if not state.terminal() and state.info()[0]!=updater and self.guarded else None
        return super().outcome(state,updater,delta,*args,**kwargs)
    def strategy(self,key,actions):
        return guarded_strategy(self.guard_context) if self.guard_context is not None else super().strategy(key,actions)
    def record_visit(self,state,traversal,owner,sampling=None):
        o=state.game.observe()
        if o.player==owner and o.street:
            _,key,_=state.info();label=f'street={o.street}:active={sum(o.active)}'
            row=self.observed.setdefault(label,dict(visits=0,missing_average=0,missing_positive_regret=0))
            row['visits']+=1;row['missing_average']+=not any(self.reference_sums.get(key,[]))
            row['missing_positive_regret']+=not any(x>0 for x in self.regrets.get(key,[]))
        super().record_visit(state,traversal,owner,sampling)


def main(output,deadline):
    if not time.time()<deadline<=time.time()+3600:raise ValueError('finite future deadline within1h required')
    with run_lock(output):
        if output.exists():raise ValueError('probe exists; preserve and review')
        parents=[Path(f'runs/bot/prefix-pilot-001/candidate-{i}-plus-500.json') for i in range(2)]
        manifest=dict(inputs=[dict(path=str(p.resolve()),sha256=digest(p)) for p in parents],stream_seeds=[9400000,9401000],replicates=64,deadline=deadline,code_sha256=digest(Path(__file__)),read_only=True,criterion='Guarded heads-up/3player owner postflop visits per node>=2xhistorical in>=3/4fixtures (ifbase0 require>=20visits), eachguardedfixture>=20visits andmaxtrace<=10000. Engineering screen only.')
        atomic_save(output.with_suffix('.manifest.json'),manifest);fixtures=[]
        for i,parent in enumerate(parents):
            saved=load_checkpoint(parent);cfg=saved['config']
            for seed in manifest['stream_seeds']:
                f=dict(candidate=i,stream_seed=seed,variants={})
                for guarded in (False,True):
                    s=ReachProbe(guarded);s.regrets=saved['regrets'];s.reference_sums=saved['sums'];nodes=[];started=time.monotonic()
                    for trial in range(64):
                        if time.time()>=deadline:raise TimeoutError('coverage probe deadline reached')
                        if available_memory()<2*1024**3:raise RuntimeError('hostreserve2GiB reached')
                        s.rng=Random(seed+trial*100+i);s.nodes=0;s.node_limit=500000;s.deadline=time.monotonic()+deadline-time.time();s.opponents={}
                        if not guarded:
                            pop=Random(seed+trial*100+i+1)
                            for p in range(6):
                                if saved['archive'] and pop.random()<cfg['population_probability']:s.opponents[p]=pop.choice(saved['archive'])
                        s.outcome(Holdem(seed+trial,6,saved['abstraction']),trial%6,{})
                        nodes.append(s.nodes)
                    visits=sum(row['visits'] for key,row in s.observed.items() if int(key.split('active=')[1])<=3)
                    f['variants']['guarded' if guarded else 'historical']=dict(groups=s.observed,nodes=sum(nodes),max_nodes=max(nodes),heads_up_or_three_player_visits=visits,visits_per_node=visits/sum(nodes),wall_seconds=time.monotonic()-started)
                    atomic_save(output.with_suffix('.progress.json'),dict(status='running',candidate=i,stream_seed=seed,guarded=guarded,completed_fixtures=fixtures))
                a,b=f['variants']['historical'],f['variants']['guarded']
                f['engineering_pass']=b['heads_up_or_three_player_visits']>=20 and b['max_nodes']<=10000 and (b['visits_per_node']>=2*a['visits_per_node'] if a['visits_per_node'] else b['heads_up_or_three_player_visits']>=20)
                fixtures.append(f)
            assert digest(parent)==manifest['inputs'][i]['sha256']
            del s,saved
        atomic_save(output,dict(status='completed',read_only=True,fixtures=fixtures,eligible_for_curriculum_pilot_review=sum(f['engineering_pass'] for f in fixtures)>=3,caveat='Deliberately changed opponent distribution, not estimator variance/payoff/strength evidence. No learning delta committed or checkpoint changed. A production curriculum would need explicit new recipe/fork and prospective independent payoff/coverage gates; scripted training profile gains would not be held-out generalization.'))

if __name__=='__main__':
    from datetime import datetime
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--deadline',required=True)
    a=p.parse_args();d=datetime.fromisoformat(a.deadline)
    if d.tzinfo is None:raise ValueError('deadline needs timezone')
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));os.nice(10);main(a.output,d.timestamp())
