"""Read-only engineering experiment; no training or model promotion."""
import argparse,hashlib,json,math,os,resource,statistics,time
from pathlib import Path
from random import Random
from poker.cfr import normalize,Holdem,TraversalLimit
from poker.trajectory import TrajectoryMCCFR
from poker.config import load_checkpoint
from poker.train_cfr import run_lock
from poker.train import atomic_save
from scripts.sustained_train import digest
from scripts.scaling_validation import available_memory


class PrefixMCCFR(TrajectoryMCCFR):
    """Enumerate at most the first two updater decisions, then sample as v2."""
    def __init__(self,seed=1,samples=8,depth=1):
        super().__init__(seed,samples)
        if type(depth) is not int or not 0<=depth<=2:raise ValueError('prefix depth must be0..2')
        self.depth=depth

    def outcome(self,state,updater,delta,inverse_own_sample=1.,scale=1.,own_depth=0):
        if not math.isfinite(inverse_own_sample):raise TraversalLimit('nonfinite importance weight')
        if hasattr(self,'pending_traversal_stats'):
            self.pending_traversal_stats['max_inverse_own_sample']=max(self.pending_traversal_stats['max_inverse_own_sample'],inverse_own_sample)
        self.visit()
        if state.terminal():return state.returns()[updater]
        self.record_visit(state,'regret',updater)
        actor,key,actions=state.info();strategy=self.strategy(key,actions)
        if actor!=updater:
            frozen=getattr(self,'opponents',{}).get(actor)
            if isinstance(frozen,str):
                from poker.population import profile_strategy
                strategy=profile_strategy(state.game.observe(),frozen)
            elif frozen is not None:strategy=normalize(frozen.get(key,[0.]*len(actions)))
            action=self.rng.choices(actions,strategy)[0]
            return self.outcome(state.child(action),updater,delta,inverse_own_sample,scale,own_depth)
        if own_depth<self.depth:
            values=[self.outcome(state.child(a),updater,delta,inverse_own_sample,scale,own_depth+1) for a in actions]
            value=sum(p*v for p,v in zip(strategy,values))
        else:
            q=[.5*p+.5/len(actions) for p in strategy];index=self.rng.choices(range(len(actions)),q)[0]
            action_value=self.outcome(state.child(actions[index]),updater,delta,inverse_own_sample/q[index],scale,own_depth+1)/q[index]
            values=[action_value if j==index else 0. for j in range(len(actions))]
            value=strategy[index]*action_value
        row=delta.setdefault(key,[0.]*len(actions))
        for j,v in enumerate(values):
            row[j]+=scale*inverse_own_sample*(v-value)
            if not math.isfinite(row[j]):raise TraversalLimit('nonfinite regret update')
        return value


def main(output,deadline,repeats=64):
    if repeats!=64:raise ValueError('frozen diagnostic has64replicates')
    if not time.time()<deadline<=time.time()+3600:raise ValueError('finite diagnostic deadline within1h required')
    with run_lock(output):
        if output.exists():raise ValueError('existing benchmark; preserve and review')
        parents=[Path(f'runs/bot/traversal-pilot-001/candidate-{i}-plus-1500.json') for i in range(2)]
        manifest=dict(parents=[dict(path=str(p.resolve()),sha256=digest(p)) for p in parents],deal_seeds=[8800000,8800001],replicates=repeats,depths=[0,1,2],node_limit=500000,deadline=deadline,code_sha256=digest(Path(__file__)),algorithm_sources={str(p):digest(p) for p in Path('poker').glob('*.py')},read_only=True)
        atomic_save(output.with_suffix('.manifest.json'),manifest)
        results=[]
        for i,parent in enumerate(parents):
            saved=load_checkpoint(parent);cfg=saved['config']
            for deal in manifest['deal_seeds']:
                state=Holdem(deal,6,saved['abstraction']);updater,key,actions=state.info()
                pop_rng=Random(deal+1000+i);opponents={}
                if saved['archive']:
                    for player in range(6):
                        if pop_rng.random()<cfg['population_probability']:opponents[player]=pop_rng.choice(saved['archive'])
                fixture=dict(candidate=i,deal_seed=deal,updater=updater,root_key_sha256=hashlib.sha256(key.encode()).hexdigest(),variants={})
                for depth in (0,1,2):
                    solver=TrajectoryMCCFR() if depth==0 else PrefixMCCFR(depth=depth)
                    solver.regrets=saved['regrets'];solver.opponents=opponents
                    values=[];nodes=[];started=time.monotonic()
                    for trial in range(repeats):
                        if time.time()>=deadline:raise TimeoutError('diagnostic deadline reached')
                        if available_memory()<2*1024**3:raise RuntimeError('hostreserve2GiB reached')
                        solver.rng=Random(deal+trial*100+i);solver.nodes=0;solver.node_limit=500000
                        solver.deadline=time.monotonic()+max(0,deadline-time.time());delta={}
                        solver.outcome(state,updater,delta)
                        values.append(delta[key]);nodes.append(solver.nodes)
                    wall=time.monotonic()-started
                    variance=sum(statistics.variance([row[j] for row in values]) for j in range(len(actions)))
                    fixture['variants'][str(depth)]=dict(mean=[statistics.mean(row[j] for row in values) for j in range(len(actions))],variance_trace=variance,mean_nodes=statistics.mean(nodes),max_nodes=max(nodes),wall_seconds=wall,variance_times_nodes=variance*statistics.mean(nodes),variance_times_seconds=variance*wall/repeats,replicate_regrets=values)
                    atomic_save(output.with_suffix('.progress.json'),dict(status='running',candidate=i,deal_seed=deal,depth=depth,completed_fixtures=results))
                base=fixture['variants']['0']
                for depth in ('1','2'):
                    v=fixture['variants'][depth]
                    v['variance_work_ratio']=v['variance_times_nodes']/base['variance_times_nodes'] if base['variance_times_nodes']>0 else None
                    v['variance_time_ratio']=v['variance_times_seconds']/base['variance_times_seconds'] if base['variance_times_seconds']>0 else None
                results.append(fixture)
            assert digest(parent)==manifest['parents'][i]['sha256']
            del solver,saved,opponents
        eligible={depth:sum(f['variants'][depth]['variance_work_ratio'] is not None and f['variants'][depth]['variance_work_ratio']<.75 for f in results)>=3 and all(f['variants'][depth]['max_nodes']<=10000 for f in results) for depth in ('1','2')}
        atomic_save(output,dict(status='completed',read_only=True,fixtures=results,eligible_for_further_pilot_investigation=eligible,criterion='variance*nodes ratio<0.75 in>=3/4fixtures; maxtrace<=10000nodes. Exploratory only, no strength/promotion.',caveat='Conditional fixed-deal/root regret estimator diagnostic, not full training or averaging variance.64replicates can miss rare tails. Wall cost and every fixture retained. No checkpoint changed.'))

if __name__=='__main__':
    from datetime import datetime
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--deadline',required=True)
    a=p.parse_args();d=datetime.fromisoformat(a.deadline)
    if d.tzinfo is None:raise ValueError('deadline needs timezone')
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));os.nice(10)
    main(a.output,d.timestamp())
