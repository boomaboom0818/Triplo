"""Read-only conditional averaging variance/cost diagnostic, no learning writes."""
import argparse,hashlib,json,math,os,resource,statistics,time
from pathlib import Path
from random import Random
from poker.cfr import Holdem,TraversalLimit
from poker.trajectory import TrajectoryMCCFR
from poker.config import load_checkpoint
from poker.train_cfr import run_lock
from poker.train import atomic_save
from scripts.sustained_train import digest
from scripts.scaling_validation import available_memory


class AveragePrefix(TrajectoryMCCFR):
    def __init__(self,depth=1):
        super().__init__()
        if type(depth) is not int or not 0<=depth<=2:raise ValueError('external prefix depth0..2 required')
        self.depth=depth

    def trajectory_average(self,state,owner,weight=1.,external_depth=0):
        if not math.isfinite(weight*weight):raise TraversalLimit('nonfinite averaging weight')
        self.visit()
        if state.terminal():return
        self.record_visit(state,'average',owner)
        actor,key,actions=state.info();strategy=self.strategy(key,actions)
        if actor==owner:
            self.record_average_weight(state,weight)
            row=getattr(self,'pending_sums',self.sums).setdefault(key,[0.]*len(actions))
            for j,p in enumerate(strategy):row[j]+=weight*p
            action=self.rng.choices(actions,strategy)[0]
            self.trajectory_average(state.child(action),owner,weight,external_depth)
        elif external_depth<self.depth:
            # Every external action is included (inclusion probability1).
            for action in actions:self.trajectory_average(state.child(action),owner,weight,external_depth+1)
        else:
            q=[.5/len(actions)+(.5 if a=='call' else 0.) for a in actions] if 'call' in actions else [1/len(actions)]*len(actions)
            j=self.rng.choices(range(len(actions)),q)[0]
            self.trajectory_average(state.child(actions[j]),owner,weight/q[j],external_depth+1)


def main(output,deadline):
    if not time.time()<deadline<=time.time()+3600:raise ValueError('future finite diagnostic deadline within1h required')
    with run_lock(output):
        if output.exists():raise ValueError('benchmark exists; preserve instead of retry')
        parents=[Path(f'runs/bot/prefix-pilot-001/candidate-{i}-plus-500.json') for i in range(2)]
        manifest=dict(parents=[dict(path=str(p.resolve()),sha256=digest(p)) for p in parents],deal_seeds=[8900000,8900001],replicates=64,depths=[0,1,2],deadline=deadline,node_limit=500000,code_sha256=digest(Path(__file__)),algorithm_sources={str(p):digest(p) for p in Path('poker').glob('*.py')},read_only=True,
            criterion='variance*nodes ratio<.75 in>=3/4fixtures and allmaxtrace<=10000; exploratory only')
        atomic_save(output.with_suffix('.manifest.json'),manifest);results=[]
        for i,parent in enumerate(parents):
            saved=load_checkpoint(parent)
            for deal in manifest['deal_seeds']:
                state=Holdem(deal,6,saved['abstraction']);owner=state.info()[0];fixture=dict(candidate=i,deal_seed=deal,owner=owner,variants={})
                for depth in (0,1,2):
                    solver=TrajectoryMCCFR() if depth==0 else AveragePrefix(depth)
                    solver.regrets=saved['regrets'];totals={};norms=[];nodes=[];started=time.monotonic()
                    for trial in range(64):
                        if time.time()>=deadline:raise TimeoutError('diagnostic deadline reached')
                        if available_memory()<2*1024**3:raise RuntimeError('hostreserve2GiB reached')
                        solver.rng=Random(deal+trial*100+i);solver.nodes=0;solver.node_limit=500000
                        solver.deadline=time.monotonic()+max(0,deadline-time.time());solver.sums={}
                        solver.trajectory_average(state,owner)
                        norms.append(math.fsum(x*x for row in solver.sums.values() for x in row));nodes.append(solver.nodes)
                        for key,row in solver.sums.items():
                            dest=totals.setdefault(key,[0.]*len(row))
                            for j,x in enumerate(row):dest[j]+=x
                    wall=time.monotonic()-started;norm_sum=math.fsum(norms);sum_norm=math.fsum(x*x for row in totals.values() for x in row)
                    variance=max(0.,(norm_sum-sum_norm/64)/63)
                    fixture['variants'][str(depth)]=dict(variance_trace=variance,sum_squared_norms=norm_sum,squared_norm_of_vector_sum=sum_norm,replicate_squared_norms=norms,replicate_nodes=nodes,mean_nodes=statistics.mean(nodes),max_nodes=max(nodes),wall_seconds=wall,variance_times_nodes=variance*statistics.mean(nodes),variance_times_seconds=variance*wall/64,unique_rows=len(totals),summed_vector_by_key_hash={hashlib.sha256(key.encode()).hexdigest():row for key,row in totals.items()})
                    atomic_save(output.with_suffix('.progress.json'),dict(status='running',candidate=i,deal_seed=deal,depth=depth,completed_fixtures=results))
                base=fixture['variants']['0']
                for depth in ('1','2'):
                    v=fixture['variants'][depth]
                    v['variance_work_ratio']=v['variance_times_nodes']/base['variance_times_nodes'] if base['variance_times_nodes']>0 else None
                    v['variance_time_ratio']=v['variance_times_seconds']/base['variance_times_seconds'] if base['variance_times_seconds']>0 else None
                results.append(fixture)
            assert digest(parent)==manifest['parents'][i]['sha256']
            del solver,saved
        eligible={depth:sum(f['variants'][depth]['variance_work_ratio'] is not None and f['variants'][depth]['variance_work_ratio']<.75 for f in results)>=3 and all(f['variants'][depth]['max_nodes']<=10000 for f in results) for depth in ('1','2')}
        atomic_save(output,dict(status='completed',read_only=True,fixtures=results,eligible_for_further_pilot_investigation=eligible,caveat='Fixed-deal conditional vector variance across correlated node updates; not independent ESS or payoff evidence.64replicates can miss rare tails. No source sums/regrets/RNG/archive/checkpoint modified. Averaging only; regret unchanged.'))

if __name__=='__main__':
    from datetime import datetime
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--deadline',required=True)
    a=p.parse_args();d=datetime.fromisoformat(a.deadline)
    if d.tzinfo is None:raise ValueError('deadline needs timezone')
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));os.nice(10)
    main(a.output,d.timestamp())
