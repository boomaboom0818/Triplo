"""Bounded read-only replay with new-epoch moments, never lifetime subtraction."""
import argparse,copy,json,math,os,resource,time
from pathlib import Path
from datetime import datetime
from poker.config import load_checkpoint
from poker.cfr import Holdem,TraversalLimit
from poker.trajectory import TrajectoryMCCFR,ALGORITHM
from poker.train import atomic_save,tuples
from poker.train_cfr import run_lock
from scripts.sustained_train import digest
from scripts.scaling_validation import available_memory


def add(table,key,weight):
    if not math.isfinite(weight*weight):raise TraversalLimit('nonfinite diagnostic weight')
    row=table.setdefault(key,dict(updates=0,total=0.0,squared_total=0.0,maximum=0.0))
    row['updates']+=1;row['total']+=weight;row['squared_total']+=weight*weight
    row['maximum']=max(row['maximum'],weight)


class ObservedTrajectory(TrajectoryMCCFR):
    def record_average_weight(self,state,weight):
        o=state.game.observe()
        add(self.observed,f'average:street={o.street}:active={sum(o.active)}',weight)
        super().record_average_weight(state,weight)

    def outcome(self,state,updater,delta,inverse_own_sample=1.0,scale=1.0):
        if not state.terminal():
            o=state.game.observe()
            if o.player==updater:
                add(self.observed,f'regret_prefix:street={o.street}:active={sum(o.active)}',inverse_own_sample*scale)
        return super().outcome(state,updater,delta,inverse_own_sample,scale)


def diagnose(checkpoint,output,iterations,deadline,max_seconds=180):
    if type(iterations) is not int or not 1<=iterations<=64:raise ValueError('1..64 diagnostic iterations required')
    if not math.isfinite(max_seconds) or max_seconds<=0 or not time.time()<deadline<=time.time()+12*3600:raise ValueError('finite future diagnostic deadline required')
    with run_lock(output):
        if output.exists():raise ValueError('diagnostic exists; preserve rather than overwrite')
        if available_memory()<5*1024**3:raise RuntimeError('diagnostic requires3GiB plus2GiB host reserve')
        sha=digest(checkpoint);saved=load_checkpoint(checkpoint);cfg=saved['config']
        if saved['algorithm']!=ALGORITHM:raise ValueError('trajectory checkpoint required')
        solver=ObservedTrajectory(cfg['seed'],cfg['trajectory_samples'])
        solver.regrets=saved['regrets'];solver.sums=saved['sums'];solver.rng.setstate(tuples(saved['rng']))
        solver.iterations=saved['iterations'];solver.nodes=saved['nodes'];archive=saved['archive']
        completed={};aborted={};status='completed';start=time.monotonic();end=min(start+max_seconds,start+deadline-time.time())
        count=0
        for _ in range(iterations):
            if available_memory()<2*1024**3:status='host memory reserve';break
            if len(set(solver.regrets)|set(solver.sums))>=saved['limits']['max_information_sets']:status='table reserve';break
            solver.observed={};opponents={}
            if archive:
                for player in range(cfg['players']):
                    if solver.rng.random()<cfg['population_probability']:opponents[player]=solver.rng.choice(archive)
            if cfg.get('opponent_population','historical-v1')!='historical-v1':raise ValueError('historical population required')
            try:solver.step(lambda seed:Holdem(seed,cfg['players'],cfg['abstraction']),cfg['players'],saved['limits']['max_iteration_nodes'],end,opponents)
            except TraversalLimit as exc:status=str(exc);aborted=solver.observed;break
            count+=1
            for key,row in solver.observed.items():
                dest=completed.setdefault(key,dict(updates=0,total=0.0,squared_total=0.0,maximum=0.0))
                for name in ('updates','total','squared_total'):dest[name]+=row[name]
                dest['maximum']=max(dest['maximum'],row['maximum'])
            if solver.iterations%cfg['archive_every']==0:
                archive.append(copy.deepcopy(solver.sums));archive=archive[-cfg['archive_size']:]
        for row in completed.values():
            row['concentration_proxy']=row['total']**2/row['squared_total'] if row['squared_total'] else None
        assert digest(checkpoint)==sha
        report=dict(status=status,checkpoint=str(checkpoint.resolve()),checkpoint_sha256=sha,diagnostic_source_sha256=digest(Path(__file__)),source_iterations=saved['iterations'],read_only=True,discarded_local_learning=True,simulated_complete_iterations=count,requested_iterations=iterations,wall_seconds=time.monotonic()-start,deadline=deadline,max_seconds=max_seconds,moments=completed,aborted_iteration_moments=aborted,traversal_stats=solver.traversal_stats,
            caveat='Concentration proxies use correlated node weights, not inferential independent ESS. Regret prefix weights exclude later action-value importance corrections. Diagnostic local learning is discarded; no training checkpoint/progress is written.')
        atomic_save(output,report)
        return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--iterations',type=int,default=64);p.add_argument('--max-seconds',type=float,default=180);p.add_argument('--deadline',required=True)
    a=p.parse_args();d=datetime.fromisoformat(a.deadline)
    if d.tzinfo is None:raise ValueError('deadline needs timezone')
    resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3));os.nice(10)
    diagnose(a.checkpoint,a.output,a.iterations,d.timestamp(),a.max_seconds)
