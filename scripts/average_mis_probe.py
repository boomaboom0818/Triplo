"""Read-only whole-path balanced MIS averaging; never a production checkpoint."""
import argparse,hashlib,json,math,os,resource,shutil,signal,statistics,time
from pathlib import Path
from random import Random
from poker.cfr import Holdem,TraversalLimit
from poker.population import profile_strategy
from poker.trajectory import TrajectoryMCCFR
from poker.config import load_checkpoint
from poker.train_cfr import run_lock
from poker.train import atomic_save
from scripts.sustained_train import digest
from scripts.scaling_validation import available_memory
from scripts.vector_variance import VectorVariance,SCHEMA

PROPOSAL='balanced-call-guarded-average-probe-v1'
CRITERIA=Path('docs/AVERAGE-MIS-PROBE.md')
PARENTS=[Path(f'runs/bot/coverage-curriculum-001/candidate-{i}-plus-300.json') for i in range(2)]
DEALS=[9900000,9900001]
REPLICATES=128


def balance_weight(log_a,log_b,scale):
    if scale<=0 or not math.isfinite(scale) or not math.isfinite(log_a) or math.isnan(log_b) or log_b==math.inf:
        raise ValueError('invalid mixture prefix/scale')
    high=max(log_a,log_b)
    log_mix=high+math.log(.5*math.exp(log_a-high)+.5*math.exp(log_b-high))
    try:weight=math.exp(math.log(scale)-log_mix)
    except OverflowError:raise TraversalLimit('nonfinite MIS averaging weight') from None
    if not math.isfinite(weight*weight):raise TraversalLimit('nonfinite MIS averaging weight')
    ratio=math.exp(log_a-log_mix)
    if ratio>2+1e-12:raise ArithmeticError('MIS pointwise2xweight bound violated')
    return weight,ratio


class Probe(TrajectoryMCCFR):
    def __init__(self):
        super().__init__();self.owner_postflop_active23=0;self.max_weight_ratio=1.
    def step(self,*args,**kwargs):
        raise RuntimeError('read-only probe; training step disabled')
    def record_average_weight(self,state,weight):
        o=state.game.observe()
        self.owner_postflop_active23+=bool(o.street and sum(o.active) in (2,3))
        super().record_average_weight(state,weight)


class BalancedAverage(Probe):
    def trajectory_average(self,state,owner,weight=1.):
        mode=self.rng.choices((0,1),(.5,.5))[0]
        self.mixture_average(state,owner,mode,weight,0.,0.)

    def mixture_average(self,state,owner,mode,scale,log_a,log_b):
        weight,ratio=balance_weight(log_a,log_b,scale)
        self.max_weight_ratio=max(self.max_weight_ratio,ratio)
        self.visit()
        if state.terminal():return
        self.record_visit(state,'average',owner,sampling='sampled')
        actor,key,actions=state.info();strategy=self.strategy(key,actions)
        if actor==owner:
            self.record_average_weight(state,weight)
            row=self.sums.setdefault(key,[0.]*len(actions))
            for i,p in enumerate(strategy):row[i]+=weight*p
            action=self.rng.choices(actions,strategy)[0]
            self.mixture_average(state.child(action),owner,mode,scale,log_a,log_b)
        else:
            qa=[.5/len(actions)+(.5 if a=='call' else 0.) for a in actions] if 'call' in actions else [1/len(actions)]*len(actions)
            qb=profile_strategy(state.game.observe(),'guarded-v2')
            index=self.rng.choices(range(len(actions)),qa if mode==0 else qb)[0]
            self.mixture_average(state.child(actions[index]),owner,mode,scale,
                log_a+math.log(qa[index]),log_b+math.log(qb[index]) if qb[index]>0 else -math.inf)


def verify(manifest):
    for path,sha in manifest['frozen'].items():
        if digest(Path(path))!=sha:raise ValueError('frozen MIS probe input/code/criteria changed')


def main(output,deadline):
    if not time.time()<deadline<=time.time()+3600:raise ValueError('finite probe deadline within1h required')
    if available_memory()<5*1024**3 or shutil.disk_usage(output.parent).free<10*1024**3:raise RuntimeError('probe memory/disk preflight reserve')
    with run_lock(output):
        if output.exists() or output.with_suffix('.manifest.json').exists():raise ValueError('existing probe; investigate rather than repeat')
        files=[*PARENTS,*Path('poker').glob('*.py'),Path(__file__),CRITERIA,Path('scripts/sustained_train.py'),Path('scripts/scaling_validation.py'),Path('scripts/vector_variance.py')]
        manifest=dict(proposal=PROPOSAL,deadline=deadline,deal_seeds=DEALS,replicates=REPLICATES,
            criteria=str(CRITERIA),frozen={str(p):digest(p) for p in files},read_only=True)
        atomic_save(output.with_suffix('.manifest.json'),manifest)
        started=time.monotonic();status='failed';error=None;results=[]
        previous={s:signal.getsignal(s) for s in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT)}
        def interrupted(signum,frame):raise TimeoutError('probe deadline/interruption')
        for sig in previous:signal.signal(sig,interrupted)
        signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.time()))
        minimum_available=available_memory()
        try:
            for i,parent in enumerate(PARENTS):
                saved=load_checkpoint(parent)
                for deal in DEALS:
                    fixture=dict(candidate=i,deal_seed=deal,variants={});state=Holdem(deal,6,saved['abstraction']);owner=state.info()[0]
                    fixture['owner']=owner
                    for name,implementation in [('baseline',Probe),('mixture',BalancedAverage)]:
                        solver=implementation();solver.regrets=saved['regrets'];norms=[];nodes=[];visits=[];totals={};max_weight=0.;max_ratio=0.;variance_estimator=VectorVariance();wall_start=time.monotonic()
                        for trial in range(REPLICATES):
                            memory=available_memory();minimum_available=min(minimum_available,memory)
                            if memory<2*1024**3 or shutil.disk_usage(output.parent).free<10*1024**3:raise RuntimeError('probe host reserve breached')
                            if time.time()>=deadline:raise TimeoutError('probe deadline reached')
                            solver.rng=Random(deal+trial*100+i);solver.nodes=0;solver.node_limit=500000;solver.deadline=time.monotonic()+deadline-time.time()
                            solver.sums={};solver.average_weight_moments={};solver.diagnostics.clear();solver.owner_postflop_active23=0;solver.max_weight_ratio=1.
                            solver.trajectory_average(state,owner)
                            variance_estimator.add(solver.sums)
                            norms.append(math.fsum(x*x for row in solver.sums.values() for x in row));nodes.append(solver.nodes);visits.append(solver.owner_postflop_active23)
                            max_ratio=max(max_ratio,solver.max_weight_ratio)
                            max_weight=max(max_weight,max((v['maximum'] for v in solver.average_weight_moments.values()),default=0.))
                            for key,row in solver.sums.items():
                                dest=totals.setdefault(key,[0.]*len(row))
                                for j,x in enumerate(row):dest[j]+=x
                        wall=time.monotonic()-wall_start;normsum=math.fsum(norms);sumnorm=math.fsum(x*x for row in totals.values() for x in row)
                        variance=variance_estimator.trace()
                        fixture['variants'][name]=dict(variance_diagnostic_schema=SCHEMA,variance_trace=variance,variance_times_nodes=variance*statistics.mean(nodes),variance_times_seconds=variance*wall/REPLICATES,
                            wall_seconds=wall,mean_nodes=statistics.mean(nodes),max_nodes=max(nodes),postflop_active23_visits=sum(visits),postflop_active23_per_node=sum(visits)/sum(nodes),
                            max_average_weight=max_weight,max_pointwise_weight_ratio=max_ratio,sum_squared_norms=normsum,squared_norm_of_vector_sum=sumnorm,
                            replicate_squared_norms=norms,replicate_nodes=nodes,replicate_postflop_active23_visits=visits,
                            summed_vector_by_key_hash={hashlib.sha256(key.encode()).hexdigest():row for key,row in totals.items()})
                        atomic_save(output.with_suffix('.progress.json'),dict(status='running',candidate=i,deal_seed=deal,variant=name,completed_fixtures=results))
                    base=fixture['variants']['baseline'];mix=fixture['variants']['mixture']
                    fixture['variance_work_ratio']=mix['variance_times_nodes']/base['variance_times_nodes'] if base['variance_times_nodes'] else None
                    fixture['variance_time_ratio']=mix['variance_times_seconds']/base['variance_times_seconds'] if base['variance_times_seconds'] else None
                    fixture['visit_rate_gain']=mix['postflop_active23_per_node']/base['postflop_active23_per_node'] if base['postflop_active23_per_node'] else None
                    results.append(fixture)
                del saved,solver
            verify(manifest)
            variance_pass=sum(f['variance_work_ratio'] is not None and f['variance_work_ratio']<.75 and f['variance_time_ratio']<1 for f in results)>=3
            coverage_pass=sum(f['visit_rate_gain'] is not None and f['visit_rate_gain']>=2 for f in results)>=3
            bounds_pass=all(v['max_nodes']<=10000 and v['max_pointwise_weight_ratio']<=2+1e-12 for f in results for v in f['variants'].values())
            atomic_save(output,dict(status='completed',proposal=PROPOSAL,read_only=True,fixtures=results,variance_pass=variance_pass,coverage_pass=coverage_pass,bounds_pass=bounds_pass,
                eligible_for_further_pilot_investigation=variance_pass and coverage_pass and bounds_pass,
                caveat='Conditional engineering only;128replicates can miss rare tails. No learning/payoff/generalization evidence. Zero owner reach remains zero. All source checkpoints unchanged.'))
            status='completed'
        except BaseException as exc:error=str(exc) or type(exc).__name__;raise
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
            for sig,handler in previous.items():signal.signal(sig,handler)
            fields={}
            for line in Path('/proc/self/status').read_text().splitlines():
                if line.startswith(('VmPeak:','VmHWM:')):
                    key,value,*_=line.split();fields[key.rstrip(':')]=int(value)*1024
            atomic_save(output.with_suffix('.metrics.json'),dict(status=status,error=error,pid=os.getpid(),wall_seconds=time.monotonic()-started,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,address_space_limit_bytes=resource.getrlimit(resource.RLIMIT_AS)[0],minimum_available_bytes=minimum_available,**fields))


if __name__=='__main__':
    from datetime import datetime
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--deadline',required=True)
    a=parser.parse_args();deadline=datetime.fromisoformat(a.deadline)
    if deadline.tzinfo is None:raise ValueError('deadline needs timezone')
    resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3));os.nice(10)
    main(a.output,deadline.timestamp())
