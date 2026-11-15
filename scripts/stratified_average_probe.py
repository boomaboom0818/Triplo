"""Fixed all-role engineering test of deterministic stratified averaging v4."""
import argparse,hashlib,json,math,os,resource,shutil,signal,statistics,time
from pathlib import Path
from random import Random
from poker.cfr import Holdem,information_key
from poker.config import load_checkpoint
from poker.trajectory import TrajectoryMCCFR
from poker.stratified_average import StratifiedMCCFR
from poker.train import atomic_save
from poker.train_cfr import run_lock
from scripts.sustained_train import digest
from scripts.scaling_validation import available_memory
from scripts.vector_variance import VectorVariance,SCHEMA

ROOT=Path('runs/bot/stratified-averaging-probe-001')
PARENTS=[Path(f'runs/bot/coverage-curriculum-001/candidate-{i}-plus-300.json') for i in range(2)]
CRITERIA=Path('docs/STRATIFIED-AVERAGING-PILOT.md')
DEALS=list(range(10500000,10500008))
REPLICATES=32


class ObservationMetrics:
    def setup(self,saved):self.saved=saved;self.groups={};self.new_keys={};self.max_ratio=1.
    def record_average_weight(self,state,weight):
        o=state.game.observe();key=information_key(o,self.saved['abstraction'])
        label=f'owner={o.player}:street={o.street}:active={sum(o.active)}'
        group=self.groups.setdefault(label,dict(updates=0,weight_total=0.,weight_squared_total=0.,max_weight=0.))
        group['updates']+=1;group['weight_total']+=weight;group['weight_squared_total']+=weight*weight;group['max_weight']=max(group['max_weight'],weight)
        if not any(x>0 for x in self.saved['sums'].get(key,[])):self.new_keys.setdefault(label,set()).add(hashlib.sha256(key.encode()).hexdigest())
        super().record_average_weight(state,weight)


class Baseline(ObservationMetrics,TrajectoryMCCFR):pass


class Stratified(ObservationMetrics,StratifiedMCCFR):
    def record_balance_ratio(self,ratio):self.max_ratio=max(self.max_ratio,ratio);super().record_balance_ratio(ratio)


def batch(solver,state,owner,stratified):
    for index in range(8):
        if stratified:solver.average_path(state,owner,0 if index<6 else 1,1/8)
        else:solver.trajectory_average(state,owner,1/8)


def verify(manifest):
    for path,sha in manifest['frozen'].items():
        if digest(Path(path))!=sha:raise ValueError('frozen stratified probe input/source/criteria changed')


def summarize(solver):
    updates=0;keys=set();owners=set();streets=set()
    for label,row in solver.groups.items():
        parts=dict(t.split('=',1) for t in label.split(':'))
        if parts['street']!='0' and parts['active'] in ('2','3'):
            updates+=row['updates'];keys.update(solver.new_keys.get(label,set()));owners.add(int(parts['owner']));streets.add(int(parts['street']))
    return dict(groups=solver.groups,new_positive_parent_missing_keys={k:sorted(v) for k,v in solver.new_keys.items()},active23_postflop_updates=updates,
        active23_new_positive_keys=len(keys),active23_owners=sorted(owners),active23_streets=sorted(streets),max_pointwise_weight_ratio=solver.max_ratio)


def main(candidate,deadline,*,implementation=None,root=None,criteria=None,additional_sources=()):
    if candidate not in (0,1) or not time.time()<deadline<=time.time()+2*3600:raise ValueError('candidate0/1 and finite deadline within2h required')
    root=ROOT if root is None else Path(root);criteria=CRITERIA if criteria is None else Path(criteria)
    root.mkdir(parents=True,exist_ok=True);output=root/f'candidate-{candidate}.json';parent=PARENTS[candidate]
    with run_lock(output):
        if output.exists() or output.with_suffix('.manifest.json').exists():raise ValueError('existing probe; investigate, never repeat unchanged')
        if available_memory()<5*1024**3 or shutil.disk_usage(root).free<10*1024**3:raise RuntimeError('stratified probe preflight reserve')
        files=[parent,*Path('poker').glob('*.py'),Path(__file__),criteria,Path('scripts/vector_variance.py'),Path('scripts/sustained_train.py'),Path('scripts/scaling_validation.py'),*additional_sources]
        manifest=dict(candidate=candidate,deadline=deadline,deals=DEALS,replicates=REPLICATES,samples=8,components=[6,2],read_only=True,
            frozen={str(p):digest(p) for p in files});atomic_save(output.with_suffix('.manifest.json'),manifest)
        started=time.monotonic();status='failed';error=None;minimum_available=available_memory()
        previous={s:signal.getsignal(s) for s in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT)}
        def interrupted(signum,frame):raise TimeoutError('stratified probe deadline/interruption')
        for sig in previous:signal.signal(sig,interrupted)
        signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.time()))
        try:
            saved=load_checkpoint(parent);solvers={'baseline':Baseline(),'stratified':(Stratified if implementation is None else implementation)()}
            for s in solvers.values():s.regrets=saved['regrets'];s.setup(saved)
            fixtures=[];total_nodes={name:0 for name in solvers}
            for owner in range(6):
                for n,deal in enumerate(DEALS):
                    fixture=dict(owner=owner,deal_seed=deal,variants={});state=Holdem(deal,6,saved['abstraction'])
                    for name,solver in solvers.items():
                        estimator=VectorVariance();nodes=[];start=time.monotonic()
                        for trial in range(REPLICATES):
                            memory=available_memory();minimum_available=min(minimum_available,memory)
                            if memory<2*1024**3 or shutil.disk_usage(root).free<10*1024**3:raise RuntimeError('stratified probe host reserve breached')
                            if time.time()>=deadline:raise TimeoutError('stratified probe deadline reached')
                            solver.rng=Random(10400000+candidate*1000000+owner*10000+n*100+trial);solver.sums={};solver.nodes=0;solver.node_limit=500000
                            solver.deadline=time.monotonic()+deadline-time.time();batch(solver,state,owner,name=='stratified');estimator.add(solver.sums);nodes.append(solver.nodes)
                        wall=time.monotonic()-start;var=estimator.trace();mean_nodes=statistics.mean(nodes)
                        fixture['variants'][name]=dict(variance_diagnostic_schema=SCHEMA,variance_trace=var,mean_nodes=mean_nodes,max_nodes=max(nodes),replicate_nodes=nodes,wall_seconds=wall,
                            variance_times_nodes=var*mean_nodes,variance_times_seconds=var*wall/REPLICATES)
                        total_nodes[name]+=sum(nodes)
                    fixtures.append(fixture)
                    atomic_save(output.with_suffix('.progress.json'),dict(time=time.time(),owner=owner,deal_index=n,completed_fixtures=len(fixtures),requested_fixtures=6*len(DEALS)))
            verify(manifest);variants={name:summarize(s) for name,s in solvers.items()}
            aggregates={}
            for name in solvers:
                aggregates[name]=dict(variance_times_nodes=math.fsum(f['variants'][name]['variance_times_nodes'] for f in fixtures),
                    variance_times_seconds=math.fsum(f['variants'][name]['variance_times_seconds'] for f in fixtures),total_nodes=total_nodes[name],
                    active23_updates_per_node=variants[name]['active23_postflop_updates']/total_nodes[name])
            a=aggregates['baseline'];b=aggregates['stratified']
            ratio=lambda field:b[field]/a[field] if a[field]>0 else None
            work=ratio('variance_times_nodes');wall=ratio('variance_times_seconds');visit=ratio('active23_updates_per_node');v=variants['stratified']
            gates=dict(variance_work_pass=work is not None and work<=1.5,variance_time_pass=wall is not None and wall<=2,
                coverage_rate_pass=visit is not None and visit>=3,new_keys_roles_streets_pass=v['active23_new_positive_keys']>=20 and v['active23_owners']==list(range(6)) and v['active23_streets']==[1,2,3],
                bounds_pass=v['max_pointwise_weight_ratio']<=4/3+1e-12 and all(z['max_nodes']<=10000 for f in fixtures for z in f['variants'].values()))
            atomic_save(output,dict(status='completed',candidate=candidate,read_only=True,fixtures=fixtures,variants=variants,aggregates=aggregates,
                variance_work_ratio=work,variance_time_ratio=wall,visit_rate_gain=visit,gates=gates,eligible_for_learning_pilot=all(gates.values()),
                caveat='Fixed conditional engineering observations, not payoff evidence or independent ESS. Old randomized50/50MIS failure remains unchanged. Zero own-target reach is not repaired. All source learning bytes unchanged.'))
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
    p=argparse.ArgumentParser();p.add_argument('--candidate',type=int,choices=(0,1),required=True);p.add_argument('--deadline',required=True)
    a=p.parse_args();d=datetime.fromisoformat(a.deadline)
    if d.tzinfo is None:raise ValueError('deadline needs timezone')
    resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3));os.nice(10);main(a.candidate,d.timestamp())
