"""Read-only all-role isolation of mean-policy coverage, support and averaging."""
import argparse,hashlib,json,math,os,resource,shutil,signal,statistics,time
from pathlib import Path
from collections import Counter
from random import Random
from poker.cfr import CFRPolicy,Holdem,information_key,normalize
from poker.game import play
from poker.policy import Policy
from poker.trajectory import TrajectoryMCCFR
from poker.config import load_checkpoint
from poker.train import atomic_save
from poker.train_cfr import run_lock
from scripts.sustained_train import digest
from scripts.scaling_validation import available_memory
from scripts.vector_variance import VectorVariance,SCHEMA

ROOT=Path('runs/bot/averaging-isolation-001')
PARENTS=[Path(f'runs/bot/coverage-curriculum-001/candidate-{i}-plus-300.json') for i in range(2)]
CRITERIA=Path('docs/SIX-ROLE-AVERAGING-ISOLATION.md')
NATURAL_DEALS=list(range(10000000,10000064))
CONTEXT_DEALS=list(range(10010000,10010004))
ROOT_DEALS=list(range(10020000,10020008))
REPLICATES=32
POPULATIONS=('calling','tight','mixed','self-average')


def classify(saved,o):
    key=information_key(o,saved['abstraction']);row=saved['sums'].get(key);reg=saved['regrets'].get(key)
    return key,'positive' if row and any(x>0 for x in row) else 'missing' if row is None else 'zero',bool(reg and any(x>0 for x in reg)),normalize(reg if reg is not None else [0.]*len(o.legal))


def observe_group(groups,label,saved,o,log_reach):
    key,source,regret,strategy=classify(saved,o)
    name=f'{label}:owner={o.player}:position={o.position}:street={o.street}:active={sum(o.active)}'
    row=groups.setdefault(name,Counter());row['decisions']+=1;row[source+'_average']+=1;row['positive_regret']+=regret
    row['zero_current_reach']+=log_reach==-math.inf
    row['tiny_current_reach']+=log_reach!=-math.inf and log_reach<math.log(1e-6)
    if source!='positive':
        row['average_miss']+=1;row['average_miss_positive_regret']+=regret
        row['average_miss_zero_current_reach']+=log_reach==-math.inf
        row['average_miss_positive_regret_zero_current_reach']+=regret and log_reach==-math.inf
    return key,source,regret,strategy


def advance_reach(log_reach,probability):
    return log_reach+math.log(probability) if probability>0 else -math.inf


class ReachPolicy(CFRPolicy):
    def __init__(self,saved,seed,groups,label):
        super().__init__(saved['sums'],seed,saved['abstraction']);self.saved=saved;self.groups=groups;self.label=label;self.log_reach=0.
    def act(self,o):
        _,_,_,strategy=observe_group(self.groups,self.label,self.saved,o,self.log_reach)
        action=super().act(o);self.log_reach=advance_reach(self.log_reach,strategy[o.legal.index(action)])
        return action


def policies(saved,population,owner,seed,groups=None):
    result=[]
    for player in range(6):
        if player==owner:
            result.append(CFRPolicy(saved['sums'],seed+player,saved['abstraction']) if groups is None else ReachPolicy(saved,seed+player,groups,'population='+population))
        elif population=='self-average':result.append(CFRPolicy(saved['sums'],seed+player,saved['abstraction']))
        else:result.append(Policy(seed+player,{},['calling','tight','aggressive'][player%3] if population=='mixed' else population))
    return result


def context(saved,deal,owner,street,active):
    state=Holdem(deal,6,saved['abstraction']);keep={(owner+j)%6 for j in range(active)};raised=False;log_reach=0.
    for _ in range(300):
        if state.terminal():raise ValueError('controlled context terminated before requested owner/street')
        o=state.game.observe()
        if o.street==street and o.player==owner:
            if sum(o.active)!=active:raise ValueError('controlled context active count mismatch')
            return state,log_reach
        if o.street>street:raise ValueError('controlled context skipped requested owner')
        action='call'
        if o.street==0:
            if o.player not in keep and 'fold' in o.legal:action='fold'
            elif active<6 and o.player in keep and not raised:
                raises=[a for a in o.legal if isinstance(a,int)]
                if raises:action=min(raises);raised=True
        if o.player==owner:
            strategy=classify(saved,o)[3];log_reach=advance_reach(log_reach,strategy[o.legal.index(action)])
        state.game.act(action)
    raise ValueError('controlled context action budget reached')


class ObservedAverage(TrajectoryMCCFR):
    def __init__(self,saved,owner,groups):
        super().__init__();self.regrets=saved['regrets'];self.saved=saved;self.owner=owner;self.groups=groups;self.keys={}
    def record_average_weight(self,state,weight):
        o=state.game.observe();key,source,_,_=classify(self.saved,o)
        label=f'owner={self.owner}:position={o.position}:street={o.street}:active={sum(o.active)}'
        row=self.groups.setdefault(label,dict(updates=0,positive_parent_average=0,new_positive_average_updates=0,total_weight=0.,squared_total_weight=0.,max_weight=0.))
        row['updates']+=1;row['positive_parent_average']+=source=='positive';row['new_positive_average_updates']+=source!='positive'
        row['total_weight']+=weight;row['squared_total_weight']+=weight*weight;row['max_weight']=max(row['max_weight'],weight)
        self.keys.setdefault(label,set()).add(key)
        super().record_average_weight(state,weight)


def verify(manifest):
    for path,sha in manifest['frozen'].items():
        if digest(Path(path))!=sha:raise ValueError('frozen six-role diagnostic input/code/criteria changed')


def main(candidate,deadline):
    if candidate not in (0,1) or not time.time()<deadline<=time.time()+2*3600:raise ValueError('candidate0/1 and finite deadline within2h required')
    ROOT.mkdir(parents=True,exist_ok=True);output=ROOT/f'candidate-{candidate}.json';parent=PARENTS[candidate]
    with run_lock(output):
        if output.exists() or output.with_suffix('.manifest.json').exists():raise ValueError('existing diagnostic; investigate rather than repeat')
        if available_memory()<5*1024**3 or shutil.disk_usage(ROOT).free<10*1024**3:raise RuntimeError('diagnostic preflight reserve')
        files=[parent,*Path('poker').glob('*.py'),Path(__file__),CRITERIA,Path('scripts/vector_variance.py'),Path('scripts/sustained_train.py'),Path('scripts/scaling_validation.py')]
        manifest=dict(candidate=candidate,deadline=deadline,natural_deals=NATURAL_DEALS,context_deals=CONTEXT_DEALS,root_deals=ROOT_DEALS,replicates=REPLICATES,populations=POPULATIONS,
            frozen={str(p):digest(p) for p in files},read_only=True)
        atomic_save(output.with_suffix('.manifest.json'),manifest)
        started=time.monotonic();status='failed';error=None;minimum_available=available_memory()
        previous={sig:signal.getsignal(sig) for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT)}
        def interrupted(signum,frame):raise TimeoutError('six-role diagnostic deadline/interruption')
        for sig in previous:signal.signal(sig,interrupted)
        signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.time()))
        def guard():
            nonlocal minimum_available
            memory=available_memory();minimum_available=min(minimum_available,memory)
            if memory<2*1024**3 or shutil.disk_usage(ROOT).free<10*1024**3:raise RuntimeError('diagnostic host reserve breached')
            if time.time()>=deadline:raise TimeoutError('six-role deadline reached')
        def progress(phase,**fields):atomic_save(output.with_suffix('.progress.json'),dict(phase=phase,time=time.time(),**fields))
        try:
            saved=load_checkpoint(parent);natural={};contexts=[];average={};fixtures=[]
            for population in POPULATIONS:
                for n,deal in enumerate(NATURAL_DEALS):
                    for owner in range(6):
                        guard();play(policies(saved,population,owner,deal*100+owner*6,natural),deal)
                    if n%8==0:progress('natural_replay',population=population,completed_deals=n+1)
            for deal in CONTEXT_DEALS:
                for owner in range(6):
                    for street in (1,2,3):
                        for active in (2,3,6):
                            guard();state,reach=context(saved,deal,owner,street,active);o=state.game.observe();key,source,regret,_=classify(saved,o)
                            contexts.append(dict(deal_seed=deal,owner=owner,position=o.position,street=street,active=active,key_sha256=hashlib.sha256(key.encode()).hexdigest(),average_source=source,positive_regret=regret,
                                zero_current_prefix_support=reach==-math.inf,log_current_prefix_reach=reach if math.isfinite(reach) else None))
                progress('controlled_contexts',completed_contexts=len(contexts))
            for owner in range(6):
                solver=ObservedAverage(saved,owner,average)
                for n,deal in enumerate(ROOT_DEALS):
                    state=Holdem(deal,6,saved['abstraction']);variance=VectorVariance();nodes=[];t=time.monotonic()
                    for trial in range(REPLICATES):
                        guard();solver.rng=Random(10100000+candidate*1000000+owner*10000+n*100+trial);solver.sums={};solver.nodes=0;solver.node_limit=500000
                        solver.deadline=time.monotonic()+deadline-time.time();solver.trajectory_average(state,owner);variance.add(solver.sums);nodes.append(solver.nodes)
                    fixtures.append(dict(owner=owner,deal_seed=deal,replicates=REPLICATES,variance_diagnostic_schema=SCHEMA,variance_trace=variance.trace(),mean_nodes=statistics.mean(nodes),max_nodes=max(nodes),replicate_nodes=nodes,wall_seconds=time.monotonic()-t))
                for label,keys in solver.keys.items():average[label]['unique_keys']=len(keys)
                progress('root_averaging',completed_owners=owner+1)
            verify(manifest)
            for row in average.values():row['weight_concentration_proxy']=row['total_weight']**2/row['squared_total_weight'] if row['squared_total_weight'] else None
            atomic_save(output,dict(status='completed',candidate=candidate,read_only=True,source_iterations=saved['iterations'],natural_groups=natural,controlled_contexts=contexts,root_averaging_groups=average,root_averaging_fixtures=fixtures,
                caveat='Isolation only. Current-zero prefix support is not floating underflow. Forced contexts are conditional, not training samples. No efficacy payoffs or independent ESS. Regrets, averages, archive and RNG checkpoint bytes unchanged.'))
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
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',type=int,choices=(0,1),required=True);parser.add_argument('--deadline',required=True)
    a=parser.parse_args();deadline=datetime.fromisoformat(a.deadline)
    if deadline.tzinfo is None:raise ValueError('deadline needs timezone')
    resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3));os.nice(10);main(a.candidate,deadline.timestamp())
