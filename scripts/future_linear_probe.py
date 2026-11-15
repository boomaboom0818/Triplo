"""Read-only paired all-role engineering of one prospective averaging clock."""
import argparse,hashlib,json,math,os,resource,signal,statistics,time
from pathlib import Path
from random import Random
from datetime import datetime
from poker.config import load_checkpoint
from poker.cfr import Holdem
from poker.joint_average import JointAverageMCCFR
from poker.future_linear_average import FutureLinearAverageMCCFR
from poker.train import atomic_save
from poker.train_cfr import run_lock
from scripts.scaling_validation import available_memory
from scripts.sustained_train import digest
import shutil

ROOT=Path('runs/bot/future-linear-probe-002')
PARENTS=[Path(f'runs/bot/joint-average-training-001/candidate-{i}-plus-2000.json') for i in range(2)]
CRITERIA=Path('docs/FUTURE-LINEAR-AVERAGING.md')
DEALS=list(range(11300000,11300004))
CLOCKS=[1,50,2000]


class ObserveWeight:
    def record_average_weight(self,state,weight):
        self.max_ratio=max(getattr(self,'max_ratio',0.),self.average_update_ratio)
        super().record_average_weight(state,weight)


class Baseline(ObserveWeight,JointAverageMCCFR):pass
class Candidate(ObserveWeight,FutureLinearAverageMCCFR):pass


def maximum_average_weight(solver):
    # Lifetime average moments use street= keys; epoch moments use average:.
    rows=solver.average_weight_moments
    if any(not key.startswith('street=') for key in rows):raise ValueError('unexpected average moment schema')
    maximum=max((row['maximum'] for row in rows.values()),default=0.)
    if bool(maximum>0)!=any(any(v>0 for v in row) for row in solver.sums.values()):
        raise ValueError('average peak instrumentation disagrees with actual contributions')
    return maximum


def selected(saved,seed):
    rng=Random(seed);cfg=saved['config'];archive=saved['archive'];result={}
    if archive:
        for player in range(6):
            if rng.random()<cfg['population_probability']:result[player]=rng.choice(archive)
    if rng.random()<.25:result={p:'guarded-v2' for p in range(6)}
    return result


def batch(solver,state,owner):
    delta={}
    for _ in range(8):
        solver.rng.randrange(2**63);solver.outcome(state,owner,delta,scale=1/8)
        solver.rng.randrange(2**63);solver.trajectory_average(state,owner,1/8)
    return delta


def main(candidate,deadline):
    end=datetime.fromisoformat(deadline)
    if end.tzinfo is None or not time.time()<end.timestamp()<=time.time()+2*3600:
        raise ValueError('finite probe deadline within2h required')
    ROOT.mkdir(parents=True,exist_ok=True)
    resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3));os.nice(10)
    def stop(signum,frame):raise TimeoutError('future linear engineering deadline')
    signal.signal(signal.SIGALRM,stop);signal.alarm(math.ceil(end.timestamp()-time.time()))
    with run_lock(ROOT/f'candidate-{candidate}'):
        out=ROOT/f'candidate-{candidate}.json';manifest_path=ROOT/f'candidate-{candidate}.manifest.json'
        if out.exists() or manifest_path.exists():raise ValueError('existing probe; no unchanged retry')
        if available_memory()<8*1024**3 or shutil.disk_usage(ROOT).free<10*1024**3:
            raise RuntimeError('probe host reserve preflight')
        files=[PARENTS[candidate],*Path('poker').glob('*.py'),Path(__file__),Path('scripts/scaling_validation.py'),Path('scripts/sustained_train.py'),CRITERIA,ROOT/'protocol.json']
        manifest=dict(candidate=candidate,deadline=deadline,read_only=True,deals=DEALS,clocks=CLOCKS,replicates=4,
            frozen={str(p):digest(p) for p in files});atomic_save(manifest_path,manifest)
        started=time.monotonic();status='failed';error=None;current=None;minimum=available_memory()
        try:
            saved=load_checkpoint(PARENTS[candidate]);assert saved['iterations']==[6906,7150][candidate]
            cases=[];times=[0.,0.];max_nodes=0;max_ratio=0.;max_weights={str(k):0. for k in CLOCKS}
            total_sources={str(k):{} for k in CLOCKS}
            for owner in range(6):
                for index,deal in enumerate(DEALS):
                    state=Holdem(deal,6,saved['abstraction'])
                    for clock in CLOCKS:
                        for trial in range(4):
                            minimum=min(minimum,available_memory())
                            if minimum<2*1024**3 or shutil.disk_usage(ROOT).free<10*1024**3:raise RuntimeError('probe reserve breached')
                            current=dict(owner=owner,deal=deal,clock=clock,replicate=trial)
                            seed=11400000+candidate*1000000+owner*10000+index*100+trial
                            solvers=[Baseline(samples=8,depth=2),Candidate(samples=8,depth=2)]
                            deltas=[];walls=[]
                            for variant,s in enumerate(solvers):
                                s.regrets=saved['regrets'];s.iterations=clock-1;s.rng=Random(seed);s.opponents=selected(saved,seed)
                                s.node_limit=10000;s.deadline=time.monotonic()+min(2.,end.timestamp()-time.time())
                                start=time.monotonic();deltas.append(batch(s,state,owner));wall=time.monotonic()-start
                                if wall>2.:raise RuntimeError('paired full batch exceeded2seconds')
                                walls.append(wall);times[variant]+=wall;max_nodes=max(max_nodes,s.nodes)
                            a,b=solvers
                            assert deltas[0]==deltas[1] and a.rng.getstate()==b.rng.getstate() and a.nodes==b.nodes
                            assert a.sums.keys()==b.sums.keys() and a.diagnostics==b.diagnostics
                            for key,row in a.sums.items():
                                assert len(row)==len(b.sums[key])
                                assert all(math.isfinite(v) and math.isfinite(v*v) for v in b.sums[key])
                                assert all(math.isclose(v*clock,w,rel_tol=1e-12,abs_tol=1e-12) for v,w in zip(row,b.sums[key]))
                            signature=hashlib.sha256(json.dumps(dict(regrets=deltas[0],rng=a.rng.getstate(),nodes=a.nodes),sort_keys=True).encode()).hexdigest()
                            max_ratio=max(max_ratio,getattr(b,'max_ratio',0.))
                            rawmax=maximum_average_weight(b)
                            max_weights[str(clock)]=max(max_weights[str(clock)],rawmax)
                            for key,value in b.diagnostics.items():
                                if key.startswith('joint-average-v7:source='):
                                    bucket=total_sources[str(clock)];bucket[key]=bucket.get(key,0)+value
                            cases.append(dict(**current,nodes=a.nodes,baseline_seconds=walls[0],candidate_seconds=walls[1],regret_rng_node_sha256=signature,
                                means_exact_k_times_baseline=True,source_counts_exact=True,max_raw_average_weight=rawmax))
                        atomic_save(ROOT/f'candidate-{candidate}.progress.json',dict(completed_pairs=len(cases),expected_pairs=288,current=current))
            for p,sha in manifest['frozen'].items():
                if digest(Path(p))!=sha:raise ValueError('frozen probe input/source changed')
            assert len(cases)==288
            gates=dict(exact_regret_rng_nodes=True,exact_weighted_target=True,exact_source_counts=True,
                bounded_batch=max_nodes<=10000,normalized_defensive_bound=max_ratio<=1+1e-12,
                time_ratio=times[1]<=2*times[0],finite_weights=all(v>0 and math.isfinite(v*v) for v in max_weights.values()))
            atomic_save(out,dict(candidate=candidate,read_only=True,inputs=manifest['frozen'],cases=cases,gates=gates,
                eligible_for_registration=all(gates.values()),total_seconds=times,time_ratio=times[1]/times[0],
                max_normalized_pointwise_ratio=max_ratio,max_raw_weight_by_clock=max_weights,source_groups_by_clock=total_sources,
                note='k-scaled target; raw variance scales k squared. No variance reduction, payoff, independent ESS or strength claim.'))
            status='completed'
        except BaseException as exc:error=repr(exc);raise
        finally:
            peaks={}
            for line in Path('/proc/self/status').read_text().splitlines():
                if line.startswith(('VmPeak:','VmHWM:')):
                    key,value,*_=line.split();peaks[key.rstrip(':')]=int(value)*1024
            atomic_save(ROOT/f'candidate-{candidate}.metrics.json',dict(pid=os.getpid(),status=status,error=error,last_case=current,
                wall_seconds=time.monotonic()-started,address_space_limit_bytes=resource.getrlimit(resource.RLIMIT_AS)[0],
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,minimum_host_available_bytes=minimum,**peaks))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--candidate',type=int,choices=[0,1],required=True);p.add_argument('--deadline',required=True)
    args=p.parse_args();main(args.candidate,args.deadline)
