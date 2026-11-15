"""Read-only inference component; never relabels checkpoints or average coverage."""
import argparse,json,os,resource,signal,time
from pathlib import Path
from collections import Counter
from poker.cfr import CFRPolicy,information_key,normalize
from poker import evaluate as evaluator
from poker.train import atomic_save
from poker.train_cfr import run_lock
from scripts.sustained_train import digest,validate_evaluation
from scripts.scaling_validation import available_memory

VARIANTS={'average':'average-v1','backoff':'average-positive-regret-backoff-v1','uniform':'uniform-control-v1'}


class BackoffPolicy(CFRPolicy):
    def __init__(self,sums,regrets,seed,abstraction,*,use_regrets,sink):
        super().__init__(sums,seed,abstraction);self.regrets=regrets;self.use_regrets=use_regrets;self.sink=sink
    def act(self,o):
        key=information_key(o,self.abstraction);average=self.sums.get(key)
        if average is not None and len(average)!=len(o.legal):raise ValueError('average action count mismatch')
        if average and any(x>0 for x in average):weights=average;source='average'
        else:
            regret=self.regrets.get(key) if self.use_regrets else None
            if regret is not None and len(regret)!=len(o.legal):raise ValueError('regret action count mismatch')
            if regret and any(x>0 for x in regret):weights=regret;source='positive_regret'
            else:weights=[0.]*len(o.legal);source='uniform'
        self.sink[f'street={o.street}:active={sum(o.active)}:source={source}']+=1
        return self.rng.choices(o.legal,normalize(weights))[0]


def evaluate_component(checkpoint,opponent,blocks,seed,variant):
    """Inject only candidate construction in this process; restore every global."""
    if variant not in VARIANTS:raise ValueError('unsupported candidate variant')
    original_load=evaluator.load_checkpoint;original_policy=evaluator.CFRPolicy
    target=checkpoint.resolve();refs={'opponent_ids':set()};sinks=[Counter() for _ in range(6)]
    counts={'candidate_instances':0,'heldout_instances':0}
    input_hashes={str(p.resolve()):digest(p) for p in (checkpoint,opponent)}
    def load(path,**kwargs):
        saved=original_load(path,**kwargs)
        if Path(path).resolve()==target:refs.update(sums=saved['sums'],regrets=saved['regrets'])
        else:refs['opponent_ids'].add(id(saved['sums']))
        return saved
    def policy(sums,policy_seed,abstraction):
        if id(sums) in refs['opponent_ids']:
            counts['heldout_instances']+=1
            return original_policy(sums,policy_seed,abstraction)
        if not (sums is refs.get('sums') or variant=='uniform' and not sums):raise ValueError('unexpected candidate table identity')
        index=counts['candidate_instances']//(6*blocks)
        if index>=6:raise ValueError('unexpected population/seat count')
        counts['candidate_instances']+=1
        return BackoffPolicy(sums,refs['regrets'],policy_seed,abstraction,use_regrets=variant=='backoff',sink=sinks[index])
    evaluator.load_checkpoint=load;evaluator.CFRPolicy=policy
    try:
        report=evaluator.evaluate(checkpoint,blocks,seed,1,opponent_checkpoints=[opponent],uniform_control=variant=='uniform')
    finally:
        evaluator.load_checkpoint=original_load;evaluator.CFRPolicy=original_policy
    if counts!={'candidate_instances':36*blocks,'heldout_instances':30*blocks}:raise ValueError('candidate/heldout injection count mismatch')
    if len(report['populations'])!=6:raise ValueError('unexpected evaluation populations')
    sources={}
    for (population,row),sink in zip(report['populations'].items(),sinks):
        n=sum(sink.values())
        if n!=row['candidate_decisions']:raise ValueError('source attribution count mismatch')
        sources[population]=dict(sink)
        row['average_missing_or_zero_fraction']=row['uniform_fallback_fraction']
        row['uniform_fallback_fraction']=sum(v for k,v in sink.items() if k.endswith('source=uniform'))/n if n else None
        row['regret_backoff_fraction']=sum(v for k,v in sink.items() if k.endswith('source=positive_regret'))/n if n else None
    for path,sha in input_hashes.items():
        if digest(Path(path))!=sha:raise ValueError('frozen component input changed')
    validate_evaluation(report,checkpoint,opponent,seed,variant=='uniform',blocks)
    report.update(candidate_policy_variant=VARIANTS[variant],inference_component_only=True,policy_sources_by_population=sources,
        policy_construction_counts=counts,raw_coverage_scope='average table only; positive-regret backoff is separately attributed and does not pass an average-coverage gate',
        component_source_sha256=digest(Path(__file__)),input_hashes=input_hashes)
    return report


def main(checkpoint,opponent,output,blocks,seed,variant,deadline):
    if not time.time()<deadline<=time.time()+12*3600:raise ValueError('finite component deadline required')
    if available_memory()<5*1024**3:raise RuntimeError('3GiB worker plus2GiB host reserve required')
    with run_lock(output):
        if output.exists():raise ValueError('component report exists; preserve rather than overwrite')
        started=time.monotonic();status='failed';error=None
        signals=(signal.SIGALRM,signal.SIGTERM,signal.SIGINT)
        previous={s:signal.getsignal(s) for s in signals}
        def timeout(signum,frame):raise TimeoutError('component evaluation deadline reached')
        def interrupted(signum,frame):raise InterruptedError('component evaluation interrupted')
        for sig in signals:signal.signal(sig,timeout if sig==signal.SIGALRM else interrupted)
        signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.time()))
        try:
            report=evaluate_component(checkpoint,opponent,blocks,seed,variant)
            atomic_save(output,report);status='completed'
        except BaseException as exc:error=str(exc) or type(exc).__name__;raise
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
            for sig,handler in previous.items():signal.signal(sig,handler)
            fields={}
            for line in Path('/proc/self/status').read_text().splitlines():
                if line.startswith(('VmPeak:','VmHWM:')):
                    key,value,*_=line.split();fields[key.rstrip(':')]=int(value)*1024
            metrics=Path(str(output)+'.metrics.json')
            row=dict(status=status,error=error,module=__name__,variant=variant,pid=os.getpid(),wall_seconds=time.monotonic()-started,
                cpu_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,address_space_limit_bytes=resource.getrlimit(resource.RLIMIT_AS)[0],**fields)
            old=json.loads(metrics.read_text()) if metrics.exists() else {};attempts=old.get('attempts',[]);attempts.append(dict(row));row['attempts']=attempts
            for key in ('peak_rss_bytes','VmPeak','VmHWM'):row[key]=max(a.get(key,0) for a in attempts)
            atomic_save(metrics,row)

if __name__=='__main__':
    from datetime import datetime
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--opponent',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--variant',choices=VARIANTS,required=True)
    p.add_argument('--blocks',type=int,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--deadline',required=True)
    a=p.parse_args();d=datetime.fromisoformat(a.deadline)
    if d.tzinfo is None:raise ValueError('deadline needs timezone')
    resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3));os.nice(10)
    main(a.checkpoint,a.opponent,a.output,a.blocks,a.seed,a.variant,d.timestamp())
