"""CPU reference trainer. Hold'em iterations can be expensive; pilot first."""
import argparse
import json
import time
import math
import platform
import signal
import subprocess
import fcntl
import shutil
import copy
import hashlib
from .abstraction import VERSION,VERSION_V2
from contextlib import contextmanager
from pathlib import Path
from .cfr import MCCFR,Kuhn,Holdem,TraversalLimit
from .train import atomic_save,tuples

_STOP_REQUESTED=False


@contextmanager
def run_lock(path):
    lock=Path(str(path)+'.lock');lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a') as handle:
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('another trainer owns this output')
        yield


def train(iterations,path,seed=1,game='kuhn',players=6,resume=None,*,
          checkpoint_every=100,max_iteration_nodes=100000,max_seconds=3600,
          max_information_sets=500000,abstraction="exact-observation-v1",
          population_probability=0.0,archive_every=100,archive_size=4,averaging_sampler="uniform",opponent_population="historical-v1",fork_population=False):
    with run_lock(path):
        return _train(iterations,path,seed,game,players,resume,checkpoint_every,
                      max_iteration_nodes,max_seconds,max_information_sets,abstraction,
                      population_probability,archive_every,archive_size,averaging_sampler,opponent_population,fork_population)


def _train(iterations,path,seed,game,players,resume,checkpoint_every,
           max_iteration_nodes,max_seconds,max_information_sets,abstraction,
           population_probability,archive_every,archive_size,averaging_sampler,opponent_population,fork_population):
    if abstraction not in ("exact-observation-v1",VERSION,VERSION_V2):raise ValueError("unsupported abstraction")
    if not 0<=population_probability<=1:raise ValueError("invalid population probability")
    for value in (checkpoint_every,max_iteration_nodes,max_information_sets,archive_every,archive_size):
        if type(value) is not int or value<1:raise ValueError('limits must be positive integers')
    if not math.isfinite(max_seconds) or max_seconds<=0:raise ValueError('invalid time budget')
    if iterations<1 or game not in ('kuhn','holdem') or not 3<=players<=6:
        raise ValueError('invalid training configuration')
    config=dict(game=game,players=2 if game=='kuhn' else players,seed=seed,
                abstraction=abstraction,population_probability=population_probability,
                archive_every=archive_every,archive_size=archive_size)
    if Path(path).exists() and (resume is None or Path(path).resolve()!=Path(resume).resolve()):
        raise ValueError('output exists; use resume or a new path')
    if averaging_sampler not in ("uniform","call-mixture-v1"):raise ValueError("unsupported averaging sampler")
    if averaging_sampler!="uniform":config["averaging_sampler"]=averaging_sampler
    if opponent_population not in ('historical-v1','diverse-v1'):raise ValueError('unsupported opponent population')
    if opponent_population!='historical-v1':
        if game!='holdem':raise ValueError('diverse profiles require holdem')
        config['opponent_population']=opponent_population
    if fork_population and (not resume or Path(path).resolve()==Path(resume).resolve()):
        raise ValueError('population fork requires a new output and explicit parent')
    recipe_fork=None
    solver=MCCFR(seed);solver.averaging_sampler=averaging_sampler;seconds=0;archive=[]
    solver.stop_requested=lambda: _STOP_REQUESTED
    if resume:
        saved=json.loads(Path(resume).read_text())
        from .population import population_fork_allowed
        compatible=saved.get('config')==config
        if fork_population:
            compatible=population_fork_allowed(saved.get('config',{}),config)
            if compatible:recipe_fork=dict(parent_sha256=hashlib.sha256(Path(resume).read_bytes()).hexdigest(),parent_config=saved['config'],iterations=saved['iterations'],kind='explicit population-diversity fork')
        else:recipe_fork=saved.get('recipe_fork')
        if (saved.get('schema')!=1 or saved.get('algorithm')!='external-sampling-mccfr-v1'
                or not compatible or saved.get('model_id')!='bot-opponent-v1'):
            raise ValueError('incompatible CFR checkpoint')
        solver.regrets=saved['regrets'];solver.sums=saved['sums'];archive=saved['archive']
        solver.rng.setstate(tuples(saved['rng']));solver.iterations=saved['iterations']
        solver.nodes=saved['nodes'];seconds=saved['seconds']
        solver.diagnostics.update(saved.get('diagnostics',{}))
        solver.average_weight_moments=saved.get('average_weight_moments',{})
    if iterations<=solver.iterations:raise ValueError('target must exceed completed iterations')
    try:revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    except (OSError,subprocess.CalledProcessError):revision='unknown'
    source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(Path(__file__).parent.glob('*.py'))}
    start=time.perf_counter();deadline=time.monotonic()+max_seconds
    stop_reason='running'
    def save():
        result=dict(schema=1,algorithm='external-sampling-mccfr-v1',model_id='bot-opponent-v1',
                    config=config,recipe_fork=recipe_fork,engine='pokerkit-0.7.7',abstraction=abstraction,archive=archive,
                    iterations=solver.iterations,regrets=solver.regrets,sums=solver.sums,
                    rng=solver.rng.getstate(),nodes=solver.nodes,diagnostics=dict(solver.diagnostics),
                    average_weight_moments=solver.average_weight_moments,
                    weight_diagnostics_started_at=saved.get('weight_diagnostics_started_at',saved['iterations']) if resume else 0,
                    diagnostics_started_at=saved.get('diagnostics_started_at',saved['iterations']) if resume else 0,seconds=seconds+time.perf_counter()-start,
                    stop_reason=stop_reason,provenance=dict(revision=revision,python=platform.python_version(),source_sha256=source_hashes),
                    limits=dict(max_iteration_nodes=max_iteration_nodes,max_seconds=max_seconds,
                                max_information_sets=max_information_sets,checkpoint_every=checkpoint_every))
        # Keep the last complete checkpoint as recovery if replacement/disk write fails.
        if Path(path).exists():
            shutil.copyfile(path,str(path)+'.previous.tmp')
            Path(str(path)+'.previous.tmp').replace(str(path)+'.previous')
        atomic_save(path,result)
        record={k:result[k] for k in ('iterations','nodes','seconds','stop_reason')}
        record.update(information_sets=len(set(solver.regrets)|set(solver.sums)),checkpoint_bytes=Path(path).stat().st_size)
        with Path(str(path)+'.progress.jsonl').open('a') as log:log.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
        return result
    factory=Kuhn if game=='kuhn' else lambda seed:Holdem(seed,players,abstraction)
    try:
        for _ in range(solver.iterations,iterations):
            if _STOP_REQUESTED:raise KeyboardInterrupt
            if len(set(solver.regrets)|set(solver.sums))>=max_information_sets:
                stop_reason='information_set_budget';break
            # Snapshot RNG before population selection for exact interrupted resume.
            rng_before=solver.rng.getstate()
            opponents={}
            if archive:
                for player in range(config['players']):
                    if solver.rng.random()<population_probability:
                        opponents[player]=solver.rng.choice(archive)
            if opponent_population=='diverse-v1':
                for player in range(config['players']):
                    if solver.rng.random()<.25:opponents[player]=solver.rng.choice(('passive-v1','pressure-v1'))
            try:solver.step(factory,config['players'],max_iteration_nodes,deadline,opponents)
            except BaseException:
                solver.rng.setstate(rng_before);raise
            if solver.iterations%archive_every==0:
                archive.append(copy.deepcopy(solver.sums));archive=archive[-archive_size:]
            if solver.iterations%checkpoint_every==0 and solver.iterations<iterations:save()
    except TraversalLimit as exc:stop_reason=str(exc)
    except KeyboardInterrupt:stop_reason='interrupted'
    if stop_reason=='running':stop_reason='target_reached'
    return save()


def stop_signal(signum,frame):
    global _STOP_REQUESTED
    _STOP_REQUESTED=True


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--game',choices=['kuhn','holdem'],default='kuhn')
    p.add_argument('--players',type=int,default=6);p.add_argument('--iterations',type=int,default=1000)
    p.add_argument('--seed',type=int,default=1);p.add_argument('--resume')
    p.add_argument('--output',default='runs/bot/cfr.json')
    p.add_argument('--config')
    p.add_argument('--checkpoint-every',type=int,default=100)
    p.add_argument('--max-iteration-nodes',type=int,default=100000)
    p.add_argument('--max-seconds',type=float,default=3600)
    p.add_argument('--max-information-sets',type=int,default=500000)
    p.add_argument('--abstraction',choices=['exact-observation-v1',VERSION,VERSION_V2],default='exact-observation-v1')
    p.add_argument('--population-probability',type=float,default=0.0)
    p.add_argument('--archive-every',type=int,default=100)
    p.add_argument('--archive-size',type=int,default=4)
    p.add_argument('--averaging-sampler',choices=['uniform','call-mixture-v1'],default='uniform')
    p.add_argument('--opponent-population',choices=['historical-v1','diverse-v1'],default='historical-v1')
    p.add_argument('--fork-population',action='store_true')
    a=p.parse_args()
    settings=dict(iterations=a.iterations,seed=a.seed,game=a.game,players=a.players,
                  checkpoint_every=a.checkpoint_every,max_iteration_nodes=a.max_iteration_nodes,
                  max_seconds=a.max_seconds,max_information_sets=a.max_information_sets,
                  abstraction=a.abstraction,population_probability=a.population_probability,
                  archive_every=a.archive_every,archive_size=a.archive_size,averaging_sampler=a.averaging_sampler,opponent_population=a.opponent_population)
    if a.config:
        cfg=json.loads(Path(a.config).read_text())
        cfg.setdefault('averaging_sampler','uniform')
        cfg.setdefault('opponent_population','historical-v1')
        if cfg.pop('schema',None)!=1 or set(cfg)!=set(settings):raise ValueError('invalid CFR run config')
        settings=cfg  # Run configuration is authoritative; CLI output/resume remain independent.
    signal.signal(signal.SIGTERM,stop_signal)
    signal.signal(signal.SIGINT,stop_signal)
    train(path=a.output,resume=a.resume,fork_population=a.fork_population,**settings)
