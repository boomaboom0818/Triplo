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
from .trajectory import TrajectoryMCCFR,ALGORITHM as TRAJECTORY_ALGORITHM,AVERAGING as TRAJECTORY_AVERAGING,fork_allowed
from .prefix import PrefixMCCFR,ALGORITHM as PREFIX_ALGORITHM,fork_allowed as prefix_fork_allowed
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
          population_probability=0.0,archive_every=100,archive_size=4,averaging_sampler="uniform",opponent_population="historical-v1",fork_population=False,
          traversal='external-v1',trajectory_samples=8,fork_traversal=False,regret_prefix_depth=2):
    with run_lock(path):
        return _train(iterations,path,seed,game,players,resume,checkpoint_every,
                      max_iteration_nodes,max_seconds,max_information_sets,abstraction,
                      population_probability,archive_every,archive_size,averaging_sampler,opponent_population,fork_population,
                      traversal,trajectory_samples,fork_traversal,regret_prefix_depth)


def _train(iterations,path,seed,game,players,resume,checkpoint_every,
           max_iteration_nodes,max_seconds,max_information_sets,abstraction,
           population_probability,archive_every,archive_size,averaging_sampler,opponent_population,fork_population,
           traversal,trajectory_samples,fork_traversal,regret_prefix_depth):
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
    if traversal not in ('external-v1','trajectory-v2','prefix-v3'):raise ValueError('unsupported traversal')
    algorithm={'external-v1':'external-sampling-mccfr-v1','trajectory-v2':TRAJECTORY_ALGORITHM,'prefix-v3':PREFIX_ALGORITHM}[traversal]
    if averaging_sampler not in ("uniform","call-mixture-v1",TRAJECTORY_AVERAGING):raise ValueError("unsupported averaging sampler")
    if (traversal!='external-v1')!=(averaging_sampler==TRAJECTORY_AVERAGING):raise ValueError('traversal/averaging version mismatch')
    if traversal!='external-v1':
        if type(trajectory_samples) is not int or not 1<=trajectory_samples<=64:raise ValueError('invalid trajectory sample count')
        config.update(traversal=traversal,trajectory_samples=trajectory_samples)
    if traversal=='prefix-v3':
        if type(regret_prefix_depth) is not int or regret_prefix_depth not in (1,2):raise ValueError('invalid regret prefix depth')
        config['regret_prefix_depth']=regret_prefix_depth
    if averaging_sampler!="uniform":config["averaging_sampler"]=averaging_sampler
    if opponent_population not in ('historical-v1','diverse-v1'):raise ValueError('unsupported opponent population')
    if opponent_population!='historical-v1':
        if game!='holdem':raise ValueError('diverse profiles require holdem')
        config['opponent_population']=opponent_population
    if fork_population and (not resume or Path(path).resolve()==Path(resume).resolve()):
        raise ValueError('population fork requires a new output and explicit parent')
    if fork_traversal and (fork_population or not resume or Path(path).resolve()==Path(resume).resolve()):
        raise ValueError('traversal fork requires a new output, explicit parent and no population change')
    recipe_fork=None
    solver=PrefixMCCFR(seed,trajectory_samples,regret_prefix_depth) if traversal=='prefix-v3' else TrajectoryMCCFR(seed,trajectory_samples) if traversal=='trajectory-v2' else MCCFR(seed)
    solver.averaging_sampler=averaging_sampler;seconds=0;archive=[]
    solver.stop_requested=lambda: _STOP_REQUESTED
    if resume:
        saved=json.loads(Path(resume).read_text())
        from .population import population_fork_allowed
        compatible=saved.get('config')==config
        if fork_population:
            compatible=population_fork_allowed(saved.get('config',{}),config)
            if compatible:recipe_fork=dict(parent_sha256=hashlib.sha256(Path(resume).read_bytes()).hexdigest(),parent_config=saved['config'],iterations=saved['iterations'],kind='explicit population-diversity fork')
        elif fork_traversal:
            compatible=(traversal=='trajectory-v2' and saved.get('algorithm')=='external-sampling-mccfr-v1' and fork_allowed(saved.get('config',{}),config)) or (traversal=='prefix-v3' and saved.get('algorithm')==TRAJECTORY_ALGORITHM and prefix_fork_allowed(saved.get('config',{}),config))
            if compatible:recipe_fork=dict(parent_sha256=hashlib.sha256(Path(resume).read_bytes()).hexdigest(),parent_algorithm=saved['algorithm'],parent_config=saved['config'],iterations=saved['iterations'],kind=f'explicit {traversal} fork',previous_recipe_fork=saved.get('recipe_fork'))
        else:recipe_fork=saved.get('recipe_fork')
        expected_algorithm=TRAJECTORY_ALGORITHM if fork_traversal and traversal=='prefix-v3' else 'external-sampling-mccfr-v1' if fork_traversal else algorithm
        if (saved.get('schema')!=1 or saved.get('algorithm')!=expected_algorithm
                or not compatible or saved.get('model_id')!='bot-opponent-v1'):
            raise ValueError('incompatible CFR checkpoint')
        solver.regrets=saved['regrets'];solver.sums=saved['sums'];archive=saved['archive']
        solver.rng.setstate(tuples(saved['rng']));solver.iterations=saved['iterations']
        solver.nodes=saved['nodes'];seconds=saved['seconds']
        solver.diagnostics.update(saved.get('diagnostics',{}))
        solver.average_weight_moments=saved.get('average_weight_moments',{})
        if traversal!='external-v1' and not fork_traversal and 'traversal_stats' in saved:solver.traversal_stats=saved['traversal_stats']
        if traversal=='prefix-v3' and not fork_traversal:solver.epoch_weight_moments=saved.get('epoch_weight_moments',{})
    if iterations<=solver.iterations:raise ValueError('target must exceed completed iterations')
    try:revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    except (OSError,subprocess.CalledProcessError):revision='unknown'
    source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(Path(__file__).parent.glob('*.py'))}
    start=time.perf_counter();deadline=time.monotonic()+max_seconds
    stop_reason='running'
    def save():
        result=dict(schema=1,algorithm=algorithm,model_id='bot-opponent-v1',
                    config=config,recipe_fork=recipe_fork,engine='pokerkit-0.7.7',abstraction=abstraction,archive=archive,
                    iterations=solver.iterations,regrets=solver.regrets,sums=solver.sums,
                    rng=solver.rng.getstate(),nodes=solver.nodes,diagnostics=dict(solver.diagnostics),
                    average_weight_moments=solver.average_weight_moments,
                    weight_diagnostics_started_at=saved.get('weight_diagnostics_started_at',saved['iterations']) if resume else 0,
                    diagnostics_started_at=saved.get('diagnostics_started_at',saved['iterations']) if resume else 0,seconds=seconds+time.perf_counter()-start,
                    stop_reason=stop_reason,provenance=dict(revision=revision,python=platform.python_version(),source_sha256=source_hashes),
                    limits=dict(max_iteration_nodes=max_iteration_nodes,max_seconds=max_seconds,
                                max_information_sets=max_information_sets,checkpoint_every=checkpoint_every))
        if traversal!='external-v1':
            result.update(traversal_stats=solver.traversal_stats,traversal_stats_started_at=(saved['iterations'] if fork_traversal else saved.get('traversal_stats_started_at',saved['iterations'])) if resume else 0)
            result.update(sampling_diagnostics_schema='actual-actions-v2',sampling_diagnostics_started_at=saved.get('sampling_diagnostics_started_at',saved['iterations']) if resume else 0)
        if traversal=='prefix-v3':result['epoch_weight_moments']=solver.epoch_weight_moments
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
    p.add_argument('--averaging-sampler',choices=['uniform','call-mixture-v1',TRAJECTORY_AVERAGING],default='uniform')
    p.add_argument('--opponent-population',choices=['historical-v1','diverse-v1'],default='historical-v1')
    p.add_argument('--fork-population',action='store_true')
    p.add_argument('--traversal',choices=['external-v1','trajectory-v2','prefix-v3'],default='external-v1')
    p.add_argument('--trajectory-samples',type=int,default=8)
    p.add_argument('--regret-prefix-depth',type=int,default=2)
    p.add_argument('--fork-traversal',action='store_true')
    a=p.parse_args()
    settings=dict(iterations=a.iterations,seed=a.seed,game=a.game,players=a.players,
                  checkpoint_every=a.checkpoint_every,max_iteration_nodes=a.max_iteration_nodes,
                  max_seconds=a.max_seconds,max_information_sets=a.max_information_sets,
                  abstraction=a.abstraction,population_probability=a.population_probability,
                  archive_every=a.archive_every,archive_size=a.archive_size,averaging_sampler=a.averaging_sampler,opponent_population=a.opponent_population,
                  traversal=a.traversal,trajectory_samples=a.trajectory_samples,regret_prefix_depth=a.regret_prefix_depth)
    if a.config:
        cfg=json.loads(Path(a.config).read_text())
        cfg.setdefault('averaging_sampler','uniform')
        cfg.setdefault('opponent_population','historical-v1')
        cfg.setdefault('traversal','external-v1')
        cfg.setdefault('trajectory_samples',8)
        cfg.setdefault('regret_prefix_depth',2)
        if cfg.pop('schema',None)!=1 or set(cfg)!=set(settings):raise ValueError('invalid CFR run config')
        settings=cfg  # Run configuration is authoritative; CLI output/resume remain independent.
    signal.signal(signal.SIGTERM,stop_signal)
    signal.signal(signal.SIGINT,stop_signal)
    train(path=a.output,resume=a.resume,fork_population=a.fork_population,fork_traversal=a.fork_traversal,**settings)
