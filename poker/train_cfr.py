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
from contextlib import contextmanager
from pathlib import Path
from .cfr import MCCFR,Kuhn,Holdem,TraversalLimit
from .train import atomic_save,tuples


@contextmanager
def run_lock(path):
    lock=Path(str(path)+'.lock');lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a') as handle:
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('another trainer owns this output')
        yield


def train(iterations,path,seed=1,game='kuhn',players=6,resume=None,*,
          checkpoint_every=100,max_iteration_nodes=100000,max_seconds=3600,
          max_information_sets=500000):
    with run_lock(path):
        return _train(iterations,path,seed,game,players,resume,checkpoint_every,
                      max_iteration_nodes,max_seconds,max_information_sets)


def _train(iterations,path,seed,game,players,resume,checkpoint_every,
           max_iteration_nodes,max_seconds,max_information_sets):
    for value in (checkpoint_every,max_iteration_nodes,max_information_sets):
        if type(value) is not int or value<1:raise ValueError('limits must be positive integers')
    if not math.isfinite(max_seconds) or max_seconds<=0:raise ValueError('invalid time budget')
    if iterations<1 or game not in ('kuhn','holdem') or not 3<=players<=6:
        raise ValueError('invalid training configuration')
    config=dict(game=game,players=2 if game=='kuhn' else players,seed=seed)
    if Path(path).exists() and (resume is None or Path(path).resolve()!=Path(resume).resolve()):
        raise ValueError('output exists; use resume or a new path')
    solver=MCCFR(seed);seconds=0
    if resume:
        saved=json.loads(Path(resume).read_text())
        if (saved.get('schema')!=1 or saved.get('algorithm')!='external-sampling-mccfr-v1'
                or saved.get('config')!=config or saved.get('model_id')!='bot-opponent-v1'):
            raise ValueError('incompatible CFR checkpoint')
        solver.regrets=saved['regrets'];solver.sums=saved['sums']
        solver.rng.setstate(tuples(saved['rng']));solver.iterations=saved['iterations']
        solver.nodes=saved['nodes'];seconds=saved['seconds']
    if iterations<=solver.iterations:raise ValueError('target must exceed completed iterations')
    try:revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    except (OSError,subprocess.CalledProcessError):revision='unknown'
    start=time.perf_counter();deadline=time.monotonic()+max_seconds
    stop_reason='running'
    def save():
        result=dict(schema=1,algorithm='external-sampling-mccfr-v1',model_id='bot-opponent-v1',
                    config=config,engine='pokerkit-0.7.7',abstraction='exact-observation-v1',
                    iterations=solver.iterations,regrets=solver.regrets,sums=solver.sums,
                    rng=solver.rng.getstate(),nodes=solver.nodes,seconds=seconds+time.perf_counter()-start,
                    stop_reason=stop_reason,provenance=dict(revision=revision,python=platform.python_version()),
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
    factory=Kuhn if game=='kuhn' else lambda seed:Holdem(seed,players)
    try:
        for _ in range(solver.iterations,iterations):
            if len(set(solver.regrets)|set(solver.sums))>=max_information_sets:
                stop_reason='information_set_budget';break
            solver.step(factory,config['players'],max_iteration_nodes,deadline)
            if solver.iterations%checkpoint_every==0 and solver.iterations<iterations:save()
    except TraversalLimit as exc:stop_reason=str(exc)
    except KeyboardInterrupt:stop_reason='interrupted'
    if stop_reason=='running':stop_reason='target_reached'
    return save()


def stop_signal(signum,frame):
    raise KeyboardInterrupt


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
    a=p.parse_args()
    settings=dict(iterations=a.iterations,seed=a.seed,game=a.game,players=a.players,
                  checkpoint_every=a.checkpoint_every,max_iteration_nodes=a.max_iteration_nodes,
                  max_seconds=a.max_seconds,max_information_sets=a.max_information_sets)
    if a.config:
        cfg=json.loads(Path(a.config).read_text())
        if cfg.pop('schema',None)!=1 or set(cfg)!=set(settings):raise ValueError('invalid CFR run config')
        settings=cfg  # Run configuration is authoritative; CLI output/resume remain independent.
    signal.signal(signal.SIGTERM,stop_signal)
    train(path=a.output,resume=a.resume,**settings)
