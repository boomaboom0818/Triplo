"""Resumable Monte Carlo policy-gradient population self-play, not CFR."""
import argparse
import copy
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from random import Random
from .game import play
from .policy import Policy, key
from .config import load_checkpoint

DEFAULT = dict(model_id="bot-opponent-v1", schema=1, players=6, seed=1, learning_rate=.02,
               checkpoint_every=100, archive_every=100, archive_size=8,
               population=['blueprint','blueprint','calling','aggressive'])


def configuration(path=None):
    config = DEFAULT.copy() if path is None else json.loads(Path(path).read_text())
    if set(config) != set(DEFAULT) or config['schema'] != 1:
        raise ValueError('unsupported training config fields/schema')
    for name in ('seed','players','checkpoint_every','archive_every','archive_size'):
        if type(config[name]) is not int or config[name] < (0 if name == 'seed' else 1):
            raise ValueError(f'invalid {name}')
    if config['model_id'] != 'bot-opponent-v1':
        raise ValueError('only bot-opponent-v1 training is active; human training is deferred')
    if config['players'] != 6:
        raise ValueError('baseline trainer currently supports six players only')
    if type(config['learning_rate']) not in (int,float) or not 0 < config['learning_rate'] <= 1:
        raise ValueError('learning_rate must be in (0,1]')
    if not isinstance(config['population'],list) or not config['population'] or any(
        x not in ('blueprint','calling','aggressive') for x in config['population']):
        raise ValueError('invalid training population')
    return config


def atomic_save(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as f:
        json.dump(value,f,allow_nan=False); f.flush(); os.fsync(f.fileno())
    os.replace(temporary,path)


def tuples(value):
    return tuple(map(tuples,value)) if isinstance(value,list) else value


def train(hands, seed, path, *, config=None, resume=None, overwrite=False):
    if type(hands) is not int or hands < 1: raise ValueError('hands must be positive')
    cfg=configuration(config)
    if seed is not None: cfg['seed']=seed
    target=Path(path)
    if target.exists() and not overwrite and (resume is None or Path(resume).resolve()!=target.resolve()):
        raise ValueError('output exists; use --resume or explicitly --overwrite')
    rng=Random(cfg['seed']); table={}; archive=[{}]; completed=0; seconds=0
    if resume:
        saved=load_checkpoint(resume, model_id=cfg['model_id'])
        state=saved.get('training')
        if state is None: raise ValueError('checkpoint has no resumable training state')
        if state['config'] != cfg: raise ValueError('resume configuration differs from checkpoint')
        table=saved['table']; archive=state['archive']; completed=saved['hands']; seconds=saved['seconds']
        rng.setstate(tuples(state['rng']))
    if hands <= completed: raise ValueError('--hands must exceed completed hands (total target)')
    start=time.perf_counter()
    try:
        revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    except (OSError,subprocess.CalledProcessError): revision='unknown'
    log=target.with_suffix('.progress.jsonl')
    def save():
        result=dict(model_id=cfg['model_id'],schema=1,algorithm='tabular-reinforce-v1',engine='pokerkit-0.7.7',
                    seed=cfg['seed'],hands=completed,seconds=seconds+time.perf_counter()-start,
                    table=table,abstraction='rank-position-board-active-pot-v1',stacks=200,players=6,
                    training=dict(config=cfg,archive=archive,rng=rng.getstate()),
                    provenance=dict(revision=revision,python=platform.python_version(),
                        config_sha256=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()))
        atomic_save(target,result)
        record=dict(hands=completed,seconds=result['seconds'],information_sets=len(table),
                    archive_size=len(archive),checkpoint_bytes=target.stat().st_size)
        with log.open('a') as f: f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
        return result
    for h in range(completed,hands):
        learner=h%6
        policies=[Policy(rng.randrange(2**32),rng.choice(archive),rng.choice(cfg['population'])) for _ in range(6)]
        policies[learner]=Policy(rng.randrange(2**32),table)
        rewards,traces,_=play(policies,rng.randrange(2**63),h%6)
        # Freeze behavior probabilities before any updates, including repeated buckets.
        trajectory=[(o,a,policies[learner].probabilities(o)) for o,a in traces[learner]]
        for o,a,probs in trajectory:
            row=table.setdefault(key(o),[0.0]*len(o.legal))
            for j in range(len(row)):
                row[j]+=cfg['learning_rate']*(rewards[learner]/100)*((o.legal[j]==a)-probs[j])
        completed=h+1
        if completed%cfg['archive_every']==0:
            archive.append(copy.deepcopy(table)); archive=archive[-cfg['archive_size']:]
        if completed%cfg['checkpoint_every']==0 and completed<hands: save()
    return save()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hands',type=int,default=1000,help='total target, including resumed hands')
    p.add_argument('--seed',type=int); p.add_argument('--config')
    p.add_argument('--output',default='runs/bot/seed1.json');p.add_argument('--resume')
    p.add_argument('--overwrite',action='store_true')
    a=p.parse_args();train(a.hands,a.seed,a.output,config=a.config,resume=a.resume,overwrite=a.overwrite)
