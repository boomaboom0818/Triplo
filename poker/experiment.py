"""Staged, reproducible bot training and validation artifacts."""
import argparse
import json
import signal
from pathlib import Path
from .train_cfr import train,stop_signal
from .train import atomic_save
from .evaluate import evaluate


def experiment(config_path,output,stages,blocks,opponents=()):
    root=Path(output)
    if root.exists():raise ValueError('use a new experiment directory; interrupted checkpoints can resume via train_cfr')
    if not stages or stages!=sorted(set(stages)) or stages[0]<1 or blocks<2:
        raise ValueError('stages must be positive increasing targets; blocks >= 2')
    cfg=json.loads(Path(config_path).read_text())
    if cfg.pop('schema',None)!=1:raise ValueError('invalid run schema')
    cfg.pop('iterations')
    root.mkdir(parents=True)
    atomic_save(root/'manifest.json',dict(config=cfg,stages=stages,blocks=blocks,
        opponents=list(opponents),evaluation_seed=2000000,model_id='bot-opponent-v1'))
    previous=None
    for stage in stages:
        checkpoint=root/f'iteration-{stage}.json'
        result=train(stage,checkpoint,resume=previous,**cfg)
        if result['iterations']!=stage:
            atomic_save(root/'status.json',dict(status='budget_stopped',checkpoint=str(checkpoint),
                completed=result['iterations'],target=stage,reason=result['stop_reason']))
            return
        for copies in (1,3):
            report=evaluate(checkpoint,blocks,2000000,copies,opponent_checkpoints=opponents)
            atomic_save(root/f'iteration-{stage}-copies-{copies}.json',report)
        previous=checkpoint
    atomic_save(root/'status.json',dict(status='completed',checkpoint=str(previous),
        warning='Experiment completion is not evidence of competitive strength'))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/cfr-bot.json')
    p.add_argument('--output',required=True)
    p.add_argument('--stages',type=int,nargs='+',default=[100,1000,10000])
    p.add_argument('--blocks',type=int,default=1000)
    p.add_argument('--opponent-checkpoints',nargs='*',default=[])
    signal.signal(signal.SIGINT,stop_signal)
    signal.signal(signal.SIGTERM,stop_signal)
    a=p.parse_args();experiment(a.config,a.output,a.stages,a.blocks,a.opponent_checkpoints)
