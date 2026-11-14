"""Paired small continuation controls versus explicitly forked diversity recipe."""
import json
from datetime import datetime
from pathlib import Path
from poker.config import load_checkpoint
from poker.train import atomic_save
from poker.train_cfr import run_lock
from poker.compare import paired_difference
from scripts.sustained_train import run_child,evaluate,digest,stage_config

ROOT=Path('runs/bot/population-pilot-001')
SOURCE=Path('runs/bot/sustained-002')
VALIDATION=Path('runs/bot/validation-extended-001/validation-750.json')
DEADLINE=datetime.fromisoformat('2026-11-14T20:31:00+00:00')
TARGET=2100
SEED=7000000


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    with run_lock(ROOT/'supervisor'):
        if (ROOT/'manifest.json').exists():raise ValueError('existing pilot: investigate before recovery')
        parents={n:SOURCE/f'{n}-2000.json' for n in ('candidate-0','candidate-1')}
        atomic_save(ROOT/'manifest.json',dict(parents={n:{'path':str(p.resolve()),'sha256':digest(p)} for n,p in parents.items()},target=TARGET,seed=SEED,blocks=1000,deadline=DEADLINE.isoformat(),validation={'path':str(VALIDATION),'sha256':digest(VALIDATION)},purpose='controlled exploratory population pilot; preserves2000 learners; original gate unchanged',profiles='25%per external seat: passive/pressure equally; otherwise historical recipe',primary='paired diverse-minus-control learned-opponent payoff; coverage/compute/tight diagnostics; no automatic promotion'))
        try:
            for name,parent in parents.items():
                saved=load_checkpoint(parent);cfg=stage_config(saved,TARGET);del saved
                checkpoints={}
                for recipe in ('control','diverse'):
                    config=dict(cfg)
                    if recipe=='diverse':config['opponent_population']='diverse-v1'
                    config_path=ROOT/f'{name}-{recipe}-config.json';atomic_save(config_path,config)
                    checkpoint=ROOT/f'{name}-{recipe}.json';checkpoints[recipe]=checkpoint
                    atomic_save(ROOT/'status.json',dict(status='training',candidate=name,recipe=recipe,target=TARGET))
                    args=['poker.train_cfr','--config',str(config_path),'--output',str(checkpoint),'--resume',str(parent)]
                    if recipe=='diverse':args+=['--fork-population']
                    run_child(args,ROOT/f'{name}-{recipe}.log',DEADLINE.timestamp())
                    after=load_checkpoint(checkpoint)
                    if after['iterations']!=TARGET:raise RuntimeError(f"{name}/{recipe} stopped at {after['iterations']}: {after['stop_reason']}")
                    del after
                reports={}
                for kind,path in [('control',checkpoints['control']),('diverse',checkpoints['diverse']),('parent',parent),('uniform',checkpoints['control'])]:
                    atomic_save(ROOT/'status.json',dict(status='evaluating',candidate=name,kind=kind))
                    reports[kind]=evaluate(path,ROOT/f'{name}-{kind}-evaluation.json',VALIDATION,SEED,kind=='uniform',DEADLINE.timestamp())
                atomic_save(ROOT/f'{name}-paired.json',{k:paired_difference(reports['diverse'],reports[k]) for k in ('control','parent','uniform')})
            atomic_save(ROOT/'status.json',dict(status='completed_pending_evidence_review',competitive_strength_established=False))
        except BaseException as exc:
            atomic_save(ROOT/'status.json',dict(status='stopped',reason=str(exc)));raise

if __name__=='__main__':main()
