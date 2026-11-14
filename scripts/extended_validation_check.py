"""Separate independent-validator continuation and additional frozen screening."""
import json,time
from datetime import datetime,timezone
from pathlib import Path
from poker.config import load_checkpoint
from poker.train import atomic_save
from poker.train_cfr import run_lock
from poker.compare import paired_difference
from scripts.sustained_train import run_child,evaluate,digest,stage_config

ROOT=Path('runs/bot/validation-extended-001')
SOURCE=Path('runs/bot/campaign-001/validation-250.json')
CURVE=Path('runs/bot/sustained-002')
DEADLINE=datetime.fromisoformat('2026-11-14T17:31:00+00:00')


def validator_config(saved):
    if saved['config']['seed']!=900001 or saved['config']['population_probability']!=0:
        raise ValueError('unexpected independent validator recipe')
    return stage_config(saved,750)


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    with run_lock(ROOT/'supervisor'):
        if (ROOT/'manifest.json').exists():raise ValueError('existing run: investigate before recovery')
        saved=load_checkpoint(SOURCE,model_id='bot-opponent-v1');cfg=validator_config(saved);del saved
        original=json.loads((CURVE/'manifest.json').read_text())['parents']
        atomic_save(ROOT/'manifest.json',dict(source={'path':str(SOURCE),'sha256':digest(SOURCE)},validator_target=750,max_information_sets=cfg['max_information_sets'],max_iteration_nodes=cfg['max_iteration_nodes'],memory_limit_bytes=3*1024**3,seed=5900000,blocks=1000,deadline=DEADLINE.isoformat(),purpose='additional independent screening; original validator and promotion criteria unchanged',candidate_parents=original,competitive_strength_established=False))
        checkpoint=ROOT/'validation-750.json';config=ROOT/'validation-config.json';atomic_save(config,cfg)
        try:
            atomic_save(ROOT/'status.json',dict(status='training_validator'))
            run_child(['poker.train_cfr','--config',str(config),'--output',str(checkpoint),'--resume',str(SOURCE)],ROOT/'validation.log',DEADLINE.timestamp())
            saved=load_checkpoint(checkpoint)
            if saved['iterations']!=750:raise RuntimeError(f"validator stopped at {saved['iterations']}: {saved['stop_reason']}; investigate before retry")
            del saved
            if digest(SOURCE)!=json.loads((ROOT/'manifest.json').read_text())['source']['sha256']:raise ValueError('original validator changed')
            # Initial screening of the new validator against an independent candidate,
            # kept distinct from evaluating the candidates against the frozen validator.
            independent=Path('runs/bot/sampling-curve-001/candidate-a-1000.json')
            atomic_save(ROOT/'status.json',dict(status='evaluating_validator'))
            evaluate(checkpoint,ROOT/'validator-quality.json',independent,5900000,False,DEADLINE.timestamp())
            for name,parent in original.items():
                candidate=CURVE/f'{name}-2000.json'
                while True:
                    if time.time()>=DEADLINE.timestamp():raise TimeoutError('deadline reached')
                    if candidate.exists():
                        data=load_checkpoint(candidate)
                        ready=data['iterations']==2000 and data['stop_reason']=='target_reached';del data
                        if ready:break
                    atomic_save(ROOT/'status.json',dict(status='waiting_for2000',candidate=name))
                    if json.loads((CURVE/'status.json').read_text())['status'] in ('stopped','completed_pending_evidence_review'):raise RuntimeError('curve stopped without2000 checkpoint')
                    time.sleep(60)
                reports={}
                if digest(Path(parent['path']))!=parent['sha256']:raise ValueError('candidate parent changed')
                for kind,path in [('candidate',candidate),('parent',Path(parent['path'])),('uniform',candidate)]:
                    atomic_save(ROOT/'status.json',dict(status='evaluating_candidate',candidate=name,kind=kind))
                    reports[kind]=evaluate(path,ROOT/f'{name}-{kind}.json',checkpoint,5900000,kind=='uniform',DEADLINE.timestamp())
                atomic_save(ROOT/f'{name}-paired.json',{k:paired_difference(reports['candidate'],reports[k]) for k in ('parent','uniform')})
            atomic_save(ROOT/'status.json',dict(status='completed_pending_evidence_review',competitive_strength_established=False))
        except BaseException as exc:
            atomic_save(ROOT/'status.json',dict(status='stopped',reason=str(exc)));raise

if __name__=='__main__':main()
