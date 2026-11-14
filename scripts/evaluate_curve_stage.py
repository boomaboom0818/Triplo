"""Supply a stage omitted by a legacy active supervisor without training writers."""
import argparse,json,time
from datetime import datetime,timezone
from pathlib import Path
from poker.config import load_checkpoint
from poker.train import atomic_save
from poker.train_cfr import run_lock
from poker.compare import paired_difference
from scripts.sampling_curve import ROOT,PARENT,VALIDATION,evaluate_cached


def ready(path,stage):
    if not path.exists():return False
    checkpoint=load_checkpoint(path,model_id='bot-opponent-v1')
    if checkpoint['iterations']>stage:raise ValueError('checkpoint exceeds stage')
    return checkpoint['iterations']==stage and checkpoint['stop_reason']=='target_reached'


def main(stage,wait,deadline):
    status=ROOT/f'evaluation-{stage}-status.json'
    with run_lock(ROOT/f'evaluation-{stage}-supervisor'):
        remaining=['candidate-a','candidate-b']
        try:
            while remaining:
                if datetime.now(timezone.utc)>=deadline:
                    atomic_save(status,dict(status='deadline_reached',remaining=remaining));return
                progressed=False
                for name in list(remaining):
                    checkpoint=ROOT/f'{name}-{stage}.json'
                    if not ready(checkpoint,stage):continue
                    refs={}
                    for kind in ('parent','uniform'):
                        path=PARENT/f'{name}.json'
                        args=['poker.evaluate','--checkpoint',str(path),'--blocks','1000','--seed','5300000','--copies','1','--opponent-checkpoints',str(VALIDATION)]
                        if kind=='uniform':args+=['--uniform-control']
                        refs[kind]=evaluate_cached(args,ROOT/f'{name}-{kind}.json')
                    for mode in ('standard','live-fallback'):
                        atomic_save(status,dict(status='evaluating',candidate=name,stage=stage,mode=mode))
                        args=['poker.evaluate','--checkpoint',str(checkpoint),'--blocks','1000','--seed','5300000','--copies','1','--opponent-checkpoints',str(VALIDATION)]
                        if mode=='live-fallback':args+=['--live-fallback']
                        result=evaluate_cached(args,ROOT/f'{name}-{stage}-{mode}.json')
                        atomic_save(ROOT/f'{name}-{stage}-{mode}-paired.json',{k:paired_difference(result,v) for k,v in refs.items()})
                    remaining.remove(name);progressed=True
                if not remaining:break
                atomic_save(status,dict(status='waiting_for_complete_checkpoints',remaining=remaining))
                if not wait:return
                state=json.loads((ROOT/'status.json').read_text())
                if state['status'] in ('stopped','completed_pending_evidence_review') and not progressed:
                    raise RuntimeError('main curve stopped/completed without required stage checkpoint')
                time.sleep(60)
            atomic_save(status,dict(status='completed',stage=stage))
        except BaseException as exc:
            atomic_save(status,dict(status='stopped',reason=str(exc),remaining=remaining));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',type=int,default=750)
    p.add_argument('--wait',action='store_true')
    p.add_argument('--deadline',default='2026-11-14T17:31:00+00:00')
    args=p.parse_args();deadline=datetime.fromisoformat(args.deadline)
    if args.stage<1 or deadline.tzinfo is None:p.error('positive stage and timezone-aware deadline required')
    main(args.stage,args.wait,deadline)
