"""Explicitly authorized exploratory continuation; no automatic sustained training."""
import hashlib,json,os
from pathlib import Path
from scripts.continuation_control import run
from poker.train import atomic_save
from poker.train_cfr import run_lock
from poker.config import load_checkpoint
from poker.compare import paired_difference

ROOT=Path('runs/bot/sampling-curve-001')
PARENT=Path('runs/bot/sampling-001')
VALIDATION=Path('runs/bot/campaign-001/validation-250.json')

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    with run_lock(ROOT/'supervisor'):
        if (ROOT/'manifest.json').exists():raise ValueError('existing curve: investigate before recovery')
        parents={n:hashlib.sha256((PARENT/f'{n}.json').read_bytes()).hexdigest() for n in ('candidate-a','candidate-b')}
        atomic_save(ROOT/'manifest.json',dict(pid=os.getpid(),parents=parents,stages=[500,750,1000],memory_limit_bytes=3*1024**3,seed=5300000,blocks=1000,purpose='exploratory extension authorized after failed payoff screen',live_fallback='separate evaluation only',sustained_training=False))
        previous={n:PARENT/f'{n}.json' for n in parents}
        try:
            references={}
            for stage in (500,750,1000):
                for name in parents:
                    parent=load_checkpoint(previous[name],model_id='bot-opponent-v1')
                    if parent['config'].get('averaging_sampler')!='call-mixture-v1':raise ValueError('unexpected parent recipe')
                    del parent
                    cfg=json.loads((PARENT/f'{name}-config.json').read_text());cfg.update(iterations=stage,max_seconds=3600)
                    config=ROOT/f'{name}-{stage}-config.json';atomic_save(config,cfg)
                    checkpoint=ROOT/f'{name}-{stage}.json'
                    atomic_save(ROOT/'status.json',dict(status='training',candidate=name,stage=stage))
                    run(['poker.train_cfr','--config',str(config),'--output',str(checkpoint),'--resume',str(previous[name])],ROOT/f'{name}-{stage}.log')
                    saved=load_checkpoint(checkpoint)
                    if saved['iterations']!=stage:raise RuntimeError(f"{name} stage {stage}: {saved['stop_reason']} at {saved['iterations']}")
                    del saved
                    previous[name]=checkpoint
                    if stage==750:continue
                    if name not in references:
                        references[name]={}
                        for kind in ('parent','uniform'):
                            atomic_save(ROOT/'status.json',dict(status='evaluating_reference',candidate=name,kind=kind))
                            output=ROOT/f'{name}-{kind}.json'
                            args=['poker.evaluate','--checkpoint',str(PARENT/f'{name}.json'),'--blocks','1000','--seed','5300000','--copies','1','--opponent-checkpoints',str(VALIDATION)]
                            if kind=='uniform':args+=['--uniform-control']
                            run(args,output);references[name][kind]=json.loads(output.read_text())
                    for kind in ('standard','live-fallback'):
                        atomic_save(ROOT/'status.json',dict(status='evaluating',candidate=name,stage=stage,kind=kind))
                        report=ROOT/f'{name}-{stage}-{kind}.json'
                        args=['poker.evaluate','--checkpoint',str(checkpoint),'--blocks','1000','--seed','5300000','--copies','1','--opponent-checkpoints',str(VALIDATION)]
                        if kind=='live-fallback':args+=['--live-fallback']
                        run(args,report)
                        result=json.loads(report.read_text())
                        atomic_save(ROOT/f'{name}-{stage}-{kind}-paired.json',{k:paired_difference(result,v) for k,v in references[name].items()})
            atomic_save(ROOT/'status.json',dict(status='completed_pending_evidence_review',sustained_training_started=False))
        except BaseException as exc:
            atomic_save(ROOT/'status.json',dict(status='stopped',reason=str(exc)))
            raise

if __name__=='__main__':main()
