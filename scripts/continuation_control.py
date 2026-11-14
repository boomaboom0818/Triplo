"""Bounded local controls and v2 pilots. Never promotes to sustained training."""
import hashlib,json,os,resource,subprocess,sys
from pathlib import Path
from poker.train import atomic_save
from poker.train_cfr import run_lock

ROOT=Path('runs/bot/continuation-001')
BASE=Path('runs/bot/campaign-001')

def limits():
    resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3))
    os.nice(10)

def run(args,log):
    with log.open('w') as handle:
        subprocess.run([sys.executable,'-m',*args],stdout=handle,stderr=subprocess.STDOUT,preexec_fn=limits,check=True)

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    with run_lock(ROOT/'supervisor'):
        if (ROOT/'manifest.json').exists():raise ValueError('existing experiment: inspect before recovery')
        manifest={'pid':os.getpid(),'memory_limit':3*1024**3,'evaluation_seed':5100000,'blocks':1000,'parents':{},'sustained_training_authorized_by_gate':False}
        for name in ('candidate-a','candidate-b'):
            manifest['parents'][name]=hashlib.sha256((BASE/f'{name}-5000.json').read_bytes()).hexdigest()
        atomic_save(ROOT/'manifest.json',manifest)
        try:
            jobs=[(n,5250,'texture-buckets-v1') for n in ('candidate-a','candidate-b')]
            jobs += [(n,stage,'texture-buckets-v2') for stage in (250,1000) for n in ('candidate-a','candidate-b')]
            for name,stage,abstraction in jobs:
                tag=f'{name}-{abstraction}-{stage}';checkpoint=ROOT/f'{tag}.json'
                config=json.loads((BASE/f'{name}-5000-config.json').read_text())
                config.update(iterations=stage,abstraction=abstraction,max_seconds=1800)
                cfg=ROOT/f'{tag}-config.json';atomic_save(cfg,config)
                args=['poker.train_cfr','--config',str(cfg),'--output',str(checkpoint)]
                if abstraction.endswith('v1'):args+=['--resume',str(BASE/f'{name}-5000.json')]
                elif stage==1000:args+=['--resume',str(ROOT/f'{name}-{abstraction}-250.json')]
                atomic_save(ROOT/'status.json',{'status':'training','job':tag})
                run(args,ROOT/f'{tag}.log')
                saved=json.loads(checkpoint.read_text())
                if saved['iterations']!=stage:raise RuntimeError(f"{tag}: {saved['stop_reason']} at {saved['iterations']}")
                del saved
                atomic_save(ROOT/'status.json',{'status':'evaluating','job':tag})
                for copies in (1,3):
                    report=ROOT/f'{tag}-copies-{copies}.json'
                    run(['poker.evaluate','--checkpoint',str(checkpoint),'--blocks','1000','--seed','5100000','--copies',str(copies),'--opponent-checkpoints',str(BASE/'validation-250.json')],report)
                    json.loads(report.read_text())
            for name in ('candidate-a','candidate-b'):
                for copies in (1,3):
                    atomic_save(ROOT/'status.json',{'status':'evaluating_baseline','candidate':name,'copies':copies})
                    run(['poker.evaluate','--checkpoint',str(BASE/f'{name}-5000.json'),'--blocks','1000','--seed','5100000','--copies',str(copies),'--opponent-checkpoints',str(BASE/'validation-250.json')],ROOT/f'{name}-baseline-copies-{copies}.json')
            atomic_save(ROOT/'status.json',{'status':'completed_pending_evidence_review','sustained_training_started':False})
        except BaseException as exc:
            atomic_save(ROOT/'status.json',{'status':'stopped','reason':str(exc)})
            raise

if __name__=='__main__':main()
