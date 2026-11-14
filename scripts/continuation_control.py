"""Bounded local controls and v2 pilots. Never promotes to sustained training."""
import argparse,hashlib,json,os,resource,subprocess,sys
from poker.config import load_checkpoint
from pathlib import Path
from poker.train import atomic_save
from poker.train_cfr import run_lock

ROOT=Path('runs/bot/continuation-001')
BASE=Path('runs/bot/campaign-001')

def limits():
    resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3))
    os.nice(10)

def run(args,log):
    with log.open('a' if args[0]=='poker.train_cfr' else 'w') as handle:
        subprocess.run([sys.executable,'-m',*args],stdout=handle,stderr=subprocess.STDOUT,preexec_fn=limits,check=True)

def checkpoint_iterations(path, config):
    saved=load_checkpoint(path,model_id='bot-opponent-v1')
    expected={k:config[k] for k in ('game','players','seed','abstraction','population_probability','archive_every','archive_size')}
    if saved['config']!=expected or not 0<=saved['iterations']<=config['iterations']:
        raise ValueError('checkpoint configuration or iteration mismatch')
    if saved['iterations']<config['iterations'] and saved['stop_reason'] not in ('running','interrupted','run time budget reached'):
        raise ValueError('checkpoint stopped for a cause requiring investigation')
    return saved['iterations']


def valid_report(path,checkpoint,copies):
    if not path.exists():return False
    r=json.loads(path.read_text())
    expected=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    opponent=hashlib.sha256((BASE/'validation-250.json').read_bytes()).hexdigest()
    if (r['checkpoint_sha256']!=expected or r['copies']!=copies or r['seed']!=5100000
        or r['opponent_checkpoint_hashes']!=['checkpoint:'+opponent]
        or len(r['populations'])!=6
        or any(v['independent_deal_blocks']!=1000 for v in r['populations'].values())):
        raise ValueError('existing evaluation provenance mismatch')
    return True


def main(resume=False,max_seconds=1800):
    ROOT.mkdir(parents=True,exist_ok=True)
    with run_lock(ROOT/'supervisor'):
        if (ROOT/'manifest.json').exists() and not resume:raise ValueError('existing experiment: use verified --resume')
        if resume and not (ROOT/'manifest.json').exists():raise ValueError('missing manifest')
        manifest={'pid':os.getpid(),'memory_limit':3*1024**3,'evaluation_seed':5100000,'blocks':1000,'parents':{},'sustained_training_authorized_by_gate':False}
        for name in ('candidate-a','candidate-b'):
            manifest['parents'][name]=hashlib.sha256((BASE/f'{name}-5000.json').read_bytes()).hexdigest()
        if resume:
            previous=json.loads((ROOT/'manifest.json').read_text())
            if previous['parents']!=manifest['parents']:raise ValueError('parent checkpoint changed')
            manifest['previous_manifest']=previous
        manifest['invocation_seconds']=max_seconds
        atomic_save(ROOT/'manifest.json',manifest)
        try:
            jobs=[(n,5250,'texture-buckets-v1') for n in ('candidate-a','candidate-b')]
            jobs += [(n,stage,'texture-buckets-v2') for stage in (250,1000) for n in ('candidate-a','candidate-b')]
            for name,stage,abstraction in jobs:
                tag=f'{name}-{abstraction}-{stage}';checkpoint=ROOT/f'{tag}.json'
                config=json.loads((BASE/f'{name}-5000-config.json').read_text())
                config.update(iterations=stage,abstraction=abstraction,max_seconds=max_seconds)
                cfg=ROOT/f'{tag}-config.json'
                args=['poker.train_cfr','--config',str(cfg),'--output',str(checkpoint)]
                completed=checkpoint_iterations(checkpoint,config) if checkpoint.exists() else 0
                if checkpoint.exists():args+=['--resume',str(checkpoint)]
                elif abstraction.endswith('v1'):args+=['--resume',str(BASE/f'{name}-5000.json')]
                elif stage==1000:args+=['--resume',str(ROOT/f'{name}-{abstraction}-250.json')]
                atomic_save(ROOT/'status.json',{'status':'training','job':tag})
                if completed<stage:
                    atomic_save(cfg,config)
                    run(args,ROOT/f'{tag}.log')
                saved=json.loads(checkpoint.read_text())
                if saved['iterations']!=stage:raise RuntimeError(f"{tag}: {saved['stop_reason']} at {saved['iterations']}")
                del saved
                atomic_save(ROOT/'status.json',{'status':'evaluating','job':tag})
                for copies in (1,3):
                    report=ROOT/f'{tag}-copies-{copies}.json'
                    if valid_report(report,checkpoint,copies):continue
                    run(['poker.evaluate','--checkpoint',str(checkpoint),'--blocks','1000','--seed','5100000','--copies',str(copies),'--opponent-checkpoints',str(BASE/'validation-250.json')],report)
                    json.loads(report.read_text())
            for name in ('candidate-a','candidate-b'):
                for copies in (1,3):
                    atomic_save(ROOT/'status.json',{'status':'evaluating_baseline','candidate':name,'copies':copies})
                    if valid_report(ROOT/f'{name}-baseline-copies-{copies}.json',BASE/f'{name}-5000.json',copies):continue
                    run(['poker.evaluate','--checkpoint',str(BASE/f'{name}-5000.json'),'--blocks','1000','--seed','5100000','--copies',str(copies),'--opponent-checkpoints',str(BASE/'validation-250.json')],ROOT/f'{name}-baseline-copies-{copies}.json')
            atomic_save(ROOT/'status.json',{'status':'completed_pending_evidence_review','sustained_training_started':False})
        except BaseException as exc:
            atomic_save(ROOT/'status.json',{'status':'stopped','reason':str(exc)})
            raise

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--max-seconds',type=int,default=1800)
    args=parser.parse_args()
    if args.max_seconds<1:parser.error('max-seconds must be positive')
    main(args.resume,args.max_seconds)
