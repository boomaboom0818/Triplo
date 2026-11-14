"""Explicitly authorized exploratory continuation; no automatic sustained training."""
import argparse,hashlib,json,os
from pathlib import Path
from scripts.continuation_control import run
from poker.train import atomic_save
from poker.train_cfr import run_lock
from poker.config import load_checkpoint
from poker.compare import paired_difference

ROOT=Path('runs/bot/sampling-curve-001')
PARENT=Path('runs/bot/sampling-001')
VALIDATION=Path('runs/bot/campaign-001/validation-250.json')

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def resume_checkpoint(path,parent,target):
    if not path.exists():return parent,False
    saved=load_checkpoint(path,model_id='bot-opponent-v1')
    original=load_checkpoint(parent,model_id='bot-opponent-v1')
    if saved['config']!=original['config'] or not original['iterations']<=saved['iterations']<=target:
        raise ValueError('resume checkpoint lineage/configuration mismatch')
    complete=saved['iterations']==target
    if not complete and saved['stop_reason'] not in ('running','interrupted','run time budget reached'):
        raise ValueError('resource/unknown failure requires investigation before resume')
    # Confirm saved RNG structure is usable, without consuming it.
    from random import Random
    from poker.train import tuples
    Random().setstate(tuples(saved['rng']))
    return path,complete


def evaluate_cached(args,path):
    checkpoint=Path(args[args.index('--checkpoint')+1])
    def validate(r):
        if (r['checkpoint_sha256']!=digest(checkpoint) or r['seed']!=5300000 or r['copies']!=1
            or r['opponent_checkpoint_hashes']!=['checkpoint:'+digest(VALIDATION)]
            or r.get('uniform_control',False)!=('--uniform-control' in args)
            or r.get('live_fallback',False)!=('--live-fallback' in args)
            or len(r['populations'])!=6
            or any(v['independent_deal_blocks']!=1000 or len(v['deal_block_bb_per_100'])!=1000 for v in r['populations'].values())):
            raise ValueError('evaluation provenance mismatch')
        return r
    if path.exists():return validate(json.loads(path.read_text()))
    temporary=Path(str(path)+'.pending')
    run(args,temporary)
    result=validate(json.loads(temporary.read_text()))
    temporary.replace(path)
    return result


def main(resume=False):
    ROOT.mkdir(parents=True,exist_ok=True)
    with run_lock(ROOT/'supervisor'):
        if (ROOT/'manifest.json').exists() and not resume:raise ValueError('existing curve: use --resume after investigation')
        if resume and not (ROOT/'manifest.json').exists():raise ValueError('missing manifest')
        parents={n:hashlib.sha256((PARENT/f'{n}.json').read_bytes()).hexdigest() for n in ('candidate-a','candidate-b')}
        manifest=dict(pid=os.getpid(),parents=parents,stages=[500,750,1000],memory_limit_bytes=3*1024**3,seed=5300000,blocks=1000,purpose='exploratory extension authorized after failed payoff screen',live_fallback='separate evaluation only',sustained_training=False,validation_sha256=digest(VALIDATION))
        if resume:
            old=json.loads((ROOT/'manifest.json').read_text())
            for key in ('parents','stages','memory_limit_bytes','seed','blocks'):
                if old[key]!=manifest[key]:raise ValueError('manifest mismatch')
            if old.get('validation_sha256',digest(VALIDATION))!=digest(VALIDATION):raise ValueError('validation changed')
            manifest['previous_manifest']=old
        atomic_save(ROOT/'manifest.json',manifest)
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
                    source,complete=resume_checkpoint(checkpoint,previous[name],stage)
                    if not complete:
                        run(['poker.train_cfr','--config',str(config),'--output',str(checkpoint),'--resume',str(source)],ROOT/f'{name}-{stage}.log')
                    saved=load_checkpoint(checkpoint)
                    if saved['iterations']!=stage:raise RuntimeError(f"{name} stage {stage}: {saved['stop_reason']} at {saved['iterations']}")
                    del saved
                    previous[name]=checkpoint
                    if name not in references:
                        references[name]={}
                        for kind in ('parent','uniform'):
                            atomic_save(ROOT/'status.json',dict(status='evaluating_reference',candidate=name,kind=kind))
                            output=ROOT/f'{name}-{kind}.json'
                            args=['poker.evaluate','--checkpoint',str(PARENT/f'{name}.json'),'--blocks','1000','--seed','5300000','--copies','1','--opponent-checkpoints',str(VALIDATION)]
                            if kind=='uniform':args+=['--uniform-control']
                            references[name][kind]=evaluate_cached(args,output)
                    for kind in ('standard','live-fallback'):
                        atomic_save(ROOT/'status.json',dict(status='evaluating',candidate=name,stage=stage,kind=kind))
                        report=ROOT/f'{name}-{stage}-{kind}.json'
                        args=['poker.evaluate','--checkpoint',str(checkpoint),'--blocks','1000','--seed','5300000','--copies','1','--opponent-checkpoints',str(VALIDATION)]
                        if kind=='live-fallback':args+=['--live-fallback']
                        result=evaluate_cached(args,report)
                        atomic_save(ROOT/f'{name}-{stage}-{kind}-paired.json',{k:paired_difference(result,v) for k,v in references[name].items()})
            atomic_save(ROOT/'status.json',dict(status='completed_pending_evidence_review',sustained_training_started=False))
        except BaseException as exc:
            atomic_save(ROOT/'status.json',dict(status='stopped',reason=str(exc)))
            raise

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume',action='store_true')
    main(parser.parse_args().resume)
