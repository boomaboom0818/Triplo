"""Local staged continuation with immutable parents, evaluation and a hard deadline."""
import argparse,hashlib,json,os,signal,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
from poker.config import load_checkpoint
from poker.train import atomic_save
from poker.train_cfr import run_lock
from poker.compare import paired_difference
from scripts.continuation_control import limits
from scripts.sampling_curve import resume_checkpoint


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def stage_config(saved,target):
    if target<=saved['iterations']:raise ValueError('stage must exceed parent iterations')
    cfg=dict(saved['config'])
    cfg.setdefault('averaging_sampler','uniform')
    cfg.update(schema=1,iterations=target,checkpoint_every=50,max_seconds=3600,
               max_information_sets=saved['limits']['max_information_sets'],
               max_iteration_nodes=saved['limits']['max_iteration_nodes'])
    return cfg


def run_child(args,log,deadline):
    remaining=deadline-time.time()
    if remaining<=0:raise TimeoutError('experiment deadline reached')
    with log.open('a' if args[0]=='poker.train_cfr' else 'w') as handle:
        process=subprocess.Popen([sys.executable,'-m',*args],stdout=handle,stderr=subprocess.STDOUT,preexec_fn=limits)
        def stop():
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
                try:process.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    process.kill();process.wait()
        def interrupted(signum,frame):
            stop();raise KeyboardInterrupt
        previous={s:signal.signal(s,interrupted) for s in (signal.SIGTERM,signal.SIGINT)}
        try:
            try:code=process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                stop();raise TimeoutError('experiment deadline reached; child stopped gracefully')
            if code:raise RuntimeError(f'child exit {code}; inspect {log}')
        finally:
            for sig,handler in previous.items():signal.signal(sig,handler)
            stop()


def validate_evaluation(r,checkpoint,validation,seed,uniform,blocks):
    if (r['checkpoint_sha256']!=digest(checkpoint) or r['seed']!=seed or r['copies']!=1
        or r.get('uniform_control',False)!=uniform or r.get('live_fallback',False)
        or r['opponent_checkpoint_hashes']!=['checkpoint:'+digest(validation)]
        or len(r['populations'])!=6
        or any(v['independent_deal_blocks']!=blocks or len(v['deal_block_bb_per_100'])!=blocks for v in r['populations'].values())):
        raise ValueError('evaluation provenance mismatch')
    return r


def evaluate(checkpoint,output,validation,seed,uniform,deadline,blocks=1000):
    if type(blocks) is not int or blocks<2:raise ValueError("at least two independent deal blocks required")
    with run_lock(output):
        if output.exists():return validate_evaluation(json.loads(output.read_text()),checkpoint,validation,seed,uniform,blocks)
        pending=Path(str(output)+'.pending')
        args=['poker.evaluate','--checkpoint',str(checkpoint),'--blocks',str(blocks),'--seed',str(seed),'--copies','1','--opponent-checkpoints',str(validation)]
        if uniform:args+=['--uniform-control']
        run_child(args,pending,deadline)
        result=validate_evaluation(json.loads(pending.read_text()),checkpoint,validation,seed,uniform,blocks);pending.replace(output)
        return result


def main(args):
    root=args.output;root.mkdir(parents=True,exist_ok=True)
    deadline=datetime.fromisoformat(args.deadline)
    if deadline.tzinfo is None:raise ValueError('deadline must have timezone')
    if args.stages!=sorted(set(args.stages)) or not args.stages:raise ValueError('positive increasing stages required')
    manifest=dict(parents={f'candidate-{i}':{'path':str(p.resolve()),'sha256':digest(p)} for i,p in enumerate(args.parents)},stages=args.stages,validation={'path':str(args.validation.resolve()),'sha256':digest(args.validation)},seed=args.seed,blocks=1000,memory_limit_bytes=3*1024**3,purpose=args.purpose)
    with run_lock(root/'supervisor'):
        path=root/'manifest.json'
        if len({v['path'] for v in manifest['parents'].values()})!=len(manifest['parents']):raise ValueError('duplicate parent checkpoints')
        if path.exists():
            if not args.resume:raise ValueError('existing output: inspect and use --resume')
            if json.loads(path.read_text())!=manifest:raise ValueError('manifest mismatch')
        elif args.resume:raise ValueError('missing manifest')
        else:atomic_save(path,manifest)
        atomic_save(root/'invocation.json',dict(pid=os.getpid(),deadline=args.deadline))
        try:
            for name,parent in manifest['parents'].items():
                saved=load_checkpoint(Path(parent['path']),model_id='bot-opponent-v1')
                stage_config(saved,args.stages[0])
            del saved
            for stage in args.stages:
                for name,parent in manifest['parents'].items():
                    if time.time()>=deadline.timestamp():raise TimeoutError('experiment deadline reached')
                    index=args.stages.index(stage)
                    source=Path(parent['path']) if index==0 else root/f'{name}-{args.stages[index-1]}.json'
                    saved=load_checkpoint(source,model_id='bot-opponent-v1');cfg=stage_config(saved,stage);del saved
                    checkpoint=root/f'{name}-{stage}.json'
                    lineage_path=root/f'{name}-{stage}-parent.json'
                    lineage={'path':str(source.resolve()),'sha256':digest(source)}
                    if lineage_path.exists():
                        if json.loads(lineage_path.read_text())!=lineage:raise ValueError('stage parent changed')
                    elif checkpoint.exists():raise ValueError('checkpoint missing parent lineage')
                    else:atomic_save(lineage_path,lineage)
                    resume,complete=resume_checkpoint(checkpoint,source,stage)
                    if not complete:
                        config=root/f'{name}-{stage}-config.json';atomic_save(config,cfg)
                        atomic_save(root/'status.json',dict(status='training',candidate=name,stage=stage))
                        run_child(['poker.train_cfr','--config',str(config),'--output',str(checkpoint),'--resume',str(resume)],root/f'{name}-{stage}.log',deadline.timestamp())
                    saved=load_checkpoint(checkpoint)
                    if saved['iterations']!=stage:raise RuntimeError(f"{name}: {saved['stop_reason']} at {saved['iterations']}")
                    del saved
                    atomic_save(root/'status.json',dict(status='evaluating',candidate=name,stage=stage))
                    refs={}
                    for kind in ('parent','uniform'):
                        refs[kind]=evaluate(Path(parent['path']),root/f'{name}-{kind}.json',args.validation,args.seed,kind=='uniform',deadline.timestamp())
                    report=evaluate(checkpoint,root/f'{name}-{stage}-evaluation.json',args.validation,args.seed,False,deadline.timestamp())
                    atomic_save(root/f'{name}-{stage}-paired.json',{k:paired_difference(report,v) for k,v in refs.items()})
            atomic_save(root/'status.json',dict(status='completed_pending_evidence_review',competitive_strength_established=False))
        except BaseException as exc:
            atomic_save(root/'status.json',dict(status='stopped',reason=str(exc) or type(exc).__name__));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--parents',nargs='+',type=Path,required=True)
    p.add_argument('--stages',nargs='+',type=int,required=True)
    p.add_argument('--validation',type=Path,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--deadline',required=True);p.add_argument('--resume',action='store_true')
    p.add_argument('--purpose',choices=['exploratory','evidence-qualified'],default='exploratory')
    main(p.parse_args())
