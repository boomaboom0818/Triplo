"""Local CPU campaign: independent learners, frozen validation and stage reports."""
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
from .train import atomic_save


def child_limits():
    # Each subprocess is bounded independently; three workers fit this 16-GiB host.
    resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3))
    os.nice(10)


def campaign(output,targets,blocks=200,resume=False):
    root=Path(output).resolve()
    if root.exists() and not resume:raise ValueError('campaign directory already exists; use --resume')
    if resume and not (root/'manifest.json').exists():raise ValueError('campaign manifest missing')
    if blocks<2:raise ValueError('at least two evaluation blocks required')
    if not targets or targets!=sorted(set(targets)) or targets[0]<1:
        raise ValueError('targets must be positive, unique and increasing')
    root.mkdir(parents=True,exist_ok=resume)
    children=[]
    def terminate(signum,frame):
        for process in list(children):
            if process.poll() is None:process.send_signal(signal.SIGTERM)
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
    configs={}
    for name,seed,population in [('candidate-a',11,.25),('candidate-b',29,.25),('validation',900001,0.0)]:
        cfg=json.loads(Path('configs/cfr-bot.json').read_text())
        cfg.update(seed=seed,population_probability=population,archive_size=2,
                   max_information_sets=250000,checkpoint_every=50,max_seconds=86400,
                   max_iteration_nodes=500000)
        configs[name]=cfg
    if resume:
        saved=json.loads((root/'manifest.json').read_text())
        if saved['targets']!=targets or saved['blocks']!=blocks or saved['configs']!=configs:
            raise ValueError('campaign resume configuration mismatch')
    atomic_save(root/'manifest.json',dict(pid=os.getpid(),targets=targets,blocks=blocks,configs=configs,
        memory_limit_per_worker_bytes=3*1024**3,validation_frozen_at=targets[0],
        warning='Independent learned validation agent has no established strength'))
    def command(args,log):
        with (root/log).open('a') as handle:
            # Runs only from this single calling thread, so preexec_fn is safe.
            process=subprocess.Popen([sys.executable,'-m',*args],stdout=handle,stderr=subprocess.STDOUT,
                                     preexec_fn=child_limits)
            children.append(process)
        return process
    try:
        previous={}
        for stage in targets:
            jobs={}
            names=list(configs) if stage==targets[0] else ['candidate-a','candidate-b']
            for name in names:
                cfg=configs[name].copy();cfg['iterations']=stage
                cfg_path=root/f'{name}-{stage}-config.json';atomic_save(cfg_path,cfg)
                checkpoint=root/f'{name}-{stage}.json'
                if checkpoint.exists():
                    existing=json.loads(checkpoint.read_text())
                    if existing['iterations']==stage:
                        previous[name]=checkpoint;continue
                    previous[name]=checkpoint
                args=['poker.train_cfr','--config',str(cfg_path),'--output',str(checkpoint)]
                if name in previous:args+=['--resume',str(previous[name])]
                jobs[name]=(command(args,f'{name}.log'),checkpoint)
            for name,(process,checkpoint) in jobs.items():
                code=process.wait()
                if code:raise RuntimeError(f'{name} failed with exit {code}; see its log')
                saved=json.loads(checkpoint.read_text())
                if saved['iterations']!=stage:
                    raise RuntimeError(f"{name} stopped at {saved['iterations']}: {saved['stop_reason']}")
                previous[name]=checkpoint
            for name in ['candidate-a','candidate-b']:
                for copies in (1,3):
                    report=root/f'{name}-{stage}-copies-{copies}.json'
                    args=['poker.evaluate','--checkpoint',str(previous[name]),'--opponent-checkpoints',
                          str(previous['validation']),'--blocks',str(blocks),'--copies',str(copies),'--seed','2000000']
                    # stdout is the JSON report; diagnostic failures are recorded alongside it.
                    with report.open('w') as handle, (root/'evaluation-errors.log').open('a') as errors:
                        process=subprocess.Popen([sys.executable,'-m',*args],stdout=handle,stderr=errors,
                                                 preexec_fn=child_limits)
                        children.append(process)
                    if process.wait():raise RuntimeError('evaluation failed; see evaluation-errors.log')
            atomic_save(root/'status.json',dict(status='stage_completed',stage=stage))
        atomic_save(root/'status.json',dict(status='completed',stage=targets[-1]))
    except BaseException as exc:
        for process in children:
            if process.poll() is None:process.send_signal(signal.SIGTERM)
        for process in children:
            if process.poll() is None:process.wait()
        atomic_save(root/'status.json',dict(status='stopped',reason=str(exc) or type(exc).__name__))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--targets',type=int,nargs='+',default=[250,1000,5000])
    p.add_argument('--blocks',type=int,default=200)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();campaign(a.output,a.targets,a.blocks,a.resume)
