"""Predeclared bounded two-worker scaling and learning qualification."""
import argparse,json,os,resource,shutil,signal,subprocess,sys,time
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from poker.config import load_checkpoint
from poker.train import atomic_save
from poker.train_cfr import run_lock
from poker.compare import paired_difference
from scripts.sampling_curve import resume_checkpoint
from scripts.sustained_train import digest,stage_config,evaluate,validate_evaluation
LIMIT=4*1024**3

def scaling_limits(limit=LIMIT):
    resource.setrlimit(resource.RLIMIT_AS,(limit,limit))
    os.nice(10)

ROOT=Path('runs/bot/scaling-validation-001')
PARENTS=[Path(f'runs/bot/population-pilot-001/candidate-{i}-control.json') for i in range(2)]
ROBUSTNESS=[Path(f'runs/bot/sustained-002/candidate-{1-i}-2000.json') for i in range(2)]
VALIDATION=Path('runs/bot/validation-extended-001/validation-750.json')
STAGES=[2350,2600,2850,3150,3400,3650,3950,4200]
DEADLINE=datetime.fromisoformat('2026-11-15T00:00:00+00:00')


def available_memory():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):return int(line.split()[1])*1024
    raise RuntimeError('cannot measure available memory')


def resource_pass(metrics,min_available,two_workers_observed=True):
    return bool(metrics) and two_workers_observed and min_available>=2*1024**3 and all(
        m['status']=='completed' and m['address_space_limit_bytes']==LIMIT
        and all(a['status']=='completed' for a in m.get('attempts',[]))
        and m['peak_rss_bytes']<.8*LIMIT and m['VmPeak']<.8*LIMIT for m in metrics)


def run_pair(jobs,deadline):
    if not jobs:return
    if len(jobs)>2:raise ValueError("protocol permits at most two workers")
    modules=[j.get('module','poker.train_cfr') for j in jobs]
    if len(set(modules))!=1 or modules[0] not in ('poker.train_cfr','poker.evaluate'):raise ValueError('uniform allowed worker modules required')
    memory_gib=4 if modules[0]=='poker.train_cfr' else 3
    worker_limit=memory_gib*1024**3
    if available_memory()<len(jobs)*worker_limit+2*1024**3:raise RuntimeError('insufficient host memory for two bounded workers')
    if len({str(j['output'].resolve()) for j in jobs})!=len(jobs):raise ValueError('duplicate outputs')
    if len(os.sched_getaffinity(0))<2:raise RuntimeError("two available CPU cores required")
    started=time.monotonic();children=[]
    def cleanup():
        for child in children:
            if child.poll() is None:child.send_signal(signal.SIGTERM)
        for child in children:
            if child.poll() is None:
                try:child.wait(timeout=120)
                except subprocess.TimeoutExpired:child.kill();child.wait()
    def interrupted(signum,frame):raise KeyboardInterrupt
    old={s:signal.signal(s,interrupted) for s in (signal.SIGTERM,signal.SIGINT)}
    try:
        with ExitStack() as stack:
            for job in jobs:
                if time.time()>=deadline:raise TimeoutError('scaling deadline reached')
                log=stack.enter_context(job['log'].open('a'))
                args=[sys.executable,'-m','scripts.limited_worker','--metrics',str(job['metrics']),'--memory-gib',str(memory_gib),job.get('module','poker.train_cfr'),*job['args']]
                children.append(subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,preexec_fn=lambda:scaling_limits(worker_limit)))
            while True:
                now=time.time()
                if now>=deadline:raise TimeoutError('scaling deadline reached')
                mem=available_memory()
                row={'time':now,'mem_available_bytes':mem,'disk_free_bytes':shutil.disk_usage(ROOT).free,'module':modules[0],'workers':[]}
                for child in children:
                    try:
                        fields={}
                        for line in Path(f'/proc/{child.pid}/status').read_text().splitlines():
                            if line.startswith(('VmRSS:','VmSize:')):
                                key,value,*_=line.split();fields[key.rstrip(':')]=int(value)*1024
                        row['workers'].append({'pid':child.pid,**fields})
                    except FileNotFoundError:pass
                with (ROOT/'health.jsonl').open('a') as handle:handle.write(json.dumps(row)+'\n')
                if mem<2*1024**3:raise RuntimeError('host memory reserve below2GiB')
                if row['disk_free_bytes']<10*1024**3:raise RuntimeError('disk reserve below10GiB')
                codes=[c.poll() for c in children]
                if any(code not in (None,0) for code in codes):raise RuntimeError(f'{modules[0]} worker failed: {codes}; inspect logs')
                if all(code==0 for code in codes):
                    with (ROOT/'batches.jsonl').open('a') as handle:handle.write(json.dumps({'workers':len(jobs),'module':modules[0],'wall_seconds':time.monotonic()-started,'metrics':[str(j['metrics']) for j in jobs]})+'\n')
                    break
                time.sleep(10)
    finally:
        cleanup()
        for sig,handler in old.items():signal.signal(sig,handler)


def evaluate_parallel(requests,deadline):
    """Up to two fixed-seed evaluations, held output locks through atomic publication."""
    with ExitStack() as stack:
        jobs=[]
        for request in requests:
            output=request['output'];checkpoint=request['checkpoint'];opponent=request['opponent']
            stack.enter_context(run_lock(output))
            if output.exists():
                validate_evaluation(json.loads(output.read_text()),checkpoint,opponent,8200000,request['uniform'],2000)
                continue
            pending=Path(str(output)+'.pending')
            if pending.exists():
                # Preserve diagnostics from an interrupted attempt before a deliberate resume.
                pending.replace(Path(str(pending)+f'.attempt-{time.time_ns()}'))
            args=['--checkpoint',str(checkpoint),'--blocks','2000','--seed','8200000','--copies','1','--opponent-checkpoints',str(opponent)]
            if request['uniform']:args+=['--uniform-control']
            jobs.append({'module':'poker.evaluate','output':output,'log':pending,'metrics':Path(str(output)+'.metrics.json'),'args':args})
        run_pair(jobs,deadline)
        for request in requests:
            output=request['output']
            if output.exists():continue
            pending=Path(str(output)+'.pending')
            validate_evaluation(json.loads(pending.read_text()),request['checkpoint'],request['opponent'],8200000,request['uniform'],2000)
            pending.replace(output)


def verify_inputs(manifest):
    for entry in [*manifest['parents'],*manifest['robustness'],manifest['validation']]:
        if digest(Path(entry['path']))!=entry['sha256']:raise ValueError('frozen input changed')
    for file,sha in manifest['algorithm_sources'].items():
        if digest(Path(file))!=sha:raise ValueError('algorithm changed during qualification')
    if digest(Path(manifest['criteria']))!=manifest['criteria_sha256']:raise ValueError('qualification criteria changed')


def main(resume=False):
    ROOT.mkdir(parents=True,exist_ok=True)
    with run_lock(ROOT/'supervisor'):
        manifest={'parents':[{'path':str(p.resolve()),'sha256':digest(p)} for p in PARENTS],
            'robustness':[{'path':str(p.resolve()),'sha256':digest(p)} for p in ROBUSTNESS],
            'validation':{'path':str(VALIDATION.resolve()),'sha256':digest(VALIDATION)},'stages':STAGES,
            'deadline':DEADLINE.isoformat(),'screen_seed':8100000,'screen_blocks':200,'confirm_seed':8200000,
            'confirm_blocks':2000,'workers':2,'memory_limit_per_worker':LIMIT,'purpose':'qualification; no competitive-strength claim',
            'criteria':'docs/SCALING-VALIDATION.md','criteria_sha256':digest(Path('docs/SCALING-VALIDATION.md')),
            'algorithm_sources':{str(p):digest(p) for p in Path('poker').glob('*.py')}}
        path=ROOT/'manifest.json'
        if path.exists():
            if not resume or json.loads(path.read_text())!=manifest:raise ValueError('existing/mismatched validation manifest')
        elif resume:raise ValueError('missing manifest')
        else:atomic_save(path,manifest)
        atomic_save(ROOT/'invocation.json',{'pid':os.getpid(),'resume':resume})
        try:
            for parent in PARENTS:
                p=load_checkpoint(parent,model_id='bot-opponent-v1')
                if p['iterations']!=2100 or p['config'].get('opponent_population','historical-v1')!='historical-v1':raise ValueError('unqualified parent recipe')
            del p
            for index,stage in enumerate(STAGES):
                verify_inputs(manifest)
                jobs=[]
                for i,parent in enumerate(PARENTS):
                    source=parent if index==0 else ROOT/f'candidate-{i}-{STAGES[index-1]}.json'
                    cfg=stage_config(load_checkpoint(source),stage)
                    checkpoint=ROOT/f'candidate-{i}-{stage}.json';lineage={'path':str(source.resolve()),'sha256':digest(source)}
                    lineage_file=ROOT/f'candidate-{i}-{stage}-parent.json'
                    if lineage_file.exists():
                        if json.loads(lineage_file.read_text())!=lineage:raise ValueError('stage parent changed')
                    elif checkpoint.exists():raise ValueError('missing stage lineage')
                    else:atomic_save(lineage_file,lineage)
                    source,complete=resume_checkpoint(checkpoint,source,stage)
                    if not complete:
                        config=ROOT/f'candidate-{i}-{stage}-config.json';atomic_save(config,cfg)
                        jobs.append({'output':checkpoint,'log':ROOT/f'candidate-{i}-{stage}.log','metrics':ROOT/f'candidate-{i}-{stage}-metrics.json',
                            'args':['--config',str(config),'--output',str(checkpoint),'--resume',str(source)]})
                atomic_save(ROOT/'status.json',{'status':'training_pair','stage':stage,'workers_to_launch':len(jobs)})
                run_pair(jobs,DEADLINE.timestamp())
                for i in range(2):
                    checkpoint=ROOT/f'candidate-{i}-{stage}.json';saved=load_checkpoint(checkpoint)
                    if saved['iterations']!=stage or saved['stop_reason']!='target_reached':raise RuntimeError(f'candidate{i} stopped at{saved["iterations"]}: {saved["stop_reason"]}')
                    del saved
                    metrics=ROOT/f'candidate-{i}-{stage}-metrics.json'
                    if not metrics.exists():raise RuntimeError('completed checkpoint lacks peak resource measurements')
                    atomic_save(ROOT/'status.json',{'status':'screening','candidate':i,'stage':stage})
                    report=evaluate(checkpoint,ROOT/f'candidate-{i}-{stage}-screen.json',VALIDATION,8100000,False,DEADLINE.timestamp(),blocks=200)
                    baseline=evaluate(PARENTS[i],ROOT/f'candidate-{i}-baseline-screen.json',VALIDATION,8100000,False,DEADLINE.timestamp(),blocks=200)
                    atomic_save(ROOT/f'candidate-{i}-{stage}-screen-paired.json',paired_difference(report,baseline))
            verify_inputs(manifest)
            requests=[]
            for i,parent in enumerate(PARENTS):
                for population,opponent in [('validation',VALIDATION),('independent',ROBUSTNESS[i])]:
                    for kind,checkpoint in [('confirm',ROOT/f'candidate-{i}-4200.json'),('baseline',parent),('uniform',ROOT/f'candidate-{i}-4200.json')]:
                        requests.append({'checkpoint':checkpoint,'opponent':opponent,'uniform':kind=='uniform','output':ROOT/f'candidate-{i}-{population}-{kind}.json'})
            for index in range(0,len(requests),2):
                atomic_save(ROOT/'status.json',{'status':'confirming_parallel','batch':index//2})
                evaluate_parallel(requests[index:index+2],DEADLINE.timestamp())
            final={}
            for i,parent in enumerate(PARENTS):
                final[str(i)]={}
                for population,opponent in [('validation',VALIDATION),('independent',ROBUSTNESS[i])]:
                    atomic_save(ROOT/'status.json',{'status':'confirming','candidate':i,'population':population})
                    candidate=evaluate(ROOT/f'candidate-{i}-4200.json',ROOT/f'candidate-{i}-{population}-confirm.json',opponent,8200000,False,DEADLINE.timestamp(),blocks=2000)
                    baseline=evaluate(parent,ROOT/f'candidate-{i}-{population}-baseline.json',opponent,8200000,False,DEADLINE.timestamp(),blocks=2000)
                    uniform=evaluate(ROOT/f'candidate-{i}-4200.json',ROOT/f'candidate-{i}-{population}-uniform.json',opponent,8200000,True,DEADLINE.timestamp(),blocks=2000)
                    uniform_pair=paired_difference(candidate,uniform)[candidate['opponent_checkpoint_hashes'][0]]
                    paired=paired_difference(candidate,baseline,confidence=.9875)
                    key=candidate['opponent_checkpoint_hashes'][0]
                    rows=[r for s,r in candidate['populations'][key]['coverage_by_street'].items() if s!='0'];n=sum(r['decisions'] for r in rows);hits=sum(r['positive_weight_rows'] for r in rows)
                    final[str(i)][population]={'primary':paired[key],'all_populations':paired,'postflop_decisions':n,'postflop_positive':hits,
                        'coverage_pass':n>=200 and hits/n>=.05,'uniform_comparison':uniform_pair,'gain_pass':paired[key]['interval'][0]>0 and uniform_pair['ci95'][0]>0}
                    atomic_save(ROOT/'learning-evidence.json',final)
            metrics=[json.loads((ROOT/f'candidate-{i}-{stage}-metrics.json').read_text()) for i in range(2) for stage in STAGES]
            health=[json.loads(line) for line in (ROOT/'health.jsonl').read_text().splitlines()]
            overlap=any(len(r['workers'])==2 and r.get('module','poker.train_cfr')=='poker.train_cfr' for r in health)
            batches=[json.loads(line) for line in (ROOT/'batches.jsonl').read_text().splitlines()]
            cpu_ratios=[sum(json.loads(Path(p).read_text())['cpu_seconds'] for p in b['metrics'])/b['wall_seconds'] for b in batches if b['workers']==2 and b.get('module','poker.train_cfr')=='poker.train_cfr']
            table=[]
            for i in range(2):
                saved=load_checkpoint(ROOT/f'candidate-{i}-4200.json')
                table.append({'candidate':i,'sets':len(set(saved['regrets'])|set(saved['sums'])),'budget':saved['limits']['max_information_sets']})
                del saved
            operational=resource_pass(metrics,min(r['mem_available_bytes'] for r in health),overlap) and any(r>=1.5 for r in cpu_ratios) and all(t['sets']<.8*t['budget'] for t in table)
            atomic_save(ROOT/'resource-evidence.json',{'operational_pass':operational,'two_workers_observed':overlap,'metrics':metrics,'cpu_core_use_ratios':cpu_ratios,'table_headroom':table,'minimum_mem_available':min(r['mem_available_bytes'] for r in health)})
            learning=all(v['coverage_pass'] and v['gain_pass'] for pair in final.values() for v in pair.values())
            atomic_save(ROOT/'status.json',{'status':'completed_pending_review','operational_pass':operational,'learning_pass':learning,'competitive_strength_established':False})
        except BaseException as exc:
            atomic_save(ROOT/'status.json',{'status':'stopped','reason':str(exc) or type(exc).__name__});raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');main(p.parse_args().resume)
