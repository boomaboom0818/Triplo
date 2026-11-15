"""One frozen read-only candidate inference proposal, with bounded local workers."""
import argparse,json,os,shutil,signal,subprocess,sys,time
from contextlib import ExitStack
from datetime import datetime,timezone
from pathlib import Path
from poker.compare import paired_difference
from poker.train import atomic_save
from poker.train_cfr import run_lock
from scripts.scaling_validation import available_memory,VALIDATION,ROBUSTNESS
from scripts.sustained_train import digest,validate_evaluation
from scripts.policy_backoff_component import VARIANTS

ROOT=Path('runs/bot/policy-backoff-001')
CHECKPOINTS=[Path(f'runs/bot/coverage-curriculum-001/candidate-{i}-plus-300.json') for i in range(2)]
CRITERIA=Path('docs/POLICY-BACKOFF-COMPONENT.md')
POLL_SECONDS=10


def verify(manifest):
    for path,sha in manifest['frozen'].items():
        if digest(Path(path))!=sha:raise ValueError('frozen backoff source/input/criteria changed: '+path)


def validate_report(report,checkpoint,opponent,seed,blocks,variant):
    validate_evaluation(report,checkpoint,opponent,seed,variant=='uniform',blocks)
    if (report.get('candidate_policy_variant')!=VARIANTS[variant] or not report.get('inference_component_only')
        or report.get('component_source_sha256')!=digest(Path('scripts/policy_backoff_component.py'))
        or report.get('input_hashes')!={str(p.resolve()):digest(p) for p in (checkpoint,opponent)}):
        raise ValueError('inference component provenance mismatch')
    for name,row in report['populations'].items():
        source=report['policy_sources_by_population'][name]
        if sum(source.values())!=row['candidate_decisions']:raise ValueError('component source attribution mismatch')
    return report


def command(job,deadline):
    return [sys.executable,'-m','scripts.policy_backoff_component','--checkpoint',str(job['checkpoint']),
            '--opponent',str(job['opponent']),'--output',str(job['output']),'--variant',job['variant'],
            '--blocks',str(job['blocks']),'--seed',str(job['seed']),
            '--deadline',datetime.fromtimestamp(deadline,timezone.utc).isoformat()]


def run_batch(jobs,deadline):
    if not jobs:return
    if len(jobs)>2 or len({j['output'].resolve() for j in jobs})!=len(jobs):raise ValueError('at most two distinct outputs required')
    if available_memory()<(3*len(jobs)+2)*1024**3:raise RuntimeError('insufficient host reserve for bounded evaluations')
    if shutil.disk_usage(ROOT).free<10*1024**3:raise RuntimeError('insufficient disk reserve for bounded evaluations')
    children=[]
    def interrupted(signum,frame):raise InterruptedError('component supervisor interrupted')
    previous={sig:signal.signal(sig,interrupted) for sig in (signal.SIGTERM,signal.SIGINT)}
    try:
        with ExitStack() as stack:
            for job in jobs:
                stack.enter_context(run_lock(Path(str(job['output'])+'.dispatch')))
                if job['output'].exists():raise ValueError('refuse duplicate/overwrite component output')
                if time.time()>=deadline:raise TimeoutError('component supervisor deadline reached')
                log=stack.enter_context(Path(str(job['output'])+'.log').open('a'))
                # Separate supervisor-owned lock; child owns its publication lock.
                children.append(subprocess.Popen(command(job,deadline),stdout=log,stderr=subprocess.STDOUT))
            while True:
                if time.time()>=deadline:raise TimeoutError('component supervisor deadline reached')
                row=dict(time=time.time(),mem_available_bytes=available_memory(),disk_free_bytes=shutil.disk_usage(ROOT).free,workers=[])
                for child in children:
                    try:
                        fields={}
                        for line in Path(f'/proc/{child.pid}/status').read_text().splitlines():
                            if line.startswith(('VmRSS:','VmSize:')):
                                key,value,*_=line.split();fields[key.rstrip(':')]=int(value)*1024
                        row['workers'].append(dict(pid=child.pid,**fields))
                    except FileNotFoundError:pass
                with (ROOT/'health.jsonl').open('a') as handle:handle.write(json.dumps(row)+'\n')
                if row['mem_available_bytes']<2*1024**3 or row['disk_free_bytes']<10*1024**3:raise RuntimeError('host memory/disk reserve breached')
                codes=[p.poll() for p in children]
                if any(c not in (None,0) for c in codes):raise RuntimeError(f'component worker failed {codes}; investigate logs/attempts before recovery')
                if all(c==0 for c in codes):break
                time.sleep(POLL_SECONDS)
    finally:
        for p in children:
            if p.poll() is None:p.terminate()
        for p in children:
            if p.poll() is None:
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:p.kill();p.wait()
        for sig,handler in previous.items():signal.signal(sig,handler)


def evaluate_phase(phase,seed,blocks,deadline,manifest):
    requests=[]
    for i,cp in enumerate(CHECKPOINTS):
        for label,op in [('validation',VALIDATION),('independent',ROBUSTNESS[i])]:
            for variant in VARIANTS:
                requests.append(dict(checkpoint=cp,opponent=op,variant=variant,seed=seed,blocks=blocks,
                    output=ROOT/f'{phase}-{i}-{label}-{variant}.json'))
    for start in range(0,len(requests),2):
        verify(manifest);batch=requests[start:start+2]
        run_batch(batch,deadline)
        for job in batch:
            validate_report(json.loads(job['output'].read_text()),job['checkpoint'],job['opponent'],seed,blocks,job['variant'])
    evidence={}
    for i in range(2):
        evidence[str(i)]={}
        for label in ('validation','independent'):
            reports={v:json.loads((ROOT/f'{phase}-{i}-{label}-{v}.json').read_text()) for v in VARIANTS}
            backoff=reports['backoff'];key=backoff['opponent_checkpoint_hashes'][0]
            pair=paired_difference(backoff,reports['average'],confidence=.9875 if phase=='confirmation' else .95)
            uniform=paired_difference(backoff,reports['uniform'])
            interval='interval' if phase=='confirmation' else 'ci95'
            evidence[str(i)][label]=dict(paired=pair,uniform=uniform,
                component_gain_pass=pair[key][interval][0]>0 and uniform[key]['ci95'][0]>0,
                material_regression=any(v[interval][1]<-10 for v in pair.values()),
                source_attribution=backoff['policy_sources_by_population'],
                raw_average_coverage={p:row['coverage_by_street'] for p,row in backoff['populations'].items()})
    atomic_save(ROOT/f'{phase}-evidence.json',evidence)
    return evidence


def main(deadline):
    if deadline.tzinfo is None or not time.time()<deadline.timestamp()<=time.time()+12*3600:raise ValueError('finite timezone-aware invocation deadline within12h required')
    ROOT.mkdir(parents=True,exist_ok=True)
    with run_lock(ROOT/'supervisor'):
        if (ROOT/'manifest.json').exists():raise ValueError('existing pilot; inspect outcome before any recovery')
        review=json.loads(Path('runs/bot/coverage-curriculum-001/qualification.json').read_text())
        if not review['all_twelve_confirmation_reports_validated'] or review['learning_status']!='FAIL':raise ValueError('completed curriculum failure review required')
        files=[*sorted(Path('poker').glob('*.py')),Path(__file__),Path('scripts/policy_backoff_component.py'),Path('scripts/sustained_train.py'),Path('scripts/scaling_validation.py'),Path('configs/bot.json'),Path('runs/bot/coverage-curriculum-001/qualification.json'),CRITERIA,*CHECKPOINTS,VALIDATION,*ROBUSTNESS]
        manifest=dict(deadline=deadline.isoformat(),proposal=VARIANTS['backoff'],screen_seed=9700000,confirm_seed=9800000,
            screen_blocks=200,confirm_blocks=2000,frozen={str(p):digest(p) for p in files},
            curriculum_qualification_sha256=digest(Path('runs/bot/coverage-curriculum-001/qualification.json')))
        atomic_save(ROOT/'manifest.json',manifest);atomic_save(ROOT/'invocation.json',dict(pid=os.getpid(),deadline=deadline.isoformat()))
        try:
            for phase,seed,blocks in [('screen',9700000,200),('confirmation',9800000,2000)]:
                atomic_save(ROOT/'status.json',dict(status=phase))
                result=evaluate_phase(phase,seed,blocks,deadline.timestamp(),manifest)
                if any(v['material_regression'] for row in result.values() for v in row.values()):
                    atomic_save(ROOT/'status.json',dict(status='completed_pending_review',component_pass=False,reason=phase+' material regression',ready_for_sustained_training=False));return
            verify(manifest)
            metrics=[json.loads(p.read_text()) for p in ROOT.glob('*.json.metrics.json')]
            health=[json.loads(s) for s in (ROOT/'health.jsonl').read_text().splitlines()]
            operational=len(metrics)==24 and all(a['status']=='completed' and a['address_space_limit_bytes']==3*1024**3 and a['VmPeak']<.8*3*1024**3 and a['peak_rss_bytes']<.8*3*1024**3 for m in metrics for a in m['attempts']) and min(h['mem_available_bytes'] for h in health)>=2*1024**3
            atomic_save(ROOT/'status.json',dict(status='completed_pending_review',operational_pass=operational,
                component_pass=all(v['component_gain_pass'] and not v['material_regression'] for row in result.values() for v in row.values()),
                raw_average_coverage_gate_replaced=False,ready_for_sustained_training=False,competitive_strength_established=False))
        except BaseException as exc:
            atomic_save(ROOT/'status.json',dict(status='stopped',reason=str(exc) or type(exc).__name__));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--deadline',required=True)
    main(datetime.fromisoformat(parser.parse_args().deadline))
