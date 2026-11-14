"""Bounded, prospectively frozen continuation of preserved learning with v2 paths."""
import argparse,json,os,time
from datetime import datetime
from pathlib import Path
from random import Random
from poker.config import load_checkpoint
from poker.compare import paired_difference
from poker.train import atomic_save,tuples
from poker.train_cfr import run_lock
from poker.trajectory import ALGORITHM,AVERAGING,fork_allowed
from scripts.sustained_train import digest,stage_config
from scripts import scaling_validation as workers

ROOT=Path('runs/bot/traversal-pilot-001')
PARENTS=[Path(f'runs/bot/scaling-validation-001/candidate-{i}-2850.json') for i in range(2)]
ADDITIONS=[50,250,750,1500]
VALIDATION=workers.VALIDATION
ROBUSTNESS=workers.ROBUSTNESS
CRITERIA=Path('docs/TRAVERSAL-PILOT.md')
TRAVERSAL='trajectory-v2'
PREFIX_DEPTH=None
EXPECTED_BASES=[2606,2850]
SCREEN_SEED=8300000
CONFIRM_SEED=8400000
ADDITIONAL_SOURCES=[]
POPULATION=None
FORK_FLAG='--fork-traversal'
REQUIRE_TIGHT_IMPROVEMENT=False


def recipe(saved):
    result={**saved['config'],'traversal':TRAVERSAL,'trajectory_samples':8,'averaging_sampler':AVERAGING}
    if PREFIX_DEPTH is not None:result['regret_prefix_depth']=PREFIX_DEPTH
    if POPULATION is not None:result['opponent_population']=POPULATION
    return result


def config(saved,target):
    cfg=stage_config(saved,target)
    cfg.update(recipe(saved))
    return cfg


def verified_resume(path,parent,target,expected):
    if not path.exists():return parent,False
    saved=load_checkpoint(path,model_id='bot-opponent-v1');original=load_checkpoint(parent,model_id='bot-opponent-v1')
    if (saved['algorithm']!=ALGORITHM or saved['config']!=expected
            or not original['iterations']<=saved['iterations']<=target
            or saved['limits']['max_iteration_nodes']!=original['limits']['max_iteration_nodes']
            or saved['limits']['max_information_sets']!=original['limits']['max_information_sets']):
        raise ValueError('trajectory resume lineage/config/limit mismatch')
    complete=saved['iterations']==target and saved['stop_reason']=='target_reached'
    if not complete and saved['stop_reason'] not in ('interrupted','run time budget reached','running'):
        raise ValueError('resource/unknown stop requires investigation; no unchanged retry')
    Random().setstate(tuples(saved['rng']))
    return path,complete


def verify(manifest):
    for entry in manifest['inputs']:
        if digest(Path(entry['path']))!=entry['sha256']:raise ValueError('frozen pilot input changed')
    for path,sha in manifest['sources'].items():
        if digest(Path(path))!=sha:raise ValueError('frozen pilot code changed')
    if digest(CRITERIA)!=manifest['criteria_sha256']:raise ValueError('pilot criteria changed')


def evaluate_requests(requests,deadline,seed,blocks):
    for index in range(0,len(requests),2):workers.evaluate_parallel(requests[index:index+2],deadline,seed=seed,blocks=blocks)


def coverage(report):
    result={}
    for pop,row in report['populations'].items():
        post=[v for s,v in row['coverage_by_street'].items() if s!='0']
        n=sum(v['decisions'] for v in post);hits=sum(v['positive_weight_rows'] for v in post)
        result[pop]=dict(decisions=n,positive=hits,fraction=hits/n if n else 0)
    return result


def tight_coverage_improvement(candidate,baseline):
    after=candidate['tight'];before=baseline['tight'];gain=after['fraction']-before['fraction']
    return dict(candidate=after,baseline=before,percentage_point_gain=100*gain,
                required_min_fraction=.05,required_min_decisions=200,required_percentage_point_gain=2.,
                passed=after['decisions']>=200 and after['fraction']>=.05 and gain>=.02-1e-12)


def main(deadline,resume=False):
    deadline=datetime.fromisoformat(deadline)
    if deadline.tzinfo is None or not time.time()<deadline.timestamp()<=time.time()+12*3600:raise ValueError('finite invocation deadline within12h required')
    ROOT.mkdir(parents=True,exist_ok=True);workers.ROOT=ROOT
    with run_lock(ROOT/'supervisor'):
        inputs=[*PARENTS,VALIDATION,*ROBUSTNESS]
        manifest=dict(algorithm=ALGORITHM,recipe=POPULATION,parents=[str(p) for p in PARENTS],additions=ADDITIONS,samples=8,
            deadline=deadline.isoformat(),screen_seed=SCREEN_SEED,screen_blocks=200,confirm_seed=CONFIRM_SEED,confirm_blocks=2000,
            inputs=[dict(path=str(p.resolve()),sha256=digest(p)) for p in inputs],
            criteria=str(CRITERIA),criteria_sha256=digest(CRITERIA),
            sources={str(p):digest(p) for p in [*sorted(Path('poker').glob('*.py')),Path(__file__),Path(workers.__file__),Path('scripts/limited_worker.py'),Path('scripts/sustained_train.py'),*ADDITIONAL_SOURCES]})
        path=ROOT/'manifest.json'
        if path.exists():
            if not resume or json.loads(path.read_text())!=manifest:raise ValueError('existing/mismatched pilot manifest')
        elif resume:raise ValueError('missing pilot manifest')
        else:atomic_save(path,manifest)
        atomic_save(ROOT/'invocation.json',dict(pid=os.getpid(),resume=resume,deadline=deadline.isoformat()))
        try:
            bases=[];expected=[]
            for i,p in enumerate(PARENTS):
                s=load_checkpoint(p,model_id='bot-opponent-v1');bases.append(s['iterations'])
                e=recipe(s)
                if s['iterations']!=EXPECTED_BASES[i] or not fork_allowed(s['config'],e):raise ValueError('unexpected source recipe/state')
                expected.append(e)
                del s
            for index,addition in enumerate(ADDITIONS):
                verify(manifest);jobs=[]
                for i,parent in enumerate(PARENTS):
                    source=parent if index==0 else ROOT/f'candidate-{i}-plus-{ADDITIONS[index-1]}.json'
                    target=bases[i]+addition;p=ROOT/f'candidate-{i}-plus-{addition}.json'
                    lineage=dict(path=str(source.resolve()),sha256=digest(source))
                    lp=ROOT/f'candidate-{i}-plus-{addition}-parent.json'
                    if lp.exists():
                        if json.loads(lp.read_text())!=lineage:raise ValueError('stage parent changed')
                    elif p.exists():raise ValueError('missing lineage')
                    else:atomic_save(lp,lineage)
                    state=load_checkpoint(source);cfg=config(state,target);del state
                    actual,complete=verified_resume(p,source,target,expected[i])
                    if not complete:
                        cp=ROOT/f'candidate-{i}-plus-{addition}-config.json';atomic_save(cp,cfg)
                        args=['--config',str(cp),'--output',str(p),'--resume',str(actual)]
                        if actual==parent:args+=[FORK_FLAG]
                        jobs.append(dict(output=p,log=ROOT/f'candidate-{i}-plus-{addition}.log',metrics=ROOT/f'candidate-{i}-plus-{addition}-metrics.json',args=args))
                atomic_save(ROOT/'status.json',dict(status='training',added_iterations=addition,workers=len(jobs)))
                workers.run_pair(jobs,deadline.timestamp())
                requests=[]
                for i,parent in enumerate(PARENTS):
                    p=ROOT/f'candidate-{i}-plus-{addition}.json';s=load_checkpoint(p)
                    if s['iterations']!=bases[i]+addition or s['stop_reason']!='target_reached':raise RuntimeError('incomplete training stage')
                    del s
                    requests.extend([dict(checkpoint=p,opponent=VALIDATION,uniform=False,output=ROOT/f'candidate-{i}-plus-{addition}-screen.json'),dict(checkpoint=parent,opponent=VALIDATION,uniform=False,output=ROOT/f'candidate-{i}-baseline-screen.json')])
                atomic_save(ROOT/'status.json',dict(status='screening',added_iterations=addition))
                evaluate_requests(requests,deadline.timestamp(),SCREEN_SEED,200)
                screens={}
                for i in range(2):
                    candidate=json.loads((ROOT/f'candidate-{i}-plus-{addition}-screen.json').read_text())
                    baseline=json.loads((ROOT/f'candidate-{i}-baseline-screen.json').read_text())
                    pair=paired_difference(candidate,baseline)
                    screens[str(i)]=dict(paired=pair,coverage=coverage(candidate))
                    atomic_save(ROOT/f'candidate-{i}-plus-{addition}-screen-paired.json',screens[str(i)])
                # Screening only stops for clear harmful evidence, never promotes.
                if any(v['ci95'][1]<-10 for row in screens.values() for v in row['paired'].values()):raise RuntimeError('screening detected material regression; investigate before further training')
            verify(manifest);requests=[];last=ADDITIONS[-1]
            for i,parent in enumerate(PARENTS):
                for population,opponent in [('validation',VALIDATION),('independent',ROBUSTNESS[i])]:
                    for kind,checkpoint in [('candidate',ROOT/f'candidate-{i}-plus-{last}.json'),('baseline',parent),('uniform',ROOT/f'candidate-{i}-plus-{last}.json')]:
                        requests.append(dict(checkpoint=checkpoint,opponent=opponent,uniform=kind=='uniform',output=ROOT/f'candidate-{i}-{population}-{kind}.json'))
            atomic_save(ROOT/'status.json',dict(status='confirmation'))
            evaluate_requests(requests,deadline.timestamp(),CONFIRM_SEED,2000)
            final={}
            for i in range(2):
                final[str(i)]={}
                for population in ('validation','independent'):
                    c,b,u=[json.loads((ROOT/f'candidate-{i}-{population}-{kind}.json').read_text()) for kind in ('candidate','baseline','uniform')]
                    key=c['opponent_checkpoint_hashes'][0];pair=paired_difference(c,b,confidence=.9875);uniform=paired_difference(c,u);cov=coverage(c)
                    final[str(i)][population]=dict(paired=pair,uniform=uniform,coverage=cov,
                        gain_pass=pair[key]['interval'][0]>0 and uniform[key]['ci95'][0]>0,
                        coverage_pass=all(row['decisions']>=200 and row['fraction']>=.05 for pop,row in cov.items() if pop in ('calling','mixed',key)),
                        material_regression=any(v['interval'][1]<-10 for v in pair.values()))
                    if REQUIRE_TIGHT_IMPROVEMENT:
                        improvement=tight_coverage_improvement(cov,coverage(b))
                        final[str(i)][population]['tight_coverage_improvement']=improvement
                        final[str(i)][population]['coverage_pass'] &= improvement['passed']
            atomic_save(ROOT/'learning-evidence.json',final)
            metrics=[json.loads((ROOT/f'candidate-{i}-plus-{a}-metrics.json').read_text()) for i in range(2) for a in ADDITIONS]
            health=[json.loads(line) for line in (ROOT/'health.jsonl').read_text().splitlines()]
            overlap=any(len(row['workers'])==2 and row['module']=='poker.train_cfr' for row in health)
            operational=workers.resource_pass(metrics,min(row['mem_available_bytes'] for row in health),overlap)
            states=[]
            for i in range(2):
                s=load_checkpoint(ROOT/f'candidate-{i}-plus-{last}.json');sets=len(set(s['regrets'])|set(s['sums']))
                operational=operational and sets<.8*s['limits']['max_information_sets'] and s['traversal_stats']['iterations']==last
                states.append(dict(candidate=i,sets=sets,limits=s['limits'],traversal_stats=s['traversal_stats']))
                del s
            # Per invocation throughput includes checkpoint I/O; fixed before launch.
            previous=0
            for j,a in enumerate(ADDITIONS):
                operational=operational and all(sum(attempt['wall_seconds'] for attempt in metrics[i*len(ADDITIONS)+j].get('attempts',[metrics[i*len(ADDITIONS)+j]]))/(a-previous)<=20 for i in range(2));previous=a
            atomic_save(ROOT/'resource-evidence.json',dict(operational_pass=operational,metrics=metrics,states=states,minimum_available=min(row['mem_available_bytes'] for row in health)))
            learning=all(v['gain_pass'] and v['coverage_pass'] and not v['material_regression'] for row in final.values() for v in row.values())
            atomic_save(ROOT/'status.json',dict(status='completed_pending_review',operational_pass=operational,learning_pass=learning,ready_for_sustained_training=operational and learning,competitive_strength_established=False))
        except BaseException as exc:
            atomic_save(ROOT/'status.json',dict(status='stopped',reason=str(exc) or type(exc).__name__));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--deadline',required=True);p.add_argument('--resume',action='store_true')
    a=p.parse_args();main(a.deadline,a.resume)
