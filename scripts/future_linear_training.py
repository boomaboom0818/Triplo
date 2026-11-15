"""Strict engineering-gated future-linear continuation; bounded staged pilot."""
import argparse,json
from pathlib import Path
from poker.future_linear_average import ALGORITHM,AVERAGING
from scripts import trajectory_pilot as driver
from scripts.sustained_train import digest,stage_config as original_stage_config

ROOT=Path('runs/bot/future-linear-training-001')
PARENTS=[Path(f'runs/bot/joint-average-training-001/candidate-{i}-plus-2000.json') for i in range(2)]
ENGINEERING=Path('runs/bot/future-linear-probe-002/qualification.json')
MATH=[Path('poker')/name for name in ('future_linear_average.py','joint_average.py','prefix.py','trajectory.py','cfr.py','population.py')]


def fork_allowed(before,after):
    old=dict(before);new=dict(after)
    if old.pop('traversal',None)!='joint-average-v7' or new.pop('traversal',None)!='future-linear-average-v8':return False
    if old.pop('averaging_sampler',None)!='regret-path-mis-average-v7' or new.pop('averaging_sampler',None)!=AVERAGING:return False
    return old==new


def verify_engineering():
    q=json.loads(ENGINEERING.read_text());rows=q.get('candidates',[])
    if (q.get('independently_verified') is not True or q.get('eligible_for_registration') is not True or len(rows)!=2
            or {r.get('candidate') for r in rows}!={0,1} or not all(r.get('all_clocks_exact') and r.get('pairs')==288 and r.get('gates') and all(r['gates'].values()) and all(r.get('max_raw_weight_by_clock',{}).get(str(k),0)>0 for k in (1,50,2000)) for r in rows)):
        raise ValueError('both independently reviewed future-linear engineering gates required')
    if set(q.get('algorithm_sources',{}))!={str(p) for p in MATH}:raise ValueError('qualified math source set changed')
    if any(digest(Path(p))!=sha for p,sha in q['algorithm_sources'].items()):raise ValueError('qualified math source changed')
    parents=q.get('parents',[])
    if len(parents)!=2 or {str(Path(p['path']).resolve()) for p in parents}!={str(p.resolve()) for p in PARENTS}:raise ValueError('qualified parent set changed')
    if any(digest(Path(p['path']))!=p['sha256'] for p in parents):raise ValueError('qualified parent bytes changed')


def stage_config(saved,target):
    cfg=original_stage_config(saved,target);cfg['max_seconds']=7200
    return cfg


def configure():
    driver.ROOT=ROOT;driver.PARENTS=PARENTS;driver.EXPECTED_BASES=[6906,7150];driver.ADDITIONS=[50,200,500,1000,2000]
    driver.ALGORITHM=ALGORITHM;driver.AVERAGING=AVERAGING;driver.TRAVERSAL='future-linear-average-v8';driver.PREFIX_DEPTH=2
    driver.POPULATION='guarded-tables-v2';driver.FORK_FLAG='--fork-traversal';driver.REQUIRE_TIGHT_IMPROVEMENT=True
    driver.fork_allowed=fork_allowed;driver.CRITERIA=Path('docs/FUTURE-LINEAR-AVERAGING.md')
    driver.SCREEN_SEED=11500000;driver.CONFIRM_SEED=11600000;driver.stage_config=stage_config
    driver.ADDITIONAL_SOURCES=[Path(__file__),ENGINEERING,Path('docs/FUTURE-LINEAR-RUNTIME.md'),Path('configs/bot.json')]


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--deadline',required=True);p.add_argument('--resume',action='store_true')
    a=p.parse_args();verify_engineering();configure();driver.main(a.deadline,a.resume)
