"""Qualified v7 continuation; strict evidence gate and finite per-stage runtime."""
import argparse,json
from pathlib import Path
from poker.joint_average import ALGORITHM,AVERAGING,fork_allowed
from scripts.sustained_train import digest,stage_config as original_stage_config
from scripts import trajectory_pilot as driver

ROOT=Path('runs/bot/joint-average-training-001')
PARENTS=[Path(f'runs/bot/coverage-curriculum-001/candidate-{i}-plus-300.json') for i in range(2)]
ENGINEERING=Path('runs/bot/joint-average-probe-001/qualification.json')
RUNTIME=Path('docs/JOINT-AVERAGING-RUNTIME.md')
ALGORITHM_SOURCES=[Path('poker')/name for name in ('joint_average.py','prefix.py','trajectory.py','cfr.py','population.py')]


def verify_engineering():
    q=json.loads(ENGINEERING.read_text());rows=q.get('candidates',[])
    if (q.get('algorithm')!=ALGORITHM or not q.get('independently_verified') or not q.get('eligible_for_learning_pilot')
        or len(rows)!=2 or {r.get('candidate') for r in rows}!={0,1} or not all(r.get('gates') and all(r['gates'].values()) for r in rows)):
        raise ValueError('both independently qualified joint-averaging learners required')
    sources=q.get('algorithm_sources',{})
    if set(sources)!={str(p) for p in ALGORITHM_SOURCES} or any(digest(Path(p))!=sha for p,sha in sources.items()):
        raise ValueError('qualified estimator source changed')
    parents=q.get('parents',[])
    if len(parents)!=2 or {str(Path(v['path']).resolve()) for v in parents}!={str(p.resolve()) for p in PARENTS}:
        raise ValueError('qualified parent set changed')
    for v in parents:
        if digest(Path(v['path']))!=v['sha256']:raise ValueError('qualified parent bytes changed')


def stage_config(saved,target):
    cfg=original_stage_config(saved,target);cfg['max_seconds']=7200
    return cfg


def configure():
    driver.ROOT=ROOT;driver.PARENTS=PARENTS;driver.EXPECTED_BASES=[4906,5150];driver.ADDITIONS=[50,200,500,1000,2000]
    driver.ALGORITHM=ALGORITHM;driver.AVERAGING=AVERAGING;driver.TRAVERSAL='joint-average-v7';driver.PREFIX_DEPTH=2
    driver.POPULATION='guarded-tables-v2';driver.FORK_FLAG='--fork-traversal';driver.REQUIRE_TIGHT_IMPROVEMENT=True
    driver.fork_allowed=fork_allowed;driver.CRITERIA=Path('docs/JOINT-AVERAGING-PILOT.md')
    driver.SCREEN_SEED=10200000;driver.CONFIRM_SEED=10300000;driver.stage_config=stage_config
    driver.ADDITIONAL_SOURCES=[Path(__file__),ENGINEERING,RUNTIME,Path('configs/bot.json')]


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--deadline',required=True);p.add_argument('--resume',action='store_true')
    a=p.parse_args();verify_engineering();configure();driver.main(a.deadline,a.resume)
