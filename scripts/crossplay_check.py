"""Additional six-player screening against the other independent seed's policy."""
import argparse,json,time
from datetime import datetime,timezone
from pathlib import Path
from poker.config import load_checkpoint
from poker.train import atomic_save
from poker.train_cfr import run_lock
from poker.compare import paired_difference
from scripts.sustained_train import evaluate,digest

ROOT=Path('runs/bot/crossplay-001')
CURVE=Path('runs/bot/sustained-001')
DEADLINE=datetime.fromisoformat('2026-11-14T17:31:00+00:00')


def verify_lineages(candidate,parent,opponent):
    if candidate['config']!=parent['config']:raise ValueError('candidate recipe changed')
    if candidate['config']['seed']==opponent['config']['seed']:raise ValueError('opponent must use independent seed')
    if opponent['sums']==candidate['sums'] or opponent['sums'] in candidate['archive']:
        raise ValueError('opponent overlaps training population')


def validate_design(stage,seed,blocks,deadline):
    if stage<1 or seed<0 or blocks<2 or deadline.tzinfo is None:
        raise ValueError('positive stage, nonnegative seed, >=2 blocks and timezone-aware deadline required')


def main(root=ROOT,curve=CURVE,stage=1500,seed=5600000,blocks=1000,deadline=DEADLINE):
    validate_design(stage,seed,blocks,deadline)
    root.mkdir(parents=True,exist_ok=True)
    with run_lock(root/'supervisor'):
        if (root/'manifest.json').exists():raise ValueError('existing check; investigate before recovery')
        original=json.loads((curve/'manifest.json').read_text())['parents']
        if set(original)!={'candidate-0','candidate-1'}:raise ValueError('two independent candidates required')
        for entry in original.values():
            if digest(Path(entry['path']))!=entry['sha256']:raise ValueError('frozen parent changed')
        atomic_save(root/'manifest.json',dict(seed=seed,blocks=blocks,stage=stage,frozen_parents=original,deadline=deadline.isoformat(),purpose='additional exploratory screening; does not replace original gate',format='one candidate vs five copies of other independent frozen parent',competitive_strength_established=False))
        pending=['candidate-0','candidate-1']
        try:
            while pending:
                if datetime.now(timezone.utc)>=deadline:raise TimeoutError('deadline reached')
                progressed=False
                for name in list(pending):
                    candidate=curve/f'{name}-{stage}.json'
                    if not candidate.exists():continue
                    saved=load_checkpoint(candidate,model_id='bot-opponent-v1')
                    if saved['iterations']!=stage or saved['stop_reason']!='target_reached':continue
                    other='candidate-1' if name=='candidate-0' else 'candidate-0'
                    parent=Path(original[name]['path']);opponent=Path(original[other]['path'])
                    old=load_checkpoint(parent);frozen=load_checkpoint(opponent)
                    verify_lineages(saved,old,frozen)
                    if old['iterations']>=stage:raise ValueError('stage must exceed frozen parent iteration')
                    del saved,old,frozen
                    atomic_save(root/f'{name}-lineage.json',dict(candidate_sha256=digest(candidate),parent_sha256=digest(parent),opponent_sha256=digest(opponent)))
                    reports={}
                    for kind,path in [('candidate',candidate),('parent',parent),('uniform',candidate)]:
                        atomic_save(root/'status.json',dict(status='evaluating',candidate=name,kind=kind))
                        reports[kind]=evaluate(path,root/f'{name}-{kind}.json',opponent,seed,kind=='uniform',deadline.timestamp(),blocks=blocks)
                    paired={k:paired_difference(reports['candidate'],reports[k]) for k in ('parent','uniform')}
                    atomic_save(root/f'{name}-paired.json',paired)
                    pending.remove(name);progressed=True
                if pending and not progressed:
                    atomic_save(root/'status.json',dict(status='waiting_for_complete_checkpoint',stage=stage,pending=pending))
                    state=json.loads((curve/'status.json').read_text())
                    if state['status'] in ('stopped','completed_pending_evidence_review'):raise RuntimeError('curve stopped without required complete checkpoint')
                    time.sleep(60)
            atomic_save(root/'status.json',dict(status='completed_pending_evidence_review',competitive_strength_established=False))
        except BaseException as exc:
            atomic_save(root/'status.json',dict(status='stopped',reason=str(exc)));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT)
    p.add_argument('--curve',type=Path,default=CURVE);p.add_argument('--stage',type=int,default=1500)
    p.add_argument('--seed',type=int,default=5600000);p.add_argument('--blocks',type=int,default=1000)
    p.add_argument('--deadline',default=DEADLINE.isoformat());a=p.parse_args()
    main(a.output,a.curve,a.stage,a.seed,a.blocks,datetime.fromisoformat(a.deadline))
