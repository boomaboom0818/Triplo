"""Additional six-player screening against the other independent seed's policy."""
import json,time
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


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    with run_lock(ROOT/'supervisor'):
        if (ROOT/'manifest.json').exists():raise ValueError('existing check; investigate before recovery')
        original=json.loads((CURVE/'manifest.json').read_text())['parents']
        for entry in original.values():
            if digest(Path(entry['path']))!=entry['sha256']:raise ValueError('frozen1000 parent changed')
        atomic_save(ROOT/'manifest.json',dict(seed=5600000,blocks=1000,stage=1500,frozen_parents=original,deadline=DEADLINE.isoformat(),purpose='additional exploratory screening; does not replace original gate',format='one candidate vs five copies of other independent1000 learner',competitive_strength_established=False))
        pending=['candidate-0','candidate-1']
        try:
            while pending:
                if datetime.now(timezone.utc)>=DEADLINE:raise TimeoutError('deadline reached')
                progressed=False
                for name in list(pending):
                    candidate=CURVE/f'{name}-1500.json'
                    if not candidate.exists():continue
                    saved=load_checkpoint(candidate,model_id='bot-opponent-v1')
                    if saved['iterations']!=1500 or saved['stop_reason']!='target_reached':continue
                    other='candidate-1' if name=='candidate-0' else 'candidate-0'
                    parent=Path(original[name]['path']);opponent=Path(original[other]['path'])
                    old=load_checkpoint(parent);frozen=load_checkpoint(opponent)
                    verify_lineages(saved,old,frozen);del saved,old,frozen
                    atomic_save(ROOT/f'{name}-lineage.json',dict(candidate_sha256=digest(candidate),parent_sha256=digest(parent),opponent_sha256=digest(opponent)))
                    reports={}
                    for kind,path in [('candidate',candidate),('parent',parent),('uniform',candidate)]:
                        atomic_save(ROOT/'status.json',dict(status='evaluating',candidate=name,kind=kind))
                        reports[kind]=evaluate(path,ROOT/f'{name}-{kind}.json',opponent,5600000,kind=='uniform',DEADLINE.timestamp())
                    paired={k:paired_difference(reports['candidate'],reports[k]) for k in ('parent','uniform')}
                    atomic_save(ROOT/f'{name}-paired.json',paired)
                    pending.remove(name);progressed=True
                if pending and not progressed:
                    atomic_save(ROOT/'status.json',dict(status='waiting_for_complete1500',pending=pending))
                    state=json.loads((CURVE/'status.json').read_text())
                    if state['status'] in ('stopped','completed_pending_evidence_review'):raise RuntimeError('curve stopped without required complete checkpoint')
                    time.sleep(60)
            atomic_save(ROOT/'status.json',dict(status='completed_pending_evidence_review',competitive_strength_established=False))
        except BaseException as exc:
            atomic_save(ROOT/'status.json',dict(status='stopped',reason=str(exc)));raise

if __name__=='__main__':main()
