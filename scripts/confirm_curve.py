"""Fresh development deals for selected 1000-stage candidates; not a final test."""
import json,os,time
from datetime import datetime,timezone
from pathlib import Path
from poker.config import load_checkpoint
from poker.train import atomic_save
from poker.train_cfr import run_lock
from poker.compare import paired_difference
from scripts.sustained_train import evaluate,digest

ROOT=Path('runs/bot/confirmation-001')
CURVE=Path('runs/bot/sampling-curve-001')
PARENT=Path('runs/bot/sampling-001')
VALIDATION=Path('runs/bot/campaign-001/validation-250.json')
DEADLINE=datetime.fromisoformat('2026-11-14T17:31:00+00:00')
SEED=5500000
BLOCKS=2000


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    with run_lock(ROOT/'supervisor'):
        if (ROOT/'manifest.json').exists():raise ValueError('existing confirmation; inspect before recovery')
        atomic_save(ROOT/'manifest.json',dict(pid=os.getpid(),seed=SEED,blocks=BLOCKS,stage=1000,deadline=DEADLINE.isoformat(),validation_sha256=digest(VALIDATION),purpose='fresh development confirmation after selection on5300000',final_test=False,primary='one-copy paired learned-opponent differences versus250 parent and uniform; both lower95 bounds >0 per seed, plus coverage>=5%/200decisions in calling,mixed,learned; review regressions separately'))
        pending=['candidate-a','candidate-b'];decisions={}
        try:
            while pending:
                if datetime.now(timezone.utc)>=DEADLINE:raise TimeoutError('confirmation deadline reached')
                progressed=False
                for name in list(pending):
                    path=CURVE/f'{name}-1000.json'
                    if not path.exists():continue
                    saved=load_checkpoint(path,model_id='bot-opponent-v1')
                    if saved['iterations']!=1000 or saved['stop_reason']!='target_reached':continue
                    original=load_checkpoint(PARENT/f'{name}.json',model_id='bot-opponent-v1')
                    if saved['config']!=original['config']:raise ValueError('recipe changed')
                    del saved,original
                    atomic_save(ROOT/f'{name}-lineage.json',dict(candidate_path=str(path),candidate_sha256=digest(path),parent_path=str(PARENT/f'{name}.json'),parent_sha256=digest(PARENT/f'{name}.json')))
                    reports={}
                    for kind,checkpoint in [('candidate',path),('parent',PARENT/f'{name}.json'),('uniform',path)]:
                        atomic_save(ROOT/'status.json',dict(status='evaluating',candidate=name,kind=kind))
                        reports[kind]=evaluate(checkpoint,ROOT/f'{name}-{kind}.json',VALIDATION,SEED,kind=='uniform',DEADLINE.timestamp(),blocks=BLOCKS)
                    paired={k:paired_difference(reports['candidate'],reports[k]) for k in ('parent','uniform')}
                    atomic_save(ROOT/f'{name}-paired.json',paired)
                    opponent=reports['candidate']['opponent_checkpoint_hashes'][0];coverage={}
                    for population,v in reports['candidate']['populations'].items():
                        post=[x for s,x in v['coverage_by_street'].items() if s!='0'];n=sum(x['decisions'] for x in post);hits=sum(x['positive_weight_rows'] for x in post)
                        coverage[population]={'decisions':n,'positive':hits,'fraction':hits/n if n else 0}
                    decisions[name]=dict(coverage=coverage,payoff_pass=all(paired[k][opponent]['ci95'][0]>0 for k in paired),coverage_pass=all(coverage[k]['decisions']>=200 and coverage[k]['fraction']>=.05 for k in ('calling','mixed',opponent)))
                    atomic_save(ROOT/'evidence.json',decisions);pending.remove(name);progressed=True
                if pending and not progressed:
                    atomic_save(ROOT/'status.json',dict(status='waiting_for_checkpoints',pending=pending))
                    state=json.loads((CURVE/'status.json').read_text())
                    if state['status'] in ('stopped','completed_pending_evidence_review'):raise RuntimeError('main curve stopped without complete1000 checkpoints')
                    time.sleep(60)
            atomic_save(ROOT/'status.json',dict(status='completed_pending_evidence_review',both_seeds_pass=all(v['coverage_pass'] and v['payoff_pass'] for v in decisions.values()),competitive_strength_established=False))
        except BaseException as exc:
            atomic_save(ROOT/'status.json',dict(status='stopped',reason=str(exc)));raise

if __name__=='__main__':main()
