"""Fresh bounded sampling pilot; evidence review precedes sustained training."""
import json,hashlib,os
from pathlib import Path
from scripts.continuation_control import run
from poker.train import atomic_save
from poker.train_cfr import run_lock
from poker.compare import paired_difference

ROOT=Path('runs/bot/sampling-001')
BASE=Path('runs/bot/continuation-001')
VALIDATION=Path('runs/bot/campaign-001/validation-250.json')

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    with run_lock(ROOT/'supervisor'):
        if (ROOT/'manifest.json').exists():raise ValueError('existing pilot; inspect before recovery')
        atomic_save(ROOT/'manifest.json',dict(pid=os.getpid(),stage=250,seeds=[11,29],recipe='call-mixture-v1',memory_limit_bytes=3*1024**3,evaluation_seed=5200000,blocks=1000,validation_sha256=hashlib.sha256(VALIDATION.read_bytes()).hexdigest(),gate='5% postflop coverage in calling/mixed/learned populations, >=200 postflop decisions each, paired learned-opponent improvement over uniform and v2 control with lower 95% bound >0 for both seeds; review required'))
        try:
            decisions={}
            for name in ('candidate-a','candidate-b'):
                cfg=json.loads((BASE/f'{name}-texture-buckets-v2-250-config.json').read_text())
                cfg.update(averaging_sampler='call-mixture-v1',max_seconds=3600)
                config=ROOT/f'{name}-config.json';atomic_save(config,cfg)
                checkpoint=ROOT/f'{name}.json'
                atomic_save(ROOT/'status.json',dict(status='training',candidate=name))
                run(['poker.train_cfr','--config',str(config),'--output',str(checkpoint)],ROOT/f'{name}.log')
                saved=json.loads(checkpoint.read_text())
                if saved['iterations']!=250:raise RuntimeError(f"{name} stopped at {saved['iterations']}: {saved['stop_reason']}")
                del saved
                reports={}
                for kind,path in [('candidate',checkpoint),('uniform',checkpoint),('control',BASE/f'{name}-texture-buckets-v2-250.json')]:
                    atomic_save(ROOT/'status.json',dict(status='evaluating',candidate=name,kind=kind))
                    output=ROOT/f'{name}-{kind}-evaluation.json'
                    args=['poker.evaluate','--checkpoint',str(path),'--blocks','1000','--seed','5200000','--copies','1','--opponent-checkpoints',str(VALIDATION)]
                    if kind=='uniform':args+=['--uniform-control']
                    run(args,output);reports[kind]=json.loads(output.read_text())
                pairs={k:paired_difference(reports['candidate'],reports[k]) for k in ('uniform','control')}
                atomic_save(ROOT/f'{name}-paired.json',pairs)
                coverage={}
                for pop,v in reports['candidate']['populations'].items():
                    post=[x for street,x in v['coverage_by_street'].items() if street!='0']
                    count=sum(x['decisions'] for x in post);hits=sum(x['positive_weight_rows'] for x in post)
                    coverage[pop]=dict(decisions=count,positive=hits,fraction=hits/count if count else 0)
                opponent=reports['candidate']['opponent_checkpoint_hashes'][0]
                decisions[name]=dict(coverage=coverage,coverage_pass=all(x['decisions']>=200 and x['fraction']>=.05 for pop,x in coverage.items() if pop in ('calling','mixed',opponent)),payoff_pass=all(pairs[k][opponent]['ci95'][0]>0 for k in pairs))
                atomic_save(ROOT/'evidence.json',decisions)
            atomic_save(ROOT/'status.json',dict(status='completed_pending_evidence_review',gate_pass=all(x['coverage_pass'] and x['payoff_pass'] for x in decisions.values()),sustained_training_started=False))
        except BaseException as exc:
            atomic_save(ROOT/'status.json',dict(status='stopped',reason=str(exc)))
            raise

if __name__=='__main__':main()
