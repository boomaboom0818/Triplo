"""Finite staged aggressive curriculum; development readiness precedes holdout.

Existing confirmation reports are never used in training/selection. No live
checkpoint is replaced by this script. Sources, controls and snapshots are frozen.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from statistics import mean, NormalDist
from scripts.live_gate import assess, digest, interval


def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2)


def readiness(candidate,champion,fallback,alpha):
    assessment=assess(candidate,champion,fallback,confirmation=False,family_alpha=alpha)
    names=[name for name in candidate['cases'] if 'aggressive' in name]
    n=candidate['deals']
    values=lambda report:[mean(report['cases'][name]['blocks'][i] for name in names) for i in range(n)]
    c=values(candidate);z=NormalDist().inv_cdf(1-alpha/2)
    target={label:interval([a-b for a,b in zip(c,values(control))],z)
            for label,control in (('incumbent',champion),('fallback',fallback))}
    ready=all(assessment['checks'][k] for k in ('aggregate_profit','beats_incumbent','beats_fallback',
            'no_material_case_regression','learned_coverage','postflop_coverage','latency')) and all(v['lower']>0 for v in target.values())
    return dict(ready_for_confirmation=ready,gate=assessment,aggressive_advantage=target)


def run(root):
    root=Path(root);m=json.loads((root/'manifest.json').read_text())
    if (root/'result.json').exists():raise ValueError('campaign already complete')
    def verify():
        for p,h in m['sources'].items():
            if digest(p)!=h:raise ValueError('source changed: '+p)
        if digest(m['incumbent'])!=m['incumbent_sha256']:raise ValueError('incumbent changed')
    verify()
    def command(args,tag):
        with (root/(tag+'.log')).open('a') as f:
            subprocess.run([sys.executable,'-m',*args],stdout=f,stderr=subprocess.STDOUT,check=True)
    def evaluate(path,tag,seed,deals,fallback=False):
        output=root/(tag+'.json')
        if not output.exists():
            args=['scripts.live_gate','evaluate','--checkpoint',str(path),'--incumbent',m['incumbent'],
                  '--seed',str(seed),'--deals',str(deals),'--out',str(output)]
            if fallback:args.append('--fallback-only')
            command(args,tag)
        r=json.loads(output.read_text())
        if r['seed']!=seed or r['deals']!=deals or r['fallback_only']!=fallback or r['incumbent_sha256']!=m['incumbent_sha256']:
            raise ValueError('cached evaluation mismatch')
        if not fallback and r['checkpoint_sha256']!=digest(path):raise ValueError('cached policy changed')
        return r
    incumbent=Path(m['incumbent'])
    working=root/'training';previous=0;stages={}
    parent_iterations=json.loads((Path(m['parent_state'])/'meta.json').read_text())['iterations']
    for minutes in m['stages_minutes']:
        snapshot=root/f'stage-{minutes}.json'
        if not snapshot.exists():
            args=['scripts.history_train','--root',str(working),'--minutes',str(minutes-previous),
                  '--seed',str(m['seed']),'--stack-bb','100','--workers',str(m['workers']),
                  '--population',str(m['population']),'--curriculum',m['curriculum'],
                  '--abstraction','history-compact-v1','--max-rows',str(m['max_rows'])]
            if previous:args+=['--resume']
            else:args+=['--initialize-from',m['parent_state']]
            command(args,f'train-{minutes}')
            progress=json.loads((working/'progress.json').read_text())
            if progress['iterations']<=parent_iterations or progress['stop_reason']!='deadline':
                write(root/f'stop-{minutes}.json',progress)
                raise RuntimeError('training stopped before planned stage; inspect resource/row limits')
            shutil.copyfile(working/'checkpoint.json',snapshot)
            write(root/f'progress-{minutes}.json',progress)
        previous=minutes
        seed=m['development_seed'];deals=m['development_deals']
        champion=evaluate(incumbent,'development-incumbent',seed,deals)
        fallback=evaluate(incumbent,'development-fallback',seed,deals,True)
        candidate=evaluate(snapshot,f'development-stage-{minutes}',seed,deals)
        stages[str(minutes)]=readiness(candidate,champion,fallback,m['family_alpha'])
        verify()
        print('stage',minutes,json.dumps(stages[str(minutes)]['aggressive_advantage']),flush=True)
    eligible=[s for s in stages if stages[s]['ready_for_confirmation']]
    # No stage qualifies: retain the stronger development snapshot for analysis,
    # leave the fresh confirmation seed untouched and the live champion unchanged.
    selected=max(stages,key=lambda s:min(v['mean'] for v in stages[s]['aggressive_advantage'].values()))
    result=dict(promote=False,confirmation_consumed=False,selected_stage=selected,stages=stages,
                reason='No stage passed development readiness; holdout left untouched')
    if eligible:
        selected=max(eligible,key=lambda s:min(v['mean'] for v in stages[s]['aggressive_advantage'].values()))
        snapshot=root/f'stage-{selected}.json'
        # Fresh development validation guards against selecting noise over stages.
        seed=m['validation_seed'];deals=m['validation_deals']
        champion=evaluate(incumbent,'validation-incumbent',seed,deals)
        fallback=evaluate(incumbent,'validation-fallback',seed,deals,True)
        candidate=evaluate(snapshot,'validation-selected',seed,deals)
        validation=readiness(candidate,champion,fallback,m['family_alpha'])
        result.update(selected_stage=selected,validation=validation,reason='Development validation failed; holdout left untouched')
        if validation['ready_for_confirmation']:
            write(root/'selection.json',dict(stage=selected,sha256=digest(snapshot)))
            seed=m['confirmation_seed'];deals=m['confirmation_deals']
            champion=evaluate(incumbent,'confirmation-incumbent',seed,deals)
            fallback=evaluate(incumbent,'confirmation-fallback',seed,deals,True)
            candidate=evaluate(snapshot,'confirmation-selected',seed,deals)
            promotion=assess(candidate,champion,fallback,family_alpha=m['family_alpha'])
            write(root/'promotion.json',promotion)
            result.update(promote=promotion['promote'],confirmation_consumed=True,reason='Fresh confirmation complete')
    verify();write(root/'result.json',result)
    print(json.dumps(dict(promote=result['promote'],confirmation_consumed=result['confirmation_consumed'],reason=result['reason'])),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    run(p.parse_args().root)
